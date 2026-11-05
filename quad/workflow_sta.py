"""Placement-RC electrical STA with GR timing anchors and unique-origin queues.

The supported SDC scope is the shipped AES/JPEG single physical clock, virtual
IO clock, and source-class ignore_clock_latency constraints. Generated clocks
and general through exceptions require a richer tag exporter. The entry point
restricts inputs to the shipped benchmarks. Min/max clock changes are paired
in the forward evaluator; placement edits refresh trees within an epoch.
"""
import copy
import time
import numpy as np
import torch
from autograd_cuda_sta import AutogradCudaSTA
from forward_sta import ForwardModel, rows, RF
from gpu_calibration import calibrate_gpu
from minmax_caps import load_ranges
from tagged_arrival import TaggedArrival
from ratio_sta import fit_ratio
from workflow_clock import shared_clock_lca


def reanchor_electrical(engine, native):
    """Refresh local timing labels on an existing placement tree and GPU graph.

    The reference supplies pin slew/arrival and arc/check delay, never RC.
    Reference pin inputs represent the preceding calibrated level, so the
    local LUT fits are parallel. The subsequent full forward is the audit.
    """
    old=engine.m
    if old.index.keys()!=native.index.keys():raise ValueError('Reference changed pin topology')
    # OpenDB can reorder an instance's pins after swapMaster/write_def. Align
    # by logical pin and arc identities, never by export row number.
    order=np.array([native.index[p['name']] for p in old.pins])
    if not np.array_equal(old.level,native.level[order]):raise ValueError('Reference changed graph levels')
    aligned=copy.copy(native);aligned.index=old.index.copy();aligned.pins=[native.pins[i] for i in order]
    aligned.ref_at=native.ref_at[order].copy();aligned.ref_slew=native.ref_slew[order].copy();aligned.level=native.level[order].copy()
    for field in ('arcs','checks'):
        original=getattr(old,field);reference=getattr(native,field)
        labels={arc_key(native,a):a['delay'] for a in reference}
        if len(original)!=len(reference) or {arc_key(old,a) for a in original}!=labels.keys():
            raise ValueError('Reference changed arc topology')
        setattr(aligned,field,[dict(a,delay=labels[arc_key(old,a)]) for a in original])
    native=aligned
    m=copy.copy(old)
    m.ref_at=native.ref_at.copy();m.ref_slew=native.ref_slew.copy()
    m.arcs=[dict(a,delay=b['delay']) for a,b in zip(old.arcs,native.arcs)]
    m.checks=[dict(a,delay=b['delay']) for a,b in zip(old.checks,native.checks)]
    m.endpoints=copy.deepcopy(native.endpoints)
    engine.set_model(m)
    sl=engine.tensor(m.ref_slew.reshape(-1));load=engine.load0.reshape(-1)
    delay=engine.zeros(len(m.arcs));transition=engine.zeros(len(m.arcs))
    cell,wire=engine.cells,engine.wires
    delay[cell]=engine.lut(engine.delay_tab[cell],sl[engine.s[cell]],load[engine.t[cell]])
    transition[cell]=engine.lut(engine.slew_tab[cell],sl[engine.s[cell]],load[engine.t[cell]]).clamp_min(0)
    impulse=engine.impulse0 if engine.wire_moment_model else engine.tau0
    delay[wire],transition[wire]=engine.wire_values(engine.tau0[wire],impulse[wire],sl[engine.s[wire]],wire)
    trans=engine.numpy(engine.peak(transition,engine.t,2*m.n).clamp_min(0)).reshape(-1,2)
    m.arc_delay_offset=np.array([a['delay'] for a in m.arcs])-engine.numpy(delay)
    m.at_ratio=np.ones_like(m.ref_at);m.at_offset=np.zeros_like(m.ref_at)
    m.slew_ratio,m.slew_offset,_=fit_ratio(trans,m.ref_slew)
    raw=engine.numpy(engine.lut(engine.check_tab,sl[engine.ct],sl[engine.cs]))
    m.check_ratio,m.check_offset,_=fit_ratio(raw,np.array([a['delay'] for a in m.checks]))
    engine.set_model(m)
    if m.analysis=='min':
        engine.capture_late_slew=sl[engine.cs]
        engine._replay_graph=None
    engine.sync()
    return native


def arc_key(model, arc):
    return (model.pins[arc['s']]['name'],model.pins[arc['t']]['name'],
            arc['fr'],arc['to'],arc['role'],arc['cond'])


def build_electrical(folder, catalog, tables, masters, analysis, cap_ranges,shared_rc=None):
    """Only native PLACEMENT SPEF is parsed by calibrate_gpu/build_sensitivity."""
    base = ForwardModel(folder/'placement'/analysis,catalog,masters,tables,analysis)
    native = ForwardModel(folder/'gr'/analysis,catalog,masters,tables,analysis)
    if base.index != native.index: raise ValueError('Reference/placement graph indexing differs')
    reference_arcs = {arc_key(native,a):a['delay'] for a in native.arcs+native.checks}
    for a in base.arcs+base.checks: a['delay']=reference_arcs[arc_key(base,a)]
    base.ref_at=native.ref_at.copy();base.ref_slew=native.ref_slew.copy()
    base.endpoints=copy.deepcopy(native.endpoints);base.cap_ranges=cap_ranges
    model, engine = calibrate_gpu(base,mode='arc_offset',shared_rc=shared_rc)
    # Origin-aware propagation does not permit pin-level arrival translations.
    model.at_ratio[:]=1.;model.at_offset[:]=0.
    engine.set_model(model)
    return model,engine,native


class TaggedEndpoints:
    def __init__(self, engine, geometry, endpoints, late_reference, early_reference, k=16):
        self.e=engine;self.m=m=engine.m;self.analysis=m.analysis
        self.rows=list(endpoints);self.k=k
        n=2*m.n
        source=np.array([2*a['s']+a['fr'] for a in m.arcs],dtype=np.int32)
        target=np.array([2*a['t']+a['to'] for a in m.arcs],dtype=np.int32)
        levels=np.repeat(m.level,2)
        physical={p['pin']:p for p in geometry}
        clock_roots=[i for i,p in enumerate(m.pins) if not m.incoming[i]
                     and p['name'] in physical and physical[p['name']]['port']=='1'
                     and physical[p['name']]['clock']=='1']
        if len(clock_roots)!=1: raise ValueError('Workflow requires one physical clock root')
        root=clock_roots[0]
        reach=np.zeros((n,2),dtype=bool);reach[2*root,0]=True;reach[2*root+1,1]=True
        clock_parent=np.full(n,-1,dtype=np.int32)
        for node in np.argsort(m.level,kind='stable'):
            for aidx in m.incoming[node]:
                a=m.arcs[aidx];s=source[aidx];t=target[aidx]
                if a['role']=='Reg Clk to Q':continue
                if reach[s].any():
                    if clock_parent[t]>=0 and clock_parent[t]!=s:
                        raise ValueError('Reconvergent clock network requires explicit clock-path tags')
                    clock_parent[t]=s;reach[t]|=reach[s]
        tags={};names=[];clock_nodes=[];tag_groups=[]
        def tag(name,group,clock):
            key=(name,group)
            if key not in tags:
                tags[key]=len(names);names.append(name);clock_nodes.append(clock);tag_groups.append(group)
            return tags[key]
        launch=np.full(len(m.arcs),-1,dtype=np.int32)
        launch_group=launch.copy();root_tag=np.full(n,-1,dtype=np.int32)
        for i,p in enumerate(m.pins):
            if m.incoming[i] or i==root:continue
            info=physical.get(p['name'])
            if info and info['port']=='1' and info['direction']=='INPUT':
                ident=tag(p['name'],0,-1);root_tag[2*i:2*i+2]=ident
        for i,a in enumerate(m.arcs):
            if a['role']!='Reg Clk to Q':continue
            edges=np.flatnonzero(reach[source[i]])
            if len(edges)!=1: raise ValueError('Ambiguous launch clock edge')
            group=1+int(edges[0]);launch_group[i]=group
            launch[i]=tag(m.pins[a['t']]['name'],group,int(source[i]))
        self.names=names;self.tag_groups=np.array(tag_groups);self.tag_clock=np.array(clock_nodes)
        group_count=5 if any(r['source_group']!='input' and int(r['is_path_delay'])
                            and not r['capture_pin'] for r in self.rows) else 3
        self.runtime=TaggedArrival(source,target,levels,launch,launch_group,root_tag,
                                   k=k,groups=group_count,analysis=m.analysis,device=engine.device)
        group_map={'input':0,'clock_rise':1,'clock_fall':2}
        groups=[];nodes=[];capture=[];capture_valid=[];ref_tags=[]
        for row in self.rows:
            group=group_map[row['source_group']]
            normalized=group>0 and bool(int(row['is_path_delay'])) and not row['capture_pin']
            groups.append(group+2 if normalized else group)
            nodes.append(2*m.index[row['pin']]+RF[row['rf']])
            cp=row['capture_pin'];crf=row['capture_rf']
            capture.append(2*m.index[cp]+RF[crf] if cp else 2*root)
            capture_valid.append(bool(cp) and not int(row['is_path_delay']))
            ref_tags.append(tags.get((row['startpoint'],group),-1))
        self.ep_nodes=engine.tensor(nodes,True);self.ep_groups=engine.tensor(groups,True)
        self.ep_pin=self.ep_nodes//2;self.capture=engine.tensor(capture,True)
        self.capture_valid=engine.tensor(capture_valid,True).bool()
        self.ref_tags=engine.tensor(ref_tags,True)
        if any(t<0 for t in ref_tags):
            bad=[r['startpoint'] for r,t in zip(self.rows,ref_tags) if t<0]
            raise ValueError('Unrepresented reference startpoints: '+str(bad[:8]))
        self.required=engine.tensor([float(r['required_ps']) for r in self.rows])
        self.reference_slack=engine.tensor([float(r['slack_ps']) for r in self.rows])
        self.late_ref=engine.tensor(late_reference.ref_at.reshape(-1))
        self.early_ref=engine.tensor(early_reference.ref_at.reshape(-1))
        # Common clock prefixes are shared between all endpoints on one clock
        # pin. Compact unique capture rows avoid E x S duplicate tables.
        unique, inverse=np.unique(capture,return_inverse=True)
        self.lca=shared_clock_lca(clock_parent,unique,clock_nodes,levels,engine.device)
        self.capture_group=engine.tensor(inverse,True)
        initial=self._cppr(self.ref_tags[:,None],self.late_ref,self.early_ref).squeeze(1)
        self.reference_cppr=initial
        check_ep=[];check_id=[]
        for ep,row in enumerate(self.rows):
            pin=m.index[row['pin']];rf=RF[row['rf']]
            for aid in m.check_incoming[pin]:
                a=m.checks[aid]
                if a['to']==rf and (not row['capture_pin'] or 2*a['s']+a['fr']==capture[ep]):
                    check_ep.append(ep);check_id.append(aid)
        self.check_ep=engine.tensor(check_ep,True);self.check_id=engine.tensor(check_id,True)
        self.reference_margin=self.margin(engine.tensor([a['delay'] for a in m.checks]))
        self.last=None

    def margin(self, checks):
        values=torch.full((len(self.rows),),-torch.inf,dtype=checks.dtype,device=checks.device)
        values=values.scatter_reduce(0,self.check_ep,checks[self.check_id],reduce='amax',include_self=True)
        return torch.where(torch.isfinite(values),values,0.)

    def _cppr(self, tags, late, early):
        nodes=self.lca[self.capture_group[:,None],tags.clamp_min(0).long()]
        spread=(late[nodes.clamp_min(0)]-early[nodes.clamp_min(0)]).clamp_min(0.)
        return torch.where((nodes>=0)&(tags>=0)&self.capture_valid[:,None],spread,0.)

    def evaluate(self, delay, at, slew, checks, late_at=None, early_at=None):
        arrivals,tags,overflow=self.runtime(delay,at.reshape(-1))
        values=arrivals[self.ep_nodes,self.ep_groups]
        origins=tags[self.ep_nodes,self.ep_groups]
        late=at.reshape(-1) if self.analysis=='max' else (self.late_ref if late_at is None else late_at.reshape(-1))
        early=at.reshape(-1) if self.analysis=='min' else (self.early_ref if early_at is None else early_at.reshape(-1))
        capture=early if self.analysis=='max' else late
        reference_capture=self.early_ref if self.analysis=='max' else self.late_ref
        dc=torch.where(self.capture_valid,capture[self.capture]-reference_capture[self.capture],0.)
        dm=self.margin(checks)-self.reference_margin
        sign=1. if self.analysis=='max' else -1.
        required=self.required+dc-sign*dm
        required=required[:,None]+sign*(self._cppr(origins,late,early)-self.reference_cppr[:,None])
        slack=sign*(required-values)
        slack=torch.where(origins>=0,slack,torch.inf)
        worst,choice=torch.min(slack,dim=1)
        selected=origins.gather(1,choice[:,None]).squeeze(1)
        req=required.gather(1,choice[:,None]).squeeze(1)
        pin=torch.full((self.m.n,),torch.inf,dtype=at.dtype,device=at.device)
        pin=pin.scatter_reduce(0,self.ep_pin,worst,reduce='amin',include_self=True)
        self.last=dict(missing_reference=(~(origins==self.ref_tags[:,None]).any(1)),
                       overflow=overflow,selected=selected,slack=worst)
        return dict(slack=worst,tns=torch.relu(-pin).sum(),wns=torch.relu(-worst.min()),
            endpoints=torch.stack((self.ep_pin,self.ep_nodes%2,req,worst),dim=1),
            selected_startpoints=selected)

    def diagnostics(self):
        if self.last is None:return {}
        delta=(self.last['slack']-self.reference_slack).abs()
        return dict(k=self.k,reference_startpoint_misses=int(self.last['missing_reference'].sum()),
            overflowing_queues=int(self.last['overflow'].sum()),finite=bool(torch.isfinite(delta).all()),
            mae_ps=float(delta.mean()),p99_abs_ps=float(torch.quantile(delta,.99)),
            max_abs_ps=float(delta.max()))


class WorkflowAutogradSTA(AutogradCudaSTA):
    export_local_delays=True

    def _graph(self, branch, epsilon, gamma, power=False):
        graph=super()._graph(branch,epsilon,gamma,power)
        if not power:
            # Unique-origin selection has its own exact piecewise derivative.
            # Ave/LSE, if requested, only smooth compatible electrical merges.
            graph.update(self.tagged.evaluate(graph['delay'],graph['at'],graph['slew'],graph['check']))
        return graph

    def tagged_forward(self,replacements=None,early_at=None):
        r=super().fast_forward(replacements,device_output=True)
        tagged=self.tagged.evaluate(r['delay'],r['at'],r['slew'],r['check'],early_at=early_at)
        r.update(endpoints=tagged['endpoints'],tns_ps=-tagged['tns'],wns_ps=-tagged['wns'])
        return r

    def prepare(self):
        r=super().prepare()
        tagged=self.tagged.evaluate(self.tensor(r['delay']),self.tensor(r['at']),self.tensor(r['slew']),self.tensor(r['check']))
        r.update(endpoints=self.numpy(tagged['endpoints']),tns_ps=-float(tagged['tns']),wns_ps=-float(tagged['wns']))
        self.b.forward=r
        return r


class WorkflowCorner:
    def __init__(self,folder,catalog,tables,masters,geometry,cap_ranges,k=16,max_k=128,shared_rc=None):
        t=time.perf_counter()
        lm,_,ln=build_electrical(folder,catalog,tables,masters,'max',cap_ranges,shared_rc)
        em,early,en=build_electrical(folder,catalog,tables,masters,'min',cap_ranges,lm)
        self.sensitivity_reference=copy.copy(lm)
        late=WorkflowAutogradSTA(lm)
        self.late,self.early=late,early
        self.native_late,self.native_early=ln,en
        self.rows={a:rows(folder/'groups'/a/'endpoints.tsv') for a in ('max','min')}
        self.geometry=geometry;self.max_k=max_k
        self.install_tags(k)
        # Increase K when the reference source was pruned. Arrival-only queues
        # cannot promise exact CPPR selection; misses remain explicit at max K.
        while True:
            self.forward()
            ds=self.diagnostics()
            if not any(v['reference_startpoint_misses'] for v in ds.values()) or k>=max_k:break
            k=min(max_k,k*2);self.install_tags(k)
        self.initialization_seconds=time.perf_counter()-t

    def install_tags(self,k):
        for analysis,e in [('max',self.late),('min',self.early)]:
            e.tagged=TaggedEndpoints(e,self.geometry,self.rows[analysis],self.native_late,self.native_early,k)

    def reanchor(self,folder):
        """Same-topology checkpoint: retain CUDA tables, graph and RC operator."""
        native={a:ForwardModel(folder/'gr'/a,self.late.m.catalog,self.late.m.masters,
                              self.late.m.tables,a) for a in ('max','min')}
        for a,e in [('max',self.late),('min',self.early)]:native[a]=reanchor_electrical(e,native[a])
        self.native_late,self.native_early=native['max'],native['min']
        self.rows={a:rows(folder/'groups'/a/'endpoints.tsv') for a in ('max','min')}
        k=self.late.tagged.k
        while True:
            self.install_tags(k);self.forward()
            if not any(v['reference_startpoint_misses'] for v in self.diagnostics().values()) or k>=self.max_k:break
            k=min(self.max_k,k*2)

    def level_audit(self,predictions):
        result={}
        clock_names={g['pin'] for g in self.geometry if g['clock']=='1'}
        clock=np.array([p['name'] in clock_names for p in self.late.m.pins])
        for a,e,native in [('max',self.late,self.native_late),('min',self.early,self.native_early)]:
            levels=[]
            for level in np.unique(e.m.level):
                mask=e.m.level==level;record={'level':int(level)}
                for key,ref in [('at',native.ref_at),('slew',native.ref_slew)]:
                    predicted=predictions[a][key][mask];expected=ref[mask]
                    finite=np.isfinite(predicted)&np.isfinite(expected)
                    valid=np.abs(predicted[finite]-expected[finite])
                    record[key+'_finite_mismatches']=int(np.count_nonzero(np.isfinite(predicted)^np.isfinite(expected)))
                    record[key+'_mae_ps']=float(valid.mean()) if len(valid) else 0.
                    record[key+'_max_ps']=float(valid.max()) if len(valid) else 0.
                levels.append(record)
            d=np.abs(predictions[a]['at'][clock]-native.ref_at[clock]);d=d[np.isfinite(d)]
            result[a]=dict(levels=levels,clock_arrival_mae_ps=float(d.mean()) if len(d) else 0.,
                           clock_arrival_max_ps=float(d.max()) if len(d) else 0.)
            skews={}
            for group in (1,2):
                nodes=np.unique(e.tagged.tag_clock[e.tagged.tag_groups==group])
                nodes=nodes[nodes>=0]
                if not len(nodes):continue
                predicted=predictions[a]['at'].reshape(-1)[nodes];reference=native.ref_at.reshape(-1)[nodes]
                skews['rise' if group==1 else 'fall']=dict(predicted_ps=float(np.ptp(predicted)),
                    reference_ps=float(np.ptp(reference)),error_ps=float(np.ptp(predicted)-np.ptp(reference)))
            result[a]['clock_sink_skew']=skews
        return result

    def forward(self,replacements=None):
        late=self.late.fast_forward(replacements,device_output=True)
        early=self.early.fast_forward(replacements,device_output=True)
        result={}
        for analysis,e,r in [('max',self.late,late),('min',self.early,early)]:
            tagged=e.tagged.evaluate(r['delay'],r['at'],r['slew'],r['check'],late['at'],early['at'])
            r.update(endpoints=tagged['endpoints'],tns_ps=-tagged['tns'],wns_ps=-tagged['wns'])
            result[analysis]=e._export_tensors(r)
        return result

    def diagnostics(self):
        return {'max':self.late.tagged.diagnostics(),'min':self.early.tagged.diagnostics()}
