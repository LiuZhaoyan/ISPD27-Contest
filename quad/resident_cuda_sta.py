"""GPU-resident levelized RF timing, analytic LUT VJP, and cached action queries.

No CPU full forward or CPU derivative tape is required by prepare().
CPU handles graph/library metadata; conditional-partition candidate propagation
explicitly falls back to the validated CPU envelope model. All measured APIs
synchronize before returning NumPy results.
"""
from types import SimpleNamespace
import time
import numpy as np
import torch
from cuda_sta import CudaSTA
from forward_sta import RF,MAX_CHECKS

class ResidentCudaSTA(CudaSTA):
    _banks={}
    def __init__(self,m,device='cuda:0'):
        self.m=m;self.device=device;self.fallback_count=0;self.action_cache={};self.b=SimpleNamespace(model=m,instance_arcs={},instance_checks={},instance_pins={})
        for p,(inst,_) in enumerate(m.pin_parts):
            if inst is not None:self.b.instance_pins.setdefault(inst,[]).append(p)
        for k,a in enumerate(m.arcs):
            if a['role']!='wire':self.b.instance_arcs.setdefault(m.pin_parts[a['t']][0],[]).append(k)
        for k,a in enumerate(m.checks):self.b.instance_checks.setdefault(m.pin_parts[a['t']][0],[]).append(k)
        self.s=self.tensor([a['s']*2+a['fr'] for a in m.arcs],True);self.t=self.tensor([a['t']*2+a['to'] for a in m.arcs],True)
        self.cs=self.tensor([a['s']*2+a['fr'] for a in m.checks],True);self.ct=self.tensor([a['t']*2+a['to'] for a in m.checks],True)
        wire=np.array([a['role']=='wire' for a in m.arcs]);self.wire=self.tensor(wire,True).bool();self.wires=self.tensor(np.flatnonzero(wire),True);self.cells=self.tensor(np.flatnonzero(~wire),True)
        port=np.asarray(getattr(m,'port_wire_parameters',np.zeros((len(m.arcs),2))))
        if port.shape!=(len(m.arcs),2) or not np.isfinite(port).all() or np.any(port<0):
            raise ValueError('Invalid primary-input wire threshold parameters')
        self.port_delay_factor=self.tensor(port[:,0]);self.port_slew_factor=self.tensor(port[:,1])
        self.port_wire=self.port_delay_factor>0
        levels=np.array([m.level[a['t']] for a in m.arcs]);self.plan=[];self.levels=[]
        has_incoming=np.array([bool(ids) for ids in m.incoming])
        for lev in np.unique(levels):
            ids=np.flatnonzero(levels==lev);nodes=np.flatnonzero((m.level==lev)&has_incoming);flat=(nodes[:,None]*2+np.array([0,1])).reshape(-1)
            ai=self.tensor(ids,True);self.levels.append(ai);self.plan.append((ai,self.tensor(ids[~wire[ids]],True),self.tensor(ids[wire[ids]],True),self.tensor(flat,True)))
        self.roots=self.tensor(np.array([i*2+rf for i in range(m.n) if not m.incoming[i] for rf in (0,1)]),True)
        bundles={};bi=[];bt=[]
        for a in m.arcs:
            key=(a['t'],a['to'],a['s'],a['fr'],a['role'])
            if key not in bundles:bundles[key]=len(bt);bt.append(a['t']*2+a['to'])
            bi.append(bundles[key])
        self.bundle=self.tensor(bi,True);self.bt=self.tensor(bt,True);self.nb=len(bt)
        bankkey=(id(m.tables),device);cached=self._banks.get(bankkey)
        if cached is None or cached[0] is not m.tables:
            tids={};tabs=[]
            for key,group in m.tables.items():
                # Frozen clock/register/tie cells still participate in timing.
                # Freeze masks restrict actions, not the propagation library.
                for tab in group.values():
                    if id(tab) not in tids:tids[id(tab)]=len(tabs);tabs.append(tab)
            if len({t.values.shape for t in tabs})!=1:raise ValueError('Nonuniform timing table shape')
            cached=(m.tables,tids,self.tensor([t.x for t in tabs]),self.tensor([t.y for t in tabs]),self.tensor([t.values for t in tabs]));self._banks[bankkey]=cached
        _,self.table_ids,self.tx,self.ty,self.z=cached
        coo=m.rc_matrix.tocoo();indices=self.tensor(np.array([coo.row,coo.col]),True)
        self.rc=torch.sparse_coo_tensor(indices,self.tensor(coo.data),coo.shape,device=device).coalesce();self.rc_t=self.rc.transpose(0,1).coalesce()
        self.netnames=list(m.by_net);nid=np.empty(m.n,dtype=np.int64)
        for i,net in enumerate(self.netnames):nid[m.by_net[net]]=i
        self.netid=self.tensor(nid,True);self.nn=len(self.netnames);self.sink=self.tensor(m.rf_sink,True).bool()
        self.ep_node=self.tensor([m.index[e['pin']]*2+RF[e['rf']] for e in m.endpoints],True)
        self.ep_pin=self.ep_node//2;self.ep_setup=self.tensor([e['role'] in MAX_CHECKS for e in m.endpoints],True).bool()
        self.ep_req=self.tensor([float(e['required_ps'])+float(e.get('path_at_shift_ps',0.))+(float(e['margin_ps']) if e['role'] in MAX_CHECKS else 0) for e in m.endpoints])
        self.set_model(m)

    def set_model(self,m):
        """Rebind a same-topology working point; topology changes need a new engine."""
        if hasattr(self,'source_groups') and getattr(m,'calibration_mode','ratio')!='arc_offset':
            raise ValueError('Source groups require frozen local arc corrections')
        if len(m.arcs)!=len(self.m.arcs) or m.n!=self.m.n or len(m.checks)!=self.cs.numel():raise ValueError('Topology changed: rebuild GPU graph')
        changed=[]
        self.m=m;self.b.model=m
        if hasattr(self,'_table_masters'):
            if self._table_masters.keys()!=m.masters.keys():raise ValueError('Instance topology changed: rebuild GPU graph')
            changed=[inst for inst,master in m.masters.items() if self._table_masters[inst]!=master]
            indices=[];delay_ids=[];slew_ids=[];check_indices=[];check_ids=[]
            for inst in changed:
                for k in self.b.instance_arcs.get(inst,[]):
                    a=m.arcs[k];tab=m.table(a,m.masters);rf=('rise','fall')[a['to']]
                    indices.append(k);delay_ids.append(self.table_ids[id(tab['cell_'+rf])]);slew_ids.append(self.table_ids[id(tab[rf+'_transition'])])
                for k in self.b.instance_checks.get(inst,[]):
                    a=m.checks[k];check_indices.append(k);check_ids.append(self.table_ids[id(m.table(a,m.masters)[('rise','fall')[a['to']]+'_constraint'])])
            if indices:
                index=self.tensor(indices,True)
                self.delay_tab[index]=self.tensor(delay_ids,True);self.slew_tab[index]=self.tensor(slew_ids,True)
            if check_indices:self.check_tab[self.tensor(check_indices,True)]=self.tensor(check_ids,True)
        else:
            delay=np.full(len(m.arcs),-1);slew=delay.copy()
            for k,a in enumerate(m.arcs):
                if a['role']=='wire':continue
                tab=m.table(a,m.masters);rf=('rise','fall')[a['to']];delay[k]=self.table_ids[id(tab['cell_'+rf])];slew[k]=self.table_ids[id(tab[rf+'_transition'])]
            self.delay_tab=self.tensor(delay,True);self.slew_tab=self.tensor(slew,True);self.check_tab=self.tensor([self.table_ids[id(m.table(a,m.masters)[('rise','fall')[a['to']]+'_constraint'])] for a in m.checks],True)
        # The second-moment baseline contains pin capacitances, not only wire
        # geometry. A committed master change must refresh it before prediction.
        self._moment_state_stale=getattr(self,'_moment_state_stale',False) or (
            getattr(self,'wire_moment_model',False) and bool(changed))
        self._table_masters=m.masters.copy()
        arc_calibrated=getattr(m,'calibration_mode','ratio')=='arc_offset'
        replay_valid=(getattr(self,'_replay_graph',None) is not None and
                      getattr(self,'arc_calibrated',False)==arc_calibrated)
        self.arc_calibrated=arc_calibrated
        def bind(name,value):
            # Captured kernels retain these addresses. Update the values in
            # place so same-topology sizing commits can reuse the timing graph.
            nonlocal replay_valid
            old=getattr(self,name,None)
            if old is not None and old.shape==value.shape:old.copy_(value)
            else:setattr(self,name,value);replay_valid=False
        bind('arc_offset',self.tensor(getattr(m,'arc_delay_offset',np.zeros(len(m.arcs)))))
        for name,value in [('ar',m.at_ratio.reshape(-1)),('ao',m.at_offset.reshape(-1)),
                           ('sr',m.slew_ratio.reshape(-1)),('so',m.slew_offset.reshape(-1)),
                           ('qr',m.check_ratio),('qo',m.check_offset)]:bind(name,self.tensor(value))
        bind('root_at',self.tensor(m.ref_at.reshape(-1))[self.roots])
        bind('root_sl',self.tensor(m.ref_slew.reshape(-1))[self.roots])
        if hasattr(self,'source_groups'):self.source_groups.update_roots(m.ref_at)
        self.load0=self.tensor(m.rf_base_load);self.tau0=self.tensor(m.rc_base_tau)
        self.impulse0=self.tensor(getattr(m,'rc_base_impulse',m.rc_base_tau))
        self.wire_moment_model=getattr(self,'wire_moment_model',False)
        self.prepared=False
        if not replay_valid:self._replay_graph=None

    def lut_parts(self,ids,x,y):
        tx=self.tx[ids];ty=self.ty[ids];z=self.z[ids];i=(torch.sum(tx<=x[:,None],1)-1).clamp(0,tx.shape[1]-2);j=(torch.sum(ty<=y[:,None],1)-1).clamp(0,ty.shape[1]-2);r=torch.arange(len(ids),device=self.device)
        dx=tx[r,i+1]-tx[r,i];dy=ty[r,j+1]-ty[r,j];u=(x-tx[r,i])/dx;v=(y-ty[r,j])/dy
        z00=z[r,i,j];z01=z[r,i,j+1];z10=z[r,i+1,j];z11=z[r,i+1,j+1]
        return ((1-u)*((1-v)*z00+v*z01)+u*((1-v)*z10+v*z11),((1-v)*(z10-z00)+v*(z11-z01))/dx,((1-u)*(z01-z00)+u*(z11-z10))/dy)

    def update_parasitics(self,load_rf,tau,rc_matrix):
        """Refresh physical RC on an unchanged timing graph, without recalibration.

        load_rf and tau refer to the engine's current baseline masters. Sparse
        derivatives must be refreshed together with geometry: retaining the old
        transfer resistance silently corrupts candidate sizing and gradients.
        Pin/arc topology changes require construction of a new engine.
        """
        load=np.asarray(load_rf,dtype=np.float64); tau=np.asarray(tau,dtype=np.float64)
        if load.shape!=(self.m.n,2) or tau.shape!=(len(self.m.arcs),):
            raise ValueError('Parasitic array shape mismatch')
        if not np.all(np.isfinite(load)) or not np.all(np.isfinite(tau)) or np.any(load<0) or np.any(tau<0):
            raise ValueError('Invalid physical RC load/delay')
        coo=rc_matrix.tocoo()
        if coo.shape!=(len(self.m.arcs),2*self.m.n):raise ValueError('RC sensitivity shape mismatch')
        if not np.all(np.isfinite(coo.data)) or np.any(coo.data<0):raise ValueError('Invalid RC sensitivity')
        indices=self.tensor(np.array([coo.row,coo.col]),True)
        rc=torch.sparse_coo_tensor(indices,self.tensor(coo.data),coo.shape,device=self.device).coalesce()
        self.load0=self.tensor(load);self.tau0=self.tensor(tau)
        self.impulse0=self.tau0;self.wire_moment_model=False;self.wire_moment_state=None
        self._moment_state_stale=False
        self.rc=rc;self.rc_t=rc.transpose(0,1).coalesce()
        self.rc_operator=None
        self.m.rf_base_load=load.copy();self.m.rc_base_tau=tau.copy()
        self.m.rc_matrix=rc_matrix.tocsr();self.m.rc_current_tau=tau.copy()
        self.m.rf_current_load=load.copy();self.prepared=False
        # Replay captures load/tau input buffers, not the sparse candidate RC
        # calculation. Geometry updates copy new inputs into those same buffers;
        # their timing equations and graph schedule are unchanged.

    def enable_capture_clock_delta(self):
        """Experimental common clock-delta approximation, not tagged min STA.

        Propagate capture-clock changes into required times instead of freezing
        every endpoint boundary. The current engine only computes max arrivals:
        this approximates early capture change by late capture change, retaining
        the initial early/late and CRPR offsets. Full min/max tagged propagation
        is required before treating this mode as signoff STA.
        """
        nodes=[];valid=[];anchor=[]
        for endpoint in self.m.endpoints:
            name=endpoint.get('capture_pin','');rf=endpoint.get('capture_rf','')
            ok=name in self.m.index and rf in ('rise','fall')
            node=self.m.index[name]*2+('rise','fall').index(rf) if ok else 0
            nodes.append(node);valid.append(ok)
            anchor.append(self.m.ref_at.reshape(-1)[node] if ok else 0.)
        if not any(valid):raise ValueError('Capture-clock metadata is absent')
        self.capture_node=self.tensor(nodes,True)
        self.capture_valid=self.tensor(valid,True).bool();self.capture_anchor=self.tensor(anchor)
        self.prepared=False;self._replay_graph=None

    def enable_source_groups(self,plan):
        """Install initial native source classes without candidate refitting.

        The caller verifies the initial DEF/SPEF/SDC metadata. General SDC
        exceptions and changing CRPR within a source class remain approximate.
        """
        import copy
        from source_group_sta import CudaSourceGroups
        if not self.arc_calibrated:raise ValueError('Source groups require arc_offset calibration')
        if getattr(self.m,'wire_slew_model',None)=='impulse' and not self.wire_moment_model:
            raise ValueError('Refresh second-moment RC before initializing source-group anchors')
        if hasattr(self,'source_groups'):raise ValueError('Source groups are already initialized')
        np.testing.assert_array_equal(plan.s,self.numpy(self.s));np.testing.assert_array_equal(plan.t,self.numpy(self.t))
        initial=self.forward();runtime=CudaSourceGroups(plan,self)
        group_at=runtime.forward(self.tensor(initial['delay']))
        native_at=self.tensor([float(e['at_ps']) for e in plan.endpoints])
        anchor=group_at[runtime.ep_group,runtime.ep_node]
        if not torch.all(torch.isfinite(anchor)):raise ValueError('Initial native endpoint class is unreachable')
        # A small within-class reference residual retains native cycle offsets
        # and baseline PathEnd selection; the classes themselves stay separate.
        self.group_endpoint_offset=native_at-anchor
        m=copy.copy(self.m);m.endpoints=[dict(e) for e in plan.endpoints];self.m=m;self.b.model=m
        self.ep_node=runtime.ep_node;self.ep_pin=self.ep_node//2
        checks={(a['t'],a['to']) for a in m.checks}
        check_mask=[(m.index[e['pin']],RF[e['rf']]) in checks for e in m.endpoints]
        self.ep_setup=self.tensor(check_mask,True).bool()
        self.ep_req=self.tensor([float(e['required_ps'])+(float(e['margin_ps']) if use else 0.)
                                 for e,use in zip(m.endpoints,check_mask)])
        self.enable_capture_clock_delta()
        # These benchmark PathDelay constraints explicitly ignore target-clock
        # latency; only their data-pin setup/recovery margin changes.
        self.capture_valid &= ~self.tensor([bool(int(e['is_path_delay'])) for e in m.endpoints],True).bool()
        self.source_groups=runtime;self.prepared=False;self._replay_graph=None

    def update_tree_parasitics(self,operator,load,tau,moment_state=None):
        if tuple(load.shape)!=(self.m.n,2) or tuple(tau.shape)!=(len(self.m.arcs),):
            raise ValueError('Tree RC array shape mismatch')
        if not torch.all(torch.isfinite(load)) or not torch.all(torch.isfinite(tau)):
            raise ValueError('Nonfinite tree RC load/delay')
        self.rc_operator=operator;self.load0=load;self.tau0=tau
        active=moment_state is not None
        if active!=self.wire_moment_model:self._replay_graph=None
        self.wire_moment_model=active;self.wire_moment_state=moment_state
        self._moment_state_stale=False
        self.impulse0=moment_state['arc_impulse'] if active else tau
        if active:self.m.rc_base_impulse=self.numpy(self.impulse0).copy()
        self.m.rf_base_load=self.numpy(load).copy();self.m.rc_base_tau=self.numpy(tau).copy()
        self.m.rf_current_load=self.m.rf_base_load.copy();self.m.rc_current_tau=self.m.rc_base_tau.copy()
        self.m.rc_matrix=operator.as_linear_operator();self.prepared=False

    def apply_rc(self,pin_delta):
        operator=getattr(self,'rc_operator',None)
        if operator is not None:return operator.apply(pin_delta)
        return torch.sparse.mm(self.rc,pin_delta.reshape(-1,1)).reshape(-1)

    def apply_rc_transpose(self,arc_gradient):
        operator=getattr(self,'rc_operator',None)
        if operator is not None:return operator.transpose(arc_gradient)
        return torch.sparse.mm(self.rc_t,arc_gradient.reshape(-1,1)).reshape(-1)

    def candidate_wire_moments(self,pin_delta):
        self._check_moment_state()
        if self.wire_moment_model:return self.rc_operator.moments_delta(pin_delta,self.wire_moment_state)
        tau=self.tau0+self.apply_rc(pin_delta)
        return tau,tau

    def wire_moments_transpose(self,delay_gradient,slew_gradient):
        direct=torch.where(self.port_wire,delay_gradient*self.port_delay_factor+slew_gradient*self.port_slew_factor,delay_gradient)
        if self.wire_moment_model:
            impulse_gradient=slew_gradient*2.197224577**2*self._impulse/self.local_slew.clamp_min(1e-300)
            impulse_gradient=torch.where(self.port_wire,0.,impulse_gradient)
            return self.rc_operator.moments_transpose(direct,impulse_gradient,self.wire_moment_state)
        standard=slew_gradient*2.197224577**2*self._tau/self.local_slew.clamp_min(1e-300)
        return self.apply_rc_transpose(direct+torch.where(self.port_wire,0.,standard))

    def wire_values(self,tau,impulse,source_slew,ids):
        alpha=self.port_delay_factor[ids];beta=self.port_slew_factor[ids]
        port=alpha>0
        return (torch.where(port,alpha*tau,tau),
                torch.where(port,source_slew+beta*tau,torch.hypot(source_slew,2.197224577*impulse)))

    def capture_clock_delta(self,at):
        if not hasattr(self,'capture_node'):return 0.
        return torch.where(self.capture_valid,at[self.capture_node]-self.capture_anchor,0.)

    def _check_moment_state(self):
        if getattr(self,'_moment_state_stale',False):
            raise RuntimeError('Baseline masters changed: refresh tree parasitics before timing prediction')

    def _candidate(self,replacements):
        self._check_moment_state()
        m=self.m;dt=self.delay_tab;st=self.slew_tab;qt=self.check_tab
        if not replacements:return dt,st,qt,self.load0,self.tau0,self.impulse0 if self.wire_moment_model else self.tau0
        dt=dt.clone();st=st.clone();qt=qt.clone();delta=np.zeros((m.n,2));masters={**m.masters,**replacements}
        for inst,target in replacements.items():
            old,new=m.cells[m.masters[inst]],m.cells[target]
            if old['family_id']!=new['family_id'] or new.get('frozen_reason'):raise ValueError('Illegal replacement')
            for k in self.b.instance_arcs.get(inst,[]):
                a=m.arcs[k]
                if m.changed_groups(a,masters):return None
                tabs=m.table(a,masters);rf=('rise','fall')[a['to']];dt[k]=self.table_ids[id(tabs['cell_'+rf])];st[k]=self.table_ids[id(tabs[rf+'_transition'])]
            for k in self.b.instance_checks.get(inst,[]):
                a=m.checks[k]
                if m.changed_groups(a,masters):return None
                qt[k]=self.table_ids[id(m.table(a,masters)[('rise','fall')[a['to']]+'_constraint'])]
            for p in self.b.instance_pins.get(inst,[]):
                name=m.pin_parts[p][1];op=old['liberty_data']['pins'][name];npin=new['liberty_data']['pins'][name]
                if getattr(m,'analysis','max')=='min':
                    delta[p]=np.asarray(m.cap_ranges[target][name]['min'])-np.asarray(m.cap_ranges[m.masters[inst]][name]['min'])
                else:delta[p]=[npin.get(rf+'_capacitance_ff',npin.get('capacitance_ff',0))-op.get(rf+'_capacitance_ff',op.get('capacitance_ff',0)) for rf in ('rise','fall')]
        dc=self.tensor(delta);net=self.zeros(self.nn*2).view(-1,2).index_add_(0,self.netid,dc*self.sink[:,None]);load=self.load0+net[self.netid]
        tau,impulse=self.candidate_wire_moments(dc)
        if torch.any(tau < -1e-5) or torch.any(load < -1e-6):raise ValueError('Negative RC load/delay')
        return dt,st,qt,load.clamp_min(0),tau.clamp_min(0),impulse.clamp_min(0)

    def forward(self,replacements=None,capture=False,_gpu_only=False):
        m=self.m;data=self._candidate(replacements or {})
        if data is None:
            if capture:raise ValueError('Cannot capture changed conditional graph')
            if hasattr(self,'capture_node'):raise ValueError('Changed conditional graph requires native validation in capture-clock mode')
            if self.arc_calibrated:raise ValueError('Changed conditional graph requires native validation in arc calibration mode')
            if self.wire_moment_model:raise ValueError('Changed conditional graph requires native validation in impulse mode')
            self.fallback_count+=1;return m.run(replacements)
        dt,st,qt,load,tau,impulse=data;n=m.n*2;at=torch.full((n,),-torch.inf,dtype=torch.float64,device=self.device);sl=self.zeros(n);delay=self.zeros(len(m.arcs));local=self.zeros(len(m.arcs));at[self.roots]=self.root_at;sl[self.roots]=self.root_sl;cap=load.reshape(-1)
        for ids,cell,wire,nodes in self.plan:
            delay[cell]=self.lut(dt[cell],sl[self.s[cell]],cap[self.t[cell]]);local[cell]=self.lut(st[cell],sl[self.s[cell]],cap[self.t[cell]]).clamp_min(0)
            delay[wire],local[wire]=self.wire_values(tau[wire],impulse[wire],sl[self.s[wire]],wire)
            if self.arc_calibrated:
                delay[cell]+=self.arc_offset[cell];delay[wire]=(delay[wire]+self.arc_offset[wire]).clamp_min(0)
            at.scatter_reduce_(0,self.t[ids],at[self.s[ids]]+delay[ids],reduce='amax',include_self=True);sl.scatter_reduce_(0,self.t[ids],local[ids],reduce='amax',include_self=True)
            at[nodes]=at[nodes]*self.ar[nodes]+self.ao[nodes];sl[nodes]=(sl[nodes]*self.sr[nodes]+self.so[nodes]).clamp_min(0)
        check=self.lut(qt,sl[self.ct],sl[self.cs])*self.qr+self.qo;qmax=self.peak(check,self.ct,n);req=self.ep_req-torch.where(self.ep_setup,qmax[self.ep_node],0.)+self.capture_clock_delta(at);slack=req-at[self.ep_node]
        if hasattr(self,'source_groups'):
            group_at=self.source_groups.forward(delay)
            data_at=group_at[self.source_groups.ep_group,self.ep_node]+self.group_endpoint_offset
            slack=req-data_at
        worst=torch.full((m.n,),torch.inf,dtype=torch.float64,device=self.device).scatter_reduce_(0,self.ep_pin,slack,reduce='amin',include_self=True);tns=worst.clamp_max(0).sum();wns=worst.min().clamp_max(0)
        ep=torch.stack((self.ep_pin,self.ep_node%2,req,slack),dim=1)
        if capture:
            self._at=at;self._sl=sl;self._delay=delay;self._tau=tau;self._impulse=impulse;self.local_slew=local;self._load=load;self._check=check;self._eps=ep;self.at_candidate=at[self.s]+delay
            wire_delay=torch.where(self.port_wire,tau*self.port_delay_factor,tau)
            self.wire_delay_active=(wire_delay+self.arc_offset>0) if self.arc_calibrated else torch.ones_like(tau,dtype=torch.bool)
            if hasattr(self,'source_groups'):self._group_at=group_at
        tensors=dict(at=at.reshape(-1,2),slew=sl.reshape(-1,2),load=load.max(1).values,
                     load_rf=load,delay=delay,check=check,endpoints=ep,tns_ps=tns,wns_ps=wns)
        if _gpu_only:return tensors
        return self._export_tensors(tensors)

    def _export_tensors(self,tensors):
        return {k:v.item() if k in ('tns_ps','wns_ps') else self.numpy(v) for k,v in tensors.items()}

    def fast_forward(self,replacements=None,device_output=False):
        """CUDA Graph replay of the same equations, including CPU result export.

        Candidate selection and sparse RC deltas stay outside capture. Graph
        creation is a one-time cost per timing graph/calibration and is invalidated
        on model/capture-clock changes. Physical RC changes reuse the graph.
        Forward results match forward(), while
        prepare()/backward() keep their existing derivative tape semantics.
        """
        if not str(self.device).startswith('cuda'):return self.forward(replacements,_gpu_only=device_output)
        data=self._candidate(replacements or {})
        if data is None:
            if device_output:raise ValueError('Device-only replay requires a stable conditional graph')
            return self.forward(replacements)
        if self._replay_graph is None:
            saved=(self.delay_tab,self.slew_tab,self.check_tab,self.load0,self.tau0,self.impulse0)
            buffers=tuple(t.clone() for t in saved)
            try:
                self.delay_tab,self.slew_tab,self.check_tab,self.load0,self.tau0,self.impulse0=buffers
                stream=torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):self.forward(_gpu_only=True)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):result=self.forward(_gpu_only=True)
                self._replay_graph=graph;self._replay_buffers=buffers;self._replay_result=result
            finally:self.delay_tab,self.slew_tab,self.check_tab,self.load0,self.tau0,self.impulse0=saved
        for dest,source in zip(self._replay_buffers,data):dest.copy_(source)
        self._replay_graph.replay()
        return self._replay_result if device_output else self._export_tensors(self._replay_result)

    def prepare(self):
        f=self.forward(capture=True);self.b.forward=f;m=self.m;na=len(m.arcs);cell=self.cells;wire=self.wires
        self.dx=self.zeros(na);self.dc=self.zeros(na);self.sx=self.zeros(na);self.sc=self.zeros(na)
        _,self.dx[cell],self.dc[cell]=self.lut_parts(self.delay_tab[cell],self._sl[self.s[cell]],self._load.reshape(-1)[self.t[cell]])
        raw,sx,sc=self.lut_parts(self.slew_tab[cell],self._sl[self.s[cell]],self._load.reshape(-1)[self.t[cell]]);self.sx[cell]=sx*(raw>0);self.sc[cell]=sc*(raw>0)
        self.sx[wire]=torch.where(self.port_wire[wire],1.,self._sl[self.s[wire]]/self.local_slew[wire].clamp_min(1e-300))
        _,qx,qy=self.lut_parts(self.check_tab,self._sl[self.ct],self._sl[self.cs]);self.check_dx=torch.stack((qx,qy),1);self.positive_slew=self._sl>0;self.b.local_slew=self.numpy(self.local_slew);self.prepared=True;self.sync();return f

    def backward(self,branch='ave',epsilon=.01,gamma_ps=1.,alpha=1.,beta=0.):
        if not self.prepared:raise ValueError('prepare() required')
        if branch not in ('hard','ave','lse') or not 0<=epsilon<1 or gamma_ps<=0 or min(alpha,beta)<0 or alpha+beta<=0:raise ValueError('Invalid custom VJP')
        m=self.m;n=m.n*2;slack=self._eps[:,3];worst=torch.full((m.n,),torch.inf,dtype=torch.float64,device=self.device).scatter_reduce_(0,self.ep_pin,slack,reduce='amin',include_self=True)
        active=((slack==worst[self.ep_pin])&(slack<0)).double();counts=self.zeros(m.n).scatter_add_(0,self.ep_pin,active);seeds=alpha*active/counts[self.ep_pin].clamp_min(1)
        globalactive=((slack==slack.min())&(slack<0)).double();seeds+=beta*globalactive/globalactive.sum().clamp_min(1)
        grouped=hasattr(self,'source_groups')
        gat=self.zeros(n) if grouped else self.zeros(n).scatter_add_(0,self.ep_node,seeds)
        gs=self.zeros(n);gd=self.zeros(len(m.arcs));gsl=self.zeros(len(m.arcs))
        group_delay=self.source_groups.delay_vjp(self._group_at,self._delay,seeds,branch,epsilon,gamma_ps) if grouped else gd
        if hasattr(self,'capture_node'):gat.scatter_add_(0,self.capture_node,-seeds*self.capture_valid)
        qpeak=self.peak(self._check,self.ct,n);qa=(self._check==qpeak[self.ct]).double();qn=self.zeros(n).scatter_add_(0,self.ct,qa);qs=self.zeros(n).scatter_add_(0,self.ep_node,seeds*self.ep_setup)
        gq=qs[self.ct]*qa/qn[self.ct].clamp_min(1)*self.qr
        gs.scatter_add_(0,self.ct,gq*self.check_dx[:,0]);gs.scatter_add_(0,self.cs,gq*self.check_dx[:,1]);wat=self.weights(self.at_candidate,branch,epsilon,gamma_ps);wsl=self.weights(self.local_slew,branch,epsilon,gamma_ps)
        for ids in self.levels[::-1]:
            dst=self.t[ids];src=self.s[ids];physical=gat[dst]*self.ar[dst]*wat[ids]
            gd[ids]=physical+(group_delay[ids] if grouped else 0.);gsl[ids]=gs[dst]*self.sr[dst]*wsl[ids]*self.positive_slew[dst]
            gat.scatter_add_(0,src,physical);gs.scatter_add_(0,src,gd[ids]*self.dx[ids]+gsl[ids]*self.sx[ids])
        pin=self.wire_moments_transpose(gd*self.wire_delay_active,gsl).reshape(-1,2)
        cell=self.cells;cap=self.zeros(n).scatter_add_(0,self.t[cell],gd[cell]*self.dc[cell]+gsl[cell]*self.sc[cell]).view(-1,2);net=self.zeros(self.nn*2).view(-1,2).index_add_(0,self.netid,cap);pin+=net[self.netid]*self.sink[:,None]
        return dict(at=self.numpy(gat).reshape(-1,2),slew=self.numpy(gs).reshape(-1,2),load=np.zeros(m.n),net_load={net:0. for net in self.netnames},pin_load=self.numpy(pin),delay=self.numpy(gd),local_slew=self.numpy(gsl),setup=self.numpy(gq),at_weights=self.numpy(wat),slew_weights=self.numpy(wsl),endpoint_seeds=self.numpy(seeds),branch=branch,epsilon=epsilon,gamma_ps=gamma_ps,alpha=alpha,beta=beta)

    def _descriptor(self,inst,target):
        m=self.m;old=m.masters[inst];key=(inst,old,target)
        if key in self.action_cache:return self.action_cache[key]
        qt=[];qx=[];qy=[];qs=[];qgroup=[];coef=[];component=[];slrows=[];slindex=[];pids=[];pdelta=[];na=len(m.arcs)
        for role,indices in [('cell',self.b.instance_arcs.get(inst,[])),('check',self.b.instance_checks.get(inst,[]))]:
            for k in indices:
                a=(m.arcs if role=='cell' else m.checks)[k];rf=('rise','fall')[a['to']];groups=m.changed_groups(a,{inst:target});before,after=groups if groups else ({0:m.table(a,m.masters)},{0:m.table(a,{inst:target})})
                kinds=[(0,'cell_'+rf),(2,rf+'_transition')] if role=='cell' else [(3,rf+'_constraint')]
                for c,kind in kinds:
                    row=len(coef);coef.append(k+(0 if c==0 else na if c==2 else 2*na));component.append(c);slrows.append(c==2);slindex.append(k if c==2 else 0)
                    for side,group in enumerate((before,after)):
                        for tabs in group.values():
                            qt.append(self.table_ids[id(tabs[kind])]);qx.append((a['s']*2+a['fr']) if role=='cell' else (a['t']*2+a['to']));qy.append((a['t']*2+a['to']) if role=='cell' else (a['s']*2+a['fr']));qs.append(role=='check');qgroup.append(row*2+side)
        for pin in self.b.instance_pins.get(inst,[]):
            name=m.pin_parts[pin][1];op=m.cells[old]['liberty_data']['pins'][name];npin=m.cells[target]['liberty_data']['pins'][name]
            for rf,edge in enumerate(('rise','fall')):
                dc=npin.get(edge+'_capacitance_ff',npin.get('capacitance_ff',0))-op.get(edge+'_capacitance_ff',op.get('capacitance_ff',0))
                if dc:pids.append(pin*2+rf);pdelta.append(dc)
        data=tuple(np.asarray(x,dtype=np.float64 if i==11 else np.int64) for i,x in enumerate((qt,qx,qy,qs,qgroup,coef,component,slrows,slindex,pids,[],pdelta)))
        self.action_cache[key]=data;return data

    def backward_power(self,slew_seed,load_seed):
        """Hard-forward slew VJP for propagated NLPM internal power.

        There is no arrival-time or setup seed. Power derivatives can be
        signed; the timing-safe mask still comes from the separate LSE pass.
        """
        if not self.prepared:raise ValueError('prepare() required')
        n=self.m.n*2;gs=self.tensor(np.asarray(slew_seed).reshape(-1)).clone();gsl=self.zeros(len(self.m.arcs))
        wsl=self.weights(self.local_slew,'hard',.01,1.)
        for ids in self.levels[::-1]:
            gsl[ids]=gs[self.t[ids]]*self.sr[self.t[ids]]*wsl[ids]*self.positive_slew[self.t[ids]]
            gs.scatter_add_(0,self.s[ids],gsl[ids]*self.sx[ids])
        pin=self.wire_moments_transpose(self.zeros(len(self.m.arcs)),gsl).reshape(-1,2)
        cap=self.tensor(load_seed).reshape(-1)+self.zeros(n).scatter_add_(0,self.t[self.cells],gsl[self.cells]*self.sc[self.cells])
        net=self.zeros(self.nn*2).view(-1,2).index_add_(0,self.netid,cap.view(-1,2))
        pin+=net[self.netid]*self.sink[:,None]
        return dict(at=np.zeros((self.m.n,2)),delay=np.zeros(len(self.m.arcs)),
                    local_slew=self.numpy(gsl),setup=np.zeros(len(self.m.checks)),pin_load=self.numpy(pin))

    def score_all(self,instances,states,gradients,tau=1e-6):
        if not self.prepared:raise ValueError('prepare() required')
        ts=time.perf_counter();m=self.m;shape=states['action_target_id'].shape;mask=states['action_mask'];pack=[[] for _ in range(11)];nd=0
        width=shape[1]
        for row in instances:
            i=int(row['instance_id']);inst=row['name']
            if row['master']!=m.masters[inst]:raise ValueError('Stale master state')
            for d in range(width):
                if not mask[i,d]:continue
                target=m.catalog['cells'][int(states['action_target_id'][i,d])]['name'];qt,qx,qy,qs,qgroup,coef,component,slrows,slindex,pids,unused,pdelta=self._descriptor(inst,target);action=i*width+d
                for j,v in enumerate((qt,qx,qy,qs,qgroup+2*nd,coef,component+action*4,slrows,slindex,pids,pdelta)):pack[j].append(v)
                nd+=len(coef)
        # Pin-cap action destinations need a separate ragged list.
        pd=[]
        for row in instances:
            i=int(row['instance_id']);inst=row['name']
            for d in range(width):
                if mask[i,d]:
                    target=m.catalog['cells'][int(states['action_target_id'][i,d])]['name'];pd.append(np.full(len(self._descriptor(inst,target)[9]),i*width+d,dtype=np.int64))
        arrays=[np.concatenate(v) if v else np.empty(0,dtype=np.float64 if j==10 else np.int64) for j,v in enumerate(pack)]
        qt,qx,qy,qs,qg,coef,dest,slrows,slindex,pids,pdelta=[self.tensor(v,j!=10) for j,v in enumerate(arrays)];pd=self.tensor(np.concatenate(pd) if pd else np.empty(0,dtype=np.int64),True)
        self.last_pack_seconds=time.perf_counter()-ts;ts=time.perf_counter();maxima=torch.full((nd*2,),-torch.inf,dtype=torch.float64,device=self.device)
        for lo in range(0,len(qt),100000):
            hi=min(lo+100000,len(qt));x=self._sl[qx[lo:hi]];y=torch.where(qs[lo:hi].bool(),self._sl[qy[lo:hi]],self._load.reshape(-1)[qy[lo:hi]]);v=self.lut(qt[lo:hi],x,y)
            clip=slrows[qg[lo:hi]//2].bool();v=torch.where(clip,v.clamp_min(0),v);maxima.scatter_reduce_(0,qg[lo:hi],v,reduce='amax',include_self=True)
        maxima=maxima.view(-1,2);delta=maxima[:,1]-maxima[:,0];old=self.local_slew[slindex];delta=torch.where(slrows.bool(),(old+delta).clamp_min(0)-old,delta);result={}
        for name,g in gradients.items():
            gr=self.tensor(np.concatenate((g['delay'],g['local_slew'],g['setup'])));scores=self.zeros(mask.size*4).scatter_add_(0,dest,gr[coef]*delta).view(-1,4);pin=self.tensor(g['pin_load'].reshape(-1));scores[:,1].scatter_add_(0,pd,pin[pids]*pdelta)
            values=self.numpy(scores).reshape(*shape,4);values[~mask]=np.nan;result[name]=values
        gate=np.full(len(instances),np.inf)
        for row in instances:
            pins=self.b.instance_pins.get(row['name'],[])
            if pins:gate[int(row['instance_id'])]=np.max(np.abs(gradients['lse']['at'][pins]))
        safe=gate<=tau;self.sync();self.last_score_seconds=time.perf_counter()-ts
        power_directions=states.get('power_direction_mask',np.array([False,True,True,False]) if width==4 else np.ones_like(mask))
        return dict(components=result,delta_loss={k:v.sum(2) for k,v in result.items()},benefit={k:-v.sum(2) for k,v in result.items()},mask=mask.copy(),gate_lse_at=gate,timing_safe=safe,power_candidate_mask=mask&safe[:,None]&power_directions)
