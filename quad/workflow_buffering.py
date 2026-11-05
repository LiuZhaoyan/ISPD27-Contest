"""Concurrent native tree-DP proposals; full tagged MCMM decides acceptance."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
from pathlib import Path
import os
import fnmatch
import subprocess
import numpy as np
from forward_sta import condition
from policy import PROHIBITED_MASTERS


def library():
    src=Path(__file__).with_name('workflow_buffer_dp.cpp')
    key=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    dest=src.parents[1]/'cache/workflow'/key/'buffer_dp.so'
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(f'.{os.getpid()}.so')
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(src),'-o',str(tmp)],check=True)
        tmp.replace(dest)
    lib=ctypes.CDLL(str(dest));p=ctypes.c_void_p;i=ctypes.c_int;d=ctypes.c_double
    lib.workflow_buffer_dp.argtypes=[i,p,p,p,p,i,i,i,p,p,p,p,p,i,i,d,d,d,p,p,p,p,p]
    lib.workflow_buffer_dp.restype=i
    return lib


def required_bounds(model,forward):
    req=np.full(2*model.n,np.inf)
    for pin,rf,value,slack in forward['endpoints']:
        node=2*int(pin)+int(rf);req[node]=min(req[node],value)
    for node in model.order[::-1]:
        for aid in model.incoming[node]:
            a=model.arcs[aid]
            if a['role']=='Reg Clk to Q':continue
            s=2*a['s']+a['fr'];t=2*a['t']+a['to']
            req[s]=min(req[s],req[t]-forward['delay'][aid])
    return req.reshape(-1,2)


class ParallelBufferDP:
    def __init__(self,catalog,tables,workers=4,frontier=24,max_per_net=1):
        if workers<1 or frontier<5 or not 1<=max_per_net<=4:raise ValueError('Invalid DP budget')
        self.lib=library();self.workers=workers;self.frontier=frontier;self.max_per_net=max_per_net
        self.buffers=[];self.tables=[]
        for c in catalog['cells']:
            if not c['name'].startswith('BUF') or c.get('frozen_reason'):continue
            if any(fnmatch.fnmatchcase(c['name'],p) for p in PROHIBITED_MASTERS):continue
            p=c['liberty_data']['pins'];ins=[n for n,x in p.items() if x['direction']=='input'];outs=[n for n,x in p.items() if x['direction']=='output']
            if len(ins)!=1 or len(outs)!=1:continue
            key=(c['name'],ins[0],outs[0],'combinational',condition(''))
            if key not in tables:continue
            self.buffers.append((c,ins[0],outs[0]));self.tables.append(tables[key])
        if not self.buffers:raise ValueError('No legal positive-unate buffer cells')

    def solve(self,network,engine,forward,required,serial):
        m=engine.m;n=len(network.nodes);root=network.pin_nodes[network.driver]
        adjacency=[[] for _ in range(n)]
        for a,b,r in network.resistors:adjacency[a].append((b,r));adjacency[b].append((a,r))
        parent=np.full(n,-2,dtype=np.int32);parent[root]=-1;r=np.zeros(n);order=[root]
        for u in order:
            for v,resistance in adjacency[u]:
                if parent[v]!=-2:continue
                parent[v]=u;r[v]=resistance;order.append(v)
        if len(order)!=n or len(network.resistors)!=n-1:raise ValueError('DP input must be a tree')
        cap=network.wire_caps.copy();q=np.full(n,np.inf);leaves=[set() for _ in range(n)]
        positions={p.name:(p.x,p.y) for p in network.pins}
        for pin,node in network.pin_nodes.items():
            if pin==network.driver or pin not in m.index:continue
            idx=m.index[pin];cap[node]+=m.rf_base_pin[idx].max();q[node]=min(q[node],required[idx].min())
            leaves[node].add(pin)
        for node in reversed(order[1:]):leaves[parent[node]].update(leaves[node])
        if not np.isfinite(q).any():return []
        driver=m.index[network.driver];slew=max(1.,float(forward['slew'][driver].max()))
        inputs=[];areas=[];powers=[];loads=[];delays=[]
        for (cell,ip,op),tab in zip(self.buffers,self.tables):
            pins=cell['liberty_data']['pins'];inputs.append(pins[ip].get('capacitance_ff',0.))
            areas.append(cell['liberty_data']['area'])
            leakage=[float(s['value']) for s in cell['liberty_data'].get('leakage_states',[]) if s.get('related_pg_pin')=='VDD']
            powers.append((max(leakage) if leakage else 0.)*1e-12)
            ys=tab['cell_rise'].y;loads.append(ys)
            delays.append([max(tab['cell_rise'](slew,y),tab['cell_fall'](slew,y)) for y in ys])
        arrays=[np.ascontiguousarray(x,dtype=np.float64) for x in (inputs,areas,powers,loads,delays)]
        nb=len(inputs);nt=arrays[3].shape[1]
        driver_r=0.
        for aid in m.incoming[driver]:
            a=m.arcs[aid]
            if a['role']=='wire':continue
            tab=m.table(a,m.masters)['cell_'+('rise','fall')[a['to']]]
            x=forward['slew'][a['s'],a['fr']];y=forward['load'][driver]
            driver_r=max(driver_r,(tab(x,y+.01)-tab(x,y))/.01)
        # Compare every plan to the same tree without insertion. Absolute
        # required-time scores from different nets are not comparable.
        base_cap=cap.copy();base_required=q.copy()
        for node in reversed(order[1:]):
            up=parent[node]
            base_required[up]=min(base_required[up],base_required[node]-r[node]*base_cap[node])
            base_cap[up]+=base_cap[node]
        baseline=float(base_required[root]-driver_r*base_cap[root])
        scores=np.empty(self.frontier);summary=np.empty((self.frontier,4));counts=np.empty(self.frontier,dtype=np.int32)
        nodes=np.empty((self.frontier,self.max_per_net),dtype=np.int32);buf=np.empty_like(nodes)
        number=self.lib.workflow_buffer_dp(n,parent.ctypes.data,r.ctypes.data,cap.ctypes.data,q.ctypes.data,
            root,nb,nt,*[x.ctypes.data for x in arrays],self.frontier,self.max_per_net,driver_r,.05,1e5,
            scores.ctypes.data,summary.ctypes.data,counts.ctypes.data,nodes.ctypes.data,buf.ctypes.data)
        if number<0:raise RuntimeError('Native buffering DP failed: '+str(number))
        proposals=[]
        for i in range(number):
            if not counts[i] or scores[i]<=baseline+1e-6:continue
            actions=[];wrapped=[]
            for j in range(counts[i]):
                node=int(nodes[i,j]);cell,ip,op=self.buffers[int(buf[i,j])]
                original=set(leaves[node]);sinks=set(original)
                for covered,replacement in wrapped:
                    if covered<=original:sinks.difference_update(covered);sinks.add(replacement)
                if not sinks or any('/' not in p for p in sinks):actions=[];break
                name=f'quad_buf_{serial}_{len(proposals)}_{j}'
                xy=np.mean([positions[p] for p in original],axis=0).tolist()
                actions.append(dict(kind='buffer',instance=name,target=cell['name'],sinks=sorted(sinks),
                    location=xy,net=network.name,area=cell['liberty_data']['area'],dp_score=float(scores[i])))
                wrapped.append((original,name+'/'+ip))
            if actions:proposals.append(dict(actions=actions,score=float(scores[i]),baseline_score=baseline,
                                            gain=float(scores[i]-baseline),net=network.name))
            if len(proposals)>=3:break
        return proposals

    def propose(self,forest,engine,forward,net_scores,limit=8,serial=0):
        required=required_bounds(engine.m,forward)
        ranked=sorted(net_scores,key=lambda n:(-net_scores[n],n))
        nets=[n for n in ranked if n in forest.networks and len(forest.networks[n].pins)>=3
              and not any(p.clock for p in forest.networks[n].pins)][:limit]
        # ctypes releases the GIL. Each independent tree runs C++ DP on a
        # worker, without touching OpenDB or global placement occupancy.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            result=list(pool.map(lambda item:self.solve(forest.networks[item[1]],engine,forward,required,
                                                        f'{serial}_{item[0]}'),enumerate(nets)))
        return sorted((proposal for group in result for proposal in group),
                      key=lambda p:(-p['gain'],p['net'],p['actions'][0]['instance']))
