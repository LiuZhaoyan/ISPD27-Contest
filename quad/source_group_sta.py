"""Explicit single-clock source classes for max-plus arrival propagation.

Input paths and each launch edge have independent arrivals. A normalized
register class restarts at clock-to-Q with zero launch-clock arrival for
the benchmark's register-to-output -ignore_clock_latency constraints.
No learned model or candidate timing is used. This is not general SDC tag STA:
generated clocks, arbitrary through exceptions and dynamic CRPR need more tags.
"""
import numpy as np
from forward_sta import RF


class SourceGroupPlan:
    def __init__(self,m,geometry,endpoints):
        self.analysis=getattr(m,'analysis','max')
        if self.analysis not in ('min','max'):raise ValueError('Expected min/max source classes')
        self.n=2*m.n;self.s=np.array([2*a['s']+a['fr'] for a in m.arcs],dtype=np.int64)
        self.t=np.array([2*a['t']+a['to'] for a in m.arcs],dtype=np.int64)
        levels=np.array([m.level[a['t']] for a in m.arcs])
        self.levels=[np.flatnonzero(levels==level) for level in np.unique(levels)]
        self.endpoints=list(endpoints);self.ep_node=np.array([2*m.index[e['pin']]+RF[e['rf']] for e in endpoints])
        pins={p.name:p for p in geometry};roots=[i for i in range(m.n) if not m.incoming[i]]
        clocks=[i for i in roots if m.pins[i]['name'] in pins and pins[m.pins[i]['name']].port and pins[m.pins[i]['name']].clock]
        if len(clocks)!=1:raise ValueError('Source groups require exactly one physical root clock port')
        self.clock_root=clocks[0]
        groups={e['source_group'] for e in endpoints}
        if not groups<= {'input','clock_rise','clock_fall'}:raise ValueError('Unsupported source group')
        self.labels=[x for x in ('input','clock_rise','clock_fall') if x in groups]
        physical_count=len(self.labels)
        normalized={e['source_group'] for e in endpoints if self.normalized(e)}
        self.labels += [x+'_normalized' for x in ('clock_rise','clock_fall') if x in normalized]
        self.g=len(self.labels);self.ep_group=np.array([self.labels.index(e['source_group']+('_normalized' if self.normalized(e) else '')) for e in endpoints])
        seed_g=[];seed_node=[]
        for g,label in enumerate(self.labels[:physical_count]):
            if label=='input':
                for p in roots:
                    info=pins.get(m.pins[p]['name'])
                    if info is not None and info.port and not info.clock and info.direction.upper()=='INPUT':
                        for rf in (0,1):seed_g.append(g);seed_node.append(2*p+rf)
            else:
                seed_g.append(g);seed_node.append(2*self.clock_root+(label=='clock_fall'))
        self.seed_group=np.array(seed_g,dtype=np.int64);self.seed_node=np.array(seed_node,dtype=np.int64)
        self.seed=np.full((self.g,self.n),-np.inf if self.analysis=='max' else np.inf)
        self.seed[self.seed_group,self.seed_node]=m.ref_at.reshape(-1)[self.seed_node]
        # Reachability of a launch edge at each sequential clock pin determines
        # which normalized class may restart at that clock-to-Q arc.
        reach=np.isfinite(self.seed[:physical_count]).copy()
        for ids in self.levels:
            np.logical_or.at(reach,(np.arange(physical_count)[:,None],self.t[ids]),reach[:,self.s[ids]])
        seq=np.array([i for i,a in enumerate(m.arcs) if a['role']=='Reg Clk to Q'],dtype=np.int64)
        self.reset=np.zeros((self.g,len(m.arcs)),dtype=bool)
        for g,label in enumerate(self.labels[physical_count:],physical_count):
            physical=self.labels.index(label.removesuffix('_normalized'))
            self.reset[g,seq]=reach[physical,self.s[seq]]

    @staticmethod
    def normalized(endpoint):
        # This mode matches the checked-in contest constraints.sdc. Unknown
        # constraint forms must be exported with a richer explicit tag model.
        return (endpoint['source_group']!='input' and bool(int(endpoint['is_path_delay']))
                and not endpoint.get('capture_pin'))

    def forward(self,delay):
        delay=np.asarray(delay);at=self.seed.copy()
        for ids in self.levels:
            values=at[:,self.s[ids]]+delay[ids]
            values=np.where(self.reset[:,ids],delay[ids],values)
            reduce=np.maximum.at if self.analysis=='max' else np.minimum.at
            reduce(at,(np.arange(self.g)[:,None],self.t[ids]),values)
        return at


class CudaSourceGroups:
    """GPU source-class propagation and hard/Ave/LSE delay VJPs.

    NLDM and physical RC remain in ResidentCudaSTA. The source-class engine
    consumes its local delays; its VJP seeds those same local delay equations.
    """
    def __init__(self,plan,engine):
        self.plan=plan;self.engine=engine;self.seed=engine.tensor(plan.seed)
        self.s=engine.tensor(plan.s,True);self.t=engine.tensor(plan.t,True)
        self.ep_group=engine.tensor(plan.ep_group,True);self.ep_node=engine.tensor(plan.ep_node,True)
        self.reset=engine.tensor(plan.reset,True).bool()
        self.schedule=[]
        for ids in plan.levels:
            ti=engine.tensor(ids,True);g,p=np.nonzero(plan.reset[:,ids])
            self.schedule.append((ti,engine.tensor(g,True),engine.tensor(p,True)))

    def update_roots(self,at):
        p=self.plan;e=self.engine
        self.seed[e.tensor(p.seed_group,True),e.tensor(p.seed_node,True)]=e.tensor(np.asarray(at).reshape(-1)[p.seed_node])

    def forward(self,delay,functional=False):
        import torch
        at=self.seed.clone();g=self.plan.g
        for ids,rg,rp in self.schedule:
            values=at[:,self.s[ids]]+delay[ids]
            # Avoid a conditional GPU synchronization; all masks are static.
            values=torch.where(self.reset[:,ids],delay[ids][None,:],values)
            index=self.t[ids][None,:].expand(g,-1)
            reduce='amax' if self.plan.analysis=='max' else 'amin'
            if functional:at=at.scatter_reduce(1,index,values,reduce=reduce,include_self=True)
            else:at.scatter_reduce_(1,index,values,reduce=reduce,include_self=True)
        return at

    def delay_vjp(self,at,delay,endpoint_seeds,branch='hard',epsilon=.01,gamma_ps=1.):
        if self.plan.analysis!='max':raise NotImplementedError('Early source-class reverse is not implemented')
        import torch
        e=self.engine;p=self.plan
        gat=torch.zeros_like(at);gat.index_put_((self.ep_group,self.ep_node),endpoint_seeds,accumulate=True)
        values=torch.where(self.reset,delay[None,:],at[:,self.s]+delay[None,:])
        weights=torch.stack([e.weights(values[g],branch,epsilon,gamma_ps) for g in range(p.g)])
        gd=torch.zeros_like(delay)
        for ids,rg,rp in self.schedule[::-1]:
            local=gat[:,self.t[ids]]*weights[:,ids]
            gd[ids]=local.sum(0)
            local=torch.where(self.reset[:,ids],0.,local)
            gat.scatter_add_(1,self.s[ids][None,:].expand(p.g,-1),local)
        return gd
