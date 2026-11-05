"""Float64 CUDA levelized NLDM forward and custom Ave/LSE reverse passes.

CPU graph/LUT packing and local action-delta construction remain explicit costs.
CUDA atomics can change last-bit summation order; never use bitwise equivalence.
"""
import time
import numpy as np
import torch
from backward_sta import endpoint_seeds, max_weights
from forward_sta import RF,MAX_CHECKS

class CudaSTA:
    def __init__(self,tape,device='cuda:0'):
        self.b=tape;self.m=m=tape.model;self.device=device
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        self.s=self.tensor([2*a['s']+a['fr'] for a in m.arcs],True)
        self.t=self.tensor([2*a['t']+a['to'] for a in m.arcs],True)
        self.cap_pin=self.tensor([a['s'] if a['role']=='wire' else a['t'] for a in m.arcs],True)
        levels=np.array([m.level[a['t']] for a in m.arcs])
        self.levels=[self.tensor(np.flatnonzero(levels==lev),True) for lev in np.unique(levels)]
        self.level_nodes=[self.tensor(np.flatnonzero(m.level==lev),True) for lev in np.unique(levels)]
        self.wire=self.tensor([a['role']=='wire' for a in m.arcs],True).bool()
        self.elmore=self.tensor([a['elmore'] for a in m.arcs])
        self.dx=self.tensor(tape.delay_dx);self.dc=self.tensor(tape.delay_dc)
        self.sx=self.tensor(tape.slew_dx);self.sc=self.tensor(tape.slew_dc)
        self.positive_slew=self.tensor(tape.forward['slew'].reshape(-1)>0,True).bool()
        bundle=np.zeros(len(m.arcs),dtype=np.int64);bt=[]
        for pin,bundles in enumerate(tape.bundles):
            for rf,groups in enumerate(bundles):
                for ids in groups:
                    bundle[ids]=len(bt);bt.append(pin*2+rf)
        self.bundle=self.tensor(bundle,True);self.bt=self.tensor(bt,True);self.nb=len(bt)
        self.at_candidate=self.tensor(tape.at_candidate);self.local_slew=self.tensor(tape.local_slew)
        self.table_ids={};tabs=[]
        def tid(table):
            key=id(table)
            if key not in self.table_ids:self.table_ids[key]=len(tabs);tabs.append(table)
            return self.table_ids[key]
        delay=[];slew=[]
        for a in m.arcs:
            if a['role']=='wire':delay.append(-1);slew.append(-1);continue
            tab=m.table(a,m.masters);rf=('rise','fall')[a['to']]
            delay.append(tid(tab['cell_'+rf]));slew.append(tid(tab[rf+'_transition']))
        self.delay_tab=self.tensor(delay,True);self.slew_tab=self.tensor(slew,True)
        self.check_tab=self.tensor([tid(m.table(a,m.masters)[('rise','fall')[a['to']]+'_constraint']) for a in m.checks],True)
        # ASAP7 uses uniform 7x7 tables. Fail closed on a different library.
        shapes={(len(t.x),len(t.y)) for t in tabs}
        if len(shapes)!=1:raise ValueError(f'Nonuniform LUT shape: {shapes}')
        self.tx=self.tensor(np.array([t.x for t in tabs]));self.ty=self.tensor(np.array([t.y for t in tabs]));self.z=self.tensor(np.array([t.values for t in tabs]))
        self.cs=self.tensor([a['s']*2+a['fr'] for a in m.checks],True);self.ct=self.tensor([a['t']*2+a['to'] for a in m.checks],True)
        self.check_dx=self.tensor(tape.check_dx)
        self.sync()

    def tensor(self,x,integer=False):return torch.as_tensor(np.asarray(x),dtype=torch.int64 if integer else torch.float64,device=self.device)
    def zeros(self,n):return torch.zeros(n,dtype=torch.float64,device=self.device)
    def sync(self):torch.cuda.synchronize(self.device)
    def numpy(self,x):return x.detach().cpu().numpy()
    def peak(self,x,index,n):
        z=torch.full((n,),-torch.inf,dtype=torch.float64,device=self.device)
        return z.scatter_reduce_(0,index,x,reduce='amax',include_self=True)
    def lut(self,ids,x,y):
        tx=self.tx[ids];ty=self.ty[ids];z=self.z[ids]
        i=(torch.sum(tx<=x[:,None],dim=1)-1).clamp(0,tx.shape[1]-2)
        j=(torch.sum(ty<=y[:,None],dim=1)-1).clamp(0,ty.shape[1]-2)
        r=torch.arange(len(ids),device=self.device)
        u=(x-tx[r,i])/(tx[r,i+1]-tx[r,i]);v=(y-ty[r,j])/(ty[r,j+1]-ty[r,j])
        return (1-u)*((1-v)*z[r,i,j]+v*z[r,i,j+1])+u*((1-v)*z[r,i+1,j]+v*z[r,i+1,j+1])

    def forward(self):
        m=self.m;n=2*m.n
        at=torch.full((n,),-torch.inf,dtype=torch.float64,device=self.device);sl=self.zeros(n);delay=self.zeros(len(m.arcs))
        roots=self.tensor([i for i in range(m.n) if not m.incoming[i]],True)
        at.view(-1,2)[roots]=self.tensor(m.ref_at)[roots];sl.view(-1,2)[roots]=self.tensor(m.ref_slew)[roots]
        cap=self.tensor(m.base_cap);corr=self.tensor(m.delay_correction);scorr=self.tensor(m.slew_correction)
        for ids,nodes in zip(self.levels,self.level_nodes):
            cell=ids[~self.wire[ids]];wire=ids[self.wire[ids]]
            d=self.zeros(len(m.arcs));s=self.zeros(len(m.arcs))
            d[cell]=self.lut(self.delay_tab[cell],sl[self.s[cell]],cap[self.cap_pin[cell]])
            s[cell]=self.lut(self.slew_tab[cell],sl[self.s[cell]],cap[self.cap_pin[cell]]).clamp_min(0)
            d[wire]=self.elmore[wire];s[wire]=torch.hypot(sl[self.s[wire]],2.197224577*self.elmore[wire])
            delay[ids]=d[ids]+corr[ids]
            at.scatter_reduce_(0,self.t[ids],at[self.s[ids]]+delay[ids],reduce='amax',include_self=True)
            sl.scatter_reduce_(0,self.t[ids],s[ids],reduce='amax',include_self=True)
            sl.view(-1,2)[nodes]=(sl.view(-1,2)[nodes]+scorr[nodes]).clamp_min(0)
        check=self.lut(self.check_tab,sl[self.ct],sl[self.cs])+self.tensor(m.check_correction)
        at,sl,delay,check=map(self.numpy,(at,sl,delay,check));at=at.reshape(-1,2);sl=sl.reshape(-1,2)
        eps=[];worst={}
        for e in m.endpoints:
            t=m.index[e['pin']];rf=RF[e['rf']];req=float(e['required_ps'])
            if e['role'] in MAX_CHECKS:req+=float(e['margin_ps'])-max(check[k] for k in m.check_incoming[t] if m.checks[k]['to']==rf)
            slack=req-at[t,rf];eps.append((t,rf,req,slack));worst[t]=min(worst.get(t,np.inf),slack)
        vals=np.array(list(worst.values()))
        return dict(at=at,slew=sl,load=m.base_cap.copy(),delay=delay,check=check,endpoints=np.array(eps),tns_ps=float(np.minimum(vals,0).sum()),wns_ps=float(min(vals.min(),0)))

    def weights(self,values,branch,epsilon,gamma):
        peaks=self.peak(values,self.bundle,self.nb);tops=self.peak(peaks,self.bt,2*self.m.n)
        finite=torch.isfinite(peaks)
        if branch=='lse':outer=torch.where(finite,torch.exp((peaks-tops[self.bt])/gamma),0.)
        else:
            threshold=tops[self.bt]-(epsilon*tops[self.bt].abs() if branch=='ave' else 0.)
            outer=(finite & (peaks>=threshold)).double()
        sums=self.zeros(2*self.m.n).scatter_add_(0,self.bt,outer)
        outer=outer/sums[self.bt].clamp_min(1e-300)
        inner=(torch.isfinite(values)&(values==peaks[self.bundle])).double()
        count=self.zeros(self.nb).scatter_add_(0,self.bundle,inner)
        return inner/count[self.bundle].clamp_min(1)*outer[self.bundle]

    def backward(self,branch='ave',epsilon=.01,gamma_ps=1.,alpha=1.,beta=0.):
        if branch not in ('hard','ave','lse') or not 0<=epsilon<1 or gamma_ps<=0:raise ValueError('Invalid custom VJP')
        m,r=self.m,self.b.forward;n=m.n
        seed=endpoint_seeds(r,alpha,beta);gat=np.zeros((n,2));gq=np.zeros(len(m.checks))
        for i,e in enumerate(m.endpoints):
            t,rf=map(int,r['endpoints'][i,:2]);gat[t,rf]+=seed[i]
            if e['role'] in MAX_CHECKS:
                ids=[k for k in m.check_incoming[t] if m.checks[k]['to']==rf]
                gq[ids]+=seed[i]*max_weights(r['check'][ids],'hard')
        gat=self.tensor(gat.reshape(-1));gq=self.tensor(gq);gs=self.zeros(2*n);gc=self.zeros(n);gd=self.zeros(len(m.arcs));gsl=self.zeros(len(m.arcs))
        gs.scatter_add_(0,self.ct,gq*self.check_dx[:,0]);gs.scatter_add_(0,self.cs,gq*self.check_dx[:,1])
        wat=self.weights(self.at_candidate,branch,epsilon,gamma_ps);wsl=self.weights(self.local_slew,branch,epsilon,gamma_ps)
        for ids in self.levels[::-1]:
            dst=self.t[ids];src=self.s[ids]
            gd[ids]=gat[dst]*wat[ids];gsl[ids]=gs[dst]*wsl[ids]*self.positive_slew[dst]
            gat.scatter_add_(0,src,gd[ids]);gs.scatter_add_(0,src,gd[ids]*self.dx[ids]+gsl[ids]*self.sx[ids])
            gc.scatter_add_(0,self.cap_pin[ids],gd[ids]*self.dc[ids]+gsl[ids]*self.sc[ids])
        values=list(map(self.numpy,(gat,gs,gc,gd,gsl,gq,wat,wsl)))
        ga,gs,gc,gd,gsl,gq,wat,wsl=values
        return dict(at=ga.reshape(-1,2),slew=gs.reshape(-1,2),load=gc,delay=gd,local_slew=gsl,setup=gq,at_weights=wat,slew_weights=wsl,endpoint_seeds=seed,net_load={net:float(gc[pins].sum()) for net,pins in m.by_net.items()},branch=branch,epsilon=epsilon,gamma_ps=gamma_ps,alpha=alpha,beta=beta)

    def score_all(self,instances,states,gradients,tau=1e-6):
        """Pack exact local deltas on CPU, then batch both branch dot-products.

        Packing dominates this prototype; reported CUDA compute excludes none
        of it from optimizer end-to-end time. No precomputed stage-3 scores used.
        """
        b=self.b;m=self.m;shape=states['action_target_id'].shape;mask=states['action_mask'].copy()
        nets=list(m.by_net);net_index={net:i for i,net in enumerate(nets)}
        dest=[[],[],[],[]];source=[[],[],[],[]];value=[[],[],[],[]];gate=np.full(len(instances),np.inf)
        start=time.perf_counter()
        for row in instances:
            i=int(row['instance_id']);inst=row['name']
            if m.masters[inst]!=row['master']:raise ValueError('Stale instance state')
            pins=b.instance_pins.get(inst,[])
            if pins:gate[i]=np.max(np.abs(gradients['lse']['at'][pins]))
            for direction in range(4):
                if not mask[i,direction]:continue
                delta=b.action_delta(inst,m.catalog['cells'][int(states['action_target_id'][i,direction])]['name']);action=i*4+direction
                chunks=[(delta['arcs'],delta['delay']),([net_index[k] for k in delta['cap']],list(delta['cap'].values())),(delta['arcs'],delta['slew']),(delta['checks'],delta['setup'])]
                for j,(ids,vals) in enumerate(chunks):
                    dest[j].extend([action]*len(ids));source[j].extend(ids);value[j].extend(vals)
        self.last_pack_seconds=time.perf_counter()-start;start=time.perf_counter()
        packed=[(self.tensor(d,True),self.tensor(s,True),self.tensor(v)) for d,s,v in zip(dest,source,value)]
        result={}
        for name,g in gradients.items():
            gs=[g['delay'],[g['net_load'][net] for net in nets],g['local_slew'],g['setup']];cols=[]
            for data,(dst,src,val) in zip(gs,packed):cols.append(self.zeros(mask.size).scatter_add_(0,dst,self.tensor(data)[src]*val))
            scores=self.numpy(torch.stack(cols,dim=1)).reshape(*shape,4);scores[~mask]=np.nan;result[name]=scores
        self.sync();self.last_score_seconds=time.perf_counter()-start
        safe=gate<=tau
        return dict(components=result,delta_loss={k:v.sum(2) for k,v in result.items()},benefit={k:-v.sum(2) for k,v in result.items()},mask=mask,gate_lse_at=gate,timing_safe=safe,power_candidate_mask=mask&safe[:,None]&np.array([False,True,True,False]))
