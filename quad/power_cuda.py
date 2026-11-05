"""GPU NLPM local action differences using the audited CPU factor weights.

CPU constructs Boolean/activity metadata. CUDA interpolates all candidate energy
tables together and reduces the signed old/new contributions per action. Tables
may have different dimensions; padding never changes their extrapolation range.
"""
import numpy as np
import torch

class PowerCuda:
    def __init__(self,power,engine):
        self.power=power;self.engine=engine;self.device=engine.device
        tabs=[];self.ids={}
        for cell in power.tables.values():
            for pin in cell['pins'].values():
                for term in pin['powers']:
                    for table in term['tables'].values():
                        if id(table) not in self.ids:self.ids[id(table)]=len(tabs);tabs.append(table)
        nx=max(len(t.x) for t in tabs);ny=max(len(t.y) for t in tabs)
        xs=np.full((len(tabs),nx),np.inf);ys=np.full((len(tabs),ny),np.inf);zs=np.zeros((len(tabs),nx,ny))
        for i,t in enumerate(tabs):xs[i,:len(t.x)]=t.x;ys[i,:len(t.y)]=t.y;zs[i,:len(t.x),:len(t.y)]=t.values
        self.x=engine.tensor(xs);self.y=engine.tensor(ys);self.z=engine.tensor(zs)
        self.nx=engine.tensor([len(t.x) for t in tabs],True);self.ny=engine.tensor([len(t.y) for t in tabs],True)
        self.cache={}
        self.total_cache={}

    def total(self,forward,replacements=None,instances=None):
        """Full NLPM at propagated candidate slews/loads, in watts."""
        e=self.engine;p=self.power;masters={**p.model.masters,**(replacements or {})}
        tables=[];sources=[];loads=[];weights=[];switch_pins=[];switch_weights=[];leakage=0.
        for inst in p.inst_pins if instances is None else instances:
            master=masters[inst]
            if master not in p.tables:continue
            key=(inst,master)
            if key not in self.total_cache:
                factors,switches,leak=p.factors(inst,master);q=[]
                for src,dst,positive,rate,tabs in factors:
                    for rf,kind in enumerate(('rise','fall')):
                        if kind in tabs:q.append((self.ids[id(tabs[kind])],src*2+(rf if positive else 1-rf) if src>=0 else -1,dst,rate*.5))
                self.total_cache[key]=(q,switches,leak)
            q,switches,leak=self.total_cache[key];leakage+=leak
            for tid,src,dst,weight in q:
                tables.append(tid);sources.append(src);loads.append(dst);weights.append(weight)
            for pin,weight in switches:switch_pins.append(pin);switch_weights.append(weight)
        sl=e.tensor(np.asarray(forward['slew']).reshape(-1));cap=e.tensor(forward['load'])
        if not bool((torch.isfinite(sl)&(sl.abs()<1e25)).all()):raise ValueError('Nonfinite full NLPM slew')
        internal=0.;switching=0.
        if tables:
            tid=e.tensor(tables,True);src=e.tensor(sources,True);dst=e.tensor(loads,True)
            x=torch.where(src>=0,sl[src.clamp_min(0)],0.);y=torch.where(dst>=0,cap[dst.clamp_min(0)],0.)
            internal=(self.lut(tid,x,y)*e.tensor(weights)).sum().item()
        if switch_pins:switching=(cap[e.tensor(switch_pins,True)]*e.tensor(switch_weights)).sum().item()
        return dict(internal_w=internal,switching_w=switching,leakage_w=leakage,total_w=internal+switching+leakage)

    def lut(self,ids,x,y):
        tx=self.x[ids];ty=self.y[ids];z=self.z[ids]
        i=torch.minimum((tx<=x[:,None]).sum(1).sub(1).clamp_min(0),self.nx[ids]-2)
        j=torch.minimum((ty<=y[:,None]).sum(1).sub(1).clamp_min(0),self.ny[ids]-2)
        row=torch.arange(len(ids),device=self.device)
        u=(x-tx[row,i])/(tx[row,i+1]-tx[row,i]);v=(y-ty[row,j])/(ty[row,j+1]-ty[row,j])
        return (1-u)*((1-v)*z[row,i,j]+v*z[row,i,j+1])+u*((1-v)*z[row,i+1,j]+v*z[row,i+1,j+1])

    def lut_parts(self,ids,x,y):
        tx=self.x[ids];ty=self.y[ids];z=self.z[ids]
        i=torch.minimum((tx<=x[:,None]).sum(1).sub(1).clamp_min(0),self.nx[ids]-2)
        j=torch.minimum((ty<=y[:,None]).sum(1).sub(1).clamp_min(0),self.ny[ids]-2)
        row=torch.arange(len(ids),device=self.device)
        dx=tx[row,i+1]-tx[row,i];dy=ty[row,j+1]-ty[row,j]
        u=(x-tx[row,i])/dx;v=(y-ty[row,j])/dy
        a=z[row,i,j];b=z[row,i,j+1];c=z[row,i+1,j];d=z[row,i+1,j+1]
        return (((1-u)*((1-v)*a+v*b)+u*((1-v)*c+v*d)),
                ((1-v)*(c-a)+v*(d-b))/dx, ((1-u)*(b-a)+u*(d-c))/dy)

    def internal_partials(self):
        """All cells' internal-power slew/load seeds in W/ps and W/fF.

        Switching due to changed input capacitance is already in local_delta,
        so it must not be counted again in these propagation seeds.
        """
        e=self.engine;p=self.power;tables=[];sources=[];loads=[];weights=[]
        for inst in p.inst_pins:
            master=p.model.masters[inst]
            if master not in p.tables:continue
            factors,_,_=p.factors(inst,master)
            for src,dst,positive,rate,tabs in factors:
                for rf,kind in enumerate(('rise','fall')):
                    if kind in tabs:
                        tables.append(self.ids[id(tabs[kind])])
                        sources.append(src*2+(rf if positive else 1-rf) if src>=0 else -1)
                        loads.append(dst);weights.append(rate*.5)
        gs=e.zeros(e.m.n*2);gc=e.zeros(e.m.n)
        if getattr(e,'gradient_backend','manual')=='autograd':
            # The scalar internal-power expression is differentiated by AD;
            # switching/leakage stay in local discrete differences, as before.
            with torch.enable_grad():
                sl=e._sl.detach().requires_grad_()
                cap=e._load.detach().requires_grad_()
                if tables:
                    ids=e.tensor(tables,True);src=e.tensor(sources,True);dst=e.tensor(loads,True);w=e.tensor(weights)
                    x=torch.where(src>=0,sl[src.clamp_min(0)],0.)
                    y=torch.where(dst>=0,cap.amax(1)[dst.clamp_min(0)],0.)
                    loss=(self.lut(ids,x,y)*w).sum()
                    a,b=torch.autograd.grad(loss,(sl,cap),allow_unused=True)
                    gs=torch.zeros_like(sl) if a is None else a
                    gcrf=torch.zeros_like(cap) if b is None else b
                else:
                    gs=torch.zeros_like(sl);gcrf=torch.zeros_like(cap)
            return e.numpy(gs).reshape(-1,2),e.numpy(gcrf)
        if tables:
            ids=e.tensor(tables,True);src=e.tensor(sources,True);dst=e.tensor(loads,True);w=e.tensor(weights)
            x=torch.where(src>=0,e._sl[src.clamp_min(0)],0.)
            y=torch.where(dst>=0,e._load.max(1).values[dst.clamp_min(0)],0.)
            _,dx,dy=self.lut_parts(ids,x,y)
            gs.scatter_add_(0,src.clamp_min(0),dx*w*(src>=0))
            gc.scatter_add_(0,dst.clamp_min(0),dy*w*(dst>=0))
        # PowerModel uses max(rise_load,fall_load). Split exact ties equally.
        active=(e._load==e._load.max(1).values[:,None]).double()
        gcrf=gc[:,None]*active/active.sum(1)[:,None]
        return e.numpy(gs).reshape(-1,2),e.numpy(gcrf)

    def descriptor(self,inst,target):
        p=self.power;m=p.model;old=m.masters[inst];key=(inst,old,target)
        if key in self.cache:return self.cache[key]
        queries=[];constant=0.
        for sign,master in ((-1.,old),(1.,target)):
            factors,switches,leakage=p.factors(inst,master);constant+=sign*leakage
            for src,dst,positive,rate,tabs in factors:
                for rf,kind in enumerate(('rise','fall')):
                    if kind in tabs:queries.append((self.ids[id(tabs[kind])],src*2+(rf if positive else 1-rf) if src>=0 else -1,dst,sign*rate*.5))
        # Equivalent cells have identical output activities/voltage and unchanged
        # output loads at a local action query, so their switching terms cancel.
        op=m.cells[old]['liberty_data']['pins'];npin=m.cells[target]['liberty_data']['pins']
        for pin,index in p.inst_pins[inst].items():
            dc=npin[pin].get('capacitance_ff',0.)-op[pin].get('capacitance_ff',0.)
            constant+=.5*p.tables[target]['voltage']**2*p.activities[m.pins[index]['name']][0]*dc*1e-15
        self.cache[key]=(queries,constant);return queries,constant

    def score(self,instances,targets,mask):
        e=self.engine;tables=[];sources=[];loads=[];weights=[];dest=[];constant=np.zeros(mask.size)
        for i,d in np.argwhere(mask):
            row=instances[i];target=e.m.catalog['cells'][targets[i,d]]['name'];q,k=self.descriptor(row['name'],target);action=int(i)*mask.shape[1]+int(d);constant[action]=k
            for table,src,load,weight in q:
                tables.append(table);sources.append(src);loads.append(load);weights.append(weight);dest.append(action)
        out=e.tensor(constant)
        if tables:
            tids=e.tensor(tables,True);src=e.tensor(sources,True);load=e.tensor(loads,True);dst=e.tensor(dest,True);weight=e.tensor(weights)
            x=torch.where(src>=0,e._sl[src.clamp_min(0)],0.)
            y=torch.where(load>=0,e._load.max(1).values[load.clamp_min(0)],0.)
            valid=torch.isfinite(x)&(x.abs()<1e25)
            # Match the CPU average over valid rise/fall slew values. All timing
            # model slews should be finite; fail rather than silently change it.
            if not bool(valid.all()):raise ValueError('Nonfinite NLPM slew: use validated CPU path')
            out.scatter_add_(0,dst,self.lut(tids,x,y)*weight)
        result=e.numpy(out).reshape(mask.shape);result[~mask]=np.nan;return result
