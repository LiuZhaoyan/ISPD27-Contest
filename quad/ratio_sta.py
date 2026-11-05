"""Hierarchical frozen-ratio calibration (paper section 3.4), in ps/fF.

Loads are calibrated first; each node then sees calibrated input AT/slew.
Ratios are differentiated through, but never refitted during a candidate test.
Zero/nonfinite denominators use an explicitly recorded affine fallback.
"""
import numpy as np
from forward_sta import RF,MAX_CHECKS
from move_forward_sta import MoveForwardModel
from backward_sta import BackwardModel,endpoint_seeds,max_weights,bundle_weights


def fit_ratio(predicted,reference,tolerance=1e-10):
    p,r=np.asarray(predicted),np.asarray(reference)
    ratio=np.ones_like(p);offset=np.zeros_like(p)
    finite=np.isfinite(p)&np.isfinite(r)&(np.abs(r)<1e25)
    safe=finite&(np.abs(p)>tolerance)&((np.signbit(p)==np.signbit(r))|(r==0))
    ratio[safe]=r[safe]/p[safe]
    fallback=finite&~safe
    offset[fallback]=r[fallback]-p[fallback]
    return ratio,offset,fallback


class RatioForwardModel(MoveForwardModel):
    def calibrate(self):
        # Derive physical load from wire + library sink pin capacitances.
        # Preserve port/external loads by obtaining their residual from reference.
        raw=np.zeros(self.n)
        for net,pins in self.by_net.items():
            sinks=0.
            for i in pins:
                inst,pin=self.pin_parts[i]
                if inst is not None:
                    data=self.cells[self.masters[inst]].get('liberty_data',{}).get('pins',{}).get(pin,{})
                    if data.get('direction')=='input':sinks+=data.get('capacitance_ff',0.)
            wire=max(float(self.pins[i].get('wire_ff',0)) for i in pins)
            external=max(0.,float(self.base_cap[pins[0]])-wire-sinks)
            raw[pins]=wire+sinks+external
        self.load_ratio,self.load_offset,lf=fit_ratio(raw,self.base_cap)
        self.raw_base_cap=raw
        candidates=np.full((self.n,2),-np.inf);transitions=np.zeros((self.n,2))
        self.delay_correction[:]=0.;self.slew_correction[:]=0.;self.check_correction[:]=0.
        # ref_at/ref_slew supply exactly the already-calibrated previous layer.
        for a in self.arcs:
            d,sl=self.arc_value(a,self.ref_slew,self.base_cap,self.masters)
            t,rf=a['t'],a['to']
            candidates[t,rf]=max(candidates[t,rf],self.ref_at[a['s'],a['fr']]+d)
            transitions[t,rf]=max(transitions[t,rf],sl)
        self.at_ratio,self.at_offset,af=fit_ratio(candidates,self.ref_at)
        self.slew_ratio,self.slew_offset,sf=fit_ratio(transitions,self.ref_slew)
        rawcheck=np.array([self.check_value(a,self.ref_slew,self.masters) for a in self.checks])
        self.check_ratio,self.check_offset,qf=fit_ratio(rawcheck,np.array([a['delay'] for a in self.checks]))
        self.calibration_fallbacks=dict(load=int(lf.sum()),at=int(af.sum()),slew=int(sf.sum()),setup=int(qf.sum()))

    def caps(self,masters):
        # The superclass returns reference net load + raw library pin deltas.
        return self.base_cap+(super().caps(masters)-self.base_cap)*self.load_ratio

    def finish(self,at,slew,cap,delays,checks):
        eps=[];worst={}
        for e in self.endpoints:
            t=self.index[e['pin']];rf=RF[e['rf']];required=float(e['required_ps'])+float(e.get('path_at_shift_ps',0.))
            if e['role'] in MAX_CHECKS:required+=float(e['margin_ps'])-max(checks[k] for k in self.check_incoming[t] if self.checks[k]['to']==rf)
            slack=required-at[t,rf];eps.append((t,rf,required,slack));worst[t]=min(worst.get(t,np.inf),slack)
        vals=np.array(list(worst.values()))
        return dict(at=at,slew=slew,load=cap,delay=delays,check=checks,endpoints=np.array(eps),tns_ps=float(np.minimum(vals,0).sum()),wns_ps=float(min(0,vals.min())))

    def run(self,replacements=None,mode='calibrated'):
        if mode!='calibrated':raise ValueError('Use the original model for raw/oracle diagnostics')
        masters={**self.masters,**(replacements or {})}
        for inst,target in (replacements or {}).items():
            if self.cells[self.masters[inst]]['family_id']!=self.cells[target]['family_id'] or self.cells[target].get('frozen_reason'):raise ValueError('Illegal replacement')
        cap=self.caps(masters);at=np.full((self.n,2),-np.inf);slew=np.zeros((self.n,2));delays=np.zeros(len(self.arcs))
        for t in self.order:
            ids=self.incoming[t]
            if not ids:at[t]=self.ref_at[t];slew[t]=self.ref_slew[t];continue
            for k in ids:
                a=self.arcs[k];d,sl=self.arc_value(a,slew,cap,masters);delays[k]=d
                rf=a['to'];at[t,rf]=max(at[t,rf],at[a['s'],a['fr']]+d);slew[t,rf]=max(slew[t,rf],sl)
            at[t]=at[t]*self.at_ratio[t]+self.at_offset[t]
            slew[t]=np.maximum(0,slew[t]*self.slew_ratio[t]+self.slew_offset[t])
        checks=np.array([self.check_value(a,slew,masters) for a in self.checks])*self.check_ratio+self.check_offset
        return self.finish(at,slew,cap,delays,checks)


class RatioBackwardModel(BackwardModel):
    def backward(self,branch='ave',epsilon=.01,gamma_ps=1.,alpha=1.,beta=0.):
        m,r=self.model,self.forward;gat=np.zeros((m.n,2));gs=np.zeros_like(gat);gc=np.zeros(m.n)
        gd=np.zeros(len(m.arcs));gsl=np.zeros_like(gd);gq=np.zeros(len(m.checks));wat=np.zeros_like(gd);wsl=np.zeros_like(gd)
        seeds=endpoint_seeds(r,alpha,beta)
        for i,e in enumerate(m.endpoints):
            t,rf=map(int,r['endpoints'][i,:2]);gat[t,rf]+=seeds[i]
            if e['role'] in MAX_CHECKS:
                ids=[k for k in m.check_incoming[t] if m.checks[k]['to']==rf];gq[ids]+=seeds[i]*max_weights(r['check'][ids],'hard')
        gq*=m.check_ratio
        for k,a in enumerate(m.checks):
            gs[a['t'],a['to']]+=gq[k]*self.check_dx[k,0];gs[a['s'],a['fr']]+=gq[k]*self.check_dx[k,1]
        for t in m.order[::-1]:
            for rf in (0,1):
                for ids,w in bundle_weights(self.at_candidate,self.bundles[t][rf],branch,epsilon,gamma_ps):wat[ids]=w
                for ids,w in bundle_weights(self.local_slew,self.bundles[t][rf],branch,epsilon,gamma_ps):wsl[ids]=w
                for k in self.groups[t][rf]:
                    a=m.arcs[k];s,fr=a['s'],a['fr']
                    gd[k]=gat[t,rf]*m.at_ratio[t,rf]*wat[k]
                    gsl[k]=gs[t,rf]*m.slew_ratio[t,rf]*wsl[k] if r['slew'][t,rf]>0 else 0.
                    gat[s,fr]+=gd[k];gs[s,fr]+=gd[k]*self.delay_dx[k]+gsl[k]*self.slew_dx[k]
                    cp=s if a['role']=='wire' else t;gc[cp]+=gd[k]*self.delay_dc[k]+gsl[k]*self.slew_dc[k]
        gc*=m.load_ratio
        return dict(at=gat,slew=gs,load=gc,delay=gd,local_slew=gsl,setup=gq,at_weights=wat,slew_weights=wsl,endpoint_seeds=seeds,net_load={net:float(gc[pins].sum()) for net,pins in m.by_net.items()},branch=branch,epsilon=epsilon,gamma_ps=gamma_ps,alpha=alpha,beta=beta)
