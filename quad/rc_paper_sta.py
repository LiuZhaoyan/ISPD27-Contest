"""Hierarchical paper timer with per-sink RC-tree load derivatives."""
import copy
import numpy as np
from ratio_sta import RatioForwardModel,RatioBackwardModel
from rc_tree import build_sensitivity

class RCPaperModel(RatioForwardModel):
    def pin_caps(self,masters):
        if getattr(self,'analysis','max')=='min':
            from minmax_caps import pin_capacitances
            return pin_capacitances(self,self.cap_ranges,'min',masters)
        result=np.zeros((self.n,2))
        for i,(inst,pin) in enumerate(self.pin_parts):
            if inst is not None:
                data=self.cells[masters[inst]].get('liberty_data',{}).get('pins',{}).get(pin,{})
                result[i]=[data.get(rf+'_capacitance_ff',data.get('capacitance_ff',0.)) for rf in ('rise','fall')]
        return result

    def calibrate(self):
        # Legacy stage-2 dumps used Pin.net() for top ports. OpenSTA top pins
        # connect through Term.net(); repair those old dumps from wire edges.
        repaired=0
        for a in self.arcs:
            if a['role']!='wire':continue
            s,t=a['s'],a['t']
            for empty,known in ((s,t),(t,s)):
                if not self.pins[empty]['net'] and self.pins[known]['net']:
                    self.pins[empty]=dict(self.pins[empty],net=self.pins[known]['net'],wire_ff=self.pins[known].get('wire_ff',0));self.base_cap[empty]=self.base_cap[known];repaired+=1
        self.by_net={}
        for i,p in enumerate(self.pins):self.by_net.setdefault(p['net'],[]).append(i)
        self.rc_matrix,self.rc_statistics=build_sensitivity(self,self.reference/'network.spef')
        self.rc_statistics['repaired_legacy_port_nets']=repaired
        self.rc_base_tau=np.array([a['elmore'] for a in self.arcs]);self.rc_pin_caps=self.pin_caps(self.masters);self.rc_current_tau=self.rc_base_tau.copy()
        super().calibrate()

    def wire_delays(self,masters):
        tau=self.rc_base_tau+self.rc_matrix@(self.pin_caps(masters)-self.rc_pin_caps).reshape(-1)
        if np.min(tau)<-1e-5:raise ValueError('Negative physical Elmore after capacitance replacement')
        return np.maximum(tau,0)

    def run(self,replacements=None,mode='calibrated'):
        self.rc_current_tau=self.wire_delays({**self.masters,**(replacements or {})})
        return super().run(replacements,mode)

    def arc_value(self,a,slew,cap,masters):
        if a['role']=='wire':
            tau=self.rc_current_tau[a['arc_index']]
            return tau,np.hypot(slew[a['s'],a['fr']],2.197224577*tau)
        return super().arc_value(a,slew,cap,masters)

class RCPaperBackward(RatioBackwardModel):
    def __init__(self,m,r):
        super().__init__(m,r)
        wire=np.array([a['role']=='wire' for a in m.arcs]);self.delay_dc[wire]=0.;self.slew_dc[wire]=0.

    def rc_gradient(self,g):
        tau_grad=g['delay']+g['local_slew']*2.197224577**2*self.forward['delay']/np.maximum(self.local_slew,1e-300)
        return np.asarray(self.model.rc_matrix.T@tau_grad).reshape(-1,2)

    def backward(self,*args,**kwargs):
        g=super().backward(*args,**kwargs);g['pin_load']=self.rc_gradient(g);return g

    def action_delta(self,inst,target):
        d=super().action_delta(inst,target);m=self.model;old=m.cells[m.masters[inst]]['liberty_data']['pins'];new=m.cells[target]['liberty_data']['pins']
        d['pin_cap']={i:np.array([new[m.pin_parts[i][1]].get(rf+'_capacitance_ff',new[m.pin_parts[i][1]].get('capacitance_ff',0))-old[m.pin_parts[i][1]].get(rf+'_capacitance_ff',old[m.pin_parts[i][1]].get('capacitance_ff',0)) for rf in ('rise','fall')]) for i in self.instance_pins.get(inst,[])}
        return d

    @staticmethod
    def score_delta(g,d):
        c=RatioBackwardModel.score_delta(g,d)
        c[1]+=sum(g['pin_load'][i]@v for i,v in d['pin_cap'].items());return c

def rebase_rc(model,forward,replacements):
    masters={**model.masters,**replacements}
    for a in model.arcs+model.checks:
        inst=model.pin_parts[a['t']][0]
        if a['role']!='wire' and inst in replacements and model.changed_groups(a,masters):return None
    m=copy.copy(model);m.masters=masters;m.base_cap=forward['load'].copy();m.envelope_pairs=set();m.ref_at=forward['at'].copy();m.ref_slew=forward['slew'].copy();m.arcs=[]
    m.rc_base_tau=model.rc_base_tau.copy()
    for k,a in enumerate(model.arcs):
        a=a.copy()
        if a['role']=='wire':a['elmore']=forward['delay'][k];m.rc_base_tau[k]=forward['delay'][k]
        m.arcs.append(a)
    m.rc_pin_caps=m.pin_caps(masters);m.rc_current_tau=m.rc_base_tau.copy();return m
