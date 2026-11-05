"""Transition-specific cell loads plus the reference route RC network.

OpenSTA connectedCap(net,max) is max(sum rise pin C,sum fall pin C)+wire C.
The cell NLDM queries use the corresponding transition's total, not the sum
of per-pin maxima. Wire resistance and frozen ratio calibration are unchanged.
"""
import numpy as np
from backward_sta import table_partials
from ratio_sta import RatioForwardModel
from rc_paper_sta import RCPaperModel,RCPaperBackward,rebase_rc

class RFPaperModel(RCPaperModel):
    def calibrate(self):
        self.rf_ready=False
        super().calibrate()
        pin=self.pin_caps(self.masters);self.rf_base_load=np.zeros((self.n,2))
        self.rf_sink=np.zeros(self.n,dtype=bool)
        for i,(inst,name) in enumerate(self.pin_parts):
            if inst is not None:self.rf_sink[i]=self.cells[self.masters[inst]].get('liberty_data',{}).get('pins',{}).get(name,{}).get('direction')=='input'
        for net,pins in self.by_net.items():
            pc=pin[pins][self.rf_sink[pins]].sum(0)
            # Same reference net and fixed external/port load for both RFs.
            fixed=float(np.max(self.base_cap[pins]))-float(pc.max())
            if fixed < -(1e-6+float(pc.max())*1e-5):raise ValueError(f'Pin capacitance exceeds native connected load: {net}: {fixed}')
            self.rf_base_load[pins]=np.maximum(0.,pc+fixed)
        self.rf_base_pin=pin;self.rf_current_load=self.rf_base_load.copy();self.rf_ready=True
        RatioForwardModel.calibrate(self)
        self.load_ratio[:]=1.;self.load_offset[:]=0.

    def loads_rf(self,masters):
        delta=self.pin_caps(masters)-self.rf_base_pin;result=self.rf_base_load.copy()
        for pins in self.by_net.values():result[pins]+=delta[pins][self.rf_sink[pins]].sum(0)
        if result.min() < -1e-6:raise ValueError('Negative transition-specific load')
        return np.maximum(result,0.)

    def caps(self,masters):
        if not self.rf_ready:return super().caps(masters)
        return self.loads_rf(masters).max(1)

    def run(self,replacements=None,mode='calibrated'):
        self.rf_current_load=self.loads_rf({**self.masters,**(replacements or {})})
        result=super().run(replacements,mode);result['load_rf']=self.rf_current_load.copy();return result

    def arc_value(self,a,slew,cap,masters):
        if a['role']=='wire' or not self.rf_ready:return super().arc_value(a,slew,cap,masters)
        x=slew[a['s'],a['fr']];y=self.rf_current_load[a['t'],a['to']];rf=('rise','fall')[a['to']]
        groups=self.changed_groups(a,masters)
        tabs=self.table(a,self.masters if groups else masters)
        d=tabs['cell_'+rf](x,y);sl=max(0.,tabs[rf+'_transition'](x,y))
        if groups:
            before,after=groups
            d+=max(t['cell_'+rf](x,y) for t in after.values())-max(t['cell_'+rf](x,y) for t in before.values())
            sl+=max(max(0.,t[rf+'_transition'](x,y)) for t in after.values())-max(max(0.,t[rf+'_transition'](x,y)) for t in before.values())
        return d,max(0.,sl)

class RFPaperBackward(RCPaperBackward):
    def __init__(self,m,r):
        m.rf_current_load=r['load_rf'].copy()
        super().__init__(m,r)
        for k,a in enumerate(m.arcs):
            if a['role']=='wire':continue
            tab=m.table(a,m.masters);rf=('rise','fall')[a['to']];x=r['slew'][a['s'],a['fr']];y=r['load_rf'][a['t'],a['to']]
            self.delay_dx[k],self.delay_dc[k]=table_partials(tab['cell_'+rf],x,y)
            if self.local_slew[k]>0:self.slew_dx[k],self.slew_dc[k]=table_partials(tab[rf+'_transition'],x,y)
        self.cell_ids=np.array([k for k,a in enumerate(m.arcs) if a['role']!='wire'],dtype=int)
        self.cell_dst=np.array([m.arcs[k]['t']*2+m.arcs[k]['to'] for k in self.cell_ids])

    def rc_gradient(self,g):
        pin=super().rc_gradient(g);ids=self.cell_ids
        cap=np.bincount(self.cell_dst,weights=g['delay'][ids]*self.delay_dc[ids]+g['local_slew'][ids]*self.slew_dc[ids],minlength=self.model.n*2).reshape(-1,2)
        for pins in self.model.by_net.values():
            sinks=np.asarray(pins)[self.model.rf_sink[pins]]
            pin[sinks]+=cap[pins].sum(0)
        # Scalar nominal-cap path is replaced, never counted a second time.
        g['load'][:]=0.;g['net_load']={net:0. for net in self.model.by_net}
        return pin

def rebase_rf(model,forward,replacements):
    m=rebase_rc(model,forward,replacements)
    if m is None:return None
    m.rf_base_load=forward['load_rf'].copy();m.rf_base_pin=m.pin_caps(m.masters);m.rf_current_load=m.rf_base_load.copy()
    return m
