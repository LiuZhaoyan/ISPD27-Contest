"""Stage-2-compatible local timing model for changed Liberty when partitions.

An equivalent-cell move may merge/split conditional timing groups. Match groups
by physical pins, role and input/output RF, and shift the old arc bundle by the
change in its worst-case raw LUT envelope. This preserves the baseline calibrated
graph while allowing all legal cell moves. It is a fixed-graph local surrogate;
STA must rebuild/recalibrate topology after accepted multi-round changes.
"""
import json
from functools import lru_cache
from forward_sta import ForwardModel,MAX_CHECKS


class MoveForwardModel(ForwardModel):
    @classmethod
    def wrap(cls,model):
        result=cls.__new__(cls)
        result.__dict__=model.__dict__.copy()
        result.envelope_pairs=set()
        return result

    @lru_cache(maxsize=None)
    def group(self,master,src,dst,role,fr,to):
        cells=self.cells[master]['liberty_data']
        found={}
        for pin,related,typ,sense,cond in cells['timing_arc_signatures']:
            if pin!=dst or related!=src:continue
            if role in MAX_CHECKS:
                if typ not in (role+'_rising',role+'_falling') or fr!=int(typ==role+'_falling'):continue
            elif role=='Reg Clk to Q':
                if typ not in ('rising_edge','falling_edge') or fr!=int(typ=='falling_edge'):continue
            elif role=='combinational':
                if not typ.startswith('combinational'):continue
                if typ=='combinational_rise' and to!=0:continue
                if typ=='combinational_fall' and to!=1:continue
                if sense=='positive_unate' and fr!=to:continue
                if sense=='negative_unate' and fr==to:continue
            else:raise ValueError(role)
            key=(master,src,dst,role,json.dumps(cond,sort_keys=True))
            if key not in self.tables:raise ValueError(f'Missing Liberty group {key}')
            rf=('rise','fall')[to]
            kind=rf+'_constraint' if role in MAX_CHECKS else 'cell_'+rf
            if kind in self.tables[key]:found[key[-1]]=self.tables[key]
        if not found:raise ValueError(f'Missing RF-compatible timing group {(master,src,dst,role,fr,to)}')
        return found

    def changed_groups(self,a,masters):
        inst,pin=self.pin_parts[a['t']]
        old,new=self.masters[inst],masters[inst]
        if old==new:return None
        src=self.pin_parts[a['s']][1]
        suffix=(src,pin,a['role'],a['fr'],a['to'])
        before=self.group(old,*suffix);after=self.group(new,*suffix)
        if before.keys()==after.keys():return None
        self.envelope_pairs.add((old,new))
        return before,after

    def arc_value(self,a,slew,cap,masters):
        if a['role']=='wire':return super().arc_value(a,slew,cap,masters)
        groups=self.changed_groups(a,masters)
        if groups is None:return super().arc_value(a,slew,cap,masters)
        before,after=groups
        x,y=slew[a['s'],a['fr']],cap[a['t']]
        rf=('rise','fall')[a['to']]
        d,sl=super().arc_value(a,slew,cap,self.masters)
        dd=max(t['cell_'+rf](x,y) for t in after.values())-max(t['cell_'+rf](x,y) for t in before.values())
        ds=max(max(0.,t[rf+'_transition'](x,y)) for t in after.values())-max(max(0.,t[rf+'_transition'](x,y)) for t in before.values())
        return d+dd,max(0.,sl+ds)

    def check_value(self,a,slew,masters):
        groups=self.changed_groups(a,masters)
        if groups is None:return super().check_value(a,slew,masters)
        before,after=groups
        x,y=slew[a['t'],a['to']],slew[a['s'],a['fr']]
        kind=('rise','fall')[a['to']]+'_constraint'
        difference=max(t[kind](x,y) for t in after.values())-max(t[kind](x,y) for t in before.values())
        return super().check_value(a,slew,self.masters)+difference
