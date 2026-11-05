"""Library NLPM internal energy, switching and conditional leakage power.

Units are ps/fF at LUT inputs, joules for table values, watts at outputs.
Activities are reference Boolean propagation, frozen across equivalent sizing.
The input-duty and output-energy weighting follow local OpenSTA Power.cc.
"""
import hashlib,itertools,pickle
from pathlib import Path
from functools import lru_cache
import numpy as np
from cell_space import entries,clean,attributes,unquote,boolean_key
from forward_sta import Table,numbers,rows
ROOT = Path(__file__).resolve().parents[1]


def load_power_tables(catalog):
    paths=sorted({c['liberty_data']['liberty'] for c in catalog['cells'] if c.get('liberty_data')})
    h=hashlib.sha256(Path(__file__).read_bytes())
    for p in paths:h.update(Path(p).read_bytes())
    target=ROOT/'artifacts/stage48/cache'/('power_'+h.hexdigest()+'.pickle');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        with target.open('rb') as f:return pickle.load(f)
    result={}
    for path in paths:
        lib=next(v for k,v in entries(clean(Path(path).read_text())) if k=='library');parts=list(entries(lib.body));la=attributes(parts)
        if la['time_unit']!='1ps' or la['voltage_unit']!='1V' or la['leakage_power_unit']!='1pW':raise ValueError('Unsupported NLPM units')
        if dict(parts)['capacitive_load_unit'][1].replace(' ','')!='1,ff':raise ValueError('Unsupported capacitance unit')
        templates={unquote(v.args):dict(entries(v.body)) for k,v in parts if k=='power_lut_template'}
        for k,c in parts:
            if k!='cell':continue
            cp=list(entries(c.body));ca=attributes(cp);data=dict(voltage=float(la['nom_voltage']),pins={},leakage=[],default_leakage=float(ca.get('cell_leakage_power',0))*1e-12)
            pg={unquote(g.args):attributes(list(entries(g.body))).get('pg_type') for k,g in cp if k=='pg_pin'}
            for k,g in cp:
                if k=='leakage_power':
                    a=attributes(list(entries(g.body)))
                    if pg.get(a.get('related_pg_pin'),'primary_power')=='primary_power':data['leakage'].append(dict(pg=a.get('related_pg_pin',''),when=boolean_key(a.get('when')),value=float(a['value'])*1e-12))
                if k!='pin':continue
                pp=list(entries(g.body));pa=attributes(pp);powers=[];senses={}
                for pk,p in pp:
                    if pk=='timing':
                        ta=attributes(list(entries(p.body)));senses.setdefault(ta.get('related_pin'),ta.get('timing_sense','positive_unate'))
                    if pk!='internal_power':continue
                    ip=list(entries(p.body));a=attributes(ip);tabs={}
                    for kind,tab in ip:
                        if kind not in ('rise_power','fall_power','power'):continue
                        z=dict(entries(tab.body));tn=unquote(tab.args)
                        if tn=='scalar':
                            val=numbers(z['values'][1])[0];table=Table([0,1],[0,1],np.full((2,2),val*1e-15))
                        else:
                            tmp=templates[tn]
                            if tmp.get('variable_1')!='input_transition_time' or tmp.get('variable_2') not in (None,'total_output_net_capacitance'):raise ValueError(f'Unsupported power axes {tmp}')
                            x=numbers(z.get('index_1',tmp.get('index_1'))[1]);val=numbers(z['values'][1])*1e-15
                            if tmp.get('variable_2') is None:y=np.array([0.,1.]);val=np.repeat(val[:,None],2,axis=1)
                            else:y=numbers(z.get('index_2',tmp.get('index_2'))[1])
                            table=Table(x,y,val)
                        if kind=='power':tabs['rise']=table;tabs['fall']=table
                        else:tabs[kind.split('_')[0]]=table
                    powers.append(dict(related=a.get('related_pin'),when=boolean_key(a.get('when')),pg=a.get('related_pg_pin',''),tables=tabs))
                data['pins'][unquote(g.args)]=dict(direction=pa['direction'],function=boolean_key(pa.get('function')),powers=powers,senses=senses)
            result[unquote(c.args)]=data
    with target.open('wb') as f:pickle.dump(result,f)
    return result


@lru_cache(maxsize=None)
def truth_masks(variables,truth):
    bits=np.array(list(itertools.product((0,1),repeat=len(variables))),dtype=np.int8)
    return bits,np.array([x=='1' for x in truth])

def probability(expr,duties):
    if expr is None:return 1.
    names=tuple(expr['variables']);bits,truth=truth_masks(names,expr['truth_table'])
    p=np.array([duties[n] for n in names]);weights=np.prod(np.where(bits,p,1-p),axis=1)
    return float(weights[truth].sum())

def sensitivity(expr,pin,duties):
    names=tuple(expr['variables']);bits,truth=truth_masks(names,expr['truth_table']);j=names.index(pin)
    flipped=np.arange(len(truth))^(1<<(len(names)-1-j));diff=truth!=truth[flipped]
    p=np.array([duties[n] for n in names]);weights=np.prod(np.where(bits,p,1-p),axis=1)
    return float(weights[diff].sum())


class PowerModel:
    def __init__(self,model,power_tables,reference=None):
        self.model=model;self.tables=power_tables;self.activities={r['pin']:(float(r['density_hz']),float(r['duty'])) for r in rows(Path(reference or model.reference)/'activity.tsv')}
        self.inst_pins={}
        for i,(inst,pin) in enumerate(model.pin_parts):
            if inst is not None:self.inst_pins.setdefault(inst,{})[pin]=i
        self.state={};self._factors={};self._leakage={}
        for inst,pins in self.inst_pins.items():
            state={pin:self.activities[model.pins[i]['name']] for pin,i in pins.items()}
            state.update(VDD=(0.,1.),VSS=(0.,0.),GND=(0.,0.))
            lib=model.cells[model.masters[inst]].get('liberty_data')
            if lib:
                for pin,p in lib['pins'].items():
                    f=p.get('function_key')
                    if f and len(f['variables'])==1 and pin in state:
                        var=f['variables'][0]
                        if var not in state:
                            density,duty=state[pin];state[var]=(density,duty if f['truth_table']=='01' else 1-duty)
                for seq in lib['behavior']['state']:
                    a,b=seq['state'].split(',')
                    if a in state and b not in state:state[b]=(state[a][0],1-state[a][1])
                    if b in state and a not in state:state[a]=(state[b][0],1-state[b][1])
            self.state[inst]=state

    def factors(self,inst,master):
        key=(inst,master)
        if key in self._factors:return self._factors[key]
        data=self.tables[master];state=self.state[inst];duties={p:v[1] for p,v in state.items()};factors=[];switches=[]
        for pin,p in data['pins'].items():
            if pin not in self.inst_pins[inst]:continue
            density=state[pin][0];output=p['direction']=='output';denom={};pduty=[]
            if output:switches.append((self.inst_pins[inst][pin],.5*data['voltage']**2*density*1e-15))
            for term in p['powers']:
                related=term['related'];fun=p['function'];when=term['when']
                if output:
                    duty=sensitivity(fun,related,duties) if fun and related in fun['variables'] else probability(when,duties) if when else .5
                    if related is None:duty=0.
                    d=state[related][0] if related else 0.;denom[term['pg']]=denom.get(term['pg'],0)+d*duty;pduty.append(d*duty)
                else:
                    duty=probability(when,duties)
                    if when:
                        outpins=[q for q in when['variables'] if q in data['pins'] and data['pins'][q]['direction']=='output']
                        if outpins:
                            f=data['pins'][outpins[0]]['function']
                            if f and pin in f['variables']:duty=sensitivity(f,pin,duties)
                    pduty.append(duty)
            for term,weight in zip(p['powers'],pduty):
                if output:weight=weight/denom[term['pg']] if denom.get(term['pg'],0)>0 else 0.
                src=term['related'] if output else pin;srcid=self.inst_pins[inst].get(src,-1)
                positive=p['senses'].get(src,'positive_unate')!='negative_unate';dst=self.inst_pins[inst][pin] if output else -1
                factors.append((srcid,dst,positive,weight*density,term['tables']))
        leakage=self.leakage(inst,master)
        self._factors[key]=(factors,switches,leakage);return self._factors[key]

    def leakage(self,inst,master):
        """Conditional leakage without constructing unused internal-power LUT queries."""
        key=(inst,master)
        if key in self._leakage:return self._leakage[key]
        data=self.tables[master];duties={p:v[1] for p,v in self.state[inst].items()}
        leakage=0.;groups={}
        for term in data['leakage']:groups.setdefault(term['pg'],[]).append(term)
        for terms in groups.values():
            cond=[t for t in terms if t['when'] is not None]
            if cond:
                probs=[probability(t['when'],duties) for t in cond]
                # Constant-true expressions override probabilistic averaging.
                certain=[t['value'] for t,p in zip(cond,probs) if p>=1.-1e-15]
                if certain:leakage+=certain[-1]
                else:leakage+=sum(t['value']*p for t,p in zip(cond,probs))+data['default_leakage']*(1-sum(p for t,p in zip(cond,probs) if t['value']>0))
            else:leakage+=terms[-1]['value']
        if not groups:leakage=data['default_leakage']
        self._leakage[key]=leakage;return leakage

    def instance(self,inst,master,forward):
        if master not in self.tables:return np.zeros(3)
        factors,switches,leak=self.factors(inst,master);internal=0.
        for src,dst,positive,rate,tabs in factors:
            energy=0.;count=0
            for rf,kind in enumerate(('rise','fall')):
                slew=forward['slew'][src,rf if positive else 1-rf] if src>=0 else 0.
                if np.isfinite(slew) and abs(slew)<1e25:
                    energy+=tabs[kind](slew,forward['load'][dst] if dst>=0 else 0.) if kind in tabs else 0.;count+=1
            internal+=rate*energy/max(count,1)
        switching=sum(forward['load'][pin]*coef for pin,coef in switches)
        return np.array([internal,switching,leak])

    def run(self,forward,replacements=None):
        masters={**self.model.masters,**(replacements or {})};per={i:self.instance(i,masters[i],forward) for i in self.inst_pins}
        total=sum(per.values(),np.zeros(3));return dict(internal_w=float(total[0]),switching_w=float(total[1]),leakage_w=float(total[2]),total_w=float(total.sum()),instances=per)

    def local_delta(self,inst,target,forward):
        # NLPM change of the sized cell, plus switching energy on its inputs'
        # upstream nets. Full candidate run also includes propagated slew/load.
        old=self.model.masters[inst];delta=self.instance(inst,target,forward)-self.instance(inst,old,forward)
        m=self.model
        for pin,i in self.inst_pins[inst].items():
            a=m.cells[old]['liberty_data']['pins'][pin];b=m.cells[target]['liberty_data']['pins'][pin]
            dc=b.get('capacitance_ff',0)-a.get('capacitance_ff',0)
            if dc:
                delta[1]+=.5*self.tables[target]['voltage']**2*self.activities[m.pins[i]['name']][0]*dc*1e-15
        return float(delta.sum())
