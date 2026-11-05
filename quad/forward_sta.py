"""Single-clock, max-delay NLDM forward model in ps/fF.

Topology, SDC boundary conditions and reference RC moments come from OpenSTA.
The independent forward pass never calls STA. Frozen baseline corrections are
an approximation to DMP effective capacitance / waveform propagation, not a
replacement for signoff STA. See reports/stage2_forward_sta.md.
"""
import csv
from functools import lru_cache
from bisect import bisect_right
import json
from pathlib import Path
import re
import numpy as np
from cell_space import entries, clean, attributes, Group, unquote, boolean_key

RF = {'rise': 0, 'fall': 1}
PROPAGATING = {'wire', 'combinational', 'Reg Clk to Q'}
MAX_CHECKS = {'setup', 'recovery'}
MIN_CHECKS = {'hold', 'removal'}
CHECKS = MAX_CHECKS | MIN_CHECKS


@lru_cache(maxsize=4096)
def condition(expr):
    return json.dumps(boolean_key(expr or None), sort_keys=True)


def numbers(value):
    return np.array([float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', value)])


class Table:
    def __init__(self, x, y, values):
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.values = np.asarray(values).reshape(len(x), len(y))
        if any(len(a) < 2 or np.any(np.diff(a) <= 0) for a in (self.x, self.y)):
            raise ValueError('Expected increasing 2D NLDM axes')

    def __call__(self, x, y):
        # OpenSTA extrapolates using the closest pair, rather than clamping.
        i = min(max(bisect_right(self.x, x) - 1, 0), len(self.x) - 2)
        j = min(max(bisect_right(self.y, y) - 1, 0), len(self.y) - 2)
        u = (x-self.x[i])/(self.x[i+1]-self.x[i])
        v = (y-self.y[j])/(self.y[j+1]-self.y[j])
        z = self.values
        return (1-u)*((1-v)*z[i,j]+v*z[i,j+1])+u*((1-v)*z[i+1,j]+v*z[i+1,j+1])


def load_tables(catalog,selected_roles=None):
    result = {}
    paths = sorted({c['liberty_data']['liberty'] for c in catalog['cells'] if c.get('liberty_data')})
    for path in paths:
        lib = next(v for k,v in entries(clean(Path(path).read_text())) if k == 'library')
        parts = list(entries(lib.body))
        la = attributes(parts)
        if la['time_unit'] != '1ps' or dict(parts)['capacitive_load_unit'][1].replace(' ', '') != '1,ff':
            raise ValueError('This model currently requires ASAP7 ps/fF tables')
        templates = {unquote(v.args): dict(entries(v.body)) for k,v in parts if k == 'lu_table_template'}
        for k,cell in parts:
            if k != 'cell': continue
            name = unquote(cell.args)
            for pk,pin in entries(cell.body):
                if pk != 'pin': continue
                for tk,timing in entries(pin.body):
                    if tk != 'timing': continue
                    ts = list(entries(timing.body)); a = attributes(ts)
                    typ = a.get('timing_type','combinational')
                    if typ not in ('combinational', 'combinational_rise', 'combinational_fall', 'rising_edge', 'falling_edge', 'setup_rising', 'setup_falling','recovery_rising','recovery_falling','hold_rising','hold_falling','removal_rising','removal_falling'): continue
                    role = next((r for r in CHECKS if typ.startswith(r+'_')), 'Reg Clk to Q' if typ.endswith('_edge') else 'combinational')
                    if selected_roles is not None and role not in selected_roles:continue
                    tabs = {}
                    for kind,tab in ts:
                        if kind not in ('cell_rise','cell_fall','rise_transition','fall_transition','rise_constraint','fall_constraint'): continue
                        t = dict(entries(tab.body)); template = templates[unquote(tab.args)]
                        expected = ('constrained_pin_transition','related_pin_transition') if role in CHECKS else ('input_net_transition','total_output_net_capacitance')
                        if (template.get('variable_1'),template.get('variable_2')) != expected: raise ValueError('Unsupported LUT axes')
                        x = numbers(t.get('index_1',template.get('index_1'))[1])
                        y = numbers(t.get('index_2',template.get('index_2'))[1])
                        tabs[kind] = Table(x,y,numbers(t['values'][1]))
                    key = (name, a.get('related_pin'), unquote(pin.args),role,condition(a.get('when')))
                    if tabs:
                        if key in result: raise ValueError(f'Duplicate timing table key: {key}')
                        result[key] = tabs
    return result


def rows(path):
    with Path(path).open() as f: return list(csv.DictReader(f,delimiter='\t'))


class ForwardModel:
    def __init__(self, reference, catalog, instance_masters, tables, analysis='max'):
        if analysis not in ('min', 'max'):raise ValueError('Expected min/max analysis')
        self.analysis=analysis
        self.check_roles=MAX_CHECKS if analysis=='max' else MIN_CHECKS
        self.reference, self.catalog, self.tables = Path(reference), catalog, tables
        self.cells = {c['name']: c for c in catalog['cells']}
        self.masters = dict(instance_masters)
        self.pins = rows(self.reference/'pins.tsv')
        self.index = {p['name']: i for i,p in enumerate(self.pins)}
        if len(self.index) != len(self.pins): raise ValueError('Bidirectional vertices not supported')
        self.n = len(self.pins)
        self.ref_at = np.array([[float(p['at_r']),float(p['at_f'])] for p in self.pins])
        self.ref_slew = np.array([[float(p['slew_r']),float(p['slew_f'])] for p in self.pins])
        self.level = np.array([int(p['level']) for p in self.pins])
        self.base_cap = np.array([float(p['cap_ff']) for p in self.pins])
        self.pin_parts = [p['name'].rsplit('/',1) if '/' in p['name'] else (None,p['name']) for p in self.pins]
        self.by_net = {}
        for i,p in enumerate(self.pins): self.by_net.setdefault(p['net'],[]).append(i)
        self.arcs = []; self.checks = []; self.incoming = [[] for _ in self.pins]
        self.outgoing = [[] for _ in self.pins]
        self.check_incoming = [[] for _ in self.pins]
        for row in rows(self.reference/'arcs.tsv'):
            if int(row['disabled']): continue
            role = row['role']
            if role not in PROPAGATING | MAX_CHECKS | {'hold','removal','width'}: raise ValueError(f'Unsupported role {role}')
            if role not in PROPAGATING | self.check_roles: continue
            s,t = self.index[row['src']],self.index[row['dst']]
            fr,to = RF[row['fr']],RF[row['to']]
            sense = row['sense']
            if role == 'combinational' and ((sense == 'positive_unate' and fr != to) or (sense == 'negative_unate' and fr == to)): continue
            a = dict(s=s,t=t,fr=fr,to=to,role=role,cond=condition(row['when']),delay=float(row['delay_ps']),elmore=float(row['elmore_ps']))
            if role in self.check_roles:
                self.check_incoming[t].append(len(self.checks)); self.checks.append(a)
            else:
                if self.level[s] >= self.level[t]: raise ValueError(f'Non-DAG edge {row}')
                self.incoming[t].append(len(self.arcs));self.outgoing[s].append(len(self.arcs));self.arcs.append(a)
        self.order = np.argsort(self.level,kind='stable')
        self.endpoints = rows(self.reference/'endpoints.tsv')
        self.delay_correction = np.zeros(len(self.arcs))
        self.slew_correction = np.zeros((self.n,2))
        self.check_correction = np.zeros(len(self.checks))
        # Verify every active library arc, including those without an arrival.
        for a in self.arcs+self.checks:
            if a['role'] != 'wire': self.table(a,self.masters)

    def table(self,a,masters):
        inst,pin = self.pin_parts[a['t']]
        srcpin = self.pin_parts[a['s']][1]
        return self.tables[(masters[inst],srcpin,pin,a['role'],a['cond'])]

    def caps(self,masters):
        cap = self.base_cap.copy()
        deltas = {}
        for i,(inst,pin) in enumerate(self.pin_parts):
            if inst is None or masters[inst] == self.masters[inst]: continue
            old = self.cells[self.masters[inst]]['liberty_data']['pins'][pin]
            new = self.cells[masters[inst]]['liberty_data']['pins'][pin]
            # Net.capacitance(max) uses nominal pin capacitance. Baseline cap
            # retains physical wire cap, port load, and all unchanged sinks.
            d = new.get('capacitance_ff',0)-old.get('capacitance_ff',0)
            net = self.pins[i]['net']; deltas[net] = deltas.get(net,0)+d
        for net,delta in deltas.items(): cap[self.by_net[net]] += delta
        if np.any(cap < -1e-8): raise ValueError('Negative net load')
        return np.maximum(cap,0)

    def arc_value(self,a,slew,cap,masters):
        s,t,fr,to = (a[x] for x in ('s','t','fr','to'))
        if a['role'] == 'wire':
            ratio = cap[s]/self.base_cap[s] if self.base_cap[s]>0 else 1.0
            tau = a['elmore']*ratio
            return tau, np.hypot(slew[s,fr],2.197224577*tau)
        tabs = self.table(a,masters); rf = ('rise','fall')[to]
        return tabs['cell_'+rf](slew[s,fr],cap[t]), max(0.,tabs[rf+'_transition'](slew[s,fr],cap[t]))

    def check_value(self,a,slew,masters):
        rf = ('rise','fall')[a['to']]
        return self.table(a,masters)[rf+'_constraint'](slew[a['t'],a['to']],slew[a['s'],a['fr']])

    def calibrate(self):
        candidate_slew = np.full((self.n,2),-np.inf)
        for k,a in enumerate(self.arcs):
            d,s = self.arc_value(a,self.ref_slew,self.base_cap,self.masters)
            self.delay_correction[k] = a['delay']-d
            candidate_slew[a['t'],a['to']] = max(candidate_slew[a['t'],a['to']],s)
        valid = np.isfinite(candidate_slew)
        self.slew_correction[valid] = self.ref_slew[valid]-candidate_slew[valid]
        for k,a in enumerate(self.checks): self.check_correction[k] = a['delay']-self.check_value(a,self.ref_slew,self.masters)

    def run(self,replacements=None,mode='calibrated'):
        if getattr(self,'analysis','max')!='max':raise NotImplementedError('Use EarlyCudaSTA for minimum-delay propagation')
        if mode not in ('oracle','raw','calibrated'): raise ValueError(mode)
        masters = dict(self.masters)
        for inst,target in (replacements or {}).items():
            if inst not in masters or target not in self.cells: raise ValueError('Unknown replacement')
            old,new = self.cells[masters[inst]],self.cells[target]
            if old['family_id']!=new['family_id'] or new.get('frozen_reason'): raise ValueError('Invalid cell family replacement')
            masters[inst]=target
        if mode == 'oracle' and replacements: raise ValueError('Oracle arc replay only supports its own exported design')
        cap = self.caps(masters)
        at = np.full((self.n,2),-np.inf); slew=np.zeros((self.n,2))
        delays=np.zeros(len(self.arcs)); pred=np.full((self.n,2),-1,dtype=np.int32)
        for t in self.order:
            ids=self.incoming[t]
            if not ids:
                at[t]=self.ref_at[t];slew[t]=self.ref_slew[t];continue
            for k in ids:
                a=self.arcs[k];s,fr,to=a['s'],a['fr'],a['to']
                if mode=='oracle': d,sl=a['delay'],self.ref_slew[t,to]
                else:
                    d,sl=self.arc_value(a,slew,cap,masters)
                    if mode=='calibrated': d+=self.delay_correction[k]
                delays[k]=d; slew[t,to]=max(slew[t,to],sl)
                candidate=at[s,fr]+d
                if candidate>at[t,to]: at[t,to]=candidate;pred[t,to]=k
            if mode=='calibrated': slew[t]=np.maximum(0,slew[t]+self.slew_correction[t])
        check=np.array([a['delay'] if mode=='oracle' else self.check_value(a,slew,masters)+(self.check_correction[k] if mode=='calibrated' else 0) for k,a in enumerate(self.checks)])
        endpoint_slack={}; endpoint_rows=[]
        for e in self.endpoints:
            t=self.index[e['pin']];rf=RF[e['rf']];required=float(e['required_ps'])
            if e['role'] in MAX_CHECKS:
                ids=[k for k in self.check_incoming[t] if self.checks[k]['to']==rf]
                if not ids: raise ValueError('Missing setup arcs')
                # SDC clock edge/latency remains fixed. Recompute setup margin
                # when endpoint data slew or sequential master changes.
                required=float(e['required_ps'])+float(e['margin_ps'])-max(check[ids])
            elif e['role']!='output setup': raise ValueError('Unsupported endpoint constraint')
            slack=required-at[t,rf]
            endpoint_slack[t]=min(endpoint_slack.get(t,np.inf),slack)
            endpoint_rows.append((t,rf,required,slack))
        es=np.array(list(endpoint_slack.values()))
        return dict(at=at,slew=slew,load=cap,delay=delays,predecessor=pred,check=check,endpoints=np.array(endpoint_rows),tns_ps=float(np.minimum(es,0).sum()),wns_ps=float(min(es.min(),0)))


def error_stats(pred,ref):
    pred,ref=np.asarray(pred),np.asarray(ref)
    valid=np.isfinite(pred)&np.isfinite(ref)&(np.abs(ref)<1e25)
    d=pred[valid]-ref[valid]
    return dict(count=int(valid.sum()),finite_mask_mismatch=int(np.sum(np.isfinite(pred)!=np.isfinite(ref))),mae_ps=float(np.mean(np.abs(d))),p99_abs_ps=float(np.quantile(np.abs(d),.99)),max_abs_ps=float(np.max(np.abs(d))),bias_ps=float(np.mean(d)))


def compare(model,result,reference):
    ps={r['name']:r for r in rows(Path(reference)/'pins.tsv')}
    at=np.array([[float(ps[p['name']]['at_'+rf]) for rf in ('r','f')] for p in model.pins])
    sl=np.array([[float(ps[p['name']]['slew_'+rf]) for rf in ('r','f')] for p in model.pins])
    ep={(r['pin'],RF[r['rf']]):float(r['slack_ps']) for r in rows(Path(reference)/'endpoints.tsv')}
    expected=[ep[(model.pins[int(t)]['name'],int(rf))] for t,rf,_,_ in result['endpoints']]
    metrics=rows(Path(reference)/'metrics.tsv')[0]
    return dict(at=error_stats(result['at'],at),slew=error_stats(result['slew'],sl),endpoint_slack=error_stats(result['endpoints'][:,3],expected),tns_ps=result['tns_ps'],wns_ps=result['wns_ps'],reference_tns_ps=float(metrics['tns_ps']),reference_wns_ps=float(metrics['wns_ps']),tns_error_ps=result['tns_ps']-float(metrics['tns_ps']),wns_error_ps=result['wns_ps']-float(metrics['wns_ps']))
