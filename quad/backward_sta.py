"""Dual custom VJPs and library-native quad scores for the stage-2 timer.

Hard max remains the forward operation. Ave/LSE are surrogate backward rules,
not derivatives of that hard forward. `hard` is a diagnostic subgradient.
All times (including gamma) are ps; calibration residuals are frozen constants.
"""
from bisect import bisect_right
import numpy as np
from forward_sta import RF,MAX_CHECKS

DIRECTIONS = ('strength_up', 'strength_down', 'vt_up', 'vt_down')
SIGNS = np.array([1., -1., 1., -1.])


def table_partials(table, x, y):
    """Analytic derivatives of the same bilinear interpolation/extrapolation."""
    i = min(max(bisect_right(table.x, x)-1, 0), len(table.x)-2)
    j = min(max(bisect_right(table.y, y)-1, 0), len(table.y)-2)
    dx, dy = table.x[i+1]-table.x[i], table.y[j+1]-table.y[j]
    u, v = (x-table.x[i])/dx, (y-table.y[j])/dy
    z = table.values
    return (((1-v)*(z[i+1,j]-z[i,j])+v*(z[i+1,j+1]-z[i,j+1]))/dx,
            ((1-u)*(z[i,j+1]-z[i,j])+u*(z[i+1,j+1]-z[i+1,j]))/dy)


def max_weights(values, branch='hard', epsilon=.01, gamma_ps=1.):
    values = np.asarray(values, dtype=float)
    w = np.zeros(values.shape)
    finite = np.isfinite(values)
    if not finite.any(): return w
    if not 0 <= epsilon < 1 or not np.isfinite(gamma_ps) or gamma_ps <= 0:
        raise ValueError('Require 0 <= epsilon < 1 and finite gamma_ps > 0')
    top = values[finite].max()
    if branch == 'hard':
        active = finite & (values == top)
        w[active] = 1/active.sum()
    elif branch == 'ave':
        # Eq. (2) for nonnegative arrivals. abs(top) extends it to negative
        # arrivals without producing an empty neighborhood.
        active = finite & (values >= top-epsilon*abs(top))
        w[active] = 1/active.sum()
    elif branch == 'lse':
        exps = np.exp((values[finite]-top)/gamma_ps)
        w[finite] = exps/exps.sum()
    else: raise ValueError(f'Unknown branch {branch}')
    return w


def objective(result, alpha=1., beta=0.):
    if not np.isfinite([alpha,beta]).all() or min(alpha,beta)<0 or alpha+beta<=0:
        raise ValueError('Need nonnegative finite objective weights, not both zero')
    return -alpha*result['tns_ps']-beta*result['wns_ps']


def endpoint_seeds(result, alpha=1., beta=0.):
    """dF/d(-slack); count TNS once per unique endpoint, including ties."""
    objective(result,alpha,beta)
    ep = result['endpoints']
    seeds = np.zeros(len(ep))
    for pin in np.unique(ep[:,0]):
        ids = np.flatnonzero(ep[:,0]==pin)
        worst = ep[ids,3].min()
        if worst < 0:
            active = ids[ep[ids,3]==worst]
            seeds[active] += alpha/len(active)
    worst = ep[:,3].min()
    if worst < 0:
        active = np.flatnonzero(ep[:,3]==worst)
        seeds[active] += beta/len(active)
    return seeds


def bundle_weights(values,bundles,branch,epsilon=.01,gamma_ps=1.):
    """Smooth distinct fanin/RF bundles; Liberty conditions remain a hard max.

    Repeating a conditional arc must not increase an input pin's LSE mass.
    Return (arc IDs, weights) for each parallel-arc bundle.
    """
    peaks=np.array([np.max(values[ids]) for ids in bundles])
    outer=max_weights(peaks,branch,epsilon,gamma_ps)
    return [(ids,weight*max_weights(values[ids],'hard')) for ids,weight in zip(bundles,outer)]


class BackwardModel:
    def __init__(self, model, forward):
        self.model, self.forward = model, forward
        m, r = model, forward
        n = len(m.arcs)
        self.local_slew = np.zeros(n)
        self.delay_dx = np.zeros(n); self.delay_dc = np.zeros(n)
        self.slew_dx = np.zeros(n); self.slew_dc = np.zeros(n)
        self.at_candidate = np.zeros(n)
        self.groups = [[[] for _ in range(2)] for _ in range(m.n)]
        self.instance_arcs = {}; self.instance_checks = {}; self.instance_pins = {}
        for pin,(inst,_) in enumerate(m.pin_parts):
            if inst is not None: self.instance_pins.setdefault(inst,[]).append(pin)
        for k,a in enumerate(m.arcs):
            s,t,fr,to = (a[key] for key in ('s','t','fr','to'))
            self.groups[t][to].append(k)
            self.at_candidate[k] = r['at'][s,fr]+r['delay'][k]
            _, sl = m.arc_value(a,r['slew'],r['load'],m.masters)
            self.local_slew[k] = sl
            if a['role']=='wire':
                dc = a['elmore']/m.base_cap[s] if m.base_cap[s]>0 else 0.
                tau = dc*r['load'][s] if m.base_cap[s]>0 else a['elmore']
                self.delay_dc[k] = dc
                if sl > 0:
                    self.slew_dx[k] = r['slew'][s,fr]/sl
                    self.slew_dc[k] = 2.197224577**2*tau*dc/sl
            else:
                inst = m.pin_parts[t][0]
                self.instance_arcs.setdefault(inst,[]).append(k)
                tabs = m.table(a,m.masters); rf = ('rise','fall')[to]
                self.delay_dx[k],self.delay_dc[k] = table_partials(tabs['cell_'+rf],r['slew'][s,fr],r['load'][t])
                if sl > 0:
                    self.slew_dx[k],self.slew_dc[k] = table_partials(tabs[rf+'_transition'],r['slew'][s,fr],r['load'][t])
        self.groups = [[np.asarray(ids,dtype=np.int32) for ids in group] for group in self.groups]
        self.bundles=[]
        for group in self.groups:
            per_rf=[]
            for ids in group:
                buckets={}
                for k in ids:
                    a=m.arcs[k]
                    buckets.setdefault((a['s'],a['fr'],a['role']),[]).append(k)
                per_rf.append([np.asarray(v,dtype=np.int32) for v in buckets.values()])
            self.bundles.append(per_rf)
        self.check_dx = np.zeros((len(m.checks),2))
        for k,a in enumerate(m.checks):
            self.instance_checks.setdefault(m.pin_parts[a['t']][0],[]).append(k)
            table = m.table(a,m.masters)[('rise','fall')[a['to']]+'_constraint']
            self.check_dx[k] = table_partials(table,r['slew'][a['t'],a['to']],r['slew'][a['s'],a['fr']])

    def backward(self, branch='ave', epsilon=.01, gamma_ps=1., alpha=1., beta=0.):
        m,r = self.model,self.forward
        gat=np.zeros((m.n,2)); gs=np.zeros_like(gat); gc=np.zeros(m.n)
        gd=np.zeros(len(m.arcs)); gsl=np.zeros_like(gd); gcheck=np.zeros(len(m.checks))
        wat=np.zeros_like(gd); wsl=np.zeros_like(gd)
        seeds=endpoint_seeds(r,alpha,beta)
        for i,e in enumerate(m.endpoints):
            t,rf=int(r['endpoints'][i,0]),int(r['endpoints'][i,1])
            gat[t,rf] += seeds[i]
            if e['role'] in MAX_CHECKS:
                ids=[k for k in m.check_incoming[t] if m.checks[k]['to']==rf]
                gcheck[ids] += seeds[i]*max_weights(r['check'][ids],'hard')
        for k,a in enumerate(m.checks):
            gs[a['t'],a['to']] += gcheck[k]*self.check_dx[k,0]
            gs[a['s'],a['fr']] += gcheck[k]*self.check_dx[k,1]
        for t in m.order[::-1]:
            for rf in (0,1):
                ids=self.groups[t][rf]
                if not len(ids): continue
                for members,weights in bundle_weights(self.at_candidate,self.bundles[t][rf],branch,epsilon,gamma_ps):wat[members]=weights
                for members,weights in bundle_weights(self.local_slew,self.bundles[t][rf],branch,epsilon,gamma_ps):wsl[members]=weights
                # Forward's post-calibration max(0, slew) has zero derivative
                # at/below zero; the local cell-table clipping is handled above.
                out_slew_grad=gs[t,rf] if r['slew'][t,rf]>0 else 0.
                for k in ids:
                    a=m.arcs[k];s,fr=a['s'],a['fr']
                    gd[k]=gat[t,rf]*wat[k]; gsl[k]=out_slew_grad*wsl[k]
                    gat[s,fr] += gd[k]
                    gs[s,fr] += gd[k]*self.delay_dx[k]+gsl[k]*self.slew_dx[k]
                    cap_pin=s if a['role']=='wire' else t
                    gc[cap_pin] += gd[k]*self.delay_dc[k]+gsl[k]*self.slew_dc[k]
        net_grad={net:float(gc[pins].sum()) for net,pins in m.by_net.items()}
        return dict(at=gat,slew=gs,load=gc,delay=gd,local_slew=gsl,setup=gcheck,
                    at_weights=wat,slew_weights=wsl,endpoint_seeds=seeds,net_load=net_grad,
                    branch=branch,epsilon=epsilon,gamma_ps=gamma_ps,alpha=alpha,beta=beta)

    def action_delta(self, instance, target):
        """Sparse local differences at the calibrated operating point, not a
        whole-circuit finite difference. Calibration constants cancel out."""
        m,r=self.model,self.forward
        old=m.cells[m.masters[instance]];new=m.cells[target]
        if old['family_id']!=new['family_id'] or new.get('frozen_reason'):
            raise ValueError('Invalid family replacement')
        # Every evaluated cell/check arc belongs to this instance; avoid a
        # full netlist dictionary copy for each of the tens of thousands moves.
        masters={instance:target}
        ids=np.asarray(self.instance_arcs.get(instance,[]),dtype=np.int32)
        dd=[];ds=[]
        for k in ids:
            a=m.arcs[k]
            d,sl=m.arc_value(a,r['slew'],r['load'],masters)
            old_d,_=m.arc_value(a,r['slew'],r['load'],m.masters)
            dd.append(d-old_d);ds.append(sl-self.local_slew[k])
        checks=np.asarray(self.instance_checks.get(instance,[]),dtype=np.int32)
        dq=np.array([m.check_value(m.checks[k],r['slew'],masters)-m.check_value(m.checks[k],r['slew'],m.masters) for k in checks])
        caps={}
        for pin in self.instance_pins.get(instance,[]):
            name=m.pin_parts[pin][1];net=m.pins[pin]['net']
            delta=new['liberty_data']['pins'][name].get('capacitance_ff',0)-old['liberty_data']['pins'][name].get('capacitance_ff',0)
            caps[net]=caps.get(net,0.)+delta
        return dict(arcs=ids,delay=np.array(dd),slew=np.array(ds),checks=checks,setup=dq,cap=caps)

    @staticmethod
    def score_delta(gradient,delta):
        components=np.array([gradient['delay'][delta['arcs']]@delta['delay'],
            sum(gradient['net_load'].get(net,0.)*value for net,value in delta['cap'].items()),
            gradient['local_slew'][delta['arcs']]@delta['slew'],
            gradient['setup'][delta['checks']]@delta['setup']])
        return components

    def score_all(self,instances,states,gradients,tau=1e-6):
        if not np.isfinite(tau) or tau<0:raise ValueError('Invalid timing-safe threshold')
        m=self.model;shape=states['action_target_id'].shape
        if shape!=(len(instances),4):raise ValueError('Unexpected stage-1 action shape')
        mask=states['action_mask'].copy()
        if np.any(mask & states['frozen'][:,None]):raise ValueError('Frozen instance has a legal action')
        scores={name:np.full((*shape,4),np.nan) for name in gradients}
        gate_at=np.zeros(len(instances))
        for row in instances:
            i=int(row['instance_id']);inst=row['name']
            if m.masters[inst]!=row['master']:raise ValueError('States are not the calibrated initial design')
            pins=self.instance_pins.get(inst,[])
            if pins:gate_at[i]=np.max(np.abs(gradients['lse']['at'][pins]))
            else:gate_at[i]=np.inf
            for d in range(4):
                if not mask[i,d]:continue
                target=m.catalog['cells'][int(states['action_target_id'][i,d])]['name']
                delta=self.action_delta(inst,target)
                for name,g in gradients.items():scores[name][i,d]=self.score_delta(g,delta)
        safe=gate_at<=tau
        return dict(components=scores,delta_loss={k:v.sum(axis=2) for k,v in scores.items()},
                    benefit={k:-v.sum(axis=2) for k,v in scores.items()},mask=mask,
                    gate_lse_at=gate_at,timing_safe=safe,
                    power_candidate_mask=mask & safe[:,None] & np.array([False,True,True,False]))

    def replay(self,delta,h,gradient,frozen_weights=True):
        """Finite-difference validation of the implemented chain rule.

        Inject h*local action deltas into the original-master LUT evaluations;
        this is a tangent experiment, not an actual discrete master swap.
        If frozen_weights, each max is replaced by its affine local custom VJP
        with weights fixed at the baseline. Endpoints use frozen objective seeds.
        Otherwise run hard max / the original TNS+WNS objective.
        """
        m,r=self.model,self.forward
        dadd=np.zeros(len(m.arcs));sadd=np.zeros(len(m.arcs));qadd=np.zeros(len(m.checks))
        dadd[delta['arcs']]=h*delta['delay'];sadd[delta['arcs']]=h*delta['slew'];qadd[delta['checks']]=h*delta['setup']
        cap=r['load'].copy()
        for net,value in delta['cap'].items():cap[m.by_net[net]]+=h*value
        at=np.full_like(r['at'],-np.inf);slew=np.zeros_like(r['slew'])
        for t in m.order:
            if not m.incoming[t]:at[t]=r['at'][t];slew[t]=r['slew'][t];continue
            for rf in (0,1):
                ids=self.groups[t][rf]
                if not len(ids):continue
                arrivals=[];transitions=[]
                for k in ids:
                    a=m.arcs[k];d,sl=m.arc_value(a,slew,cap,m.masters)
                    arrivals.append(at[a['s'],a['fr']]+d+m.delay_correction[k]+dadd[k])
                    transitions.append(sl+sadd[k])
                if frozen_weights:
                    w=gradient['at_weights'][ids];valid=w>0
                    at[t,rf]=r['at'][t,rf]+np.dot(w[valid],np.asarray(arrivals)[valid]-self.at_candidate[ids][valid])
                    sl=self.local_slew[ids].max()+gradient['slew_weights'][ids]@(np.asarray(transitions)-self.local_slew[ids])
                else:at[t,rf]=max(arrivals);sl=max(0.,max(transitions))
                slew[t,rf]=max(0.,sl+m.slew_correction[t,rf])
        checks=np.array([m.check_value(a,slew,m.masters)+m.check_correction[k]+qadd[k] for k,a in enumerate(m.checks)])
        slacks=[]
        for i,e in enumerate(m.endpoints):
            t,rf=int(r['endpoints'][i,0]),int(r['endpoints'][i,1]);required=float(e['required_ps'])
            if e['role'] in MAX_CHECKS:
                ids=[k for k in m.check_incoming[t] if m.checks[k]['to']==rf]
                margin=(r['check'][ids].max()+max_weights(r['check'][ids],'hard')@(checks[ids]-r['check'][ids])) if frozen_weights else max(checks[ids])
                required+=float(e['margin_ps'])-margin
            slacks.append(required-at[t,rf])
        if frozen_weights:
            return objective(r,gradient['alpha'],gradient['beta'])-gradient['endpoint_seeds']@(np.asarray(slacks)-r['endpoints'][:,3])
        per_pin={}
        for row,sl in zip(r['endpoints'],slacks):per_pin[row[0]]=min(per_pin.get(row[0],np.inf),sl)
        es=np.array(list(per_pin.values()))
        return -gradient['alpha']*np.minimum(es,0).sum()-gradient['beta']*min(0.,es.min())
