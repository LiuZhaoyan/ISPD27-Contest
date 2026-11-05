"""Sizing-first budgets, low-conflict batches, and MCMM acceptance guards."""
from dataclasses import dataclass, asdict
import math


@dataclass
class WorkflowConfig:
    sizing_rounds:int=20
    topk:int=128
    max_topk:int=512
    search_space:str='family'
    salvage_attempts:int=6
    reference_salvage_attempts:int=2
    power_interval:int=4
    refresh_every:int=8
    max_refresh_every:int=16
    reference_move_budget:int=512
    reference_rc_fraction:float=.05
    tag_k:int=8
    tag_max_k:int=64
    buffer_interval:int=8
    buffer_max_interval:int=16
    buffer_stall:int=2
    max_buffer_batches:int=4
    max_new_buffers:int=16
    buffer_nets:int=32
    buffers_per_batch:int=4
    buffers_per_net:int=2
    dp_frontier:int=24
    workers:int=4
    tail_rounds:int=3
    max_seconds:float=0.
    power_growth:float=.05
    wns_guard_ps:float=1.
    mae_ps:float=1.
    p99_ps:float=5.
    initial_repair:bool=True
    final_repair:bool=True

    def validate(self):
        nonnegative=('sizing_rounds','tail_rounds','max_buffer_batches','max_new_buffers','reference_salvage_attempts')
        for name in nonnegative:
            if getattr(self,name)<0:raise ValueError(name+' must be nonnegative')
        for name in ('topk','max_topk','salvage_attempts','power_interval','refresh_every','max_refresh_every','reference_move_budget','tag_k','tag_max_k','buffer_interval','buffer_max_interval','buffer_stall','buffer_nets','buffers_per_batch','buffers_per_net','workers'):
            if getattr(self,name)<1:raise ValueError(name+' must be positive')
        if self.search_space not in ('family','neighbors'):raise ValueError('Invalid search_space')
        if self.topk>self.max_topk or self.refresh_every>self.max_refresh_every:raise ValueError('Invalid adaptive budget range')
        if self.buffer_max_interval<self.buffer_interval or self.buffers_per_net>4:raise ValueError('Invalid buffer budget')
        if self.tag_max_k<self.tag_k or self.tag_max_k>256 or self.dp_frontier<5:raise ValueError('Invalid queue/DP budget')
        for name in ('max_seconds','power_growth','wns_guard_ps','mae_ps','p99_ps','reference_rc_fraction'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError('Invalid '+name)
        if not 0<self.reference_rc_fraction<=1:raise ValueError('Invalid reference_rc_fraction')
        return self


def select_batch(candidates,limit,diagnostics=None,resource_limits=None):
    """Symmetric timing read/write conflicts plus physical region claims.

    This is a sufficient local conflict screen, not a proof of global timing
    independence. Shared downstream paths are checked by joint full STA.
    """
    if limit<1:raise ValueError('Positive batch limit required')
    selected=[];writes=set();reads=set();regions=set();used=set()
    stats=dict(nonpositive=0,duplicate_gate=0,timing_conflict=0,region_conflict=0,resource_budget=0,selected=0)
    consumed={};resource_limits=resource_limits or {}
    for a in sorted(candidates,key=lambda x:(-x['gain'],x['instance'],x.get('direction',0))):
        if not math.isfinite(a['gain']) or a['gain']<=0:stats['nonpositive']+=1;continue
        if a['instance'] in used:stats['duplicate_gate']+=1;continue
        aw=set(a['writes']);ar=set(a['reads']);ap=set(a.get('regions',()))
        if aw&writes or aw&reads or ar&writes:stats['timing_conflict']+=1;continue
        if ap&regions:stats['region_conflict']+=1;continue
        cost=a.get('resources',{})
        if any(consumed.get(k,0.)+cost.get(k,0.)>v+1e-15 for k,v in resource_limits.items()):
            stats['resource_budget']+=1;continue
        selected.append(a);writes|=aw;reads|=ar;regions|=ap;used.add(a['instance'])
        for k,v in cost.items():consumed[k]=consumed.get(k,0.)+v
        if len(selected)>=limit:break
    stats['selected']=len(selected)
    if diagnostics is not None:diagnostics.update(stats)
    return selected


def select_buffer_batch(proposals,budget,resource_limits,resource_cost,diagnostics=None):
    """Admit whole DP plans, allowing a cheaper alternative on the same net."""
    selected=[];used=set();regions=set();consumed={};skipped=0
    for proposal in proposals:
        actions=proposal['actions']
        if proposal['net'] in used or len(selected)+len(actions)>budget:continue
        claimed={(int(a['location'][0]//2),int(a['location'][1]//.27)) for a in actions}
        if claimed&regions:continue
        cost=resource_cost(proposal)
        if any(consumed.get(k,0.)+cost.get(k,0.)>v+1e-15 for k,v in resource_limits.items()):
            skipped+=1;continue
        selected.extend(actions);used.add(proposal['net']);regions.update(claimed)
        for k,v in cost.items():consumed[k]=consumed.get(k,0.)+v
        if len(selected)>=budget:break
    if diagnostics is not None:diagnostics.update(resource_budget=skipped,selected=len(selected),nets=len(used),resources=consumed)
    return selected


def objective(metrics,baseline):
    total=0.
    for c,r in metrics.items():
        b=baseline[c]
        total+=(-r['setup_tns_ns']/max(abs(b['setup_tns_ns']),.001)+
                -.25*r['setup_wns_ns']/max(abs(b['setup_wns_ns']),.001)+
                .05*r['dynamic_power_w']/max(b['dynamic_power_w'],1e-12)+
                .05*r['leakage_power_w']/max(b['leakage_power_w'],1e-12))/len(metrics)
    return total


def accept(before,after,baseline,config,mode='timing',native=False):
    reasons=[]
    for c,r in after.items():
        old=before[c];base=baseline[c]
        if not all(math.isfinite(float(v)) for v in r.values()):reasons.append(c+':nonfinite');continue
        # Preserve an already violated starting design until the explicit hold
        # closure stage; legal designs may not cross the contest tolerance.
        for key,tol in [('hold_wns_ns',.01),('hold_tns_ns',1.)]:
            floor=min(old[key],-tol)
            if r[key]<floor-(1e-6 if native else .0002):reasons.append(c+':'+key)
        if r['setup_wns_ns']<old['setup_wns_ns']-config.wns_guard_ps/1000:reasons.append(c+':wns_guard')
        for key in ('dynamic_power_w','leakage_power_w'):
            if r[key]>base[key]*(1+config.power_growth)+1e-12:reasons.append(c+':'+key)
        if native:
            for key,tol in [('slew_violation_sum_ps',.001),('cap_violation_sum_ff',1e-6),('fanout_violation_sum',1e-6)]:
                if r[key]>max(tol,old[key])+1e-8:reasons.append(c+':'+key)
            if r['global_routing_overflow'] or r['max_global_routing_overflow']:reasons.append(c+':routing')
    if mode=='power':
        if sum(r['dynamic_power_w']+r['leakage_power_w'] for r in after.values())>=sum(r['dynamic_power_w']+r['leakage_power_w'] for r in before.values()):reasons.append('power_not_reduced')
        if any(after[c]['setup_tns_ns']<before[c]['setup_tns_ns']-.001 for c in after):reasons.append('power_tns_guard')
    elif mode=='tail':
        if min(r['setup_wns_ns'] for r in after.values())<=min(r['setup_wns_ns'] for r in before.values())+1e-6:reasons.append('wns_not_improved')
        if any(after[c]['setup_tns_ns']<before[c]['setup_tns_ns']-.001 for c in after):reasons.append('tail_tns_guard')
    if objective(after,baseline)>=objective(before,baseline)-1e-8:reasons.append('objective_not_improved')
    return not reasons,reasons


def buffering_due(rounds_since,stalls,batches,inserted,config,final=False):
    due=(rounds_since>=config.buffer_interval and stalls>=config.buffer_stall
         or rounds_since>=config.buffer_max_interval
         or final and rounds_since>=min(2,config.buffer_interval))
    return (due and batches<config.max_buffer_batches and inserted<config.max_new_buffers)


def reference_due(rounds,moves,changed_nets,total_nets,interval,config):
    if not rounds:return None
    if rounds>=interval:return 'accepted_batch_budget'
    if moves>=config.reference_move_budget:return 'accepted_move_budget'
    if changed_nets/max(total_nets,1)>=config.reference_rc_fraction:return 'changed_rc_budget'
    return None


def erc_revert_actions(reference_instances,current_instances,geometry,violated_pins):
    """Revert changed gates incident to violated nets, not the entire epoch."""
    original={r['name']:r['master'] for r in reference_instances}
    current={r['name']:r for r in current_instances}
    bad_nets={r['net'] for r in geometry if r['pin'] in violated_pins}
    touched={r['pin'].rsplit('/',1)[0] for r in geometry
             if r['net'] in bad_nets and '/' in r['pin']}
    actions=[]
    for name in sorted(touched&original.keys()&current.keys()):
        row=current[name]
        if row['master']!=original[name] and row['frozen']!='1' and row['status']=='PLACED':
            actions.append(dict(kind='size',instance=name,odb_name=row['odb_name'],
                                old=row['master'],target=original[name]))
    return actions


def accuracy_pass(errors,config):
    return all(not r.get('missing',0) and not r.get('extra',0)
               and r['mae_ps']<=config.mae_ps and r['p99_abs_ps']<=config.p99_ps
               for corner in errors.values() for r in corner.values())


def tighten_budget(interval,topk):
    # More frequent references first, then reduce the physical trust region.
    return (max(1,interval//2),topk) if interval>1 else (1,max(1,topk//2))
