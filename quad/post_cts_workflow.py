"""Sizing-first post-CTS optimization with tagged GPU STA and sparse GR audits.

Run: python quad/post_cts_workflow.py aes runs/my_new_run
Outputs: solution/{aes,jpeg}.{def,v}, report.json and timed native evidence.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from forward_sta import rows
from mcmm_model import load_corner
from minmax_caps import load_ranges
from native_metrics import NativeScore
from model_runtime import feasible_constraints
from nlpm import PowerModel
from policy import contest_targets
from power_cuda import PowerCuda
from sta_server import ROOT,CORNERS
from workflow_buffering import ParallelBufferDP
from workflow_native import NativeWorkflow
from workflow_policy import WorkflowConfig,select_batch,select_buffer_batch,objective,accept,buffering_due,accuracy_pass,tighten_budget,reference_due,erc_revert_actions
from workflow_rc import PlacementForest,attach_forest,commit_sizing_pairs
from workflow_sta import WorkflowCorner


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,(np.floating,float)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


class PostCTSWorkflow:
    def __init__(self,benchmark,out,config):
        self.benchmark=benchmark;self.out=Path(out).resolve();self.config=config.validate()
        self.out.mkdir(parents=True,exist_ok=False)
        self.started=time.perf_counter();self.native=None;self.pairs={};self.power={}
        self.catalogs={};self.tables={};self.ranges={};self.power_tables={}
        self.report=dict(status='RUNNING',benchmark=benchmark,parameters=asdict(config),
            steps=[],checkpoints=[],stages={},rc_backend='openroad_native_placement',
            reference='independent_global_routing',gradient='autograd_with_unique_startpoint_CUDA',
            scope='shipped AES/JPEG single-clock MCMM constraints',
            optimization_buffers=0,buffer_batches=0,sizing_actions=0)
        self.checkpoint_id=0;self.rounds_since_reference=0;self.rounds_since_buffer=0;self.stalls=0
        self.tabu={};self.net_scores={};self.last_gradients={}
        self.refresh_interval=config.refresh_every;self.batch_limit=config.topk
        self.epoch_moves=0;self.epoch_changed_nets=set()
        self.score=NativeScore(benchmark,NativeScore.provided_baseline(benchmark))
        sources=[p for p in (ROOT/'quad').iterdir() if p.suffix in ('.py','.cu','.cpp','.tcl')]
        self.report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}

    def save(self):
        self.report['elapsed_seconds']=time.perf_counter()-self.started
        write_json(self.out/'report.json',finite_json(self.report))

    def timed(self,name,fn):
        t=time.perf_counter();result=fn()
        self.report['stages'][name]=self.report['stages'].get(name,0.)+time.perf_counter()-t
        return result

    def load_libraries(self):
        for c in CORNERS:
            self.catalogs[c],self.tables[c],_=load_corner(ROOT/'cache',c,include_min_checks=True)
            self.ranges[c]=load_ranges(self.catalogs[c],ROOT/'cache/minmax_caps')
            with (ROOT/'cache'/c/'power.pkl').open('rb') as f:self.power_tables[c]=pickle.load(f)
        self.cells=self.catalogs['TC']['cells'];self.ids={c['name']:c['id'] for c in self.cells}
        self.neighbors=contest_targets(self.catalogs['TC'],self.config.search_space)

    def build_models(self,path,native,reuse=False):
        if not reuse:
            self.pairs={};self.power={};gc.collect();torch.cuda.empty_cache()
        self.instances=rows(path/'instances.tsv')
        for i,r in enumerate(self.instances):r['instance_id']=i
        self.masters={r['name']:r['master'] for r in self.instances}
        geometry=rows(path/'pins_geometry.tsv');self.geometry={r['pin']:r for r in geometry}
        self.geometry_path=path/'pins_geometry.tsv'
        shared_rc=None;forests={};shared_bridges={}
        for c in CORNERS:
            if reuse:
                pair=self.pairs[c]
                pair.geometry=geometry
                pair.reanchor(path/c)
                self.report['checkpoints'][-1].setdefault('heldout_level_audit',{})[c]=pair.level_audit(self.predictions[c])
                self.power[c].power.model=pair.late.m
                continue
            pair=WorkflowCorner(path/c,self.catalogs[c],self.tables[c],self.masters,geometry,self.ranges[c],
                                self.config.tag_k,self.config.tag_max_k,shared_rc)
            shared_rc=pair.sensitivity_reference
            spef=path/f'placement/network_{c}.spef';key=hashlib.sha256(spef.read_bytes()).hexdigest()
            if key not in forests:forests[key]=PlacementForest(spef,path/'pins_geometry.tsv')
            forest=forests[key]
            attach_forest(pair,forest,shared=shared_bridges.get(key))
            shared_bridges.setdefault(key,pair.late_bridge)
            self.pairs[c]=pair
            pm=PowerModel(pair.late.m,self.power_tables[c]);self.power[c]=PowerCuda(pm,pair.late)
            print('MODEL',c,'tag_k',pair.late.tagged.k,'nets',len(forest.networks),flush=True)
        self.power_scale={}
        self.predictions=self.forward()
        for c in CORNERS:
            p=self.power[c].total(self.predictions[c]['max'])
            self.power_scale[c]={'dynamic':native[c]['dynamic_power_w']/max(p['internal_w']+p['switching_w'],1e-12),
                                 'leakage':native[c]['leakage_power_w']/max(p['leakage_w'],1e-12)}
        self.current=self.measure(self.predictions,native)
        diagnostics={c:p.diagnostics() for c,p in self.pairs.items()}
        if not all(v['finite'] for d in diagnostics.values() for v in d.values()):
            raise RuntimeError('Nonfinite tagged endpoint calibration')
        self.report['checkpoints'][-1]['calibration']=diagnostics
        self.report['checkpoints'][-1]['calibration_reference_path']=str(path)
        self.report['checkpoints'][-1]['calibration_budget_pass']=all(
            v['mae_ps']<=self.config.mae_ps and v['p99_abs_ps']<=self.config.p99_ps
            for d in diagnostics.values() for v in d.values())
        self.report['checkpoints'][-1]['level_audit']={c:p.level_audit(self.predictions[c]) for c,p in self.pairs.items()}
        self.report['checkpoints'][-1]['gpu_graph_reused']=reuse
        if not self.report['checkpoints'][-1]['calibration_budget_pass']:
            raise RuntimeError('Reference calibration exceeds the configured slack accuracy budget')
        self.reference_path=path;self.rounds_since_reference=0

    def forward(self,replacements=None):
        return {c:p.forward(replacements) for c,p in self.pairs.items()}

    def measure(self,predictions,template,replacements=None):
        result={c:dict(r) for c,r in template.items()}
        for c,p in predictions.items():
            power=self.power[c];power.power.model=self.pairs[c].late.m
            watts=power.total(p['max'],replacements)
            result[c].update(setup_tns_ns=p['max']['tns_ps']/1000,setup_wns_ns=p['max']['wns_ps']/1000,
                hold_tns_ns=p['min']['tns_ps']/1000,hold_wns_ns=p['min']['wns_ps']/1000,
                dynamic_power_w=(watts['internal_w']+watts['switching_w'])*self.power_scale[c]['dynamic'],
                leakage_power_w=watts['leakage_w']*self.power_scale[c]['leakage'])
        return result

    def calibration_error(self,path,predictions):
        result={}
        for c,pair in self.pairs.items():
            result[c]={}
            for analysis in ('max','min'):
                predicted={}
                for pin,rf,req,slack in predictions[c][analysis]['endpoints']:
                    key=(pair.late.m.pins[int(pin)]['name'],int(rf))
                    predicted[key]=min(predicted.get(key,np.inf),float(slack))
                reference={}
                endpoint_file=path/c/'audit'/analysis/'endpoints.tsv'
                if not endpoint_file.exists():endpoint_file=path/c/'gr'/analysis/'endpoints.tsv'
                for row in rows(endpoint_file):
                    key=(row['pin'],('rise','fall').index(row['rf']))
                    reference[key]=min(reference.get(key,np.inf),float(row['slack_ps']))
                missing=reference.keys()-predicted.keys();extra=predicted.keys()-reference.keys()
                delta=np.array([predicted[k]-reference[k] for k in reference.keys()&predicted.keys()])
                result[c][analysis]=dict(missing=len(missing),extra=len(extra),count=len(delta),
                    mae_ps=float(np.abs(delta).mean()) if len(delta) else np.inf,
                    p99_abs_ps=float(np.quantile(np.abs(delta),.99)) if len(delta) else np.inf,
                    max_abs_ps=float(np.abs(delta).max()) if len(delta) else np.inf)
        return result

    def checkpoint(self,reason,initial=False):
        path=self.out/f'reference_{self.checkpoint_id:03}';self.checkpoint_id+=1
        previous=self.current if not initial else None
        native,timing=self.native.reference(path,models=initial,timing_only=not initial)
        record=dict(path=str(path),reason=reason,native=native,seconds=timing)
        accepted=True;force_rebuild=False
        if not initial:
            record['heldout_prediction_error']=self.calibration_error(path,self.predictions)
            record['heldout_budget_pass']=accuracy_pass(record['heldout_prediction_error'],self.config)
            if not record['heldout_budget_pass']:
                self.refresh_interval,self.batch_limit=tighten_budget(self.refresh_interval,self.batch_limit)
                record['next_reference_interval']=self.refresh_interval
                record['next_batch_limit']=self.batch_limit
            accepted,reasons=accept(self.reference_metrics,native,self.baseline,self.config,native=True)
            # An unchanged epoch is a valid no-op checkpoint, not a regression.
            if not self.rounds_since_reference and reasons==['objective_not_improved']:accepted=True
            record['accepted']=accepted;record['reasons']=reasons
        self.report['checkpoints'].append(record)
        if not accepted:
            salvaged=self.salvage_reference(path,reasons)
            if salvaged is not None:
                path,native,record=salvaged;accepted=True;force_rebuild=True
        if initial:
            self.baseline=native;self.report['initial']=native
        if accepted:
            self.reference_metrics=native;self.reference_snapshot=path
            recalibrate=initial or force_rebuild or not record['heldout_budget_pass']
            record['recalibrated']=recalibrate
            if recalibrate:
                if not initial:record['calibration_export']=self.timed('calibration_exports',lambda:self.native.complete_reference(path))
                self.timed('gpu_recalibration' if not initial else 'gpu_model_build',
                           lambda:self.build_models(path,native,reuse=not initial and not force_rebuild))
            else:
                # Keep the same electrical/timing anchors when held-out native
                # endpoints pass. Checking a reference is not a mandatory refit.
                self.rounds_since_reference=0
                self.refresh_interval=min(self.config.max_refresh_every,self.refresh_interval+1)
                self.batch_limit=min(self.config.max_topk,max(self.batch_limit+1,int(self.batch_limit*1.5)))
                record.update(next_reference_interval=self.refresh_interval,next_batch_limit=self.batch_limit)
            self.best_metrics=native;self.best_snapshot=path
        else:
            self.native.restore(self.reference_snapshot)
            self.timed('rollback_calibration_exports',lambda:self.native.complete_reference(self.reference_snapshot))
            self.timed('gpu_model_rebuild_after_reference_rollback',lambda:self.build_models(self.reference_snapshot,self.reference_metrics))
            self.stalls+=1
            self.batch_limit=max(1,self.batch_limit//2)
        self.epoch_moves=0;self.epoch_changed_nets=set()
        self.save();return accepted

    def salvage_reference(self,path,reasons):
        """Bounded, native-validated reversion of ERC-causing sizing moves.

        A placement-legal epoch can have a few routed slew/cap violations. Its
        measured setup gain should not be discarded before trying to undo the
        changed cells on those nets. Hold, routing and power failures retain
        the conservative full-epoch rollback.
        """
        for attempt in range(self.config.reference_salvage_attempts):
            if not reasons or any(r.split(':')[-1] not in ('slew_violation_sum_ps','cap_violation_sum_ff','fanout_violation_sum') for r in reasons):return None
            violated=set()
            for report in (path/'metrics').glob('*_max_*.rpt'):
                violated.update(line.split()[0] for line in report.read_text().splitlines() if '(VIOLATED)' in line)
            actions=erc_revert_actions(rows(self.reference_snapshot/'instances.tsv'),rows(path/'instances.tsv'),
                                       rows(path/'pins_geometry.tsv'),violated)
            if not actions:return None
            self.timed('reference_salvage_legalization',lambda:self.native.apply(actions))
            retry=self.out/f'reference_{self.checkpoint_id:03}';self.checkpoint_id+=1
            native,timing=self.native.reference(retry,models=False,timing_only=True)
            ok,reasons=accept(self.reference_metrics,native,self.baseline,self.config,native=True)
            record=dict(path=str(retry),reason='targeted_erc_reversion',source_reference=str(path),
                        actions=actions,native=native,seconds=timing,accepted=ok,reasons=reasons,
                        heldout_scope='Original epoch error is retained; salvaged geometry is rebuilt before further optimization.')
            self.report['checkpoints'].append(record)
            iteration=max((s['iteration'] for s in self.report['steps']),default=0)
            for action in actions:self.tabu[(action['instance'],action['old'])]=iteration+4
            self.report['reverted_sizing_actions']=self.report.get('reverted_sizing_actions',0)+len(actions)
            if ok:return retry,native,record
            path=retry
        return None

    def candidates(self,mode,iteration):
        target=self.neighbors[[self.ids[r['master']] for r in self.instances]];gain=np.zeros(target.shape)
        mask=target>=0
        for i,r in enumerate(self.instances):
            old=self.cells[self.ids[r['master']]]
            if r['frozen']=='1' or r['status']!='PLACED' or old.get('frozen_reason') or old['liberty_data']['cell_type']!='combinational':mask[i]=False;continue
            for d in range(target.shape[1]):
                if not mask[i,d]:continue
                new=self.cells[target[i,d]]
                if sorted(map(str,old['liberty_data']['timing_arc_signatures']))!=sorted(map(str,new['liberty_data']['timing_arc_signatures'])):mask[i,d]=False
                if self.tabu.get((r['name'],new['name']),-1)>=iteration:mask[i,d]=False
                if mode=='power' and not (new['nominal_strength']<=old['nominal_strength'] and new['theta']>=old['theta']):mask[i,d]=False
        self.net_scores={};self.last_gradients={}
        for c,pair in self.pairs.items():
            e=pair.late;e.prepare()
            b=self.baseline[c]
            g=e.backward('hard',alpha=1/max(abs(b['setup_tns_ns'])*1000,1.),
                         beta=(1. if mode=='tail' else .25)/max(abs(b['setup_wns_ns'])*1000,1.))
            self.last_gradients[c]=g
            active=np.array([any(abs(g['delay'][a])>0 for a in e.b.instance_arcs.get(r['name'],[]))
                             or any(np.any(g['pin_load'][p]) for p in e.b.instance_pins.get(r['name'],[]))
                             for r in self.instances])
            local=mask if mode=='power' else mask&active[:,None]
            state=dict(action_target_id=target,action_mask=local,frozen=~mask.any(1))
            scored=e.score_all(self.instances,state,dict(timing=g,lse=g))
            if mode=='power':
                p=self.power[c];p.power.model=e.m
                delta=p.score(self.instances,target,local)
                gs,gc=p.internal_partials();prop=e.backward_power(gs,gc)
                delta+=e.score_all(self.instances,state,dict(power=prop,lse=g))['delta_loss']['power']
                gain-=np.where(local,delta*self.power_scale[c]['dynamic'],0.)/len(CORNERS)
            else:gain+=np.where(local,scored['benefit']['timing'],0.)/len(CORNERS)
            for aid,a in enumerate(e.m.arcs):
                if a['role']=='wire':
                    net=e.m.pins[a['s']]['net']
                    self.net_scores[net]=self.net_scores.get(net,0.)+abs(g['delay'][aid])*max(a['delay'],0.)
        m=self.pairs['TC'].late.m;candidates=[]
        for i,d in np.argwhere(mask&(gain>1e-12)):
            r=self.instances[int(i)];inst=r['name'];new=self.cells[target[i,d]];old=self.cells[self.ids[r['master']]]
            # Leakage is additive at fixed Boolean activity. Charge the actual
            # per-corner NLPM delta before admission, so a costly far-away
            # master cannot exclude a cheaper alternative for the same gate.
            resources={}
            for c,power in self.power.items():
                p=power.power
                resources['leakage:'+c]=(p.leakage(inst,new['name'])-p.leakage(inst,old['name']))*self.power_scale[c]['leakage']
            value=float(gain[i,d])
            if mode!='power':value-=sum(.05*v/max(self.baseline[k.split(':')[1]]['leakage_power_w'],1e-12)
                                        for k,v in resources.items())/len(CORNERS)
            if value<=1e-12:continue
            pins=self.pairs['TC'].late.b.instance_pins.get(inst,[])
            nets={m.pins[p]['net'] for p in pins if m.pins[p]['net']}
            writes={'inst:'+inst}|{'net:'+n for n in nets};reads=set()
            for p in pins:
                for aid in m.incoming[p]:
                    a=m.arcs[aid];up=m.pin_parts[a['s']][0]
                    if up and up!=inst:reads.add('inst:'+up)
            xy=[self.geometry[m.pins[p]['name']] for p in pins if m.pins[p]['name'] in self.geometry]
            regions=set()
            # Same-footprint swaps do not claim extra placement space. Net and
            # instance conflicts still screen their electrical interactions.
            footprint=lambda cell:(cell['physical']['width_um'],cell['physical']['height_um'])
            if xy and footprint(old)!=footprint(new):
                x=np.mean([float(p['x_um']) for p in xy]);y=np.mean([float(p['y_um']) for p in xy])
                regions={f'{int(x//2)}:{int(y//.27)}'}
            candidates.append(dict(kind='size',instance=inst,odb_name=r['odb_name'],old=r['master'],target=new['name'],
                direction=int(d),gain=value,resources=resources,writes=sorted(writes),reads=sorted(reads),regions=sorted(regions),
                coordinate=[old['nominal_strength'],old['theta']],
                target_coordinate=[new['nominal_strength'],new['theta']],
                predicted_loss_change=-value,
                coordinate_step=[new['nominal_strength']-old['nominal_strength'],new['theta']-old['theta']],
                search_kind='axis_neighbor' if d<4 else 'family_alternative'))
        return candidates

    def batch_resource_limits(self):
        return {'leakage:'+c:.98*max(0.,self.baseline[c]['leakage_power_w']*(1+self.config.power_growth)-r['leakage_power_w'])
                for c,r in self.current.items()}

    def buffer_resources(self,proposal):
        # New buffers have no cached instance activity. Use the largest
        # conditional leakage in each supply group, then validate with GR STA.
        resources={}
        for c,power in self.power.items():
            leakage=0.
            for action in proposal['actions']:
                data=power.power.tables[action['target']];groups={}
                for term in data['leakage']:groups.setdefault(term['pg'],[]).append(term['value'])
                leakage+=sum(max(v+[data['default_leakage']]) for v in groups.values()) if groups else data['default_leakage']
            resources['leakage:'+c]=leakage*self.power_scale[c]['leakage']
        return resources

    def salvage(self,actions,predictions):
        blame=[]
        for action in actions:
            cost=0.
            for c,pair in self.pairs.items():
                e=pair.late;g=self.last_gradients[c];ids=set(e.b.instance_arcs.get(action['instance'],[]))
                for pin in e.b.instance_pins.get(action['instance'],[]):ids.update(e.m.incoming[pin])
                ids=list(ids)
                cost+=float(np.dot(g['delay'][ids],predictions[c]['max']['delay'][ids]-self.predictions[c]['max']['delay'][ids]))
            blame.append((cost,-action['gain'],action['instance']))
        count=max(1,len(actions)//2)
        removed={x[2] for x in sorted(blame,reverse=True)[:count]}
        return [a for a in actions if a['instance'] not in removed],sorted(removed)

    def sizing_step(self,iteration,mode):
        start=time.perf_counter();record=dict(iteration=iteration,mode=mode,kind='sizing',accepted=False)
        candidates=self.timed('gradients_and_ranking',lambda:self.candidates(mode,iteration))
        selection={}
        actions=select_batch(candidates, min(8,self.batch_limit) if mode=='tail' else self.batch_limit,selection,
                             self.batch_resource_limits())
        record.update(candidates=len(candidates),candidate_gates=len({a['instance'] for a in candidates}),
                      batch_limit=self.batch_limit,selection=selection,proposed=len(actions),salvage=[],evaluations=[])
        before=self.current
        ok=False
        for attempt in range(self.config.salvage_attempts):
            if not actions:break
            replacements={a['instance']:a['target'] for a in actions}
            predictions=self.timed('gpu_joint_evaluation',lambda:self.forward(replacements))
            measured=self.measure(predictions,before,replacements)
            ok,reasons=accept(before,measured,self.baseline,self.config,mode)
            record['evaluations'].append(dict(size=len(actions),accepted=ok,reasons=reasons,
                objective=objective(measured,self.baseline)))
            if ok:break
            record['rejected_by']=reasons
            if attempt+1<self.config.salvage_attempts:
                actions,removed=self.salvage(actions,predictions);record['salvage'].append(removed)
        else:ok=False
        if not actions or not ok:
            self.stalls+=1;record['status']='NO_ACCEPTABLE_BATCH'
            self.batch_limit=max(1,self.batch_limit//2)
        else:
            snapshot=self.native.snapshot(self.out/f'transaction_{iteration:03}')
            old_models={c:(p.late.m,p.early.m) for c,p in self.pairs.items()}
            old_forests={c:(p.forest.networks,p.forest.signatures) for c,p in self.pairs.items()}
            applied=False
            try:
                self.timed('legalization',lambda:self.native.apply(actions))
                physical=self.out/f'placement_{iteration:03}'
                rc_export=self.timed('native_placement_rc',lambda:self.native.placement(physical,self.geometry_path))
                commit_sizing_pairs(self.pairs,replacements,predictions)
                changes={};batch_changed_nets=set()
                def refresh():
                    updated={}
                    for c,p in self.pairs.items():
                        if rc_export['geometry_reused']:
                            changes[c]=dict(changed_nets=0,geometry_reused=True)
                        else:
                            if id(p.forest) not in updated:
                                updated[id(p.forest)]=p.forest.refresh(physical/f'network_{c}.spef',physical/'pins_geometry.tsv')
                            changes[c]=updated[id(p.forest)]
                            batch_changed_nets.update(p.forest.changed_nets)
                            p.late_bridge.refresh();p.early_bridge.refresh()
                self.timed('gpu_rc_refresh',refresh)
                predictions=self.timed('gpu_joint_evaluation',self.forward)
                measured=self.measure(predictions,before)
                electrical=self.timed('native_placement_erc',lambda:self.native.electrical(physical/'erc'))
                for c in CORNERS:measured[c].update(electrical[c])
                ok,reasons=accept(before,measured,self.baseline,self.config,mode,native=True)
                record.update(after=measured,rc_changes=changes,rejected_by=reasons)
                if ok:
                    self.current=measured;self.predictions=predictions
                    self.instances=rows(physical/'instances.tsv')
                    for i,r in enumerate(self.instances):r['instance_id']=i
                    self.masters={r['name']:r['master'] for r in self.instances}
                    self.geometry={r['pin']:r for r in rows(physical/'pins_geometry.tsv')}
                    self.geometry_path=physical/'pins_geometry.tsv'
                    for a in actions:self.tabu[(a['instance'],a['old'])]=iteration+3
                    self.rounds_since_reference+=1;self.report['sizing_actions']+=len(actions)
                    self.epoch_moves+=len(actions)
                    self.epoch_changed_nets.update(batch_changed_nets)
                    self.stalls=0;applied=True;record.update(accepted=True,status='ACCEPTED',actions=actions)
                else:
                    record['status']='PHYSICAL_GUARD_REJECTED';self.stalls+=1
                    self.batch_limit=max(1,self.batch_limit//2)
                    for a in actions:self.tabu[(a['instance'],a['target'])]=iteration+2
            finally:
                if not applied:
                    self.native.restore(snapshot)
                    for c,p in self.pairs.items():
                        p.forest.networks,p.forest.signatures=old_forests[c];p.forest.generation+=1
                        p.late.set_model(old_models[c][0]);p.early.set_model(old_models[c][1])
                        p.late_bridge.refresh();p.early_bridge.refresh()
            due=reference_due(self.rounds_since_reference,self.epoch_moves,len(self.epoch_changed_nets),
                              len(self.pairs['TC'].forest.networks),self.refresh_interval,self.config)
            if applied and due:self.timed('reference_checkpoints',lambda:self.checkpoint(due))
        self.rounds_since_buffer+=1;record['seconds']=time.perf_counter()-start
        self.report['steps'].append(record);self.save()
        print('SIZING',iteration,mode,record['status'],len(actions),'seconds',round(record['seconds'],3),flush=True)

    def buffer_step(self,iteration):
        start=time.perf_counter();record=dict(iteration=iteration,kind='buffering',accepted=False)
        self.timed('buffer_net_gradients',lambda:self.candidates('timing',iteration))
        pair=self.pairs['TC']
        dp=ParallelBufferDP(self.catalogs['TC'],self.tables['TC'],workers=self.config.workers,
                            frontier=self.config.dp_frontier,max_per_net=self.config.buffers_per_net)
        proposals=self.timed('parallel_buffer_dp',lambda:dp.propose(pair.forest,pair.late,self.predictions['TC']['max'],
            self.net_scores,self.config.buffer_nets,iteration))
        record['proposals']=len(proposals)
        # At most one best DP plan per net, and a small total insertion budget.
        budget=min(self.config.buffers_per_batch,self.config.max_new_buffers-self.report['optimization_buffers'])
        record['selection']={}
        selected=select_buffer_batch(proposals,budget,self.batch_resource_limits(),self.buffer_resources,record['selection'])
        if not selected:record['status']='NO_BUFFER_PROPOSAL'
        else:
            snapshot=self.native.snapshot(self.out/f'buffer_transaction_{iteration:03}')
            committed=False
            try:
                self.native.apply(selected)
                path=self.out/f'reference_{self.checkpoint_id:03}';self.checkpoint_id+=1
                native,timing=self.native.reference(path,models=False)
                ok,reasons=accept(self.reference_metrics,native,self.baseline,self.config,native=True)
                checkpoint=dict(path=str(path),reason='buffer_topology_change',native=native,seconds=timing,accepted=ok,reasons=reasons)
                self.report['checkpoints'].append(checkpoint)
                record.update(actions=selected,accepted=ok,rejected_by=reasons)
                if ok:
                    checkpoint['calibration_export']=self.timed('calibration_exports',lambda:self.native.complete_reference(path))
                    self.timed('gpu_topology_rebuild',lambda:self.build_models(path,native))
                    self.reference_metrics=native;self.reference_snapshot=path;self.best_metrics=native;self.best_snapshot=path
                    self.report['optimization_buffers']+=len(selected);self.stalls=0;record['status']='ACCEPTED'
                    self.epoch_moves=0;self.epoch_changed_nets=set()
                    committed=True
                else:record['status']='REFERENCE_REJECTED'
            finally:
                if not committed:self.native.restore(snapshot)
        self.rounds_since_buffer=0;self.report['buffer_batches']+=1
        record['seconds']=time.perf_counter()-start;self.report['steps'].append(record);self.save()
        print('BUFFER',iteration,record['status'],record['proposals'],flush=True)

    def ensure_legality(self,label):
        records=[]
        for attempt in range(4):
            measured,timing=self.native.reference(self.out/f'{label}_legality_{attempt}',models=False)
            records.append(dict(metrics=measured,seconds=timing))
            if feasible_constraints(measured,self.score.config):
                self.report[label+'_legality_repair']=records;return measured
            if attempt==3:break
            for c in ('WC','TC','BC'):
                r=measured[c];limits=self.score.config['scenarios'][c]
                erc=(r['slew_violation_sum_ps']>.001 or r['cap_violation_sum_ff']>1e-6 or r['fanout_violation_sum']>1e-6)
                hold=(r['hold_wns_ns'] < -limits['hold_wns_tolerance_ns'] or r['hold_tns_ns'] < -limits['hold_tns_tolerance_ns'])
                if erc or hold:
                    self.native.repair(erc=erc,hold=hold,iterations=50,slew_margin=5.*(attempt+1),routed=True,corners=(c,))
                    # A structural repair invalidates routes before another
                    # corner uses them. Rebuild via the next measured retry.
                    break
        self.report[label+'_legality_repair']=records
        raise RuntimeError(label+' GR/ERC/hold closure exceeded its bounded repair budget')

    def run(self):
        try:
            self.timed('libraries',self.load_libraries)
            self.native=self.timed('native_load',lambda:NativeWorkflow(self.benchmark,self.out/'native'))
            original={r['name']:r['master'] for r in rows(self.out/'native/instances.tsv')}
            if self.config.initial_repair:
                self.timed('initial_erc_repair',lambda:self.native.repair(erc=True,hold=False,iterations=20))
                self.timed('initial_gr_legality_closure',lambda:self.ensure_legality('initial'))
            self.timed('initial_reference',lambda:self.checkpoint('initial',initial=True))
            self.report['initial_legal']=feasible_constraints(self.reference_metrics,self.score.config)
            repaired=self.masters.copy()
            self.report['initial_repair_changes']=dict(added_cells=len(repaired.keys()-original.keys()),
                added_buffers=sum(repaired[n].startswith('BUF') for n in repaired.keys()-original.keys()),
                changed_masters=sum(repaired[n]!=original[n] for n in original.keys()&repaired.keys()))
            if not self.report['initial_legal'] and self.config.initial_repair:
                raise RuntimeError('Initial repair did not produce a legal design; inspect native reports')
            for iteration in range(self.config.sizing_rounds+self.config.tail_rounds):
                if self.config.max_seconds and time.perf_counter()-self.started>=self.config.max_seconds:
                    self.report['stop_reason']='time_budget_at_batch_boundary';break
                tail=iteration>=self.config.sizing_rounds
                mode='tail' if tail else ('power' if (iteration+1)%self.config.power_interval==0 else 'timing')
                self.sizing_step(iteration,mode)
                if mode=='power' and not self.report['steps'][-1]['accepted']:
                    # A rejected power sweep must not consume a whole sizing
                    # opportunity when timing candidates remain plentiful.
                    self.sizing_step(iteration,'timing')
                if not tail and buffering_due(self.rounds_since_buffer,self.stalls,self.report['buffer_batches'],self.report['optimization_buffers'],self.config,
                                               final=iteration+1==self.config.sizing_rounds):
                    # Confirm outstanding sizing before changing topology.
                    if self.rounds_since_reference:self.timed('reference_checkpoints',lambda:self.checkpoint('before_buffering'))
                    self.buffer_step(iteration)
            if self.rounds_since_reference:self.timed('reference_checkpoints',lambda:self.checkpoint('final_sizing'))
            final=self.reference_metrics;validated=self.reference_snapshot
            if self.config.final_repair and not feasible_constraints(final,self.score.config):
                self.timed('final_erc_hold_repair',lambda:self.ensure_legality('final'))
                validated=None
            final_path=self.out/'final_reference'
            final,timing=self.timed('final_validation',lambda:self.native.final_reference(final_path,validated,self.reference_metrics))
            self.report['final_reference_validation']=timing
            self.native.output(self.out/'solution')
            final_masters={r['name']:r['master'] for r in rows(final_path/'instances.tsv')}
            self.report['retained_changes']=dict(
                sizing_vs_repaired=sum(final_masters[n]!=repaired[n] for n in repaired.keys()&final_masters.keys()),
                added_cells_vs_repaired=len(final_masters.keys()-repaired.keys()),
                added_cells_vs_input=len(final_masters.keys()-original.keys()))
            audited=[c['heldout_budget_pass'] for c in self.report['checkpoints'] if 'heldout_budget_pass' in c]
            self.report['heldout_accuracy_budget_pass']=all(audited) if audited else None
            self.report.update(final=final,final_legal=feasible_constraints(final,self.score.config),
                objective_before=objective(self.baseline,self.baseline),objective_after=objective(final,self.baseline),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),status='WORKFLOW_COMPLETE',
                validation='Native MCMM/placement/clock checks complete; independent evaluator recorded separately')
            if not self.report['final_legal']:self.report['status']='FINAL_CONSTRAINTS_FAILED'
            self.save();return self.report
        except BaseException as exc:
            self.report.update(status='FAILED',error=repr(exc));self.save();raise
        finally:
            if self.native:self.native.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark',choices=('aes','jpeg'));p.add_argument('output',type=Path)
    defaults=WorkflowConfig()
    for name,value in asdict(defaults).items():
        if isinstance(value,bool):p.add_argument('--no-'+name.replace('_','-'),dest=name,action='store_false',default=value)
        else:p.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    a=p.parse_args();config=WorkflowConfig(**{k:getattr(a,k) for k in asdict(defaults)})
    report=PostCTSWorkflow(a.benchmark,a.output,config).run()
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds','sizing_actions','optimization_buffers','final_legal')},indent=2))
    if report['status']!='WORKFLOW_COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
