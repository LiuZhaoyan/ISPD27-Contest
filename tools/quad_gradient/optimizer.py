#!/usr/bin/env python3
"""Top-K batch sizing optimizer; physical acceptance is fresh-route MCMM.

The path-union gradient is explicitly approximate. A completed tool run is not
proof of M3 prediction quality or contest net benefit. See manifest.json.
"""
import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from actions import DIRECTIONS
from data_io import ROOT, rows, save
from objective import TCSetupObjective
from physical import Runner, native_baseline


@dataclass(frozen=True)
class Checkpoint:
    path: object
    loss: float
    masters: dict
    diagnostics: list = field(default_factory=list)


@dataclass
class SearchResult:
    best: Checkpoint
    current: Checkpoint
    frozen: set
    events: list
    stop_reason: str


def select_batch(scores, frozen, k):
    """Global Top-K first, then deduplicate instances without filling gaps."""
    eligible = [a for a in scores if a.get('valid', False)
                and a['instance'] not in frozen and math.isfinite(a['benefit'])
                and a['benefit'] > 0]
    eligible.sort(key=lambda a: (-a['benefit'], a['instance'], DIRECTIONS.index(a['direction'])))
    top = eligible[:k]
    selected = {}
    for action in top:
        selected.setdefault(action['instance'], dict(action))
    return eligible, top, selected


def search(initial, physical, max_batches=30, seconds=10800, clock=time.monotonic,
           on_event=None):
    """Apply once per batch; commit physical state and bookkeeping atomically.

    refresh(checkpoint, batch) returns scored legal-neighbor actions.
    apply(checkpoint, actions, batch) returns (complete checkpoint, guard failures).
    A rejected/failed operation must never mutate its input checkpoint.
    """
    if not 1 <= max_batches <= 30 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Expected 1..30 batches and a finite positive time budget')
    start = clock()
    current = best = initial
    frozen = set()
    history = {name: [master] for name, master in initial.masters.items()}
    events = []
    k, patience, minimum_patience = 100, 0, 0
    reason = 'batch_budget'

    def terminal(event, reason):
        event.update(terminal_reason=reason, improved_best=False, next_K=k,
                     shrink_patience=patience, K10_patience=minimum_patience,
                     current_loss=current.loss, best_loss=best.loss,
                     after_checkpoint=str(current.path), best_checkpoint=str(best.path),
                     frozen_total=sorted(frozen), elapsed_seconds=clock()-start)
        events.append(event)
        if on_event:
            on_event(events)

    for batch in range(max_batches):
        if clock() - start >= seconds:
            reason = 'time_budget'
            break
        event = dict(batch=batch, K=k, eligible=0, top_k=0, deduplicated=0,
                     actual_modifications=0, frozen=[], freeze_requested=[],
                     guard_failures=[], diagnostics=[], rollback=False, accepted=False,
                     physical_attempt=False, before_checkpoint=str(current.path))
        try:
            scores = physical.refresh(current, batch)
            if clock() - start >= seconds:
                reason = 'time_budget'
                terminal(event, reason)
                break
            eligible, top, selected = select_batch(scores, frozen, k)
            event.update(eligible=len(eligible), top_k=len(top), deduplicated=len(selected))
            if not selected:
                reason = 'no_candidates'
                terminal(event, reason)
                break
            pending_freeze = set()
            for name, a in selected.items():
                if a['current'] != current.masters[name]:
                    raise ValueError('Scored action does not match current checkpoint')
                if len(history[name]) >= 2 and a['target'] == history[name][-2]:
                    pending_freeze.add(name)
                    a.update(target=best.masters[name], direction='freeze_restore')
            changes = [a for a in selected.values() if a['target'] != current.masters[a['instance']]]
            event.update(actions=list(selected.values()), freeze_requested=sorted(pending_freeze),
                         attempted_modifications=len(changes))
            event['physical_attempt'] = True
            candidate, failures = physical.apply(current, changes, batch)
            failures = list(failures)
            if not math.isfinite(candidate.loss):
                failures.append('nonfinite_TC_loss')
            expected = dict(current.masters)
            expected.update({a['instance']: a['target'] for a in changes})
            if candidate.masters != expected:
                failures.append('checkpoint_master_mismatch')
            event.update(guard_failures=failures, diagnostics=candidate.diagnostics, candidate_loss=candidate.loss if math.isfinite(candidate.loss) else None,
                         candidate_checkpoint=str(candidate.path))
            if not failures:
                current = candidate
                frozen.update(pending_freeze)
                for a in changes:
                    history[a['instance']].append(a['target'])
                    history[a['instance']] = history[a['instance']][-2:]
                event.update(accepted=True, actual_modifications=len(changes), frozen=sorted(pending_freeze))
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            event['guard_failures'] = ['physical_operation_failed']
            event['error'] = str(error)
        improved = event['accepted'] and current.loss < best.loss
        if improved:
            best = current
        patience = 0 if improved else patience + 1
        # Count only batches which START at K=10, independently of shrink patience.
        minimum_patience = (0 if improved else minimum_patience + 1) if k == 10 else 0
        if event['guard_failures'] or patience >= 3:
            k = max(10, k // 2)
            patience = 0  # one shrink per batch, including simultaneous triggers
        event.update(rollback=not event['accepted'], improved_best=improved, next_K=k,
                     shrink_patience=patience, K10_patience=minimum_patience,
                     current_loss=current.loss, best_loss=best.loss,
                     after_checkpoint=str(current.path), best_checkpoint=str(best.path),
                     frozen_total=sorted(frozen), elapsed_seconds=clock()-start)
        events.append(event)
        if on_event:
            on_event(events)
        if clock() - start >= seconds:
            reason = 'time_budget'
            break
        if minimum_patience >= 5:
            reason = 'K10_patience'
            break
    return SearchResult(best, current, frozen, events, reason)


def guards(baseline, candidate):
    failures=[]
    if set(baseline) != {'BC','TC','WC'} or set(candidate) != set(baseline):
        return ['scene_coverage']
    for scene, cand in candidate.items():
        for metric in ('global_routing_overflow','max_global_routing_overflow'):
            if cand[metric] != 0: failures.append(scene+':'+metric)
    return failures


def degradation_diagnostics(baseline, candidate, config):
    """Record hold/ERC degradation without gating internal exploration."""
    diagnostics=[]
    for scene in baseline.keys() & candidate.keys():
        base, cand = baseline[scene], candidate[scene]
        for metric in ('hold_tns','hold_wns'):
            tolerance=config['scenarios'][scene][metric+'_tolerance_ns']
            before=max(0.,-base[metric+'_ns']-tolerance)
            after=max(0.,-cand[metric+'_ns']-tolerance)
            if after > before+1e-9:
                diagnostics.append(dict(metric=scene+':'+metric, baseline=base[metric+'_ns'],
                                        candidate=cand[metric+'_ns'], tolerance_ns=tolerance))
        for metric in ('slew_violation_sum_ps','cap_violation_sum_ff','fanout_violation_sum'):
            if cand[metric]>base[metric]+1e-8:
                diagnostics.append(dict(metric=scene+':'+metric, baseline=base[metric], candidate=cand[metric]))
    return sorted(diagnostics, key=lambda record: record['metric'])


def checkpoint_masters(graph):
    return {r['instance']:r['master'] for r in rows(Path(graph)/'snapshot.cells.tsv')}


class BatchPhysical:
    """Fresh-route validation and graph refresh; no proxy acceptance gate."""
    def __init__(self, runner, platform_dir, reference, baseline, config,
                 initial_scores):
        self.runner, self.platform_dir = runner, platform_dir
        self.reference, self.baseline, self.config = reference, baseline, config
        self.initial_scores = initial_scores
        self.scores = {}

    def refresh(self, checkpoint, batch):
        if batch == 0:
            return self.initial_scores
        scores = self.scores.get(str(checkpoint.path))
        if scores is None:
            graph = self.runner.probe(checkpoint.path, f'refresh-{batch}', graph=True)
            scores, _ = self.runner.analyze(graph, self.platform_dir, self.reference)
        return scores

    def apply(self, checkpoint, actions, batch):
        evaluated = self.runner.evaluate(f'batch-{batch}-fresh', checkpoint.path, actions)
        summary = evaluated.summary
        failures = guards(self.baseline, summary)
        loss = TCSetupObjective(self.reference).value((summary['TC']['setup_tns_ns'], summary['TC']['setup_wns_ns']))
        # Reuse evaluate's clean graph. Refresh remains part of the transaction,
        # even on the final batch; a failed model must not publish a checkpoint.
        masters = checkpoint_masters(evaluated.graph)
        scores, _ = self.runner.analyze(evaluated.graph, self.platform_dir, self.reference)
        # Publish only after graph parsing/model construction/scoring succeeds.
        self.scores[str(evaluated.path)] = scores
        return Checkpoint(evaluated.path, loss, masters,
                          degradation_diagnostics(self.baseline, summary, self.config)), failures


def optimize(input_dir, platform_dir, output_dir, top, steps=30, search_seconds=10800):
    """Run sizing from a freshly prepared native checkpoint."""
    benchmark=input_dir.name
    if (benchmark not in {'aes','jpeg'} or input_dir.resolve()!=(ROOT/'benchmarks'/benchmark).resolve()
            or platform_dir.resolve()!=(ROOT/'platform/asap7').resolve()):
        raise ValueError('Only packaged AES/JPEG inputs and ASAP7 are supported')
    if top!=json.loads((input_dir/'benchmark.json').read_text())['top_module']:
        raise ValueError('Top module does not match benchmark metadata')
    if steps<1 or steps>30: raise ValueError('Batches must be 1..30')
    if not math.isfinite(search_seconds) or not 0<search_seconds<=10800:
        raise ValueError('Search seconds must be in (0, 10800]')
    output_dir.mkdir(parents=True,exist_ok=True)
    if any((output_dir/f'{benchmark}.{suffix}').exists() for suffix in ('def','v')):
        raise ValueError('Refusing to overwrite existing candidate outputs')
    work=output_dir/'quad_gradient'
    work.mkdir()
    runner=Runner(work,benchmark)
    start=time.monotonic()
    runner.deadline=start+search_seconds
    config=json.loads((ROOT/'evaluator/config/scoring_config.json').read_text())
    manifest=dict(status='running',implementation_complete=False,contest_net_gain=None,
                  tag_compatibility='unverified_deferred',config=config,
                  input_dir=str(input_dir),platform_dir=str(platform_dir),
                  source_dir=str(ROOT/'tools/quad_gradient'),events=[],
                  tools={key:dict(path=shutil.which(os.environ.get(key,default)))
                      for key,default in [('OPENROAD_EXE','openroad'),('KEPLER_FORMAL_EXE','kepler-formal')]})
    save(work/'manifest.json',manifest)
    try:
        runner.command([os.environ.get('OPENROAD_EXE','openroad'),'-version'],work/'version.log')
        current=runner.initialize(input_dir,top)
        graph=runner.probe(current,'initial-graph',graph=True)
        baseline=native_baseline(graph)
        ref=(baseline['TC']['setup_tns_ns'],baseline['TC']['setup_wns_ns'])
        current_loss=TCSetupObjective(ref).value(ref)
        manifest['initialization']=dict(method='minimal_native',
                                        checkpoint=str(current),reference_tc_ns=list(ref),guard_baseline=baseline)
        scores,analysis=runner.analyze(graph,platform_dir,ref)
        manifest['M1']=analysis
        threshold=0.
        plan=dict(improvement_threshold=threshold,epsilon_ns=1e-9,max_batches=steps,
                  search_seconds=search_seconds,initial_K=100,minimum_K=10,
                  shrink='floor half, once per batch; reset shrink patience',
                  shrink_patience=3,K10_patience=5,
                  direction_order=['drive_down','drive_up','vt_down','vt_up'],
                  best_comparison='candidate_loss < best_loss',
                  wire_rc='current routed checkpoint; refreshed every batch',
                  model=analysis['approximations'])
        save(work/'frozen_plan.json',plan)
        save(work/'manifest.json',manifest)
        initial=Checkpoint(current,current_loss,checkpoint_masters(graph))
        physical=BatchPhysical(runner,platform_dir,ref,baseline,config,scores)
        remaining=runner.deadline-time.monotonic()
        if remaining<=0:
            manifest.update(status='no_improvement',stop_reason='time_budget',
                            final_checkpoint=str(current),best_loss=current_loss,
                            public_certification='not_run',contest_score=None)
            save(work/'updates.json',[])
            return
        def record_events(events):
            events[-1]['tool_elapsed_seconds']=time.monotonic()-start
            save(work/'updates.json',events)
            manifest['events']=events
            save(work/'manifest.json',manifest)
        result=search(initial,physical,max_batches=steps,seconds=remaining,
                      on_event=record_events)
        save(work/'updates.json',result.events)
        current=Path(result.best.path)  # restore entire best, never last exploration
        manifest.update(M4=dict(accepted=sum(e['accepted'] for e in result.events),
                               rejected=sum(e['physical_attempt'] and not e['accepted'] for e in result.events),
                               attempts=len(result.events),
                               physical_attempts=sum(e['physical_attempt'] for e in result.events),final_checkpoint=str(current),
                               last_exploration_checkpoint=str(result.current.path)),
                        stop_reason=result.stop_reason,best_loss=result.best.loss,
                        frozen=sorted(result.frozen),implementation_complete=True,
                        public_certification='pending_independent_launcher',contest_score=None)
        manifest['modified_instances']=sum(master!=initial.masters[name]
                                            for name,master in result.best.masters.items())
        changed=result.best.masters!=initial.masters
        if result.best is initial or not changed:
            manifest.update(status='no_improvement',public_certification='not_run',
                            score_limitation='No modified best design; no valid optimization submission or score')
        else:
            feedback=json.loads((current/'feedback.json').read_text())
            manifest['best_physical_metrics']=feedback['scenarios']
            manifest['best_physical_metrics_provenance']=feedback['provenance']
            shutil.copyfile(current/'evaluated.def',output_dir/f'{benchmark}.def')
            shutil.copyfile(current/'evaluated.v',output_dir/f'{benchmark}.v')
            manifest.update(status='candidate_exported',outputs={name:str(output_dir/name) for name in (f'{benchmark}.def',f'{benchmark}.v')})
    except BaseException as error:
        manifest.update(status='failed',error=str(error))
        if time.monotonic()>=runner.deadline:
            manifest['stop_reason']='time_budget'
        for name in (f'{benchmark}.def',f'{benchmark}.v'):
            (output_dir/name).unlink(missing_ok=True)
        raise
    finally:
        runner.close()
        manifest.update(tool_seconds=time.monotonic()-start)
        save(work/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input_dir','platform_dir','output_dir'): parser.add_argument(name,type=Path)
    parser.add_argument('top_module')
    parser.add_argument('--max-batches','--steps',dest='steps',type=int,default=30)
    parser.add_argument('--search-seconds',type=float,default=10800)
    args=parser.parse_args()
    optimize(args.input_dir.resolve(),args.platform_dir.resolve(),args.output_dir.resolve(),args.top_module,
             args.steps,args.search_seconds)


if __name__=='__main__': main()
