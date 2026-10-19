#!/usr/bin/env python3
"""Four-argument AES sizing prototype; all final acceptance is fresh-route MCMM.

The path-union gradient is explicitly approximate. A completed tool run is not
proof of M3 prediction quality or contest net benefit. See manifest.json.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from actions import ROOT
from analyze import analyze
from m0 import sha, input_fingerprint, rows
from model import tc_loss
sys.path.insert(0, str(ROOT/'evaluator'))
from score import read_summary
from internal.legality import verify_certificate


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def guards(baseline, candidate, config):
    failures=[]
    if set(baseline) != {'BC','TC','WC'} or set(candidate) != set(baseline):
        return ['scene_coverage']
    for scene, base in baseline.items():
        cand=candidate[scene]
        for metric in ('hold_tns','hold_wns'):
            tolerance=config['scenarios'][scene][metric+'_tolerance_ns']
            before=max(0.,-base[metric+'_ns']-tolerance)
            after=max(0.,-cand[metric+'_ns']-tolerance)
            if after > before+1e-9: failures.append(scene+':'+metric)
        for metric in ('slew_violation_sum_ps','cap_violation_sum_ff','fanout_violation_sum'):
            if cand[metric]>base[metric]+1e-8: failures.append(scene+':'+metric)
        for metric in ('global_routing_overflow','max_global_routing_overflow'):
            if cand[metric] != 0: failures.append(scene+':'+metric)
    return failures


def acceptance_failures(baseline, candidate, config, current_loss, reference, threshold):
    loss=tc_loss((candidate['TC']['setup_tns_ns'],candidate['TC']['setup_wns_ns']),reference)
    failures=guards(baseline,candidate,config)
    if not math.isfinite(loss) or current_loss-loss<=threshold:
        failures.append('fresh_TC_loss_not_improved')
    return loss,failures


def native(folder):
    result={}
    for row in rows(Path(folder)/'native.tsv'):
        values=(float(row['tns_ps'])/1000,float(row['wns_ps'])/1000)
        if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite native timing')
        result[(row['scene'],row['analysis'])]=values
    if len(result)!=6: raise ValueError('Missing native MCMM statistics')
    return result


class Runner:
    def __init__(self, workspace, benchmark):
        self.workspace=Path(workspace)
        self.benchmark=benchmark
        self.commands=[]

    def command(self, command, log, env=None, timeout=1800):
        start=time.monotonic()
        record=dict(command=[str(x) for x in command],log=str(log),seconds=None,returncode=None)
        self.commands.append(record)
        print('Running:',log,flush=True)
        save(self.workspace/'commands.json',self.commands)
        try:
            with Path(log).open('x') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                                      env=env,cwd=ROOT,timeout=timeout)
            record['returncode']=result.returncode
        finally:
            record['seconds']=time.monotonic()-start
            save(self.workspace/'commands.json',self.commands)
        if result.returncode: raise RuntimeError(f'Command failed ({result.returncode}): {log}')

    def probe(self, baseline, name, action=None, graph=False):
        folder=self.workspace/name
        folder.mkdir()
        env=os.environ.copy()
        env.update(QG_ROOT=str(ROOT),QG_BASELINE=str(baseline),QG_OUTPUT=str(folder),
                   QG_EXPORT_GRAPH=str(int(graph)),MCMM_CONFIG=str(ROOT/'benchmarks'/self.benchmark/'mcmm.tcl'))
        if action:
            (folder/'action.tsv').write_text(action['instance']+'\t'+action['target']+'\n')
            env.update(QG_ACTION=str(folder/'action.tsv'),QG_LEGALIZE=str(int(action['direction'].startswith('drive'))))
            save(folder/'action.json',action)
        else: env['QG_ACTION']=''
        self.command([os.environ.get('OPENROAD_EXE','openroad'),'-exit',str(ROOT/'tools/quad_gradient/backend.tcl')],
                     folder/'openroad.log',env)
        native(folder)
        return folder

    def evaluate(self, name, deffile, verilog):
        folder=self.workspace/name
        self.command([sys.executable,str(ROOT/'evaluator/internal/run.py'),self.benchmark,str(folder),
                      '--def',str(deffile),'--verilog',str(verilog)],self.workspace/(name+'.log'),timeout=1800)
        verify_certificate(folder/'summary.csv',require_runtime=False)
        return folder


def optimize(input_dir, platform_dir, output_dir, top, steps=2):
    if top!='aes_cipher_top' or input_dir.name!='aes' or platform_dir.resolve()!=(ROOT/'platform/asap7').resolve():
        raise ValueError('Initial prototype supports only public AES / ASAP7')
    if steps<1 or steps>4: raise ValueError('Steps must be 1..4')
    output_dir.mkdir(parents=True,exist_ok=True)
    work=output_dir/'quad_gradient'
    work.mkdir()
    runner=Runner(work,'aes')
    start=time.monotonic()
    config=json.loads((ROOT/'evaluator/config/scoring_config.json').read_text())
    protected=(input_dir,platform_dir,ROOT/'evaluator')
    before=input_fingerprint(protected)
    manifest=dict(status='running',implementation_complete=False,contest_net_gain=None,
                  tag_compatibility='unverified_deferred',config=config,
                  sources=input_fingerprint((ROOT/'tools/quad_gradient',)),inputs=before,events=[],
                  tools={key:dict(path=shutil.which(os.environ.get(key,default)),
                      sha256=sha(shutil.which(os.environ.get(key,default))))
                      for key,default in [('OPENROAD_EXE','openroad'),('KEPLER_FORMAL_EXE','kepler-formal')]})
    save(work/'manifest.json',manifest)
    try:
        runner.command([os.environ.get('OPENROAD_EXE','openroad'),'-version'],work/'version.log')
        r0=runner.evaluate('r0',input_dir/'input.def',input_dir/'input.v')
        _,baseline=read_summary(r0/'summary.csv')
        ref=(baseline['TC']['setup_tns_ns'],baseline['TC']['setup_wns_ns'])
        current=r0
        current_loss=tc_loss(ref,ref)
        graph=runner.probe(current,'initial-graph',graph=True)
        scores,analysis=analyze(graph,platform_dir/'lib/NLDM',ref)
        manifest['M1']=analysis
        threshold=1e-7
        plan=dict(improvement_threshold=threshold,epsilon_ns=1e-9,steps=steps,
                  wire_rc='current routed R0; geometry effects separate',
                  model=analysis['approximations'])
        save(work/'frozen_plan.json',plan)
        save(work/'manifest.json',manifest)
        accepted=0
        attempts=[]
        for step in range(steps):
            if step:
                graph=runner.probe(current,f'refresh-{step}',graph=True)
                scores,analysis=analyze(graph,platform_dir/'lib/NLDM',ref)
            # Every trial starts in a fresh process from the complete physical
            # checkpoint; rollback is switching the reference, never master-only.
            for index,action in enumerate([s for s in scores if s['benefit']>threshold][:3]):
                event=dict(step=step,action=action,before_checkpoint=str(current),accepted=False)
                try:
                    trial=runner.probe(current,f'step-{step}-trial-{index}',action)
                    proxy_loss=tc_loss(native(trial)[('TC','max')],ref)
                    event['proxy_benefit']=current_loss-proxy_loss
                    if current_loss-proxy_loss<=threshold:
                        event['reason']='proxy_loss_not_improved'
                    else:
                        evaluated=runner.evaluate(f'step-{step}-trial-{index}-fresh',trial/'candidate.def',trial/'candidate.v')
                        _,summary=read_summary(evaluated/'summary.csv')
                        fresh_loss,failures=acceptance_failures(baseline,summary,config,current_loss,ref,threshold)
                        event.update(fresh_benefit=current_loss-fresh_loss,guard_failures=failures,evaluation=str(evaluated))
                        if not failures:
                            current,current_loss=evaluated,fresh_loss
                            accepted+=1
                            event.update(accepted=True,reason='fresh_MCMM_accepted')
                        else: event['reason']='fresh_MCMM_rejected'
                except (RuntimeError,ValueError,subprocess.SubprocessError) as error:
                    event['reason']=str(error)
                event['after_checkpoint']=str(current)
                attempts.append(event)
                save(work/'updates.json',attempts)
                if event['accepted']: break
            else: break
        manifest.update(M4=dict(accepted=accepted,rejected=sum(not e['accepted'] for e in attempts),
                               attempts=len(attempts),final_checkpoint=str(current)),
                        implementation_complete=True)
        if not accepted:
            raise RuntimeError('No physically accepted improvement; refusing to export R0 as optimization')
        # Fresh evaluated pair and its timing/parasitics remain one checkpoint.
        shutil.copyfile(current/'evaluated.def',output_dir/'aes.def')
        shutil.copyfile(current/'evaluated.v',output_dir/'aes.v')
        manifest.update(status='candidate_exported',outputs={name:sha(output_dir/name) for name in ('aes.def','aes.v')})
    except BaseException as error:
        manifest.update(status='failed',error=str(error))
        raise
    finally:
        manifest.update(tool_seconds=time.monotonic()-start,inputs_unchanged=before==input_fingerprint(protected),
                        sources_unchanged=manifest['sources']==input_fingerprint((ROOT/'tools/quad_gradient',)))
        if not manifest['inputs_unchanged'] or not manifest['sources_unchanged']:
            manifest.update(status='failed',error='Protected inputs or tool sources changed')
            for name in ('aes.def','aes.v'):
                (output_dir/name).unlink(missing_ok=True)
        save(work/'manifest.json',manifest)
    if not manifest['inputs_unchanged'] or not manifest['sources_unchanged']:
        raise ValueError('Protected inputs or tool sources changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input_dir','platform_dir','output_dir'): parser.add_argument(name,type=Path)
    parser.add_argument('top_module')
    parser.add_argument('--steps',type=int,default=2)
    args=parser.parse_args()
    optimize(args.input_dir.resolve(),args.platform_dir.resolve(),args.output_dir.resolve(),args.top_module,
             args.steps)


if __name__=='__main__': main()
