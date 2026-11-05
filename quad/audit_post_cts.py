"""Measure resident MCMM STA separately from reference export and physical work.

Input is a complete native reference produced by post_cts_workflow.py. Accuracy
here is at a calibration point; held-out accuracy belongs to workflow reports.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from post_cts_workflow import PostCTSWorkflow,write_json,finite_json
from workflow_policy import WorkflowConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark',choices=('aes','jpeg'))
    p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--repeats',type=int,default=10)
    a=p.parse_args()
    if a.repeats<3:p.error('--repeats must be at least 3')
    reference=a.reference.resolve()
    native=json.loads((reference/'reference.json').read_text())['metrics']
    w=PostCTSWorkflow(a.benchmark,a.output,WorkflowConfig(sizing_rounds=0,tail_rounds=0))
    w.baseline=native;w.report['checkpoints'].append(dict(reason='stored_reference',path=str(reference)))
    w.timed('libraries',w.load_libraries)
    w.timed('gpu_model_build',lambda:w.build_models(reference,native))
    samples=[]
    for i in range(a.repeats+2):
        t=time.perf_counter();result=w.forward();torch.cuda.synchronize()
        if i>=2:samples.append(time.perf_counter()-t)
    t=time.perf_counter();candidates=w.candidates('timing',0);torch.cuda.synchronize();gradient=time.perf_counter()-t
    previous={c:{analysis:result[c][analysis]['endpoints'].copy() for analysis in ('max','min')} for c in result}
    w.timed('same_topology_recalibration',lambda:w.build_models(reference,native,reuse=True))
    maximum=max(float(np.abs(w.predictions[c][analysis]['endpoints']-previous[c][analysis]).max())
                for c in result for analysis in ('max','min'))
    if maximum>1e-3:raise RuntimeError('Repeated calibration changed endpoint results: '+str(maximum))
    w.report.update(status='AUDIT_COMPLETE',accuracy_scope='calibration_point_only',
        resident_mcmm_forward=dict(repeats=a.repeats,median_seconds=float(np.median(samples)),
            p99_seconds=float(np.quantile(samples,.99)),samples_seconds=samples),
        gradient_and_ranking_seconds=gradient,candidate_count=len(candidates),
        reanchor_max_endpoint_change_ps=maximum,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    w.save()
    print(json.dumps({k:w.report[k] for k in ('status','stages','resident_mcmm_forward',
        'gradient_and_ranking_seconds','reanchor_max_endpoint_change_ps')},indent=2))


if __name__=='__main__':main()
