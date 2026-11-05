"""Four-argument contest adapter for the sizing-first post-CTS workflow."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from post_cts_workflow import PostCTSWorkflow
from workflow_policy import WorkflowConfig
from sta_server import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,value in asdict(WorkflowConfig()).items():
        if isinstance(value,bool):p.add_argument('--no-'+name.replace('_','-'),dest=name,action='store_false',default=value)
        else:p.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    p.add_argument('input_dir',type=Path);p.add_argument('platform_dir',type=Path)
    p.add_argument('output_dir',type=Path);p.add_argument('top_module')
    a=p.parse_args();spec=json.loads((a.input_dir/'benchmark.json').read_text());benchmark=spec['design']
    if (benchmark not in ('aes','jpeg') or a.input_dir.resolve()!=ROOT/'benchmarks'/benchmark
        or a.platform_dir.resolve()!=ROOT/'platform/asap7' or a.top_module!=spec['top_module']):
        raise ValueError('This workflow currently supports the supplied AES/JPEG platform and constraints')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    config=WorkflowConfig(**{k:getattr(a,k) for k in asdict(WorkflowConfig())})
    report=PostCTSWorkflow(benchmark,out/'workflow',config).run()
    if report['status']!='WORKFLOW_COMPLETE':raise RuntimeError('Workflow did not pass final physical constraints')
    for suffix in ('.def','.v'):
        shutil.copyfile(out/'workflow/solution'/(benchmark+suffix),out/(benchmark+suffix))


if __name__=='__main__':main()
