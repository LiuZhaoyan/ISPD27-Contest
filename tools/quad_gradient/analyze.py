#!/usr/bin/env python3
"""Build the bounded TC adjoint and retain every score and rejection."""
import argparse
from contextlib import redirect_stdout
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from actions import ActionCatalog, enumerate_actions
from data_io import load_analysis_inputs
import torch

from model import Library, build_timing_model
from objective import TCSetupObjective
from gradient import backward
from scoring import score_actions


def analyze(folder, libdir, reference, *, library=None, catalog=None):
    """Rebuild dynamic state; optionally reuse a fixed-run parsed library."""
    if library is not None:
        library.verify_folder(libdir)
    return _analyze(folder, libdir, reference, library=library, catalog=catalog)


def _analyze(folder, libdir, reference, *, library=None, catalog=None):
    """Trusted fixed-run entry point; caller owns library lifetime/validation."""
    folder = Path(folder)
    if catalog is None:
        catalog = ActionCatalog.load(folder/'snapshot.library.tsv')
    design, timing_data = load_analysis_inputs(folder, catalog)
    actions = enumerate_actions(design, catalog)
    (folder/'actions.json').write_text(json.dumps(actions, indent=2)+'\n')
    # The validated production backend is CPU float64. CUDA remains an offline
    # experiment until it also clears the frozen end-to-end performance gate.
    torch.set_num_threads(1)
    if library is None:
        library = Library(libdir)
    model = build_timing_model(timing_data, design, library)
    seeds = TCSetupObjective(reference).seeds(model)
    sensitivities = backward(model, seeds)
    scores, errors = score_actions([action for action in actions if action['valid']],
                                   model, sensitivities, library)
    scores.sort(key=lambda x: (-x['benefit'], x['instance'], x['direction']))
    for name, data in [('scores',scores), ('model_rejections',errors)]:
        (folder/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    summary = dict(numerical_backend=dict(engine='pytorch', device='cpu', dtype='float64',
                                         version=torch.__version__, cpu_threads=torch.get_num_threads()),
                   instances=len(actions)//4, decisions=len(actions), valid=sum(a['valid'] for a in actions),
                   invalid_reasons=dict(Counter(a['reason'] for a in actions if not a['valid'])),
                   model_rejections=len(errors), graph_nodes=len(model.nodes), graph_edges=len(model.edges),
                   positive_actions=sum(s['benefit']>0 for s in scores),
                   approximations=['Path-union coverage, pooled native tags; not a legal tagged full graph',
                       'Wire delay fixed; wire slew derivative 1 with native offset',
                       'Ceff load derivative 1; LUT waveform correction derivative 0',
                       'Clock slew fixed until refresh; no clock gradient',
                       'No hidden ratio calibration: native additive offsets refreshed every accepted step'])
    (folder/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    return scores, summary


def main():
    """Serve analysis requests with one fixed-technology Library per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libdir', type=Path, required=True)
    args = parser.parse_args()
    # Technology is immutable for this worker's lifetime; no per-request checks.
    with redirect_stdout(sys.stderr):
        library = Library(args.libdir)
    catalog = None
    for line in sys.stdin:
        request = json.loads(line)
        folder = Path(request['graph'])
        try:
            with (folder/'analysis.log').open('x') as log, redirect_stdout(log):
                if catalog is None:
                    catalog = ActionCatalog.load(folder/'snapshot.library.tsv')
                _analyze(folder, args.libdir, request['reference'], library=library, catalog=catalog)
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error': 'Analysis failed; see analysis-worker.log'}), flush=True)
            return 1
        print(json.dumps({'ok': True}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
