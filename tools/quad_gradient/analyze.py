#!/usr/bin/env python3
"""Build the bounded TC adjoint and retain every score and rejection."""
import argparse
import json
from collections import Counter
from pathlib import Path
from actions import enumerate_actions
from m0 import rows
from model import Library, Adjoint


def analyze(folder, libdir, reference):
    folder = Path(folder)
    views = {}
    for row in rows(folder/'views.tsv'):
        views.setdefault(row['master'], set()).add(row['view'])
    views = {k: sorted(v) for k,v in views.items()}
    actions = enumerate_actions(folder/'snapshot', views)
    (folder/'actions.json').write_text(json.dumps(actions, indent=2)+'\n')
    library = Library(libdir)
    model = Adjoint(folder, library, reference)
    scores, errors = [], []
    for action in actions:
        if not action['valid']:
            continue
        try:
            scores.append(model.score(action))
        except (ValueError, KeyError) as error:
            errors.append(dict(action, reason=str(error)))
    scores.sort(key=lambda x: (-x['benefit'], x['instance'], x['direction']))
    for name, data in [('scores',scores), ('model_rejections',errors)]:
        (folder/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    summary = dict(instances=len(actions)//4, decisions=len(actions), valid=sum(a['valid'] for a in actions),
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


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--libdir',type=Path,default=Path('platform/asap7/lib/NLDM'))
    parser.add_argument('--reference',nargs=2,type=float,default=[-44.544634826,-.282912722])
    args=parser.parse_args()
    _, summary=analyze(args.folder,args.libdir,args.reference)
    print(json.dumps(summary,indent=2))
