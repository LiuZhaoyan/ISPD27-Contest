"""Discrete working-point differences: benefit = -(delay + slew + load)."""
import math
import torch
from model import lookup_values, validate_axes


def score_actions(actions, timing, sensitivities, library):
    queries, terms, valid, upstream, errors = [], [], [], [], []
    for action in actions:
        try:
            local_queries, local_terms = [], []
            for i in timing.instance_edges.get(action['instance'], ()):
                e = timing.edges[i]
                gd, gs = sensitivities.edge_adjoint.get(i, (0., 0.))
                if not (gd or gs):
                    continue
                for component, coefficient, kind, base in (
                        (0, gd, e['delay_kind'], e['delay_lut']),
                        (1, gs, e['slew_kind'], e['slew_lut'])):
                    table = library.table(e['sample'], kind, action['target'])
                    validate_axes(table)
                    local_queries.append((table, e['x'], e['y']))
                    local_terms.append((component, coefficient, base))
            load = 0.
            for pin in timing.pins.get(action['instance'], ()):
                driver = timing.drivers.get(pin['net'])
                if driver:
                    for transition in ('rise', 'fall'):
                        delta = (library.cap(action['target'], pin['pin'], transition)-
                                 library.cap(action['current'], pin['pin'], transition))
                        load += sensitivities.load.get((driver, transition), 0.)*delta
            queries.extend(local_queries)
            terms.extend((len(valid)*3+c, coefficient, base) for c, coefficient, base in local_terms)
            valid.append(action)
            upstream.append(load)
        except (ValueError, KeyError) as error:
            errors.append(dict(action, reason=str(error)))
    result = torch.zeros((len(valid), 3), dtype=torch.float64, device=sensitivities.device)
    if queries:
        points = result.new_tensor([(x, y) for _, x, y in queries])
        values = lookup_values(queries, points)
        ix = torch.tensor([t[0] for t in terms], device=sensitivities.device)
        coefficient = result.new_tensor([t[1] for t in terms])
        base = result.new_tensor([t[2] for t in terms])
        result = result.flatten().index_add(0, ix, coefficient*(values-base)).reshape(-1, 3)
    result[:, 2] = result.new_tensor(upstream)
    scores = []
    for action, (own, downstream, upstream) in zip(valid, result.cpu().tolist()):
        delta = own+downstream+upstream
        if not math.isfinite(delta):
            errors.append(dict(action, reason='Nonfinite action benefit'))
        else:
            scores.append(dict(action, benefit=-delta, own_delay=own,
                               upstream_load=upstream, downstream_slew=downstream))
    return scores, errors

