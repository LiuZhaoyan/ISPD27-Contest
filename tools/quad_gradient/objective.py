"""Normalized TC setup objective: physical values in ns, model seeds per ps.

Full native statistics define the value; path-union coverage limits the seeds.
"""
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSeeds:
    arrival: dict
    slew: dict


def tc_loss(native, reference, epsilon=1e-9):
    tns, wns = native
    rt, rw = reference
    return -tns/abs(rt+epsilon) + max(0., -wns)/abs(rw+epsilon)


class TCSetupObjective:
    def __init__(self, reference):
        self.reference = reference  # (TNS, WNS), ns

    def value(self, metrics):
        return tc_loss(metrics, self.reference)

    def seeds(self, timing):
        ga, gs = defaultdict(float), defaultdict(float)
        endpoint_states = timing.endpoint_states
        reference = self.reference
        worst = min(v for states in endpoint_states.values() for v in states.values())
        worst_endpoints = [ep for ep, states in endpoint_states.items() if abs(min(states.values())-worst)<=1e-6]
        for ep, states in endpoint_states.items():
            minimum = min(states.values())
            active = [n for n, s in states.items() if abs(s-minimum)<=1e-6]
            seed = (1/(1000*abs(reference[0]+1e-9)) if minimum < 0 else 0)
            if ep in worst_endpoints and worst < 0:
                seed += 1/(1000*abs(reference[1]+1e-9)*len(worst_endpoints))
            for node in active:
                ga[node] += seed/len(active)
                checks = timing.checks.get(node, ())
                if checks:
                    # Ambiguous clock/conditional checks are not silently averaged.
                    if len(checks) != 1:
                        raise ValueError('Nonunique setup seed at ' + str(node))
                    gs[node] += seed/len(active)*checks[0][0]
                elif '/' in node[0]:
                    raise ValueError('Missing setup data-slew seed at ' + str(node))
        return ObjectiveSeeds(ga, gs)
