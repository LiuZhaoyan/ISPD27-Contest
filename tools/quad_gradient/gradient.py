"""Frozen AveSmooth local perturbation and its CPU float64 adjoint."""
from collections import defaultdict
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Sensitivities:
    edge_adjoint: dict
    load: dict
    ga: dict
    gs: dict
    device: object


def ave(values, epsilon=0.01):
    maximum = max(values)
    # Absolute neighborhood extends the paper rule to zero/negative arrivals.
    active = [i for i, value in enumerate(values) if value >= maximum-epsilon*abs(maximum)-1e-12]
    return [1/len(active) if i in active else 0 for i in range(len(values))]


def backward(timing, seeds, *, device='cpu'):
    """Return fresh sensitivities; neither model nor objective seeds are mutated."""
    torch.set_num_threads(1)
    device = torch.device(device)
    nodes = list(timing.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    levels, batches = {}, defaultdict(list)
    aw, sw = [0.]*len(timing.edges), [0.]*len(timing.edges)
    for node in timing.order:
        incoming = timing.incoming.get(node, ())
        levels[node] = max((levels[timing.edges[i]['a']]+1 for i in incoming), default=0)
        batches[levels[node]].extend(incoming)
        arrival_weights = ave([timing.nodes[timing.edges[i]['a']][0]+timing.edges[i]['delay'] for i in incoming]) if incoming else []
        slew_weights = ave([timing.edges[i]['slew'] for i in incoming]) if incoming else []
        for i, wa, ws in zip(incoming, arrival_weights, slew_weights):
            aw[i], sw[i] = wa, ws
    tensor = lambda data: torch.tensor(data, dtype=torch.float64, device=device)
    src = torch.tensor([node_index[e['a']] for e in timing.edges], device=device, dtype=torch.long)
    dst = torch.tensor([node_index[e['b']] for e in timing.edges], device=device, dtype=torch.long)
    coefficients = tensor([[e[k] for k in ('ds', 'dc', 'ss', 'sc')] for e in timing.edges]).reshape(-1, 4)
    aw, sw = tensor(aw), tensor(sw)
    delay = torch.zeros(len(timing.edges), dtype=torch.float64, device=device, requires_grad=True)
    slew = torch.zeros_like(delay, requires_grad=True)
    cap = torch.zeros(len(nodes), dtype=torch.float64, device=device, requires_grad=True)
    arrival_injection = torch.zeros_like(cap, requires_grad=True)
    slew_injection = torch.zeros_like(cap, requires_grad=True)
    arrival, transition = arrival_injection, slew_injection
    # Frozen AveSmooth weights define the local perturbation, not a new timer.
    for level in sorted(batches):
        indices = batches[level]
        if not indices:
            continue
        ix = torch.tensor(indices, device=device)
        a, b = src[ix], dst[ix]
        ds, dc, ss, sc = coefficients[ix].unbind(1)
        da = arrival[a] + ds*transition[a] + dc*cap[b] + delay[ix]
        dt = ss*transition[a] + sc*cap[b] + slew[ix]
        arrival = arrival.index_add(0, b, aw[ix]*da)
        transition = transition.index_add(0, b, sw[ix]*dt)
    objective = (arrival*tensor([seeds.arrival.get(n, 0.) for n in nodes]) +
                 transition*tensor([seeds.slew.get(n, 0.) for n in nodes])).sum()
    objective = objective + 0*(delay.sum()+slew.sum()+cap.sum())
    gd, gs, load, ga, gtransition = torch.autograd.grad(
        objective, (delay, slew, cap, arrival_injection, slew_injection))
    edge_adjoint = dict(enumerate(zip(gd.detach().cpu().tolist(), gs.detach().cpu().tolist())))
    load = defaultdict(float, zip(nodes, load.detach().cpu().tolist()))
    ga = defaultdict(float, zip(nodes, ga.detach().cpu().tolist()))
    gs = defaultdict(float, zip(nodes, gtransition.detach().cpu().tolist()))

    return Sensitivities(edge_adjoint, load, ga, gs, device)
