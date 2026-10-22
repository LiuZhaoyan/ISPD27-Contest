"""TC path-union working-point model. SI-facing reports converted to ps/fF here.

Tag pooling and fixed wire-delay/waveform-offset Jacobians are approximations,
not a replacement timer. Accurate acceptance always uses OpenSTA/evaluator.
"""
import bisect
from collections import defaultdict, deque
import gzip
import math
from pathlib import Path
import re

import torch

from data_io import groups, attribute, condition, NUMBER


def lookup(table, x, y):
    """Bilinear value and analytical partials, with linear boundary extrapolation."""
    axes, values = table
    weights = []
    for coords, v in zip(axes, (x, y)):
        if len(coords) < 2 or any(b <= a for a, b in zip(coords, coords[1:])):
            raise ValueError('Invalid NLDM axes')
        i = min(max(bisect.bisect_right(coords, v)-1, 0), len(coords)-2)
        step = coords[i+1]-coords[i]
        weights.append((i, (v-coords[i])/step, step))
    (i, u, dx), (j, v, dy) = weights
    a, b, c, d = values[i][j], values[i+1][j], values[i][j+1], values[i+1][j+1]
    return ((1-u)*(1-v)*a + u*(1-v)*b + (1-u)*v*c + u*v*d,
            ((1-v)*(b-a)+v*(d-c))/dx, ((1-u)*(c-a)+u*(d-b))/dy)


class Library:
    """Immutable TT data and per-instance caches, reusable across native refreshes."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.cells = {}
        for path in sorted(self.folder.glob('*TT*lib*')):
            text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
            if attribute(text, 'time_unit') != '1ps' or not re.search(r'capacitive_load_unit\s*\(\s*1\s*,\s*ff\s*\)', text):
                raise ValueError('Unsupported Liberty units')
            for name, body in groups(text, 'cell'):
                if name in self.cells:
                    raise ValueError('Duplicate TT cell: ' + name)
                self.cells[name] = dict(groups(body, 'pin'))
        self.cache = {}
        self._cap_cache = {}

    def verify_folder(self, folder):
        """Bind the parsed library to its directory; do not reread/hash sources.

        Callers keep this directory immutable for the run. Reload Library
        explicitly to use files that were changed in place.
        """
        if Path(folder).resolve() != self.folder:
            raise ValueError('Library directory does not match libdir')

    def cap(self, cell, pin, transition):
        key = (cell, pin, transition)
        if key not in self._cap_cache:
            body = self.cells[cell][pin]
            value = attribute(body, transition + '_capacitance')
            if value is None:
                value = attribute(body, 'capacitance')
            if value is None:
                raise ValueError('Missing input capacitance')
            # Failed/missing queries are not cached. Each Library owns its cache.
            self._cap_cache[key] = float(value)
        return self._cap_cache[key]

    def table(self, sample, kind, cell=None):
        cell = cell or sample['liberty_cell']
        output = sample['to_pin'].rsplit('/', 1)[-1]
        related = sample['from_pin'].rsplit('/', 1)[-1]
        timing_type = ('setup_' + ('rising' if sample['from_transition'] == 'rise' else 'falling')
                       if sample['kind'] == 'setup' else
                       ('rising_edge' if sample['from_transition'] == 'rise' else 'falling_edge')
                       if sample['role'] == 'Reg Clk to Q' else 'combinational')
        key = (cell, output, related, timing_type, sample['to_transition'], sample['sense'], sample['when'], kind)
        if key in self.cache:
            return self.cache[key]
        timings = [b for _, b in groups(self.cells[cell][output], 'timing')
                   if attribute(b, 'related_pin') == related
                   and attribute(b, 'timing_type', 'combinational') in
                       ({timing_type, 'combinational_' + sample['to_transition']} if sample['kind'] != 'setup' else {timing_type})
                   and condition(attribute(b, 'when', '')) == condition(sample['when'])
                   and (sample['kind'] == 'setup' or attribute(b, 'timing_sense') == sample['sense'])]
        if len(timings) != 1:
            raise ValueError('Ambiguous/missing candidate timing group: ' + str(key))
        tables = list(groups(timings[0], kind))
        if len(tables) != 1:
            raise ValueError('Missing unique table: ' + str(key))
        body = tables[0][1]
        axes = []
        for i in (1, 2):
            match = re.search(r'index_' + str(i) + r'\s*\(\s*"([^"]+)"\s*\)', body)
            if not match:
                raise ValueError('Missing explicit table axes')
            axes.append([float(x) for x in match[1].split(',')])
        value = re.search(r'values\s*\((.*?)\)\s*;', body, re.S)
        matrix = [[float(x) for x in line.split(',')] for line in re.findall(r'"([^"]+)"', value[1])]
        if len(matrix) != len(axes[0]) or any(len(r) != len(axes[1]) for r in matrix):
            raise ValueError('Invalid NLDM shape')
        self.cache[key] = (axes, matrix)
        return self.cache[key]


def inputs(text, setup=False):
    names = ('constrained_pin_transition', 'related_pin_transition') if setup else ('input_net_transition', 'total_output_net_capacitance')
    result = []
    for name in names:
        values = re.findall(r'\b' + name + r'\s*=\s*(' + NUMBER + ')', text)
        if not values or len(set(values)) != 1:
            raise ValueError('Missing/ambiguous native lookup input: ' + name)
        result.append(float(values[0]))
    return result


def validate_axes(table):
    if any(len(axis) < 2 or any(b <= a for a, b in zip(axis, axis[1:])) for axis in table[0]):
        raise ValueError('Invalid NLDM axes')


def lookup_values(queries, points):
    """Batch NLDM tables of equal shape; select the right interval at knots."""
    groups = defaultdict(list)
    for i, (table, _, _) in enumerate(queries):
        axes, matrix = table
        validate_axes(table)
        groups[(len(axes[0]), len(axes[1]))].append(i)
    result = torch.zeros(len(queries), dtype=torch.float64, device=points.device)
    for indices in groups.values():
        ix = torch.tensor(indices, device=points.device)
        xaxis = points.new_tensor([queries[i][0][0][0] for i in indices])
        yaxis = points.new_tensor([queries[i][0][0][1] for i in indices])
        tables = points.new_tensor([queries[i][0][1] for i in indices])
        xy = points[ix]
        x, y = xy[:, 0].contiguous(), xy[:, 1].contiguous()
        i = (torch.searchsorted(xaxis, x[:, None], right=True)[:, 0]-1).clamp(0, xaxis.shape[1]-2)
        j = (torch.searchsorted(yaxis, y[:, None], right=True)[:, 0]-1).clamp(0, yaxis.shape[1]-2)
        row = torch.arange(len(indices), device=points.device)
        u = (x-xaxis[row, i])/(xaxis[row, i+1]-xaxis[row, i])
        v = (y-yaxis[row, j])/(yaxis[row, j+1]-yaxis[row, j])
        a, b = tables[row, i, j], tables[row, i+1, j]
        c, d = tables[row, i, j+1], tables[row, i+1, j+1]
        values = (1-u)*(1-v)*a + u*(1-v)*b + (1-u)*v*c + u*v*d
        result = result.index_copy(0, ix, values)
    return result


def lookup_many(queries, device='cpu'):
    if not queries:
        return []
    points = torch.tensor([(x, y) for _, x, y in queries], dtype=torch.float64,
                          device=device, requires_grad=True)
    values = lookup_values(queries, points)
    partials, = torch.autograd.grad(values.sum(), points)
    return torch.cat((values[:, None], partials), dim=1).detach().cpu().tolist()


class TimingModel:
    def __init__(self, timing_data, design, library):
        self.nodes = {}
        paths = defaultdict(list)
        for r in timing_data.paths:
            key = (r['pin'], r['transition'])
            arrival, slew = float(r['arrival_ps']), float(r['slew_ps'])
            if not all(math.isfinite(v) for v in (arrival, slew)):
                raise ValueError('Invalid native point')
            old = self.nodes.get(key)
            if old is None or arrival > old[0]:
                self.nodes[key] = (arrival, slew)
            paths[r['path']].append((key, float(r['slack_ps']), r['endpoint']))
        self.edges = []
        wires = set()
        for path in paths.values():
            for (a, _, _), (b, _, _) in zip(path, path[1:]):
                if a[0].rsplit('/', 1)[0] != b[0].rsplit('/', 1)[0]:
                    wires.add((a, b))
        for a, b in sorted(wires):
            self.edges.append(dict(a=a, b=b, delay=self.nodes[b][0]-self.nodes[a][0],
                                   slew=self.nodes[b][1], ds=0., dc=0., ss=1., sc=0., sample=None))
        self.checks = defaultdict(list)
        queries, setup_queries = [], []
        for sample in timing_data.arcs:
            a = (sample['from_pin'], sample['from_transition'])
            b = (sample['to_pin'], sample['to_transition'])
            # Own inclusion and launch-boundary expansion in this traversal.
            # Off-union arcs may have no report, so request it only after filtering.
            if b not in self.nodes:
                continue
            text = timing_data.read_report(sample)
            x, y = inputs(text, sample['kind'] == 'setup')
            if a not in self.nodes and sample['kind'] != 'setup':
                if sample['role'] != 'Reg Clk to Q':
                    continue
                # Protected launch clock is a fixed boundary, but its Q load
                # response must contribute to first-stage input-cap actions.
                self.nodes[a] = (self.nodes[b][0]-float(sample['graph_delay_s'])*1e12, x)
            if sample['kind'] == 'setup':
                kind = ('rise' if b[1] == 'rise' else 'fall') + '_constraint'
                setup_queries.append((b, sample, x, y, len(queries)))
                queries.append((library.table(sample, kind), x, y))
            else:
                dk = 'cell_' + b[1]
                sk = b[1] + '_transition'
                dt, st = library.table(sample, dk), library.table(sample, sk)
                waveform = re.search(r'Driver waveform slew = (' + NUMBER + ')', text)
                self.edges.append(dict(a=a, b=b, delay=float(sample['graph_delay_s'])*1e12,
                                       slew=float(waveform[1]) if waveform else None,
                                       sample=sample, x=x, y=y, query=len(queries),
                                       delay_kind=dk, slew_kind=sk))
                queries.extend(((dt, x, y), (st, x, y)))
        values = lookup_many(queries)
        for b, sample, x, y, query in setup_queries:
            self.checks[b].append((values[query][1], sample, x, y))
        for e in self.edges:
            if e['sample'] is not None:
                query = e.pop('query')
                delay, ds, dc = values[query]
                slew, ss, sc = values[query+1]
                e.update(delay_lut=delay, ds=ds, dc=dc, slew_lut=slew, ss=ss, sc=sc)
                if e['slew'] is None:
                    e['slew'] = slew
        self.incoming = defaultdict(list)
        successors = defaultdict(set)
        indegree = dict.fromkeys(self.nodes, 0)
        for i, e in enumerate(self.edges):
            self.incoming[e['b']].append(i)
            if e['b'] not in successors[e['a']]:
                successors[e['a']].add(e['b'])
                indegree[e['b']] += 1
        queue = deque(sorted(n for n, d in indegree.items() if not d))
        self.order = []
        while queue:
            a = queue.popleft()
            self.order.append(a)
            for b in sorted(successors[a]):
                indegree[b] -= 1
                if not indegree[b]:
                    queue.append(b)
        if len(self.order) != len(self.nodes):
            raise ValueError('Path-union graph contains a cycle')
        endpoint_states = defaultdict(dict)
        for path in paths.values():
            node, slack, endpoint = path[-1]
            endpoint_states[endpoint][node] = min(endpoint_states[endpoint].get(node, math.inf), slack)
        self.endpoint_states = endpoint_states
        pins = design.pins
        self.drivers = {}
        for r in pins:
            if r['instance'] and r['direction'] == 'OUTPUT':
                self.drivers[r['net']] = r['instance']+'/'+r['pin']
        self.pins = defaultdict(list)
        for r in pins:
            if r['instance'] and r['direction'] == 'INPUT':
                self.pins[r['instance']].append(r)
        self.instance_edges = defaultdict(list)
        for i, edge in enumerate(self.edges):
            if edge['sample'] is not None:
                self.instance_edges[edge['sample']['to_pin'].rsplit('/',1)[0]].append(i)



def build_timing_model(timing_data, design, library):
    """Build ps/fF working-point facts; never seed or propagate sensitivities.

    Wire delay is fixed, wire slew derivative is 1, Ceff derivative is 1,
    and LUT waveform-offset derivative is 0. Clock slew is fixed until the
    next native refresh; protected launch Q still responds to its output load.
    """
    return TimingModel(timing_data, design, library)
