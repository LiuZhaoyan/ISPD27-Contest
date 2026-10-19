"""TC path-union adjoint prototype. SI-facing reports converted to ps/fF here.

Tag pooling and fixed wire-delay/waveform-offset Jacobians are approximations,
not a replacement timer. Accurate acceptance always uses OpenSTA/evaluator.
"""
import bisect
from collections import defaultdict, deque
import gzip
import math
from pathlib import Path
import re

from m0 import groups, attribute, condition, NUMBER, rows


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
    def __init__(self, folder):
        self.cells = {}
        for path in sorted(Path(folder).glob('*TT*lib*')):
            text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
            if attribute(text, 'time_unit') != '1ps' or not re.search(r'capacitive_load_unit\s*\(\s*1\s*,\s*ff\s*\)', text):
                raise ValueError('Unsupported Liberty units')
            for name, body in groups(text, 'cell'):
                if name in self.cells:
                    raise ValueError('Duplicate TT cell: ' + name)
                self.cells[name] = dict(groups(body, 'pin'))
        self.cache = {}

    def cap(self, cell, pin, transition):
        body = self.cells[cell][pin]
        value = attribute(body, transition + '_capacitance', attribute(body, 'capacitance'))
        if value is None:
            raise ValueError('Missing input capacitance')
        return float(value)

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


def ave(values, epsilon=0.01):
    maximum = max(values)
    # Absolute neighborhood extends the paper rule to zero/negative arrivals.
    active = [i for i, value in enumerate(values) if value >= maximum-epsilon*abs(maximum)-1e-12]
    return [1/len(active) if i in active else 0 for i in range(len(values))]


def tc_loss(native, reference, epsilon=1e-9):
    tns, wns = native
    rt, rw = reference
    return -tns/abs(rt+epsilon) + max(0., -wns)/abs(rw+epsilon)


class Adjoint:
    def __init__(self, folder, library, reference):
        self.folder, self.library, self.reference = Path(folder), library, reference
        self.nodes = {}
        paths = defaultdict(list)
        for r in rows(self.folder/'paths.tsv'):
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
        for sample in rows(self.folder/'arcs.tsv'):
            a = (sample['from_pin'], sample['from_transition'])
            b = (sample['to_pin'], sample['to_transition'])
            if b not in self.nodes:
                continue
            text = (self.folder/sample['report_file']).read_text()
            x, y = inputs(text, sample['kind'] == 'setup')
            if a not in self.nodes and sample['kind'] != 'setup':
                if sample['role'] != 'Reg Clk to Q':
                    continue
                # Protected launch clock is a fixed boundary, but its Q load
                # response must contribute to first-stage input-cap actions.
                self.nodes[a] = (self.nodes[b][0]-float(sample['graph_delay_s'])*1e12, x)
            if sample['kind'] == 'setup':
                kind = ('rise' if b[1] == 'rise' else 'fall') + '_constraint'
                _, ds, _ = lookup(library.table(sample, kind), x, y)
                self.checks[b].append((ds, sample, x, y))
            else:
                dk = 'cell_' + b[1]
                sk = b[1] + '_transition'
                dt, st = library.table(sample, dk), library.table(sample, sk)
                delay, ds, dc = lookup(dt, x, y)
                slew, ss, sc = lookup(st, x, y)
                waveform = re.search(r'Driver waveform slew = (' + NUMBER + ')', text)
                self.edges.append(dict(a=a, b=b, delay=float(sample['graph_delay_s'])*1e12,
                                       slew=float(waveform[1]) if waveform else slew,
                                       ds=ds, dc=dc, ss=ss, sc=sc, sample=sample, x=x, y=y,
                                       delay_kind=dk, slew_kind=sk, delay_lut=delay, slew_lut=slew))
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
        self.ga, self.gs = defaultdict(float), defaultdict(float)
        worst = min(v for states in endpoint_states.values() for v in states.values())
        worst_endpoints = [ep for ep, states in endpoint_states.items() if abs(min(states.values())-worst)<=1e-6]
        for ep, states in endpoint_states.items():
            minimum = min(states.values())
            active = [n for n, s in states.items() if abs(s-minimum)<=1e-6]
            seed = (1/(1000*abs(reference[0]+1e-9)) if minimum < 0 else 0)
            if ep in worst_endpoints and worst < 0:
                seed += 1/(1000*abs(reference[1]+1e-9)*len(worst_endpoints))
            for node in active:
                self.ga[node] += seed/len(active)
                checks = self.checks[node]
                if checks:
                    # Ambiguous clock/conditional checks are not silently averaged.
                    if len(checks) != 1:
                        raise ValueError('Nonunique setup seed at ' + str(node))
                    self.gs[node] += seed/len(active)*checks[0][0]
                elif '/' in node[0]:
                    raise ValueError('Missing setup data-slew seed at ' + str(node))
        self.load = defaultdict(float)
        self.edge_adjoint = {}
        for node in reversed(self.order):
            indices = self.incoming[node]
            if not indices:
                continue
            aw = ave([self.nodes[self.edges[i]['a']][0]+self.edges[i]['delay'] for i in indices])
            sw = ave([self.edges[i]['slew'] for i in indices])
            for i, wa, ws in zip(indices, aw, sw):
                e = self.edges[i]
                gd, gs = self.ga[node]*wa, self.gs[node]*ws
                self.edge_adjoint[i] = (gd, gs)
                self.ga[e['a']] += gd
                self.gs[e['a']] += gd*e['ds'] + gs*e['ss']
                self.load[node] += gd*e['dc'] + gs*e['sc']
        pins = rows(self.folder/'snapshot.pins.tsv')
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

    def score(self, action):
        own = downstream = upstream = 0.
        for i in self.instance_edges[action['instance']]:
            e = self.edges[i]
            s = e['sample']
            gd, gs = self.edge_adjoint.get(i, (0.,0.))
            if not (gd or gs):
                continue
            d = lookup(self.library.table(s,e['delay_kind'],action['target']),e['x'],e['y'])[0]
            t = lookup(self.library.table(s,e['slew_kind'],action['target']),e['x'],e['y'])[0]
            own += gd*(d-e['delay_lut'])
            downstream += gs*(t-e['slew_lut'])
        for pin in self.pins[action['instance']]:
            driver = self.drivers.get(pin['net'])
            if driver:
                for transition in ('rise','fall'):
                    delta = (self.library.cap(action['target'],pin['pin'],transition)-
                             self.library.cap(action['current'],pin['pin'],transition))
                    upstream += self.load[(driver,transition)]*delta
        delta = own+downstream+upstream
        if not math.isfinite(delta):
            raise ValueError('Nonfinite action benefit')
        return dict(action, benefit=-delta,
                    own_delay=own, upstream_load=upstream, downstream_slew=downstream)
