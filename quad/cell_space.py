"""ISPD26 library-native cell space. Standard-library parser plus NumPy arrays.

This is a deliberately bounded reader for the distributed ASAP7 Liberty/LEF and
flat contest netlists, not a general Verilog elaborator or a placement checker.
"""
import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path
import re

VT = {'SL': ('SLVT', 0), 'L': ('LVT', 1), 'R': ('RVT', 2)}
DIRECTIONS = ('strength_up', 'strength_down', 'vt_up', 'vt_down')
CELL_NAME = re.compile(r'^(.*?)x(\d+(?:p\d+)?|p\d+)([A-Za-z]*)_ASAP7_75t_(SL|L|R)$')
BRACES = re.compile(r'"(?:\\.|[^"\\])*"|[{}]', re.S)
COMMENTS = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*', re.S)
ENTRY = re.compile(r'\s*([\w]+)\s*([:(])')


def clean(text):
    return COMMENTS.sub(lambda m: m[0] if m[0].startswith('"') else ' ', text)


def unquote(text):
    text = text.strip()
    return text[1:-1] if text.startswith('"') and text.endswith('"') else text


@dataclass
class Group:
    kind: str
    args: str
    body: str


def entries(text):
    """Yield immediate attributes/groups, skipping nested LUT bodies intact."""
    pos = 0
    while pos < len(text):
        m = ENTRY.match(text, pos)
        if not m:
            if text[pos:].strip():
                raise ValueError('Unsupported Liberty syntax: ' + text[pos:pos + 100])
            break
        key, op = m.groups()
        pos = m.end()
        # Liberty strings may contain both parentheses and semicolons.
        quoted = False
        escaped = False
        level = 1 if op == '(' else 0
        start = pos
        while pos < len(text):
            c = text[pos]
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = not quoted
            elif not quoted:
                if op == '(' and c == '(':
                    level += 1
                elif op == '(' and c == ')':
                    level -= 1
                    if level == 0:
                        break
                elif op == ':' and c == ';':
                    break
                elif op == ':' and c == '\n' and ENTRY.match(text, pos):
                    # Distributed SEQ libraries omit ';' after some area values.
                    # Accept their newline delimiter as OpenSTA does, without
                    # merging the following clock_gating attribute into area.
                    break
            pos += 1
        if pos == len(text):
            raise ValueError('Unterminated Liberty entry ' + key)
        value = text[start:pos].strip()
        pos += 1
        if op == ':':
            yield key, unquote(value)
            continue
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if text[pos] == ';':
            pos += 1
            yield key, ('complex', value)
            continue
        if text[pos] != '{':
            raise ValueError('Expected group body for ' + key)
        depth = 1
        start = pos + 1
        for brace in BRACES.finditer(text, start):
            if brace[0] == '{':
                depth += 1
            elif brace[0] == '}':
                depth -= 1
            if depth == 0:
                yield key, Group(key, value, text[start:brace.start()])
                pos = brace.end()
                break
        else:
            raise ValueError('Unterminated group ' + key)


def attributes(items):
    return {k: v for k, v in items if isinstance(v, str)}


def boolean_key(expression):
    """Canonical truth table; fail closed for unsupported Liberty expressions."""
    if expression is None:
        return None
    token_re = re.compile(r'\s*([A-Za-z_][\w\[\]]*|[01]|[!~\'&*|+^()])')
    tokens = []
    pos = 0
    while pos < len(expression.rstrip()):
        m = token_re.match(expression, pos)
        if not m:
            raise ValueError('Unsupported Boolean expression: ' + expression)
        tokens.append(m[1])
        pos = m.end()
    cursor = 0

    def atom():
        nonlocal cursor
        t = tokens[cursor]
        cursor += 1
        if t in ('!', '~'):
            node = ('not', atom())
        elif t == '(':
            node = expr(0)
            if cursor >= len(tokens) or tokens[cursor] != ')':
                raise ValueError('Unbalanced Boolean expression')
            cursor += 1
        elif re.fullmatch(r'[A-Za-z_][\w\[\]]*|[01]', t):
            node = ('var', t)
        else:
            raise ValueError('Invalid Boolean atom ' + t)
        while cursor < len(tokens) and tokens[cursor] == "'":
            cursor += 1
            node = ('not', node)
        return node

    precedence = {'|': 1, '+': 1, '^': 2, '&': 3, '*': 3}

    def expr(minimum):
        nonlocal cursor
        node = atom()
        while cursor < len(tokens) and tokens[cursor] in precedence and precedence[tokens[cursor]] >= minimum:
            op = tokens[cursor]
            cursor += 1
            node = (op, node, expr(precedence[op] + 1))
        return node

    tree = expr(0)
    if cursor != len(tokens):
        raise ValueError('Unconsumed Boolean tokens: ' + expression)
    names = sorted(set(t for t in tokens if re.match(r'[A-Za-z_]', t)))
    if len(names) > 12:
        raise ValueError('Truth table exceeds bounded cell support')

    def evaluate(node, values):
        op = node[0]
        if op == 'var':
            return bool(int(node[1])) if node[1] in ('0', '1') else values[node[1]]
        if op == 'not':
            return not evaluate(node[1], values)
        a, b = evaluate(node[1], values), evaluate(node[2], values)
        return (a and b) if op in ('&', '*') else (a or b) if op in ('|', '+') else a != b
    truth = ''.join('1' if evaluate(tree, dict(zip(names, bits))) else '0'
                    for bits in itertools.product((False, True), repeat=len(names)))
    return {'variables': names, 'truth_table': truth}


def parse_standard_lib(path):
    library = next(v for k, v in entries(clean(path.read_text())) if k == 'library')
    contents = list(entries(library.body))
    attrs = attributes(contents)
    cap = next(v[1] for k, v in contents if k == 'capacitive_load_unit')
    amount, unit = (x.strip() for x in cap.split(','))
    scale_ff = float(amount) * {'ff': 1.0, 'pf': 1000.0, 'nf': 1e6}[unit.lower()]
    units = {'time': attrs['time_unit'], 'capacitance': cap,
             'leakage_power': attrs.get('leakage_power_unit'), 'capacitance_to_ff': scale_ff}
    cells = {}
    for kind, cell in contents:
        if kind != 'cell':
            continue
        parts = list(entries(cell.body))
        a = attributes(parts)
        pins = {}
        sequential = []
        arcs = []
        leakage = []
        for k, v in parts:
            if k == 'pin':
                pe = list(entries(v.body))
                pa = attributes(pe)
                pin = unquote(v.args)
                pins[pin] = {'direction': pa['direction'], 'function': pa.get('function'),
                             'function_key': boolean_key(pa.get('function')),
                             'three_state_key': boolean_key(pa.get('three_state')),
                             'flags': {x: y for x, y in pa.items() if x == 'clock' or x.startswith('clock_gate_')}}
                for x in ('capacitance', 'rise_capacitance', 'fall_capacitance', 'max_capacitance'):
                    if x in pa:
                        pins[pin][x + '_ff'] = float(pa[x]) * scale_ff
                for tk, tv in pe:
                    if tk == 'timing':
                        ta = attributes(list(entries(tv.body)))
                        arcs.append([pin, ta.get('related_pin'), ta.get('timing_type', 'combinational'), ta.get('timing_sense'), boolean_key(ta.get('when'))])
            elif k in ('ff', 'latch'):
                sa = attributes(list(entries(v.body)))
                for x in ('clocked_on', 'clocked_on_also', 'next_state', 'enable', 'data_in', 'clear', 'preset', 'power_down_function'):
                    if x in sa:
                        sa[x] = boolean_key(sa[x])
                sequential.append({'kind': k, 'state': re.sub(r'\s+', '', v.args), 'attributes': sa})
            elif k == 'leakage_power':
                leakage.append(attributes(list(entries(v.body))))
        behavior = {'pins': {p: {x: v for x, v in data.items() if x in ('direction', 'function_key', 'three_state_key', 'flags')}
                             for p, data in sorted(pins.items())}, 'state': sequential,
                    'clock_gating': a.get('clock_gating_integrated_cell')}
        cell_type = 'clock_gate' if a.get('clock_gating_integrated_cell') else 'sequential' if sequential else 'tie' if all(
            p.get('function') in ('0', '1', None) for p in pins.values()) else 'combinational'
        signature = hashlib.sha256(json.dumps(behavior, sort_keys=True).encode()).hexdigest()
        cells[unquote(cell.args)] = dict(liberty=str(path), library=unquote(library.args), units=units,
                                       area=float(a['area']), dont_use=a.get('dont_use') == 'true',
                                       pins=pins, behavior=behavior, behavior_sha256=signature,
                                       timing_arc_signatures=arcs, leakage_states=leakage, cell_type=cell_type)
    return cells


def parse_lef(path):
    text = path.read_text()
    macros = {}
    for m in re.finditer(r'^MACRO\s+(\S+)\s*\n', text, re.M):
        name = m[1]
        end = re.search(r'^END\s+' + re.escape(name) + r'\s*$', text[m.end():], re.M)
        if not end:
            raise ValueError('Unterminated LEF macro ' + name)
        body = text[m.end():m.end() + end.start()]
        size = re.search(r'\bSIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;', body)
        cls = re.search(r'\bCLASS\s+([^;]+);', body)
        site = re.search(r'\bSITE\s+(\S+)\s*;', body)
        pins = {}
        for pm in re.finditer(r'^\s*PIN\s+(\S+)\s*\n', body, re.M):
            tail = body[pm.end():]
            pend = re.search(r'^\s*END\s+' + re.escape(pm[1]) + r'\s*$', tail, re.M)
            if not pend:
                raise ValueError('Unterminated LEF pin')
            pin = tail[:pend.start()]
            direction = re.search(r'\bDIRECTION\s+(\w+)\s*;', pin)
            use = re.search(r'\bUSE\s+(\w+)\s*;', pin)
            pins[pm[1]] = {'direction': direction[1].lower() if direction else None, 'use': use[1] if use else 'SIGNAL'}
        macros[name] = {'lef': str(path), 'width_um': float(size[1]), 'height_um': float(size[2]),
                        'class': cls[1].strip(), 'site': site[1] if site else None, 'pins': pins}
    return macros


def nominal_state(name):
    m = CELL_NAME.fullmatch(name)
    if not m:
        return None
    prefix, drive, variant, vt = m.groups()
    strength = Decimal(drive.replace('p', '.'))
    return {'series': prefix + (':' + variant if variant else ''), 'nominal_strength': float(strength),
            'strength_label': drive, 'vt': VT[vt][0], 'theta': VT[vt][1]}


def build_catalog(repo):
    library = {}
    for path in sorted((repo / 'Platform/ASAP7/lib').glob('asap*.lib')):
        parsed = parse_standard_lib(path)
        if set(parsed) & set(library):
            raise ValueError('Duplicate cell master definitions')
        library.update(parsed)
    lefs = {}
    for path in sorted((repo / 'Platform/ASAP7/lef').glob('*.lef')):
        parsed = parse_lef(path)
        if set(parsed) & set(lefs):
            raise ValueError('Duplicate physical masters')
        lefs.update(parsed)
    groups = list(csv.reader((repo / 'equiv_check/asap7_equivalent_cell_list.csv').open()))
    membership = {}
    for i, row in enumerate(groups):
        for master in row:
            if master in membership:
                raise ValueError('Cell in multiple official groups')
            membership[master] = i
    cells = []
    for master in sorted(set(library) | set(lefs)):
        lib, lef = library.get(master), lefs.get(master)
        cell = {'id': len(cells), 'name': master, 'official_group': membership.get(master),
                'liberty_data': lib, 'physical': lef, 'family_id': -1, 'kappa': -1, 'theta': -1}
        nominal = nominal_state(master) if lib else None
        if lib and not lef:
            raise ValueError('Standard cell lacks LEF: ' + master)
        if lib:
            for pin, info in lib['pins'].items():
                if info['direction'] == 'internal':
                    continue
                if pin not in lef['pins'] or lef['pins'][pin]['direction'] != info['direction']:
                    raise ValueError('Liberty/LEF pin mismatch: ' + master + '/' + pin)
        if nominal:
            if master not in membership:
                raise ValueError('Standard cell absent from official equivalence groups: ' + master)
            cell.update(nominal)
        cell['frozen_reason'] = ('physical_only' if not lib and not (lef and lef['class'].startswith('BLOCK'))
                                 else 'macro' if not lib else 'dont_use' if lib['dont_use']
                                 else 'tie_cell' if lib['cell_type'] == 'tie' else None)
        cells.append(cell)
    # Keep implementation series separate inside official equivalence groups.
    # Same-strength BUF/BUF:f, ICG/ICG:DC, HB1..4 and INV/CKINVDC must not collide.
    grouped = {}
    for c in cells:
        if c.get('series'):
            key = (c['official_group'], c['series'], c['liberty_data']['behavior_sha256'])
            grouped.setdefault(key, []).append(c)
    families = []
    for key, members in sorted(grouped.items()):
        strengths = sorted(set(c['nominal_strength'] for c in members))
        family = {'id': len(families), 'official_group': key[0], 'series': key[1], 'behavior_sha256': key[2],
                  'strengths': strengths, 'vt_order': ['SLVT', 'LVT', 'RVT'], 'cell_ids': [], 'grid': {}}
        for c in members:
            c['family_id'] = family['id']
            c['kappa'] = strengths.index(c['nominal_strength'])
            coord = f'{c["kappa"]},{c["theta"]}'
            if coord in family['grid']:
                raise ValueError('Duplicate two-dimensional state: ' + c['name'])
            family['grid'][coord] = c['id']
            family['cell_ids'].append(c['id'])
        families.append(family)
    return cells, families


def action_rows(cells, families, available=None):
    """Nearest existing legal neighbor on ONE axis. -1 is invalid; never clamp."""
    available = set(range(len(cells))) if available is None else set(available)
    rows = []
    for src in cells:
        candidates = [] if src['family_id'] < 0 else [cells[i] for i in families[src['family_id']]['cell_ids']
                                                     if i in available and not cells[i]['frozen_reason']]
        for direction in DIRECTIONS:
            axis = 'kappa' if direction.startswith('strength') else 'theta'
            other = 'theta' if axis == 'kappa' else 'kappa'
            sign = 1 if direction.endswith('up') else -1
            neighbors = [c for c in candidates if c[other] == src[other] and sign * (c[axis] - src[axis]) > 0]
            reason = src['frozen_reason'] or ('not_in_design_libraries' if src['id'] not in available else None)
            dst = min(neighbors, key=lambda c: abs(c[axis] - src[axis])) if neighbors and not reason else None
            row = {'source_id': src['id'], 'source': src['name'], 'direction': direction,
                   'target_id': dst['id'] if dst else -1, 'target': dst['name'] if dst else '',
                   'valid': bool(dst), 'invalid_reason': '' if dst else reason or 'no_axis_neighbor',
                   'delta_kappa': dst['kappa'] - src['kappa'] if dst else 0,
                   'delta_theta': dst['theta'] - src['theta'] if dst else 0,
                   'footprint_changes': False, 'requires_legalization': False}
            if dst:
                assert src['official_group'] == dst['official_group']
                assert src['liberty_data']['behavior_sha256'] == dst['liberty_data']['behavior_sha256']
                if src['physical']['site'] != dst['physical']['site'] or src['physical']['height_um'] != dst['physical']['height_um']:
                    raise ValueError('Incompatible placement site/height')
                row['footprint_changes'] = src['physical']['width_um'] != dst['physical']['width_um']
                row['requires_legalization'] = row['footprint_changes']
            rows.append(row)
    return rows


def identifier(name):
    # Matches contest exporter normalization of escaped hierarchical/bus names.
    return name.replace('\\', '')


def def_components(path):
    active = False
    count = 0
    expected = None
    buffer = ''
    with path.open() as f:
        for line in f:
            if not active:
                m = re.match(r'^COMPONENTS\s+(\d+)\s*;', line)
                if m:
                    active = True
                    expected = int(m[1])
                continue
            if line.startswith('END COMPONENTS'):
                if buffer.strip() or count != expected:
                    raise ValueError('DEF component count mismatch')
                return
            buffer += line
            if ';' not in buffer:
                continue
            m = re.match(r'\s*-\s+(\S+)\s+(\S+)(.*);\s*$', buffer, re.S)
            if not m:
                raise ValueError('Unsupported DEF component: ' + buffer[:120])
            placement = re.search(r'\+\s+(FIXED|COVER|PLACED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\S+)', m[3])
            if not placement:
                raise ValueError('Component lacks placement: ' + m[1])
            count += 1
            yield identifier(m[1]), m[2], placement[1], int(placement[2]), int(placement[3]), placement[4]
            buffer = ''
    raise ValueError('Missing END COMPONENTS')


def netlist_instances(path, known_masters):
    """Read flat named-port contest instances; reject unknown master candidates."""
    current = None
    body = []
    start = re.compile(r'^\s*(\S+)\s+(\S+)\s*\(')
    with path.open() as f:
        for line in f:
            if current is None:
                m = start.match(line)
                if not m or m[1] == 'module' or m[1].startswith('//'):
                    continue
                if m[1] not in known_masters:
                    raise ValueError('Unknown netlist master: ' + m[1])
                current = (identifier(m[2]), m[1])
                body = [line[m.end():]]
            else:
                body.append(line)
            if re.search(r'\)\s*;', line):
                text = ''.join(body)
                ports = re.findall(r'\.([^\s(]+)\s*\(', text)
                empty_ports = re.fullmatch(r'\s*\)\s*;\s*', text) is not None
                if (not ports and not empty_ports) or len(set(ports)) != len(ports):
                    raise ValueError('Invalid named port list: ' + current[0])
                yield current[0], current[1], ports
                current = None
                body = []
    if current:
        raise ValueError('Unterminated netlist instance')
