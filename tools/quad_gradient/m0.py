#!/usr/bin/env python3
"""Read-only AES M0 probe. Not an optimizer or a contestant submission."""
import argparse
import bisect
import csv
import hashlib
import gzip
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCENES = {"BC", "TC", "WC"}
# Frozen before the design probe: report rounding / single-precision accumulation.
TOLERANCE_PS = 0.05


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(paths):
    return {str(p): sha(p) for p in sorted(set(paths)) if p.is_file()
            and "__pycache__" not in p.parts}


def input_fingerprint(roots):
    return fingerprint(p for root in roots for p in root.rglob("*") if p.is_file())


def validate_output(baseline, output):
    output.relative_to(ROOT / ".runs")
    if output == baseline or baseline in output.parents:
        raise ValueError("Probe output must not be inside the protected baseline")


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def check_aggregates(folder):
    """Compare native ps reports with worst-per-endpoint aggregation, not path sums."""
    endpoints = {}
    for row in rows(folder / "endpoints.tsv"):
        key = (row["scene"], row["analysis"])
        slack = float(row["slack_ps"])
        if not math.isfinite(slack):
            raise ValueError("Non-finite endpoint slack")
        group = endpoints.setdefault(key, {})
        name = row["endpoint"]
        group[name] = min(group.get(name, math.inf), slack)
    groups = {}
    native = rows(folder / "native.tsv")
    expected = {(scene, analysis) for scene in SCENES for analysis in ("max", "min")}
    seen = [(r["scene"], r["analysis"]) for r in native]
    passed = len(seen) == 6 and set(seen) == expected and set(endpoints) == expected
    for row in native:
        key = (row["scene"], row["analysis"])
        values = endpoints.get(key, {}).values()
        wns = min(values, default=math.inf)
        tns = math.fsum(min(0, v) for v in values)
        errors = {"wns_ps": abs(wns - float(row["wns_ps"])),
                  "tns_ps": abs(tns - float(row["tns_ps"]))}
        ok = all(math.isfinite(v) and v <= TOLERANCE_PS for v in errors.values())
        passed = passed and ok
        groups["/".join(key)] = dict(endpoint_count=len(values), wns_ps=wns,
                                    tns_ps=tns, errors=errors, passed=ok)
    return dict(passed=passed, tolerance_ps=TOLERANCE_PS, groups=groups)


def check_observations(folder):
    """Cross-check exported operating points against independent native path text."""
    units = (folder / "units.rpt").read_text()
    units_ok = all(re.search(r"\b" + name + r"\s+" + unit + r"\b", units)
                   for name, unit in (("time", "1ps"), ("capacitance", "1fF"), ("distance", "1um"), ("power", "1pW")))
    masters = rows(folder / "masters.tsv")
    mismatches = [r["instance"] for r in masters if r["sta_master"] != r["odb_master"]]
    comparisons = []
    cap_errors = []
    cap_report_differences = []
    report_cache = {}
    matched_groups = set()
    unmatched_points = []
    for point in rows(folder / "points.tsv"):
        key = (point["scene"], point["analysis"])
        if key not in report_cache:
            # A full-clock report repeats capture pins: use the data branch only.
            text = (folder / ("_".join(key) + "_path.rpt")).read_text().split("data arrival time")[0]
            entries = {}
            for line in text.splitlines():
                match = re.match(r"^\s*([\d.eE+\-\s]+)\s+([\^v])\s+(\S+)\s+\(", line)
                if match:
                    numbers = [float(v) for v in match[1].split()]
                    entries[(match[3], "rise" if match[2] == "^" else "fall")] = numbers
            report_cache[key] = entries
        values = report_cache[key].get((point["pin"], point["transition"]))
        if values and len(values) >= 3:
            matched_groups.add(key)
            slew = point["slew_ps"].split()
            comparisons.append(len(slew) == 1 and
                               abs(float(slew[0]) - values[-3]) <= TOLERANCE_PS and
                               abs(float(point["arrival_ps"]) - values[-1]) <= TOLERANCE_PS)
            if len(values) == 5 and point["total_cap_F"]:
                cap_report_differences.append(float(point["total_cap_F"]) * 1e15 - values[1])
        else:
            unmatched_points.append({k: point[k] for k in ("scene", "analysis", "pin", "transition")})
        if point["total_cap_F"]:
            total, pin, wire = (float(point[k]) for k in ("total_cap_F", "pin_cap_F", "wire_cap_F"))
            cap_errors.append(abs(total - pin - wire) <= max(1e-22, abs(total) * 1e-5)
                              and all(math.isfinite(v) and v >= 0 for v in (total, pin, wire)))
    expected = {(scene, analysis) for scene in SCENES for analysis in ("max", "min")}
    coverage_ok = matched_groups == expected
    return dict(passed=bool(units_ok and masters and not mismatches and coverage_ok and comparisons and
                            all(comparisons) and cap_errors and all(cap_errors)),
                all_groups_compared=coverage_ok, matched_groups=sorted("/".join(k) for k in matched_groups),
                unmatched_points=unmatched_points,
                units_match=bool(units_ok), master_count=len(masters), master_mismatches=mismatches,
                native_point_comparisons=len(comparisons), native_points_match=all(comparisons),
                cap_sum_checks=len(cap_errors), cap_sums_match=all(cap_errors),
                connected_minus_report_cap_fF=cap_report_differences,
                cap_note="Native path Cap can be effective load; differences are recorded, not forced to zero.")


NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def compare_lookup(text, table, label, tolerance):
    """Evaluate the supplied Liberty table, independently of native interpolation."""
    if len(table['axes']) != 2:
        raise ValueError('Only two-dimensional NLDM tables are supported')
    # A gate report has separate delay and slew lookup blocks.
    blocks = re.split(r'(?=^------- )', text, flags=re.M)
    matches = [b for b in blocks if re.search(r'^' + re.escape(label) + r' = ', b, re.M)]
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous native lookup block: ' + label)
    text = matches[0]
    inputs = []
    for axis in table['axes']:
        matches = re.findall(r'\b' + re.escape(axis) + r'\s*=\s*(' + NUMBER + ')', text)
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous native input axis: ' + axis)
        inputs.append(float(matches[0]))
    result = re.findall(r'^' + re.escape(label) + r' = (' + NUMBER + r')\s*$', text, re.M)
    scale = re.findall(r'^PVT scale factor = (' + NUMBER + r')\s*$', text, re.M)
    if len(result) != 1 or scale != ['1.000000000']:
        raise ValueError('Unsupported native result/PVT scaling')
    weights = []
    for axis, value in zip(('index_1', 'index_2'), inputs):
        coords = table[axis]
        if len(coords) < 2 or any(b <= a for a, b in zip(coords, coords[1:])):
            raise ValueError('Expected increasing two-dimensional table axes')
        i = min(max(bisect.bisect_right(coords, value) - 1, 0), len(coords) - 2)
        weights.append((i, (value - coords[i]) / (coords[i + 1] - coords[i])))
    (i, x), (j, y) = weights
    values = table['values']
    computed = ((1-x)*(1-y)*values[i][j] + x*(1-y)*values[i+1][j] +
                (1-x)*y*values[i][j+1] + x*y*values[i+1][j+1])
    native = float(result[0])
    error = abs(computed - native)
    return dict(passed=math.isfinite(error) and error <= tolerance, inputs=inputs,
                axes=table['axes'], native_ps=native, reconstructed_ps=computed,
                error_ps=error, tolerance_ps=tolerance)


def prepare_plan(path, baseline, output):
    raw = path.read_bytes()
    plan = json.loads(raw)
    if (plan['schema'] != 1 or plan['baseline_name'] != baseline.name or
            plan['scene'] != 'TC' or plan['mode'] != 'functional_TC' or
            plan['analysis'] != 'max' or plan['transitions'] != ['rise', 'fall'] or
            plan['state_limit'] != 32):
        raise ValueError('Unsupported closure plan or baseline identity')
    for key in ('tolerance_ps', 'tolerance_cap_fF'):
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError('Closure tolerances must be finite and positive')
    for key in ('endpoints', 'drivers'):
        if not plan[key] or len(plan[key]) != len(set(plan[key])):
            raise ValueError('Closure samples must be nonempty and distinct')
    # Written BEFORE starting the measurement process. Never tuned from its output.
    (output / 'closure_plan.json').write_bytes(raw)
    env = dict(QG_CLOSURE_SCENE=plan['scene'], QG_CLOSURE_FANIN=plan['fanin_output'],
               QG_CLOSURE_LIMIT=str(plan['state_limit']))
    for key, prefix in [('endpoints', 'ENDPOINT'), ('drivers', 'DRIVER')]:
        env['QG_CLOSURE_' + key.upper()] = str(len(plan[key]))
        for i, pin in enumerate(plan[key]):
            env[f'QG_CLOSURE_{prefix}_{i}'] = pin
    return plan, hashlib.sha256(raw).hexdigest(), env


def groups(text, kind):
    """Read balanced named Liberty groups, skipping braces inside strings/comments."""
    start = re.compile(r'\b' + re.escape(kind) + r'\s*\(([^)]*)\)\s*\{')
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*|[{}]', re.S)
    pos = 0
    while match := start.search(text, pos):
        depth = 1
        for token in tokens.finditer(text, match.end()):
            if token[0] == '{':
                depth += 1
            elif token[0] == '}':
                depth -= 1
                if depth == 0:
                    yield match[1].strip().strip('"'), text[match.end():token.start()]
                    pos = token.end()
                    break
        else:
            raise ValueError('Unbalanced Liberty group ' + kind)


def attribute(text, name, default=None):
    value = re.search(r'\b' + re.escape(name) + r'\s*:\s*("[^"]*"|[^;]+);', text)
    return value[1].strip().strip('"') if value else default


def condition(text):
    # Only simple conjunctions are canonicalized; never erase arbitrary grouping.
    clean = re.sub(r'\s+', '', text).replace('*', '&')
    if not clean:
        return ''
    if '!(' in clean or clean.count('(') != clean.count(')'):
        raise ValueError('Unsupported grouped negation or parentheses: ' + text)
    unwrapped = clean.replace('(', '').replace(')', '')
    if not re.fullmatch(r'!?\w+(?:&!?\w+)*', unwrapped) or '+' in clean or '|' in clean:
        raise ValueError('Unsupported conditional expression: ' + text)
    return '&'.join(sorted(unwrapped.split('&')))


@lru_cache(maxsize=32)
def library_cell(libdir, cell):
    found = []
    for path in sorted(Path(libdir).glob('*TT*lib*')):
        text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
        # Reject unit changes rather than silently assuming report/user units.
        if not re.search(r'cell\s*\(\s*' + re.escape(cell) + r'\s*\)', text):
            continue
        if attribute(text, 'time_unit') != '1ps' or not re.search(r'capacitive_load_unit\s*\(\s*1\s*,\s*ff\s*\)', text):
            raise ValueError('Unsupported Liberty units')
        bodies = [b for name, b in groups(text, 'cell') if name == cell]
        found.extend((str(path), text, body) for body in bodies)
    if len(found) != 1:
        raise ValueError('Ambiguous or missing TT Liberty cell: ' + cell)
    return found[0]


def liberty_table(libdir, sample, table_kind):
    path, text, body = library_cell(str(libdir), sample['liberty_cell'])
    pins = [b for name, b in groups(body, 'pin') if name == sample['to_pin'].rsplit('/', 1)[-1]]
    if len(pins) != 1:
        raise ValueError('Missing/ambiguous Liberty output or data pin')
    timing_type = ('setup_' + ('rising' if sample['from_transition'] == 'rise' else 'falling')
                   if sample['kind'] == 'setup' else 'combinational')
    candidates = [b for _, b in groups(pins[0], 'timing')
                  if attribute(b, 'related_pin') == sample['from_pin'].rsplit('/', 1)[-1]
                  and attribute(b, 'timing_type', 'combinational') == timing_type
                  and condition(attribute(b, 'when', '')) == condition(sample['when'])
                  and (sample['kind'] == 'setup' or attribute(b, 'timing_sense') == sample['sense'])]
    if len(candidates) != 1:
        raise ValueError('Selected arc does not bind uniquely to a Liberty timing group')
    tables = list(groups(candidates[0], table_kind))
    if len(tables) != 1:
        raise ValueError('Selected arc has no unique ' + table_kind)
    template, table_text = tables[0]
    templates = [b for name, b in groups(text, 'lu_table_template') if name == template]
    if len(templates) != 1:
        raise ValueError('Missing/ambiguous Liberty template')
    axes = [attribute(templates[0], f'variable_{i}') for i in (1, 2)]
    coords = {}
    for i in (1, 2):
        value = re.search(r'index_' + str(i) + r'\s*\(\s*"([^"]+)"\s*\)', table_text)
        if not value:
            raise ValueError('Missing explicit table axis')
        coords[f'index_{i}'] = [float(x) for x in value[1].split(',')]
    value = re.search(r'values\s*\((.*?)\)\s*;', table_text, re.S)
    matrix = [[float(x) for x in row.split(',')] for row in re.findall(r'"([^"]+)"', value[1])]
    if len(matrix) != len(coords['index_1']) or any(len(row) != len(coords['index_2']) for row in matrix):
        raise ValueError('Liberty matrix dimensions disagree')
    return dict(axes=axes, values=matrix, **coords), dict(file=path, cell=sample['liberty_cell'],
                pin=sample['to_pin'].rsplit('/', 1)[-1], related_pin=sample['from_pin'].rsplit('/', 1)[-1],
                timing_type=timing_type, when=sample['when'], template=template, table=table_kind)


def setup_endpoint(folder, endpoint, sample, lookup, tolerance):
    text = (folder / (endpoint['sample_id'] + '_path.rpt')).read_text()
    capture = text.split('data arrival time', 1)[1].split('data required time', 1)[0]
    pin_lines = [line for line in capture.splitlines() if sample['from_pin'] + ' (' in line]
    if len(pin_lines) != 1:
        raise ValueError('Native capture clock pin not uniquely reported')
    line = pin_lines[0]
    prefix = line.split(sample['from_pin'], 1)[0]
    numbers = [float(x) for x in re.findall(NUMBER, prefix)]
    clock_rf = 'rise' if '^' in prefix else 'fall' if 'v' in prefix else None
    if clock_rf != sample['from_transition'] or len(numbers) < 3:
        raise ValueError('Capture clock transition or slew is unverified')
    clock_slew, clock_arrival = numbers[-3], numbers[-1]
    adjustments = []
    setup_lines = []
    for line in capture.split(pin_lines[0], 1)[1].splitlines():
        if not line.strip():
            continue
        if 'library setup time' in line:
            setup_lines.append(line)
            break
        if not any(name in line for name in ('clock reconvergence pessimism', 'clock uncertainty')):
            raise ValueError('Unexplained capture required-time adjustment: ' + line)
        values = re.findall(NUMBER, line)
        if len(values) < 2:
            raise ValueError('Missing native capture adjustment value')
        adjustments.append(float(values[0]))
    if len(setup_lines) != 1:
        raise ValueError('Missing selected native library setup check')
    data = text.split('data arrival time', 1)[0]
    data_lines = [line for line in data.splitlines() if sample['to_pin'] + ' (' in line]
    if len(data_lines) != 1:
        raise ValueError('Native data pin not uniquely reported')
    prefix = data_lines[0].split(sample['to_pin'], 1)[0]
    data_values = [float(x) for x in re.findall(NUMBER, prefix)]
    data_rf = 'rise' if '^' in prefix else 'fall' if 'v' in prefix else None
    required = clock_arrival + math.fsum(adjustments) - lookup['reconstructed_ps']
    arrival = float(endpoint['arrival_s']) * 1e12
    errors = dict(required_ps=abs(required - float(endpoint['required_s']) * 1e12),
                  slack_ps=abs(required - arrival - float(endpoint['slack_ps'])),
                  margin_vs_pure_check_ps=abs(float(endpoint['margin_s']) * 1e12 - lookup['native_ps']),
                  data_slew_ps=abs(data_values[-3] - lookup['inputs'][0]),
                  clock_slew_ps=abs(clock_slew - lookup['inputs'][1]))
    passed = (endpoint['unconstrained'] == '0' and endpoint['check_role'] == 'setup' and
              data_rf == sample['to_transition'] and not sample['when'] and
              all(math.isfinite(v) and v <= tolerance for v in errors.values()))
    return dict(passed=passed, required_ps=required, arrival_ps=arrival,
                capture_clock_arrival_ps=clock_arrival, capture_adjustments_ps=adjustments,
                errors=errors, margin_semantics='PathEndCheck::margin is derated check arc delay; compared separately to pure LUT')


def expected_load_arcs(libdir, driver, cell):
    _, _, body = library_cell(str(libdir), cell)
    pin_name = driver.rsplit('/', 1)[-1]
    inst_name = driver.rsplit('/', 1)[0]
    result = set()
    for name, pin in groups(body, 'pin'):
        if name != pin_name:
            continue
        for _, timing in groups(pin, 'timing'):
            if attribute(timing, 'timing_type', 'combinational') != 'combinational':
                continue
            sense = attribute(timing, 'timing_sense')
            for rf in ('rise', 'fall'):
                if not list(groups(timing, 'cell_' + rf)):
                    continue
                from_rfs = ([rf] if sense == 'positive_unate' else
                            ['fall' if rf == 'rise' else 'rise'] if sense == 'negative_unate' else ['rise', 'fall'])
                for from_rf in from_rfs:
                    result.add((inst_name + '/' + attribute(timing, 'related_pin'),
                                condition(attribute(timing, 'when', '')), sense, from_rf, rf))
    return result


def cap_correspondence(connected, pi, table_loads, tolerance):
    total, pin, wire = connected
    c2, c1, ceff = pi
    errors = [abs(total - pin - wire)] + [abs(load - ceff) for load in table_loads]
    return dict(passed=bool(table_loads) and all(math.isfinite(e) and e <= tolerance for e in errors),
                errors_fF=errors, pi_minus_connected_fF=c1+c2-total,
                effective_minus_connected_fF=ceff-total,
                connected_transition_semantics='Net::capacitance uses Sta::connectedCap(Net), reducing rise/fall by min/max; not transition-specific Pi input')


def native_report_cap(text, driver, rf):
    points = []
    for line in text.split('data arrival time', 1)[0].splitlines():
        match = re.match(r'^\s*([\d.eE+\-\s]+)\s+([\^v])\s+(\S+)\s+\(', line)
        if match and match[3] == driver and match[2] == ('^' if rf == 'rise' else 'v'):
            values = [float(v) for v in match[1].split()]
            if len(values) == 5:
                points.append(values[1])
    if len(points) != 1 or not math.isfinite(points[0]):
        raise ValueError('Missing or ambiguous native report Cap for driver/transition')
    return points[0]


def setup_coverage(endpoints, samples, plan):
    expected = {(pin, rf) for pin in plan['endpoints'] for rf in plan['transitions']}
    observed = {
        'native_endpoints': [(r['endpoint'], r['data_transition']) for r in endpoints],
        'selected_arc_samples': [(r['to_pin'], r['to_transition']) for r in samples if r['kind'] == 'setup']}
    checks = {name: dict(passed=set(keys) == expected and len(keys) == len(expected),
                        missing=sorted(expected-set(keys)), extra=sorted(set(keys)-expected),
                        duplicates=sorted(key for key in set(keys) if keys.count(key) > 1))
              for name, keys in observed.items()}
    return dict(passed=all(c['passed'] for c in checks.values()), expected=len(expected), checks=checks)


def evaluate_closure(folder, plan):
    libdir = Path(__file__).resolve().parents[2] / 'platform/asap7/lib/NLDM'
    samples = rows(folder / 'closure_samples.tsv')
    endpoints = rows(folder / 'closure_endpoints.tsv')
    states = rows(folder / 'closure_states.tsv')
    relations = rows(folder / 'closure_relations.tsv')
    boundary = rows(folder / 'closure_boundary.tsv')
    interfaces = (folder / 'closure_interfaces.txt').read_text()
    setup_results, load_results, unmatched = [], [], []
    loads = rows(folder / 'closure_load_points.tsv')
    expected_endpoints = {(pin, rf) for pin in plan['endpoints'] for rf in plan['transitions']}
    setup_coverage_check = setup_coverage(endpoints, samples, plan)
    for sample in samples:
        try:
            if sample['scene'] != plan['scene'] or sample['mode'] != plan['mode']:
                raise ValueError('Sample scene/mode differs from frozen plan')
            text = (folder / sample['report_file']).read_text()
            rf = sample['to_transition']
            kind = ('rise_' if rf == 'rise' else 'fall_') + ('constraint' if sample['kind'] == 'setup' else 'transition')
            if sample['kind'] == 'setup':
                table, identity = liberty_table(libdir, sample, kind)
                lookup = compare_lookup(text, table, 'Check', plan['tolerance_ps'])
                if lookup['axes'] != ['constrained_pin_transition', 'related_pin_transition']:
                    raise ValueError('Unsupported check axis semantics')
                matches = [r for r in endpoints if r['endpoint'] == sample['to_pin'] and r['data_transition'] == rf]
                if len(matches) != 1:
                    raise ValueError('No unique native selected endpoint for setup arc')
                endpoint = setup_endpoint(folder, matches[0], sample, lookup, plan['tolerance_ps'])
                graph_error = abs(float(sample['native_delay_s']) * 1e12 - lookup['native_ps'])
                setup_results.append(dict(sample_id=sample['sample_id'], table=identity, lookup=lookup,
                                          endpoint=endpoint, graph_error_ps=graph_error,
                                          passed=lookup['passed'] and endpoint['passed'] and graph_error <= plan['tolerance_ps']))
            elif sample['kind'] == 'load':
                values = {}
                identities = {}
                for label, table_kind in [('Delay', 'cell_' + rf), ('Slew', kind)]:
                    table, identities[label] = liberty_table(libdir, sample, table_kind)
                    values[label] = compare_lookup(text, table, label, plan['tolerance_ps'])
                    if values[label]['axes'] != ['input_net_transition', 'total_output_net_capacitance']:
                        raise ValueError('Unsupported driver axis semantics')
                pi = re.search(r'Pi model C2=(' + NUMBER + r') Rpi=(' + NUMBER + r') C1=(' + NUMBER + r'), Ceff=(' + NUMBER + ')', text)
                waveform = re.search(r'Driver waveform slew = (' + NUMBER + ')', text)
                points = [r for r in loads if r['driver'] == sample['to_pin'] and r['transition'] == rf and r['scene'] == plan['scene']]
                if pi is None or waveform is None or len(points) != 1:
                    raise ValueError('Native Pi/Ceff/waveform or connected load point missing/ambiguous')
                c2, resistance, c1, ceff = map(float, pi.groups())
                point = points[0]
                total, pin, wire = [float(point[key]) * 1e15 for key in ('total_cap_F', 'pin_cap_F', 'wire_cap_F')]
                report_cap = native_report_cap((folder / point['report_file']).read_text(), sample['to_pin'], rf)
                cap_check = cap_correspondence([total, pin, wire], [c2, c1, ceff],
                                              [value['inputs'][1] for value in values.values()], plan['tolerance_cap_fF'])
                graph_error = abs(float(sample['native_delay_s']) * 1e12 - values['Delay']['native_ps'])
                load_results.append(dict(sample_id=sample['sample_id'], driver=sample['to_pin'], transition=rf,
                    tables=identities, lookups=values, connected_cap_fF=dict(total=total, pin=pin, wire=wire),
                    pi_model=dict(c2_fF=c2, c1_fF=c1, rpi_kohm=resistance, effective_cap_fF=ceff),
                    effective_minus_connected_fF=ceff-total, report_cap_fF=report_cap,
                    report_cap_file=point['report_file'], connected_minus_report_fF=total-report_cap,
                    effective_minus_report_fF=ceff-report_cap, waveform_slew_ps=float(waveform[1]),
                    native_output_slew_ps=float(point['output_slew_ps']), graph_error_ps=graph_error,
                    cap_correspondence=cap_check, passed=all(v['passed'] for v in values.values()) and
                    cap_check['passed'] and
                    graph_error <= plan['tolerance_ps']))
            else:
                raise ValueError('Unknown sample kind')
        except (ValueError, KeyError, IndexError, TypeError) as error:
            unmatched.append(dict(sample_id=sample.get('sample_id'), kind=sample.get('kind'), reason=str(error)))
    expected_setup_samples = len(expected_endpoints)
    setup_ok = (setup_coverage_check['passed'] and len(setup_results) == expected_setup_samples and
                len([s for s in samples if s['kind'] == 'setup']) == expected_setup_samples and
                all(r['passed'] for r in setup_results) and not any(r['kind'] == 'setup' for r in unmatched))
    load_coverage = []
    waveform_checks = []
    for driver in plan['drivers']:
        selected = [s for s in samples if s['kind'] == 'load' and s['to_pin'] == driver]
        try:
            cells = {s['liberty_cell'] for s in selected}
            if len(cells) != 1:
                raise ValueError('Missing or ambiguous driver master')
            expected = expected_load_arcs(libdir, driver, next(iter(cells)))
            observed = [(s['from_pin'], condition(s['when']), s['sense'], s['from_transition'], s['to_transition']) for s in selected]
            missing, extra = sorted(expected-set(observed)), sorted(set(observed)-expected)
            load_coverage.append(dict(driver=driver, expected=len(expected), observed=len(observed), missing=missing, extra=extra,
                                      passed=bool(expected) and not missing and not extra and len(observed) == len(expected)))
        except ValueError as error:
            load_coverage.append(dict(driver=driver, passed=False, reason=str(error)))
        for rf in plan['transitions']:
            results = [r for r in load_results if r['driver'] == driver and r['transition'] == rf]
            error = abs(max((r['waveform_slew_ps'] for r in results), default=math.inf) -
                        max((r['native_output_slew_ps'] for r in results), default=-math.inf))
            waveform_checks.append(dict(driver=driver, transition=rf, error_ps=error,
                                        passed=math.isfinite(error) and error <= plan['tolerance_ps']))
    load_ok = (bool(load_results) and len(load_results) == len([s for s in samples if s['kind'] == 'load']) and
               all(r['passed'] for r in load_results + load_coverage + waveform_checks) and
               not any(r['kind'] == 'load' for r in unmatched) and
               {r['to_pin'] for r in samples if r['kind'] == 'load'} == set(plan['drivers']))
    tag_check = dict(passed=False, reason='Installed Tcl references do not establish native tag-to-tag compatibility, rejected pairs, or required-state validity; no legality inferred from structure.',
                     boundary=boundary, structural_arc_rows=len(relations), raw_state_rows=len(states),
                     zero_required_states=sum(float(r['required_s'] or 0) == 0 for r in states if r['truncated'] == '0'),
                     truncated=sum(r['truncated'] == '1' for r in states),
                     required_validity='unverified', native_reference='probe.log QG_NATIVE_STATE_BEGIN/END blocks',
                     interface_inventory_present=bool(interfaces), native_legal_relation_oracle=False,
                     coverage_passed=False, numerical_state_reconstruction_passed=False)
    checks = dict(tagged_predecessors=tag_check,
                  setup_binding=dict(passed=setup_ok, reason='' if setup_ok else 'Setup sample identity, coverage or numeric correspondence incomplete',
                                     coverage=setup_coverage_check, samples=setup_results),
                  load_semantics=dict(passed=load_ok, reason='' if load_ok else 'Driver arc coverage or working-point output correspondence incomplete',
                                      coverage=load_coverage, waveform_checks=waveform_checks, samples=load_results))
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks, unmatched=unmatched,
                net_slew_rows=len(rows(folder / 'closure_net_slews.tsv')),
                scope='Bounded M0 interface identity/value checks only. No graph completeness, derivatives or network Jacobian validation.')


def run(baseline, output, closure_plan=None):
    baseline = baseline.resolve()
    output = output.resolve()
    validate_output(baseline, output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"milestone": "M0", "status": "failed", "optimizer_implemented": False}
    protected = []
    before = {}
    source = {}
    try:
        sys.path.insert(0, str(ROOT / "evaluator/internal"))
        from legality import verify_certificate
        verify_certificate(baseline / "summary.csv", require_runtime=False)
        cfg = json.loads((baseline / "run_config.json").read_text())
        if cfg["BENCHMARK"] != "aes" or cfg["RUN_RESIZER"] != "0":
            raise ValueError("M0 requires a certified AES baseline without Resizer")
        for key, path in (("INPUT_DEF", ROOT / "benchmarks/aes/input.def"),
                          ("INPUT_VERILOG", ROOT / "benchmarks/aes/input.v")):
            if cfg["sha256"][key] != sha(path):
                raise ValueError(f"Baseline input mismatch: {key}")
        if json.loads((baseline / "structural_validation.json").read_text())["changed_from_baseline"]:
            raise ValueError("M0 requires unchanged R0, not a candidate")
        protected = (ROOT / "benchmarks/aes", ROOT / "platform/asap7", ROOT / "evaluator", baseline)
        before = input_fingerprint(protected)
        source = fingerprint(Path(__file__).parent.glob("*"))
        env = os.environ.copy()
        env.update(QG_BASELINE=str(baseline), QG_OUTPUT=str(output),
                   QG_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        plan = None
        if closure_plan is not None:
            plan, plan_hash, closure_env = prepare_plan(closure_plan, baseline, output)
            env.update(closure_env)
            report['closure_plan_sha256'] = plan_hash
        exe = env.get("OPENROAD_EXE", "/install/OpenROAD/bin/openroad")
        command = [exe, "-no_init", "-no_splash", "-exit",
                   str(Path(__file__).with_name("probe.tcl"))]
        report.update(command=command, source_sha256=source, inputs_sha256=before,
                      baseline=str(baseline), openroad_sha256=sha(exe),
                      scoring_policy=json.loads((ROOT / "evaluator/config/scoring_config.json").read_text()))
        if plan:
            native_root = Path('/OpenROAD/src/sta')
            native_files = [native_root / p for p in ('search/Search.i', 'search/Search.cc', 'search/PathEnd.cc',
                'search/Tag.cc', 'graph/Graph.i', 'dcalc/GraphDelayCalc.cc', 'dcalc/DmpCeff.cc',
                'dcalc/DelayCalcBase.cc', 'liberty/TableModel.cc', 'dcalc/DelayCalc.tcl',
                'search/Sta.cc', 'network/Network.i')]
            report['native_interface_source_sha256'] = fingerprint(native_files)
            if len(report['native_interface_source_sha256']) != len(native_files):
                raise ValueError('Installed interface source evidence missing')
        report["openroad_version"] = subprocess.check_output([exe, "-version"], text=True).strip()
        with (output / "probe.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=1800)
        if "QG_M0_EXPORT_COMPLETE" not in (output / "probe.log").read_text():
            raise ValueError("OpenROAD did not complete the probe")
        report["aggregation"] = check_aggregates(output)
        report["observations"] = check_observations(output)
        # Baseline fresh-process reports are an independent identity check.
        baseline_rows = list(csv.DictReader((baseline / "timing.csv").open()))
        comparisons = []
        for row in baseline_rows:
            for analysis, prefix in (("max", "setup"), ("min", "hold")):
                actual = report["aggregation"]["groups"][row["corner"] + "/" + analysis]
                comparisons.append(all(abs(actual[k] - float(row[field])) <= TOLERANCE_PS
                                       for k, field in (("wns_ps", prefix + "_worst_slack_ps"),
                                                        ("tns_ps", prefix + "_tns_ps"))))
        report["baseline_timing_matches"] = len(comparisons) == 6 and all(comparisons)
        report["capabilities"] = rows(output / "capabilities.tsv")
        report['closure'] = evaluate_closure(output, plan) if plan else {
            'passed': False, 'checks': {name: {'passed': False, 'reason': 'No preregistered closure experiment'}
                                     for name in ('tagged_predecessors', 'setup_binding', 'load_semantics')}}
        report['gaps'] = [f'{name}: {check["reason"]}' for name, check in report['closure']['checks'].items()
                          if not check['passed']]
        report['required_checks'] = {
            'endpoint_aggregation': report['aggregation']['passed'],
            'baseline_timing': report['baseline_timing_matches'],
            'native_observations': report['observations']['passed'],
            **{name: check['passed'] for name, check in report['closure']['checks'].items()}}
        report['status'] = ('passed' if all(report['required_checks'].values()) else
                            'blocked' if all(report['required_checks'][key] for key in
                                             ('endpoint_aggregation', 'baseline_timing', 'native_observations')) else 'failed')
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["inputs_unchanged"] = bool(before) and before == input_fingerprint(protected)
        if not report["inputs_unchanged"]:
            report["status"] = "failed"
        report["elapsed_s"] = time.monotonic() - started
        report['required_checks'] = report.get('required_checks', {})
        report['required_checks']['input_identity'] = report['inputs_unchanged']
        report['source_unchanged'] = bool(source) and source == fingerprint(Path(__file__).parent.glob('*'))
        report['required_checks']['probe_source_identity'] = report['source_unchanged']
        if not report['source_unchanged']:
            report['status'] = 'failed'
        report["m0_passed"] = report['status'] == 'passed' and all(report['required_checks'].values())
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "m0_passed", "inputs_unchanged", "elapsed_s")}))
    return 0 if report['m0_passed'] else 2 if report["status"] == "blocked" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Certified unchanged AES evaluator run")
    parser.add_argument("output", type=Path, help="New directory under .runs/")
    parser.add_argument('--closure-plan', type=Path, help='Preregistered bounded M0 interface experiments')
    args = parser.parse_args()
    return run(args.baseline, args.output, args.closure_plan)


if __name__ == "__main__":
    sys.exit(main())
