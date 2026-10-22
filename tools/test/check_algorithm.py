#!/usr/bin/env python3
"""Algorithm, batch-search and runtime regressions; no physical E2E runner."""
import contextlib
import copy
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/quad_gradient'))
import torch
import actions
import analyze
import optimizer
import physical
from types import SimpleNamespace
from data_io import DesignView, load_timing_data
from objective import TCSetupObjective, ObjectiveSeeds, tc_loss
from gradient import backward
from optimizer import Checkpoint, search
from model import Library, build_timing_model, lookup_many, lookup_values
import make_eval_params
from internal import legality

REFERENCE = (-1., -.1)
SEED = 1 / (1000 * abs(-1. + 1e-9)) + 1 / (1000 * abs(-.1 + 1e-9))


@contextlib.contextmanager
def patch_roots(root):
    with patch.object(optimizer, 'ROOT', root), patch.object(physical, 'ROOT', root):
        yield


def table(path, columns, records, delimiter='\t'):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(columns.split())
        writer.writerows(records)


def snapshot(folder, paths, arcs, pins=(), transition='rise', slew=1.):
    folder.mkdir(exist_ok=True)
    table(folder / 'paths.tsv', 'path pin transition arrival_ps slew_ps slack_ps endpoint', paths)
    records = []
    for i, (kind, a, b, role, delay) in enumerate(arcs):
        report = f'{i}.rpt'
        x = slew if role == 'combinational' else 1.
        (folder / report).write_text(f'input_net_transition = {x}\ntotal_output_net_capacitance = 1\n'
                                    'constrained_pin_transition = 1\nrelated_pin_transition = 1\n')
        records.append((kind, a, b, role, transition, transition, 'current', delay * 1e-12, report))
    table(folder / 'arcs.tsv', 'kind from_pin to_pin role from_transition to_transition liberty_cell graph_delay_s report_file', records)
    table(folder / 'snapshot.pins.tsv', 'instance pin direction net signal net_signal',
          [(*pin, 'SIGNAL', 'SIGNAL') for pin in pins])
    table(folder / 'snapshot.cells.tsv', 'instance master macro physical status', [])
    table(folder / 'snapshot.library.tsv', 'master pin direction function tristate', [])
    table(folder / 'views.tsv', 'master view', [])


class ToyLibrary(Library):
    """Independent affine NLDM oracle; real Library cap parsing/cache is exercised."""
    def table(self, sample, kind, cell=None):
        if cell == 'missing':
            raise KeyError('missing table')
        if sample['kind'] == 'setup':
            c, x, y = 0., -.5, 1.
        elif sample['role'] == 'Reg Clk to Q':
            c, x, y = (1., 0., 2.) if kind.startswith('cell_') else (1., 0., 3.)
        else:
            c, x, y = (2., 4., 5.) if kind.startswith('cell_') else (3., 6., 7.)
        if cell in ('candidate', 'better'):
            if kind.startswith('cell_'):
                x += 1. if cell == 'candidate' else -5.
            else:
                c += 2.
        return (([0., 2.], [0., 2.]), [[c, c + 2*y], [c + 2*x, c + 2*x + 2*y]])


def library(folder):
    folder.mkdir()
    cells = ''.join(f'cell({name}) {{ pin(A) {{ rise_capacitance : {cap}; '
                    f'fall_capacitance : {cap + 1}; }} pin(B) {{ capacitance : 3; }} }}'
                    for name, cap in [('current', 1.), ('candidate', 1.5), ('better', 1.5)])
    (folder / 'toy_TT.lib').write_text('library(toy) { time_unit : "1ps"; '
                                     'capacitive_load_unit(1, ff);' + cells + '}')
    return ToyLibrary(folder)


def state_fixture(root):
    groups=root/'evaluator/third_party/displacement/asap7_equivalent_cell_list.csv'
    groups.parent.mkdir(parents=True)
    groups.write_text('AND,AND2\nBUF\nCLKBUF\n')
    cells=[['u','AND',0,0,'N',0,0,'PLACED'],
           ['clk','CLKBUF',0,0,'N',0,0,'FIRM']]
    pins=[['','A','a','INPUT','SIGNAL','SIGNAL'],
          ['','B','b','INPUT','SIGNAL','SIGNAL'],
          ['','Y','y','OUTPUT','SIGNAL','SIGNAL'],
          ['','CLK','clock','INPUT','CLOCK','CLOCK'],
          ['u','A','a','INPUT','SIGNAL','SIGNAL'],
          ['u','B','b','INPUT','SIGNAL','SIGNAL'],
          ['u','Y','y','OUTPUT','SIGNAL','SIGNAL'],
          ['clk','A','clock','INPUT','CLOCK','CLOCK'],
          ['clk','Y','clock_out','OUTPUT','CLOCK','CLOCK']]
    lib=[]
    for master in ('AND','AND2','BUF','CLKBUF'):
        buffer=master in ('BUF','CLKBUF')
        for pin in (('A','Y') if buffer else ('A','B','Y')):
            lib.append([master,pin,'output' if pin=='Y' else 'input',
                        ('A' if buffer else 'A&B') if pin=='Y' else '',
                        '',int(buffer),0])
    before,after=root/'before',root/'after'
    for prefix in (before,after):
        write_snapshot(prefix,cells,pins,lib)
    return before,after,cells,pins,lib,groups


def write_snapshot(prefix,cells,pins,lib,*,grid=False,attrs='',die=1000):
    table(Path(str(prefix)+'.cells.tsv'),'instance master x y orientation macro physical status',cells)
    table(Path(str(prefix)+'.pins.tsv'),'instance pin net direction signal net_signal',pins)
    table(Path(str(prefix)+'.library.tsv'),'master pin direction function tristate buffer inverter',lib)
    components=''.join(f'- {c[0]} {c[1]} + PLACED ( {c[2]} {c[3]} ) {c[4]} {attrs if c[0]=="u" else ""} ;\n' for c in cells)
    Path(str(prefix)+'.def').write_text(
        f'VERSION 5.8 ;\nDESIGN toy ;\nDIEAREA ( 0 0 ) ( {die} {die} ) ;\n'+
        ('GCELLGRID X 0 DO 10 STEP 100 ;\n' if grid else '')+
        f'COMPONENTS {len(cells)} ;\n'+components+'END COMPONENTS\nEND DESIGN\n')


class StateValidationTests(unittest.TestCase):
    def test_parses_each_snapshot_file_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            before,after,cells,pins,lib,groups=state_fixture(root)
            cells[0][1]='AND2'
            write_snapshot(after,cells,pins,lib)
            with patch_roots(root), \
                    patch.object(legality,'def_records',wraps=legality.def_records) as defs, \
                    patch.object(legality,'table',wraps=legality.table) as tables, \
                    patch.object(physical,'rows',wraps=physical.rows) as extra:
                report=physical.validate_state(before,after)
            self.assertTrue(report['passed'])
            self.assertTrue(report['changed_from_baseline'])
            self.assertEqual(Counter(str(c.args[0]) for c in defs.call_args_list),
                             {str(before)+'.def':1,str(after)+'.def':1})
            reads=Counter(str(c.args[0]) for c in tables.call_args_list+extra.call_args_list)
            self.assertEqual(reads,{str(prefix)+suffix:1 for prefix,suffix in
                ((before,'.cells.tsv'),(before,'.pins.tsv'),(before,'.library.tsv'),
                 (after,'.cells.tsv'),(after,'.pins.tsv'))})

    def test_merged_boundaries_reuse_middle_snapshot_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            before,applied,cells,pins,lib,groups=state_fixture(root)
            cells[0][1]='AND2'
            write_snapshot(applied,cells,pins,lib)
            routed=root/'routed'
            write_snapshot(routed,cells,pins,lib,grid=True)
            # B/C deliberately have no catalog: reuse must be explicit, not a
            # missing-file bypass of either boundary's full structural checks.
            for prefix in (applied,routed): Path(str(prefix)+'.library.tsv').unlink()
            with patch.object(legality,'def_records',wraps=legality.def_records) as defs, \
                    patch.object(legality,'table',wraps=legality.table) as tables:
                context=physical.ValidationContext.load(before,applied,str(before)+'.library.tsv',groups)
                apply=context.validate()
                route=context.advance(routed)
                post_route=route.validate(True)
            self.assertIs(route.before,context.after)
            self.assertIs(route.library,context.library)
            self.assertIs(route.groups,context.groups)
            self.assertEqual(Counter(str(c.args[0]) for c in defs.call_args_list),
                             {str(p)+'.def':1 for p in (before,applied,routed)})
            self.assertEqual(Counter(str(c.args[0]) for c in tables.call_args_list),
                {**{str(p)+suffix:1 for p in (before,applied,routed)
                    for suffix in ('.cells.tsv','.pins.tsv')},str(before)+'.library.tsv':1})
            self.assertEqual(apply,legality.check(before,applied,str(before)+'.library.tsv',groups))
            self.assertEqual(post_route,legality.check(applied,routed,str(before)+'.library.tsv',groups,True))
            self.assertEqual(apply['transformations']['changed_masters'],1)
            self.assertEqual(post_route['transformations']['changed_masters'],0)
            with self.assertRaisesRegex(ValueError,'Fixed DEF data changed: GCELLGRID'):
                route.validate()  # post_route must never relax A->B.
            Path(str(before)+'.library.tsv').unlink()
            with self.assertRaises(FileNotFoundError):
                physical.ValidationContext.load(before,applied,str(before)+'.library.tsv',groups)

    def test_sizing_and_route_changes_still_pass(self):
        for post_route in (False,True):
            with self.subTest(post_route=post_route), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                before,after,cells,pins,lib,groups=state_fixture(root)
                cells[0][1]='AND2'
                cells[0][2]=10
                write_snapshot(after,cells,pins,lib,grid=post_route)
                context=physical.ValidationContext.load(before,after,str(before)+'.library.tsv',groups)
                self.assertEqual(context.check(post_route),legality.check(
                    before,after,str(before)+'.library.tsv',groups,post_route))
                with patch_roots(root):
                    self.assertTrue(physical.validate_state(before,after,post_route)['passed'])

    def test_invalid_changes_remain_rejected(self):
        for case in ('wiring','protected','added','removed','pin_semantics',
                     'duplicate_instance','duplicate_pin','attributes','floorplan','pre_route_grid'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                before,after,cells,pins,lib,groups=state_fixture(root)
                attrs=''; die=1000; grid=False
                if case=='wiring':
                    # AND is symmetric: public transforms allow this swap, but
                    # the tool's sizing-only connectivity assertion must refuse.
                    pins[4][2],pins[5][2]=pins[5][2],pins[4][2]
                elif case=='protected': cells[1][1]='BUF'
                elif case=='added':
                    cells.append(['buf','BUF',0,0,'N',0,0,'PLACED'])
                    pins.extend([['buf','A','a','INPUT','SIGNAL','SIGNAL'],
                                 ['buf','Y','new','OUTPUT','SIGNAL','SIGNAL']])
                    pins[4][2]='new'
                elif case=='removed':
                    cells.pop(0); pins[:]=[p for p in pins if p[0]!='u']
                elif case=='pin_semantics':
                    cells[0][1]='AND2'
                    next(r for r in lib if r[:2]==['AND2','Y'])[3]='A|B'
                    write_snapshot(before,[['u','AND',0,0,'N',0,0,'PLACED'],cells[1]],pins,lib)
                elif case=='duplicate_instance': cells.append(cells[0].copy())
                elif case=='duplicate_pin': pins.append(pins[4].copy())
                elif case=='attributes': attrs='+ HALO 1 2 3 4'
                elif case=='floorplan': die=2000
                elif case=='pre_route_grid': grid=True
                write_snapshot(after,cells,pins,lib,grid=grid,attrs=attrs,die=die)
                def outcome(check):
                    try: return True,check()
                    except ValueError as error: return False,str(error)
                context=physical.ValidationContext.load(before,after,str(before)+'.library.tsv',groups)
                self.assertEqual(outcome(context.check),outcome(lambda: legality.check(
                    before,after,str(before)+'.library.tsv',groups)))
                with patch_roots(root), self.assertRaises(ValueError):
                    physical.validate_state(before,after)

    def test_private_context_is_immutable_and_does_not_patch_public_checker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            before,after,_,_,_,groups=state_fixture(root)
            public=tuple(getattr(legality,key) for key in ('check','table','def_records','read_library','Design'))
            context=physical.ValidationContext.load(before,after,str(before)+'.library.tsv',groups)
            first=context.check()
            self.assertEqual(context.check(),first)
            self.assertIn('COMPONENTS',context.before.records)
            for row in (context.before.cells[0],context.before.pins[0],context.library['AND']['Y']):
                with self.assertRaises(TypeError): row['master']='changed'
            self.assertEqual(public,tuple(getattr(legality,key) for key in
                ('check','table','def_records','read_library','Design')))

    def test_no_cross_call_path_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            before,after,cells,pins,lib,_=state_fixture(root)
            with patch_roots(root):
                self.assertTrue(physical.validate_state(before,after)['passed'])
                pins[4][2],pins[5][2]=pins[5][2],pins[4][2]
                write_snapshot(after,cells,pins,lib)
                with self.assertRaisesRegex(ValueError,'Sizing changed connectivity'):
                    physical.validate_state(before,after)

    def test_library_has_no_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'toy_TT.lib'
            source.write_text('library(toy) { time_unit : "1ps"; capacitive_load_unit(1, ff); '
                              'cell(C) { pin(A) { capacitance : 1; } } }')
            with patch.object(hashlib,'sha256',side_effect=AssertionError('tool must not hash files')):
                library=Library(root)
                source.write_text(source.read_text().replace('capacitance : 1','capacitance : 2'))
                self.assertEqual(library.cap('C','A','rise'),1.)
                self.assertEqual(Library(root).cap('C','A','rise'),2.)


class AlgorithmChecks(unittest.TestCase):
    def test_lookup_and_loss(self):
        nonlinear = (([0., 1., 3.], [0., 2., 5.]),
                     [[0., 4., 25.], [1., 7., 31.], [9., 19., 49.]])
        affine = (([0., 2.], [0., 4.]), [[0., 12.], [4., 24.]])
        actual = lookup_many([(nonlinear, 1., 2.), (nonlinear, -1., 6.), (affine, 1., 2.)])
        torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64),
                                   torch.tensor([[7., 6., 8.], [25., 7., 6.], [10., 4., 4.]], dtype=torch.float64))
        for x, y in ((1., 2.), (-1., 6.)):
            point = torch.tensor([[x, y]], dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda p: lookup_values([(affine, x, y)], p), (point,)))
        with self.assertRaisesRegex(ValueError, 'Invalid NLDM axes'):
            lookup_many([((([0., 0.], [0., 1.]), [[0., 1.], [1., 2.]]), 0., 0.)])
        self.assertAlmostEqual(tc_loss(REFERENCE, (-2., -.2), .01), 1/1.99 + .1/.19)
        self.assertEqual(tc_loss((0., .1), REFERENCE), 0.)
        self.assertLess(tc_loss(REFERENCE, (-2., -.2)), tc_loss((-2., -.2), (-2., -.2)))

    def test_benefit_cache_and_native_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib = library(root / 'lib')
            conditional = root / 'conditional'
            conditional.mkdir()
            (conditional / 'toy_TT.lib').write_text('''library(toy) {
                time_unit : "1ps"; capacitive_load_unit(1, ff);
                cell("current") { comment : "quoted }"; /* ignored { */
                    pin(A) { capacitance : 1; } pin(B) { capacitance : 3; }
                    pin(Y) { timing() { related_pin : "A"; when : "(A)*!B";
                        timing_sense : positive_unate;
                        cell_rise(T) { index_1("0, 2"); index_2("0, 4");
                            values("0, 12", "4, 24"); }
                    } }
                }
            }''')
            parsed = Library(conditional)
            sample = dict(kind='load', role='combinational', liberty_cell='current',
                          from_pin='g/A', to_pin='g/Y', from_transition='rise',
                          to_transition='rise', sense='positive_unate', when='!B & A')
            self.assertEqual(parsed.table(sample, 'cell_rise'), ([[0., 2.], [0., 4.]], [[0., 12.], [4., 24.]]))
            self.assertEqual(parsed.cap('current', 'A', 'rise'), 1.)
            with self.assertRaisesRegex(ValueError, 'Unsupported grouped negation'):
                parsed.table(dict(sample, when='!(A&B)'), 'cell_rise')
            folder = root / 'graph'
            arc_data = [('load', 'launch/CLK', 'launch/Q', 'Reg Clk to Q', 3.),
                        ('load', 'gate/A', 'gate/Y', 'combinational', 11.),
                        ('setup', 'ff/CLK', 'ff/D', 'setup', .5)]
            base = dict(instance='gate', current='current', direction='vt_down', valid=True)
            choices = [dict(base, target=name) for name in ('candidate', 'better', 'missing')]
            choices += [dict(base, target='candidate', direction='drive_up'),
                        dict(base, target='better', valid=False, reason='protected')]
            results = []
            for slew in (1., 1.5):
                paths = [(0, pin, 'rise', i, slew if pin == 'gate/A' else 1., -100., 'ff/D')
                         for i, pin in enumerate(('launch/Q', 'gate/A', 'gate/Y', 'ff/D'))]
                snapshot(folder, paths, arc_data, [('launch', 'Q', 'OUTPUT', 'net'),
                                                  ('gate', 'A', 'INPUT', 'net')], slew=slew)
                with patch.object(analyze, 'enumerate_actions', return_value=choices):
                    warm, summary = analyze.analyze(folder, root / 'lib', REFERENCE, library=lib)
                    cold, _ = analyze.analyze(folder, root / 'lib', REFERENCE, library=ToyLibrary(root / 'lib'))
                self.assertEqual(warm, cold)
                self.assertEqual([s['target'] for s in warm], ['better', 'candidate', 'candidate'])
                self.assertEqual([s['direction'] for s in warm[1:]], ['drive_up', 'vt_down'])
                self.assertEqual(summary['model_rejections'], 1)
                self.assertEqual(summary['invalid_reasons'], {'protected': 1})
                self.assertEqual(json.loads((folder / 'scores.json').read_text()), warm)
                score = warm[-1]
                for key, expected in [('own_delay', slew*SEED), ('downstream_slew', -SEED),
                                      ('upstream_load', 2.5*SEED), ('benefit', -(slew + 1.5)*SEED)]:
                    self.assertAlmostEqual(score[key], expected, places=13)
                results.append(score['benefit'])
            self.assertNotEqual(*results)
            self.assertEqual(lib.cap('candidate', 'A', 'fall'), 2.5)
            self.assertEqual(lib.cap('candidate', 'B', 'rise'), 3.)
            source = root / 'lib/toy_TT.lib'
            source.write_text(source.read_text().replace('rise_capacitance : 1.5', 'rise_capacitance : 9.5'))
            # Fixed-run libraries no longer hash or reread modified source files.
            with patch.object(hashlib,'sha256',side_effect=AssertionError('tool must not hash files')), \
                    patch.object(analyze,'enumerate_actions',return_value=choices):
                warm, _ = analyze.analyze(folder, root / 'lib', REFERENCE, library=lib)
                self.assertEqual(lib.cap('candidate','A','rise'),1.5)
                self.assertEqual(ToyLibrary(root/'lib').cap('candidate','A','rise'),9.5)
            with self.assertRaisesRegex(ValueError,'Library directory does not match libdir'):
                lib.verify_folder(root/'other-lib')

    def test_objective_endpoint_states_and_units(self):
        nodes = [(name, transition) for name, transition in
                 [('a/D', 'rise'), ('a/D', 'fall'), ('b/D', 'rise'), ('c/D', 'rise')]]
        timing = SimpleNamespace(endpoint_states={
            'a': {nodes[0]: -100., nodes[1]: -100.+5e-7},
            'b': {nodes[2]: -100.+5e-7}, 'c': {nodes[3]: 0.}},
            checks={node: [(-.5, None, 0., 0.)] for node in nodes})
        objective = TCSetupObjective(REFERENCE)
        seeds = objective.seeds(timing)
        tns = 1/(1000*abs(REFERENCE[0]+1e-9))
        wns = 1/(1000*abs(REFERENCE[1]+1e-9))
        for node, expected in zip(nodes, [(tns+wns/2)/2, (tns+wns/2)/2, tns+wns/2, 0.]):
            self.assertAlmostEqual(seeds.arrival[node], expected, places=15)
            self.assertAlmostEqual(seeds.slew[node], -.5*expected, places=15)
        self.assertEqual(objective.value(REFERENCE), tc_loss(REFERENCE, REFERENCE))
        timing.endpoint_states = {'a': {nodes[0]: 2., nodes[1]: 1.}}
        self.assertEqual(sum(objective.seeds(timing).arrival.values()), 0.)
        timing.checks[nodes[1]] *= 2
        with self.assertRaisesRegex(ValueError, 'Nonunique setup seed'):
            objective.seeds(timing)
        del timing.checks[nodes[1]]
        with self.assertRaisesRegex(ValueError, 'Missing setup data-slew seed'):
            objective.seeds(timing)

    def test_model_build_and_backward_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib = library(root/'lib')
            graph = root/'graph'
            snapshot(graph, [(0, 'g/A', 'rise', 0., 1., -10., 'g/Y'),
                             (0, 'g/Y', 'rise', 11., 1., -10., 'g/Y')],
                     [('load', 'g/A', 'g/Y', 'combinational', 11.),
                      ('setup', 'ff/CLK', 'g/Y', 'setup', 0.)])
            # An off-union arc never required a native report in the old path.
            arc_path = graph/'arcs.tsv'
            arc_path.write_text(arc_path.read_text()+
                'load\toff/A\toff/Y\tcombinational\trise\trise\tcurrent\t0\tmissing.rpt\n')
            timing = build_timing_model(load_timing_data(graph), SimpleNamespace(pins=[]), lib)
            self.assertEqual(len(timing.edges), 1)
            self.assertFalse(any(hasattr(timing, name) for name in
                                 ('ga', 'gs', 'load', 'edge_adjoint', 'reference')))
            seeds = TCSetupObjective(REFERENCE).seeds(timing)
            original_model, original_seeds = copy.deepcopy(timing.__dict__), copy.deepcopy(seeds)
            first = backward(timing, seeds)
            other = backward(timing, ObjectiveSeeds(
                {n: 2*v for n, v in seeds.arrival.items()},
                {n: 2*v for n, v in seeds.slew.items()}))
            again = backward(timing, seeds)
            self.assertEqual(timing.__dict__, original_model)
            self.assertEqual(seeds, original_seeds)
            self.assertEqual(first, again)
            for index, pair in first.edge_adjoint.items():
                for value, doubled in zip(pair, other.edge_adjoint[index]):
                    self.assertAlmostEqual(doubled, 2*value, places=15)
            for node, value in first.load.items():
                self.assertAlmostEqual(other.load[node], 2*value, places=15)

    def test_model_owns_launch_boundary_report_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib = library(root/'lib')
            graph = root/'graph'
            snapshot(graph, [(0, 'g/A', 'rise', 0., 1., -10., 'g/Y'),
                             (0, 'g/Y', 'rise', 11., 1., -10., 'g/Y')],
                     [('load', 'g/A', 'ff/CLK', 'combinational', 1.),
                      ('load', 'ff/CLK', 'g/Y', 'Reg Clk to Q', 2.),
                      ('load', 'g/A', 'ff/CLK', 'combinational', 1.)])
            # Before the launch boundary is added this arc is off-union.
            (graph/'0.rpt').unlink()
            data = load_timing_data(graph)
            self.assertEqual(len(data.arcs), 3)
            with patch.object(data, 'read_report', wraps=data.read_report) as read_report:
                timing = build_timing_model(data, SimpleNamespace(pins=[]), lib)
            self.assertEqual([call.args[0]['report_file'] for call in read_report.call_args_list],
                             ['1.rpt', '2.rpt'])
            self.assertEqual(timing.nodes[('ff/CLK', 'rise')], (9., 1.))
            self.assertEqual(len(timing.edges), 2)

    def test_avesmooth_fanin_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib = library(root / 'lib')
            for transition in ('rise', 'fall'):
                for arrivals, weights in (((100., 99.5), (.5, .5)), ((100., 99.), (.5, .5)),
                                          ((100., 98.999), (1., 0.)), ((-2., -2.01), (.5, .5)),
                                          ((0., -1e-13), (.5, .5)), ((0., -2e-12), (1., 0.))):
                    with self.subTest(transition=transition, arrivals=arrivals):
                        paths = [(i, pin, transition, arrival, 1., -100., 'g/Y')
                                 for i, a in enumerate(arrivals)
                                 for pin, arrival in ((f'g/{"AB"[i]}', a), ('g/Y', max(arrivals)))]
                        snapshot(root / 'fanin', paths,
                                 [('load', 'g/A', 'g/Y', 'combinational', 0.),
                                  ('load', 'g/B', 'g/Y', 'combinational', 0.),
                                  ('setup', 'ff/CLK', 'g/Y', 'setup', 0.)], transition=transition)
                        timing = build_timing_model(load_timing_data(root/'fanin'),
                                                    SimpleNamespace(pins=[]), lib)
                        model = backward(timing, TCSetupObjective(REFERENCE).seeds(timing))
                        for pin, weight in zip(('g/A', 'g/B'), weights):
                            self.assertAlmostEqual(model.ga[(pin, transition)], SEED*weight, places=13)
                            self.assertAlmostEqual(model.gs[(pin, transition)], SEED*(4*weight - 1.5), places=13)

    def test_candidates_and_checkpoint_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masters = [f'BUFx{x}_ASAP7_75t_R' for x in ('1', '1p5', '2', '4', '8')] + ['BUFx2_ASAP7_75t_L']
            table(root / 's.library.tsv', 'master pin direction function tristate',
                  [(m, p, d, f, '') for m in masters for p, d, f in [('A', 'input', ''), ('Y', 'output', 'A')]])
            table(root / 's.cells.tsv', 'instance master macro physical status',
                  [('u', masters[2], 0, 0, 'PLACED'), ('fixed', masters[2], 0, 0, 'LOCKED')])
            table(root / 's.pins.tsv', 'instance pin direction net signal net_signal', [])
            (root / 'groups.csv').write_text(','.join(masters) + '\n')
            views = {m: ['LEF', 'BC', 'TC', 'WC'] for m in masters}
            views[masters[3]].remove('WC')
            catalog = actions.ActionCatalog.load(root/'s.library.tsv', root/'groups.csv')
            design = DesignView.load(root/'s', catalog.library, views)
            with patch.object(legality, 'read_library', side_effect=AssertionError('catalog reread')), \
                    patch.object(Path, 'open', side_effect=AssertionError('action I/O')):
                decisions = actions.enumerate_actions(design, catalog)
                self.assertEqual(decisions, actions.enumerate_actions(design, catalog))
            self.assertTrue(all(not a['valid'] and a['reason'] == 'protected' for a in decisions if a['instance'] == 'fixed'))
            self.assertEqual({a['direction']: a['target'] for a in decisions if a['valid']},
                             dict(drive_down=masters[0], drive_up=masters[4], vt_down=masters[5]))

            def summary(tns=-1., wns=-.1, hold=-.004):
                return {s: dict(setup_tns_ns=tns, setup_wns_ns=wns, hold_tns_ns=-.2, hold_wns_ns=hold,
                                slew_violation_sum_ps=0., cap_violation_sum_ff=0., fanout_violation_sum=0.,
                                global_routing_overflow=0., max_global_routing_overflow=0.) for s in ('BC', 'TC', 'WC')}

            def action(target):
                return dict(instance='u', current='initial', target=target, direction='vt_down', benefit=.1, valid=True)

            policy = json.loads(make_eval_params.DEFAULT_CONFIG.read_text())
            self.assertFalse(optimizer.guards(summary(), summary(-.9, -.09, -.009)))
            degraded = summary(-.9, -.09, -.02)
            for row in degraded.values():
                row.update(hold_tns_ns=-1.1, slew_violation_sum_ps=1.,
                           cap_violation_sum_ff=1., fanout_violation_sum=1.)
            self.assertFalse(optimizer.guards(summary(), degraded))
            diagnostics = optimizer.degradation_diagnostics(summary(), degraded, policy)
            self.assertEqual(len(diagnostics), 15)
            self.assertIn('TC:hold_wns', [d['metric'] for d in diagnostics])
            self.assertEqual(optimizer.guards(summary(), {}), ['scene_coverage'])
            for field in ('max_global_routing_overflow', 'global_routing_overflow'):
                guarded = summary(-.9, -.09)
                guarded['BC'][field] = 1.
                self.assertIn(f'BC:{field}', optimizer.guards(summary(), guarded))
            outcomes = dict(bad=summary(-.9, -.09, -.02), good=summary(-.7, -.07),
                            nogain=summary(-.9, -.09), better=summary(-.8, -.08))
            outcomes['bad']['TC']['global_routing_overflow'] = 1.
            outcomes['better'] = degraded
            scores = {'initial-graph': [action('bad')],
                      'refresh-1': [action('better')], 'batch-0-fresh-sta': [],
                      'batch-1-fresh-sta': [dict(action('good'), current='better')]}
            summaries, probes, analyzed, evaluations = {}, [], [], []

            class Runner:
                def __init__(self, workspace, benchmark): self.workspace = workspace
                def command(self, *args, **kwargs): pass
                def close(self): pass
                def analyze(self, folder, *args):
                    analyzed.append(folder.name)
                    if folder.name == 'batch-2-fresh-sta':
                        raise ValueError('post-evaluation model failed')
                    return scores[folder.name], {'approximations': []}
                def initialize(self, input_dir, top):
                    folder = self.workspace / 'initial'
                    folder.mkdir()
                    summaries[folder] = summary()
                    for filename in ('evaluated.def', 'evaluated.v', 'parasitics_BC.spef',
                                     'parasitics_TC.spef', 'parasitics_WC.spef'):
                        (folder / filename).write_text('initial')
                    return folder
                def evaluate(self, name, baseline, candidate):
                    evaluations.append(name)
                    folder = self.workspace / name
                    folder.mkdir()
                    optimizer.save(folder / 'actions.json', candidate)
                    target = candidate[0]['target']
                    summaries[folder] = outcomes[target]
                    for filename in ('evaluated.def', 'evaluated.v', 'parasitics_BC.spef',
                                     'parasitics_TC.spef', 'parasitics_WC.spef'):
                        (folder / filename).write_text(name)
                    clean = self.probe(folder, name+'-sta', graph=True)
                    optimizer.save(folder/'feedback.json', dict(scenarios=outcomes[target],
                                   provenance='tool_private_fresh_route_clean_native_sta', certified=False,
                                   clean_sta=str(clean)))
                    return physical.PhysicalResult(folder,outcomes[target],clean)
                def probe(self, baseline, name, graph=False):
                    probes.append((name, baseline, graph))
                    folder = self.workspace / name
                    folder.mkdir()
                    master = {'batch-0-fresh': 'bad', 'batch-1-fresh': 'better',
                              'batch-2-fresh': 'good'}.get(baseline.name, 'initial')
                    table(folder / 'snapshot.cells.tsv', 'instance master', [('u', master)])
                    native = summaries[baseline]
                    table(folder / 'native.tsv', 'scene analysis tns_ps wns_ps',
                          [(s, kind, row[f'{prefix}_tns_ns']*1000, row[f'{prefix}_wns_ns']*1000)
                           for s, row in native.items() for kind, prefix in [('max', 'setup'), ('min', 'hold')]])
                    for scene in ('BC', 'TC', 'WC'):
                        for kind in ('max_slew', 'max_capacitance', 'max_fanout'):
                            (folder / f'{scene}_{kind}.rpt').write_text('')
                    return folder

            feedback_reads=[]
            read_text=Path.read_text
            def track_read(path,*args,**kwargs):
                if path.name=='feedback.json': feedback_reads.append(path)
                return read_text(path,*args,**kwargs)

            for path in ('benchmarks/aes', 'platform/asap7', 'evaluator/config'):
                (root / path).mkdir(parents=True)
            (root / 'benchmarks/aes/benchmark.json').write_text(json.dumps({'top_module': 'aes_cipher_top'}))
            (root / 'evaluator/config/scoring_config.json').write_bytes(make_eval_params.DEFAULT_CONFIG.read_bytes())
            executable = root / 'executable'
            executable.write_text('fixture')
            with patch.multiple(optimizer, ROOT=root, Runner=Runner), \
                    patch.object(optimizer.shutil, 'which', return_value=str(executable)), \
                    patch.object(hashlib,'sha256',side_effect=AssertionError('tool must not hash files')), \
                    patch.object(Path,'read_text',track_read):
                optimizer.optimize(root / 'benchmarks/aes', root / 'platform/asap7', root / 'out', 'aes_cipher_top', 3)
                work = root / 'out/quad_gradient'
                events = json.loads((work / 'updates.json').read_text())
                manifest = json.loads((work / 'manifest.json').read_text())
                for key in ('inputs_unchanged','sources_unchanged','inputs','sources'):
                    self.assertNotIn(key,manifest)
                self.assertEqual(manifest['outputs'],{name:str(root/'out'/name) for name in ('aes.def','aes.v')})
                self.assertTrue(all(set(tool)=={'path'} for tool in manifest['tools'].values()))
                self.assertEqual(manifest['initialization']['method'], 'minimal_native')
                self.assertEqual(manifest['initialization']['reference_tc_ns'], list(REFERENCE))
                self.assertNotIn('r0', evaluations)
                self.assertEqual([e['accepted'] for e in events], [False, True, False])
                self.assertEqual(events[2]['actions'][0]['current'], 'better')
                self.assertEqual(events[2]['before_checkpoint'], events[1]['after_checkpoint'])
                self.assertEqual(events[2]['after_checkpoint'], events[1]['after_checkpoint'])
                self.assertEqual(events[2]['best_checkpoint'], events[1]['best_checkpoint'])
                self.assertEqual(events[2]['frozen_total'], events[1]['frozen_total'])
                self.assertIn('post-evaluation model failed', events[2]['error'])
                self.assertEqual(analyzed, ['initial-graph', 'batch-0-fresh-sta', 'refresh-1',
                                            'batch-1-fresh-sta', 'batch-2-fresh-sta'])
                self.assertEqual([(name, graph) for name, _, graph in probes],
                                 [('initial-graph', True), ('batch-0-fresh-sta', True),
                                  ('refresh-1', True), ('batch-1-fresh-sta', True),
                                  ('batch-2-fresh-sta', True)])
                self.assertEqual(evaluations, ['batch-0-fresh','batch-1-fresh','batch-2-fresh'])
                # Only final export reads the saved feedback; batches use the result object.
                self.assertEqual(feedback_reads,[work/'batch-1-fresh/feedback.json'])
                self.assertEqual(events[0]['before_checkpoint'], events[0]['after_checkpoint'])
                self.assertIn('TC:global_routing_overflow', events[0]['guard_failures'])
                self.assertEqual(events[1]['guard_failures'], [])
                self.assertEqual(events[1]['diagnostics'], diagnostics)
                self.assertEqual(manifest['events'], events)
                self.assertEqual([p.name for name, p, _ in probes if name == 'refresh-1'], ['initial'])
                for name in ('aes.def', 'aes.v'):
                    self.assertEqual((root / 'out' / name).read_text(), 'batch-1-fresh')
                plan = json.loads((work / 'frozen_plan.json').read_text())
                self.assertEqual(plan['epsilon_ns'], make_eval_params.EPSILON_NS)
                self.assertEqual(plan['improvement_threshold'], 0.)
                self.assertEqual(plan['best_comparison'], 'candidate_loss < best_loss')
                scores['initial-graph'] = [action('bad')]
                optimizer.optimize(root / 'benchmarks/aes', root / 'platform/asap7', root / 'declined', 'aes_cipher_top', 1)
                declined = json.loads((root / 'declined/quad_gradient/manifest.json').read_text())
                self.assertEqual(declined['status'], 'no_improvement')
                self.assertFalse((root / 'declined/aes.def').exists())
                self.assertFalse((root / 'declined/aes.v').exists())
                graph = root / 'out/quad_gradient/initial-graph'
                self.assertEqual(physical.native_baseline(graph)['TC']['hold_wns_ns'], -.004)
                (graph / 'TC_max_capacitance.rpt').write_text('broken (VIOLATED)\n')
                with self.assertRaisesRegex(ValueError, 'Unrecognized dedicated ERC'):
                    physical.native_baseline(graph)
                (graph / 'native.tsv').write_text((graph / 'native.tsv').read_text() + 'TC\tmax\t-1000\t-100\n')
                with self.assertRaisesRegex(ValueError, 'Duplicate native MCMM'):
                    physical.native(graph)

    def test_backend_launch_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            baseline,input_dir=root/'checkpoint',root/'input'
            runner=physical.Runner(root,'aes')
            calls=[]
            def command(argv,log,env=None):
                calls.append((argv,log,env))
                if env is not None and env['QG_INITIALIZE']=='1':
                    for name in ['evaluated.def']+[f'parasitics_{s}.spef' for s in ('BC','TC','WC')]:
                        (log.parent/name).write_text('initial')
            with patch_roots(root), patch.object(runner,'command',command), \
                    patch.object(physical,'native',return_value={}), \
                    patch.object(physical,'validate_state',return_value={'passed':True}), \
                    patch.dict(os.environ,dict(OPENROAD_EXE='/test/openroad',QG_INITIALIZE='1',
                        QG_ROUTE='1',QG_ACTION='/stale',QG_EXPORT_GRAPH='1',
                        QG_BASELINE='/stale',QG_OUTPUT='/stale',MCMM_CONFIG='/stale')):
                runner.probe(baseline,'native')
                runner.probe(baseline,'graph',graph=True)
                runner.initialize(input_dir,'top')
            backends=[call for call in calls if call[2] is not None]
            self.assertEqual(len(backends),3)
            for (argv,log,env),(name,source,initialize,graph) in zip(backends,
                    [('native',baseline,'0','0'),('graph',baseline,'0','1'),('initial',input_dir,'1','0')]):
                self.assertEqual(argv,['/test/openroad','-exit',str(root/'tools/quad_gradient/backend.tcl')])
                self.assertEqual(log,root/name/'openroad.log')
                self.assertEqual(env['QG_ROOT'],str(root))
                self.assertEqual(env['QG_BASELINE'],str(source))
                self.assertEqual(env['QG_OUTPUT'],str(root/name))
                self.assertEqual(env['QG_INITIALIZE'],initialize)
                self.assertEqual(env['QG_EXPORT_GRAPH'],graph)
                self.assertEqual(env['QG_ROUTE'],'0')
                self.assertEqual(env['QG_ACTION'],'')
                mcmm=input_dir if initialize=='1' else root/'benchmarks/aes'
                self.assertEqual(env['MCMM_CONFIG'],str(mcmm/'mcmm.tcl'))
            self.assertEqual(backends[2][2]['QG_TOP'],'top')

    def test_private_fresh_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,_,cells,pins,lib,_=state_fixture(root)
            (root/'benchmarks/aes').mkdir(parents=True)
            optimizer.save(root/'benchmarks/aes/benchmark.json', {'top_module':'top'})
            baseline=root/'initial'; baseline.mkdir()
            batch=[dict(instance='u',target='AND2')]
            netlist='module top(); endmodule;'
            runner=physical.Runner(root,'aes')
            calls=[]
            clean_error=None
            bad_route=None
            def command(argv, log, env=None, **kwargs):
                calls.append((argv,env))
                folder=log.parent
                if env is None:
                    optimizer.save(log,[dict(declared_cap_ff=1.,exported_cap_ff=1.)]*3)
                    return
                self.assertEqual(env['QG_ROOT'],str(root))
                self.assertEqual(env['QG_OUTPUT'],str(folder))
                self.assertEqual(env['MCMM_CONFIG'],str(root/'benchmarks/aes/mcmm.tcl'))
                self.assertEqual(env['QG_INITIALIZE'],'0')
                if env['QG_ROUTE']=='1':
                    self.assertEqual(env['QG_BASELINE'],str(baseline))
                    self.assertEqual(Path(env['QG_ACTION']).read_text(),'u\tAND2\n')
                    self.assertEqual(env['QG_EXPORT_GRAPH'],'0')
                    write_snapshot(folder/'before',cells,pins,lib)
                    applied=[r.copy() for r in cells]; applied[0][1]='AND2'
                    write_snapshot(folder/'evaluated',applied,pins,lib,grid=bad_route=='apply_grid')
                    routed_pins=[r.copy() for r in pins]
                    if bad_route=='route_wiring':
                        routed_pins[4][2],routed_pins[5][2]=routed_pins[5][2],routed_pins[4][2]
                    write_snapshot(folder/'snapshot',applied,routed_pins,lib,grid=True)
                    for prefix in ('evaluated','snapshot'):
                        (folder/(prefix+'.library.tsv')).unlink()
                    if bad_route=='missing_catalog': (folder/'before.library.tsv').unlink()
                    (folder/'evaluated.v').write_text(netlist)
                    table(folder/'overflow.tsv','total max',[(3,2)])
                    for scene in ('BC','TC','WC'):
                        (folder/f'parasitics_{scene}.spef').write_text('fresh')
                    if bad_route=='missing_checkpoint': (folder/'parasitics_WC.spef').unlink()
                else:
                    self.assertEqual(env['QG_ACTION'],'')
                    self.assertEqual(env['QG_ROUTE'],'0')
                    self.assertEqual(env['QG_EXPORT_GRAPH'],'1')
                    routed=Path(env['QG_BASELINE'])
                    self.assertEqual((routed/'parasitics_TC.spef').read_text(),'fresh')
                    state=json.loads((routed/'state-check.json').read_text())
                    self.assertEqual(set(state),{'apply','post_route'})
                    self.assertEqual(state['apply']['transformations']['changed_masters'],1)
                    self.assertEqual(state['post_route']['transformations']['changed_masters'],0)
                    if clean_error is not None: raise clean_error
                    (folder/'loaded.v').write_text(netlist if bad_route!='clean_netlist' else 'changed;')
                    table(folder/'snapshot.pins.tsv','instance pin direction net',[])
                    table(folder/'snapshot.cells.tsv','instance master',[])
                    table(folder/'native.tsv','scene analysis tns_ps wns_ps',
                          [(s,a,-1000,-100) for s in ('BC','TC','WC') for a in ('max','min')])
                    table(folder/'paths.tsv','path pin transition arrival_ps slew_ps slack_ps endpoint',[])
                    table(folder/'arcs.tsv','kind sample_id',[])
                    for scene in ('BC','TC','WC'):
                        (folder/f'{scene}_parasitic_annotation.rpt').write_text(
                            'Found 0 unannotated drivers.\nFound 0 partially unannotated drivers.\n')
                        for kind in ('max_slew','max_capacitance','max_fanout'):
                            (folder/f'{scene}_{kind}.rpt').write_text('')
            with patch_roots(root), patch.object(runner,'command',command), \
                    patch.object(hashlib,'sha256',side_effect=AssertionError('tool must not hash files')):
                result=runner.evaluate('fresh',baseline,batch)
                folder=result.path
                self.assertEqual([env['QG_ROUTE'] for _,env in calls if env is not None],['1','0'])
                self.assertEqual(json.loads((folder/'actions.json').read_text()),batch)
                for i,error in enumerate((RuntimeError('graph export failed'),
                                          subprocess.TimeoutExpired(['openroad'],.1))):
                    with self.subTest(error=error):
                        clean_error=error
                        with self.assertRaises(type(error)):
                            runner.evaluate(f'failed-{i}',baseline,batch)
                        self.assertFalse((root/f'failed-{i}'/'feedback.json').exists())
                clean_error=None
                for bad_route,error in (('apply_grid','Fixed DEF data changed: GCELLGRID'),
                        ('route_wiring','Sizing changed connectivity'),
                        ('missing_catalog','before.library.tsv'),
                        ('missing_checkpoint','Missing fresh checkpoint file'),
                        ('clean_netlist','Clean STA netlist differs')):
                    with self.subTest(bad_route=bad_route):
                        name='failed-'+bad_route
                        with self.assertRaisesRegex((ValueError,OSError),error):
                            runner.evaluate(name,baseline,batch)
                        self.assertFalse((root/name/'feedback.json').exists())
                        self.assertEqual((root/(name+'-sta')).exists(),bad_route=='clean_netlist')
                count=len(calls)
                with self.assertRaisesRegex(ValueError,'Invalid TSV action field'):
                    runner.evaluate('invalid',baseline,[dict(instance='u\n',target='AND2')])
                self.assertEqual(len(calls),count)
            feedback=json.loads((folder/'feedback.json').read_text())
            self.assertEqual(folder,root/'fresh')
            self.assertEqual(result.summary,feedback['scenarios'])
            self.assertEqual(result.graph,root/'fresh-sta')
            self.assertFalse(feedback['certified'])
            self.assertEqual(feedback['provenance'],'tool_private_fresh_route_clean_native_sta')
            self.assertNotIn('sha256',feedback)
            self.assertEqual(set(feedback['state_checks']),{'apply','post_route'})
            self.assertEqual(Path(feedback['clean_sta']),root/'fresh-sta')
            self.assertTrue((Path(feedback['clean_sta'])/'paths.tsv').is_file())
            self.assertTrue((Path(feedback['clean_sta'])/'arcs.tsv').is_file())
            self.assertEqual(feedback['scenarios']['TC']['setup_tns_ns'],-1.)
            self.assertEqual(len(optimizer.guards(feedback['scenarios'],feedback['scenarios'])),6)
            self.assertFalse((folder/'summary.csv').exists())
            self.assertFalse(any('internal/run.py' in str(arg) for argv,_ in calls for arg in argv))

    def test_sizing_master_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            groups=root/'evaluator/third_party/displacement/asap7_equivalent_cell_list.csv'
            groups.parent.mkdir(parents=True)
            groups.write_text('BUF,INV\nOTHER\n')
            before,after=root/'before',root/'after'
            for prefix,master in ((before,'BUF'),(after,'INV')):
                Path(str(prefix)+'.def').write_text('END DESIGN\n')
                table(Path(str(prefix)+'.cells.tsv'),'instance master',[('u',master)])
                table(Path(str(prefix)+'.pins.tsv'),'instance pin net',[('u','Y','n')])
            lib={'BUF':{'Y':dict(direction='output',function='A',tristate='')},
                 'INV':{'Y':dict(direction='output',function='A',tristate='')}}
            with patch_roots(root), \
                    patch.object(physical.ValidationContext,'check',return_value={'passed':True}), \
                    patch.object(legality,'read_library',return_value=lib):
                self.assertTrue(physical.validate_state(before,after)['passed'])
                for key,value in (('direction','input'),('function','!A'),('tristate','EN')):
                    with self.subTest(key=key):
                        original=lib['INV']['Y'][key]
                        lib['INV']['Y'][key]=value
                        with self.assertRaisesRegex(ValueError,'pin semantics'):
                            physical.validate_state(before,after)
                        lib['INV']['Y'][key]=original
                lib['INV']['Z']=lib['INV']['Y'].copy()
                with self.assertRaisesRegex(ValueError,'library pins'):
                    physical.validate_state(before,after)
                del lib['INV']['Z']
                for content in ('BUF\nINV\n','BUF\n'):
                    groups.write_text(content)
                    with self.assertRaisesRegex(ValueError,'equivalence group'):
                        physical.validate_state(before,after)

    def test_annotation_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            table(root/'snapshot.pins.tsv','instance pin direction net',[
                ('u','Y','OUTPUT','dangling'),('v','Y','OUTPUT',''),
                ('w','Y','OUTPUT','loaded'),('sink','A','INPUT','loaded'),
                ('x','Y','OUTPUT','out'),('','OUT','OUTPUT','out'),
                ('y','Y','OUTPUT','io'),('sink','B','INOUT','io'),
                ('z','Y','OUTPUT','topio'),('','IO','INOUT','topio'),
                ('','IN','INPUT','unused')])
            good='Found 3 unannotated drivers.\n u/Y\n v/Y\n IN\nFound 0 partially unannotated drivers.\n'
            for scene in ('BC','TC','WC'):
                (root/f'{scene}_parasitic_annotation.rpt').write_text(good)
            self.assertEqual(physical.validate_annotation(root)['TC']['unannotated_dangling_drivers'],3)
            path=root/'TC_parasitic_annotation.rpt'
            bad=[good.replace('3 unannotated','2 unannotated'),good+'extra\n',
                 good.replace(' u/Y','u/Y'),good.replace(' v/Y',' u/Y'),
                 good.replace('0 partially','1 partially'),'',
                 good.replace(' u/Y',' unknown/Y')]
            bad += [good.replace(' u/Y',' '+name+'/Y') for name in ('w','x','y','z')]
            for text in bad:
                with self.subTest(report=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError): physical.validate_annotation(root)
            path.unlink()
            with self.assertRaisesRegex(ValueError,'Missing annotation report'):
                physical.validate_annotation(root)

    def test_evaluation_parameters(self):
        source = ROOT / 'evaluator/reference_results/aes/baseline.csv'
        original = source.read_bytes()
        config = make_eval_params.DEFAULT_CONFIG
        with patch.object(hashlib,'sha256',side_effect=AssertionError('tool must not hash files')):
            params = make_eval_params.generate(source)
            self.assertEqual(params, make_eval_params.generate(source))
        self.assertEqual(params['eligibility'], 'not_certified')
        self.assertEqual(params['units']['summary_time'], 'ns')
        self.assertNotIn('inputs',params)
        reference = params['algorithm']['reference_tc_ns']
        self.assertEqual(params['algorithm']['normalization_denominators_ns'], [abs(x + 1e-9) for x in reference])
        self.assertAlmostEqual(params['algorithm']['baseline_loss'], sum(-x/abs(x + 1e-9) for x in reference), places=13)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'params.json'
            argv = ['--baseline', str(source), '--output', str(output)]
            make_eval_params.main(argv)
            self.assertEqual(json.loads(output.read_text()), params)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                make_eval_params.main(argv)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(json.loads(output.read_text()), params)
            with source.open() as stream:
                records = list(csv.DictReader(stream))
            columns = list(records[0])
            for bad in ('nan', 'inf', '0.1', '-1e-9'):
                changed = [dict(row, setup_tns_ns=bad) if row['corner'] == 'TC' else row for row in records]
                table(root / 'bad.csv', ' '.join(columns), [[r[c] for c in columns] for r in changed], ',')
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    make_eval_params.generate(root / 'bad.csv')
            table(root / 'bad.csv', ' '.join(columns), [[r[c] for c in columns] for r in records if r['corner'] != 'TC'], ',')
            with self.assertRaisesRegex(ValueError, 'BC/TC/WC'):
                make_eval_params.generate(root / 'bad.csv')
            wrong_units = [c.replace('setup_tns_ns', 'setup_tns_ps') for c in columns]
            table(root / 'bad.csv', ' '.join(wrong_units), [[r[c] for c in columns] for r in records], ',')
            with self.assertRaisesRegex(ValueError, 'timing in ns'):
                make_eval_params.generate(root / 'bad.csv')
            policy = json.loads(config.read_text())
            policy['epsilon']['time_ns'] = 0
            (root / 'bad.json').write_text(json.dumps(policy))
            with self.assertRaises(ValueError):
                make_eval_params.generate(source, root / 'bad.json')
        self.assertEqual(source.read_bytes(), original)


def action(name, target='B', benefit=1., direction='drive_up', current='A', valid=True):
    return dict(instance=name, current=current, target=target, benefit=benefit,
                direction=direction, valid=valid)


class Physical:
    """Controlled physical boundary; checkpoints include all physical artifacts."""
    def __init__(self, candidates, responses):
        self.candidates = iter(candidates)
        self.responses = iter(responses)
        self.applied = []
        self.refreshed = []

    def refresh(self, checkpoint, batch):
        self.refreshed.append(checkpoint)
        return next(self.candidates, [])

    def apply(self, checkpoint, actions, batch):
        self.applied.append((checkpoint, actions))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        loss, failures = response
        masters = dict(checkpoint.masters)
        masters.update({a['instance']: a['target'] for a in actions})
        return Checkpoint(f'complete-def-v-spef-{batch}', loss, masters), failures


class BatchSearchTests(unittest.TestCase):
    def initial(self):
        return Checkpoint('initial-def-v-spef', 1., {f'i{i:03}': 'A' for i in range(120)})

    def test_global_top_k_before_dedup_without_refill(self):
        candidates = [action('i000', benefit=10), action('i000', 'C', 9, 'vt_up')]
        candidates += [action(f'i{i:03}', benefit=8) for i in range(1, 100)]
        physical = Physical([candidates], [(0.9, [])])
        result = search(self.initial(), physical, max_batches=1)
        event = result.events[0]
        self.assertEqual((event['eligible'], event['top_k'], event['deduplicated']), (101, 100, 99))
        self.assertEqual(len(physical.applied[0][1]), 99)
        self.assertNotIn('i099', [a['instance'] for a in physical.applied[0][1]])
        self.assertEqual(result.best.loss, 0.9)

    def test_eligibility_small_set_and_deterministic_ties(self):
        candidates = [action('i002', benefit=0), action('i003', benefit=-1),
                      action('i004', valid=False), action('i001', benefit=1e-12),
                      action('i000', 'C', direction='vt_up'), action('i000')]
        physical = Physical([candidates, []], [(0.9, [])])
        result = search(self.initial(), physical)
        self.assertEqual([(a['instance'], a['target']) for a in physical.applied[0][1]],
                         [('i000', 'B'), ('i001', 'B')])
        self.assertEqual(result.events[0]['eligible'], 3)
        self.assertEqual(result.stop_reason, 'no_candidates')

    def test_safe_degradation_continues_but_best_is_retained(self):
        physical = Physical([[action('i000')], [action('i001')], [action('i002')]],
                            [(0.8, []), (0.95, []), (0.9, [])])
        result = search(self.initial(), physical, max_batches=3)
        self.assertEqual(result.best.path, 'complete-def-v-spef-0')
        self.assertEqual(result.current.loss, 0.9)
        self.assertEqual(physical.applied[2][0].loss, 0.95)
        self.assertEqual(result.stop_reason, 'batch_budget')

    def test_rejected_or_failed_batches_restore_entire_checkpoint(self):
        initial = self.initial()
        physical = Physical([[action('i000')]] * 3,
                            [(0.5, ['BC:hold_tns']), RuntimeError('route failed'), (0.8, [])])
        result = search(initial, physical, max_batches=3)
        self.assertTrue(all(checkpoint is initial for checkpoint, _ in physical.applied))
        self.assertEqual([e['K'] for e in result.events], [100, 50, 25])
        self.assertEqual([e['rollback'] for e in result.events], [True, True, False])
        self.assertEqual(result.best.masters['i000'], 'B')

    def test_shrink_floor_reset_and_independent_K10_patience(self):
        physical = Physical([[action('i000')]] * 30, [(1.1, ['overflow'])] * 30)
        result = search(self.initial(), physical)
        self.assertEqual([e['K'] for e in result.events], [100, 50, 25, 12, 10, 10, 10, 10, 10])
        self.assertEqual(result.stop_reason, 'K10_patience')
        self.assertEqual(result.events[-1]['K10_patience'], 5)
        self.assertEqual(result.best.path, 'initial-def-v-spef')

    def test_three_nonimprovements_shrink_once_and_reset(self):
        physical = Physical([[action(f'i{i:03}')] for i in range(7)],
                            [(1.1, [])] * 5 + [(1.2, ['overflow']), (1.1, [])])
        result = search(self.initial(), physical, max_batches=7)
        self.assertEqual([e['K'] for e in result.events], [100, 100, 100, 50, 50, 50, 25])
        self.assertEqual(result.events[5]['shrink_patience'], 0)

    def test_every_strict_improvement_replaces_best(self):
        physical = Physical([[action('i000')], [action('i001')]],
                            [(1. - 0.5e-7, []), (1. - 2e-7, [])])
        result = search(self.initial(), physical, max_batches=2)
        self.assertEqual([e['improved_best'] for e in result.events], [True, True])

    def test_equal_loss_does_not_replace_best(self):
        physical = Physical([[action('i000')]], [(1., [])])
        result = search(self.initial(), physical, max_batches=1)
        self.assertFalse(result.events[0]['improved_best'])
        self.assertEqual(result.best.path, 'initial-def-v-spef')

    def test_oscillation_restores_best_and_freezes_only_on_commit(self):
        physical = Physical([[action('i000')], [action('i000', 'A', current='B')],
                             [action('i000', 'A', current='B'), action('i001')]],
                            [(0.8, []), (0.85, []), (0.7, [])])
        result = search(self.initial(), physical, max_batches=3)
        self.assertEqual(physical.applied[1][1], [])  # best has B: validated no-op freeze
        self.assertEqual(result.frozen, {'i000'})
        self.assertEqual(result.events[2]['eligible'], 1)
        self.assertEqual(result.best.masters['i000'], 'B')

    def test_freeze_restoration_failure_rolls_back_history_and_freeze(self):
        physical = Physical([[action('i000')], [action('i000', 'A', current='B')],
                             [action('i000', 'A', current='B')]],
                            [(1.1, []), (0.9, ['ERC']), (0.8, [])])
        result = search(self.initial(), physical, max_batches=3)
        self.assertIs(physical.applied[1][0], physical.applied[2][0])
        self.assertEqual(result.events[1]['frozen_total'], [])
        self.assertEqual(result.events[2]['freeze_requested'], ['i000'])
        self.assertEqual(result.best.masters['i000'], 'A')

    def test_failed_trial_does_not_create_oscillation(self):
        physical = Physical([[action('i000')], [action('i000')]],
                            [(0.8, ['ERC']), (0.7, [])])
        result = search(self.initial(), physical, max_batches=2)
        self.assertEqual(result.frozen, set())
        self.assertEqual(result.events[1]['freeze_requested'], [])

    def test_time_budget_includes_refresh_and_prevents_apply(self):
        times = iter([0, 0, 11, 11])
        physical = Physical([[action('i000')]], [])
        result = search(self.initial(), physical, seconds=10, clock=lambda: next(times))
        self.assertEqual(result.stop_reason, 'time_budget')
        self.assertEqual(physical.applied, [])
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0]['terminal_reason'], 'time_budget')
        self.assertFalse(result.events[0]['physical_attempt'])
        self.assertEqual(result.events[0]['elapsed_seconds'], 11)

    def test_improvement_resets_both_patience_counters_at_minimum_K(self):
        physical = Physical([[action(f'i{i:03}')] for i in range(12)],
                            [(1.1, ['overflow'])] * 4 + [(1.1, [])] * 2 +
                            [(0.8, [])] + [(0.9, [])] * 5)
        result = search(self.initial(), physical)
        self.assertEqual(result.events[6]['K10_patience'], 0)
        self.assertEqual(result.events[6]['shrink_patience'], 0)
        self.assertEqual(result.events[-1]['K10_patience'], 5)
        self.assertEqual(result.best.loss, 0.8)
        self.assertEqual(result.stop_reason, 'K10_patience')

    def test_post_physical_timeout_preserves_prior_best(self):
        times = iter([0, 0, 0, 11, 11])
        physical = Physical([[action('i000')]], [TimeoutError('physical deadline')])
        result = search(self.initial(), physical, seconds=10, clock=lambda: next(times))
        self.assertEqual(result.stop_reason, 'time_budget')
        self.assertTrue(result.events[0]['rollback'])
        self.assertEqual(result.best.path, 'initial-def-v-spef')

    def test_nonfinite_physical_loss_is_rejected(self):
        physical = Physical([[action('i000')]], [(float('nan'), [])])
        result = search(self.initial(), physical, max_batches=1)
        self.assertIn('nonfinite_TC_loss', result.events[0]['guard_failures'])
        self.assertTrue(result.events[0]['rollback'])

    def test_empty_search_is_normal_no_improvement(self):
        result = search(self.initial(), Physical([[]], []))
        self.assertEqual(result.stop_reason, 'no_candidates')
        self.assertEqual(result.best.path, 'initial-def-v-spef')
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0]['terminal_reason'], 'no_candidates')
        self.assertFalse(result.events[0]['physical_attempt'])
        self.assertFalse(result.events[0]['rollback'])


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] != 'Z'
    except FileNotFoundError:
        return False


class RuntimeTests(unittest.TestCase):
    def test_command_timeout_kills_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root/'child.pid'
            code = ("import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)")
            runner = physical.Runner(root, 'aes')
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    runner.command([sys.executable, '-c', code], root/'command.log', timeout=1)
                pid = int(pidfile.read_text())
                self.assertFalse(alive(pid))
                self.assertLess(runner.commands[0]['seconds'], 5)
            finally:
                if pidfile.exists() and alive(int(pidfile.read_text())):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)

    def test_outer_launcher_kills_command_and_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root/'pids.json'
            child = ("import subprocess,sys,time,os,json; from pathlib import Path; "
                     "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                     f"Path({str(pidfile)!r}).write_text(json.dumps([os.getpid(),p.pid])); time.sleep(60)")
            code = (f"import sys; sys.path.insert(0,{str(Path(optimizer.__file__).parent)!r}); "
                    "from optimizer import Runner; from pathlib import Path; "
                    f"Runner({str(root)!r},'aes').command([sys.executable,'-c',{child!r}],Path({str(root/'outer.log')!r}))")
            outer = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
            pids = []
            try:
                deadline = time.monotonic()+15
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                pids = json.loads(pidfile.read_text())
                self.assertTrue(all(os.getpgid(pid) == outer.pid for pid in pids))
                os.killpg(outer.pid, signal.SIGKILL)
                outer.wait(timeout=5)
                deadline = time.monotonic()+5
                while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertFalse(any(alive(pid) for pid in pids))
            finally:
                if outer.poll() is None:
                    os.killpg(outer.pid, signal.SIGKILL)
                    outer.wait(timeout=5)
                for pid in pids:
                    if alive(pid): os.kill(pid, signal.SIGKILL)

    def test_analysis_worker_obeys_remaining_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root/'tools/quad_gradient/analyze.py'
            worker.parent.mkdir(parents=True)
            worker.write_text('import time\ntime.sleep(60)\n')
            runner = physical.Runner(root, 'aes')
            runner.deadline = time.monotonic()+.2
            start = time.monotonic()
            with patch_roots(root), self.assertRaises(subprocess.TimeoutExpired):
                runner.analyze(root, root, (-1., -.1))
            self.assertLess(time.monotonic()-start, 5)
            self.assertFalse((root/'scores.json').exists())
            self.assertIsNone(runner.analysis_worker)

    def test_persistent_analysis_worker_reuses_library_refreshes_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libdir = root/'lib/NLDM'
            libdir.parent.mkdir()
            library(libdir)
            worker = root/'tools/quad_gradient/analyze.py'
            worker.parent.mkdir(parents=True)
            # Exercise the actual worker loop and numerical analysis across the
            # subprocess seam; only Liberty's LUT oracle/action enumeration are fixtures.
            worker.write_text(f'''
import sys
sys.path.insert(0, {str(ROOT/'tools/test')!r})
import check_algorithm as fixture
import model, analyze
from pathlib import Path
class CountingLibrary(fixture.ToyLibrary):
    def __init__(self, folder):
        super().__init__(folder)
        self.verify_folder = lambda *a: (_ for _ in ()).throw(AssertionError('per-request validation'))
        with Path({str(root/'loads')!r}).open('a') as out: out.write('load\\n')
analyze.Library = CountingLibrary
original_catalog_load = analyze.ActionCatalog.load
class CountingCatalog:
    @staticmethod
    def load(*args):
        with Path({str(root/'catalog-loads')!r}).open('a') as out: out.write('load\\n')
        return original_catalog_load(*args)
analyze.ActionCatalog = CountingCatalog
analyze.enumerate_actions = lambda *a: [dict(instance='gate', current='current',
    direction='vt_down', valid=True, target='candidate')]
sys.exit(analyze.main())
''')
            runner = physical.Runner(root, 'aes')
            runner.deadline = time.monotonic()+30
            process = None
            try:
                results = []
                for i, slew in enumerate((1., 1.5)):
                    graph = root/f'graph-{i}'
                    snapshot(graph, [(0, pin, 'rise', n, slew if pin == 'gate/A' else 1., -100., 'ff/D')
                                     for n, pin in enumerate(('launch/Q', 'gate/A', 'gate/Y', 'ff/D'))],
                             [('load', 'launch/CLK', 'launch/Q', 'Reg Clk to Q', 3.),
                              ('load', 'gate/A', 'gate/Y', 'combinational', 11.),
                              ('setup', 'ff/CLK', 'ff/D', 'setup', .5)],
                             [('launch', 'Q', 'OUTPUT', 'net'), ('gate', 'A', 'INPUT', 'net')], slew=slew)
                    with patch_roots(root):
                        try:
                            scores, summary = runner.analyze(graph, root, REFERENCE)
                        except RuntimeError:
                            self.fail((root/'analysis-worker.log').read_text())
                    results.append(scores[0]['benefit'])
                    self.assertEqual(summary['graph_nodes'], 5)
                    if process is None: process = runner.analysis_worker
                    self.assertIs(runner.analysis_worker, process)
                    self.assertEqual(os.getpgid(process.pid), os.getpgrp())
                self.assertNotEqual(*results)
                self.assertEqual((root/'loads').read_text(), 'load\n')
                self.assertEqual((root/'catalog-loads').read_text(), 'load\n')
                self.assertIsNone(process.poll())
                runner.close()
                self.assertIsNotNone(process.poll())
                self.assertIsNone(runner.analysis_worker)
                second = root/'second-run'
                second.mkdir()
                runner = physical.Runner(second, 'aes')
                runner.deadline = time.monotonic()+30
                broken = root/'broken'
                broken.mkdir()
                with patch_roots(root), self.assertRaisesRegex(RuntimeError, 'Analysis failed'):
                    runner.analyze(broken, root, REFERENCE)
                self.assertIsNone(runner.analysis_worker)
            finally:
                runner.close()
            self.assertIsNotNone(process.poll())
            self.assertIsNone(runner.analysis_worker)

    def test_analysis_worker_failure_and_cancel_kill_descendants(self):
        for mode in ('timeout', 'error', 'cancel'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                worker = root/'tools/quad_gradient/analyze.py'
                worker.parent.mkdir(parents=True)
                pidfile = root/'child.pid'
                worker.write_text(f'''
import subprocess, sys, time, json
from pathlib import Path
sys.stdin.readline()
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path({str(pidfile)!r}).write_text(str(p.pid))
if {mode!r} != 'timeout': print(json.dumps({{'error': 'injected failure'}}), flush=True)
time.sleep(60)
''')
                runner = physical.Runner(root, 'aes')
                runner.deadline = time.monotonic()+2
                real_select = physical.select.select
                def wait(*args):
                    ready = real_select(*args)
                    if mode == 'cancel': raise KeyboardInterrupt()
                    return ready
                expected = {'timeout': subprocess.TimeoutExpired, 'error': RuntimeError,
                            'cancel': KeyboardInterrupt}[mode]
                try:
                    with patch_roots(root), patch.object(physical.select, 'select', wait):
                        with self.assertRaises(expected):
                            runner.analyze(root, root, REFERENCE)
                    self.assertIsNone(runner.analysis_worker)
                    self.assertIsNone(runner.analysis_log)
                    self.assertFalse(alive(int(pidfile.read_text())))
                    with self.assertRaisesRegex(RuntimeError, 'start a new optimization run'):
                        runner.analyze(root, root, REFERENCE)
                finally:
                    runner.close()
                    if pidfile.exists() and alive(int(pidfile.read_text())):
                        os.kill(int(pidfile.read_text()), signal.SIGKILL)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
