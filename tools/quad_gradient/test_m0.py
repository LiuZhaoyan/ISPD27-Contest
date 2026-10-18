"""Behavior tests for the read-only M0 evidence checker."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("m0", Path(__file__).with_name("m0.py"))
m0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m0)


class EvidenceTests(unittest.TestCase):
    def test_endpoint_states_are_reduced_once_per_scene_and_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with (folder / "endpoints.tsv").open("w") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["scene", "analysis", "endpoint", "slack_ps"])
                for scene, slack in [("TC", -4), ("BC", -2), ("WC", -8)]:
                    for analysis in ("max", "min"):
                        for endpoint, value in [("a/D", slack), ("a/D", -1), ("b/D", -3), ("c/D", 2)]:
                            w.writerow([scene, analysis, endpoint, value])
            with (folder / "native.tsv").open("w") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["scene", "analysis", "wns_ps", "tns_ps"])
                for scene, wns, tns in [("TC", -4, -7), ("BC", -3, -5), ("WC", -8, -11)]:
                    for analysis in ("max", "min"):
                        w.writerow([scene, analysis, wns, tns])
            report = m0.check_aggregates(folder)
            self.assertTrue(report["passed"])
            self.assertEqual(report["groups"]["TC/max"]["endpoint_count"], 3)
            # Missing scene coverage must never be reported as success.
            lines = (folder / "native.tsv").read_text().splitlines()
            (folder / "native.tsv").write_text("\n".join(lines[:-1]) + "\n")
            self.assertFalse(m0.check_aggregates(folder)["passed"])

    def test_observations_detect_wrong_units_and_master_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "masters.tsv").write_text("instance\tsta_master\todb_master\nu1\tBUF\tBUF\n")
            (folder / "units.rpt").write_text(" time 1ps\n capacitance 1fF\n distance 1um\n power 1pW\n")
            header = "scene\tanalysis\tpin\ttransition\tarrival_ps\tslew_ps\ttotal_cap_F\tpin_cap_F\twire_cap_F\n"
            points = []
            for scene in ("BC", "TC", "WC"):
                for analysis in ("max", "min"):
                    points.append(f"{scene}\t{analysis}\tu1/Y\trise\t12\t3\t2e-15\t1.5e-15\t0.5e-15\n")
                    (folder / f"{scene}_{analysis}_path.rpt").write_text(
                        " 1 2.000000 3.000000 4.000000 12.000000 ^ u1/Y (BUF)\n")
            (folder / "points.tsv").write_text(header + "".join(points))
            self.assertTrue(m0.check_observations(folder)["passed"])
            (folder / "masters.tsv").write_text("instance\tsta_master\todb_master\nu1\tBUF\tINV\n")
            self.assertFalse(m0.check_observations(folder)["passed"])
            (folder / "masters.tsv").write_text("instance\tsta_master\todb_master\nu1\tBUF\tBUF\n")
            (folder / "units.rpt").write_text(" time 1ns\n capacitance 1fF\n distance 1um\n")
            self.assertFalse(m0.check_observations(folder)["passed"])
            (folder / "units.rpt").write_text(" time 1ps\n capacitance 1fF\n distance 1um\n power 1pW\n")
            (folder / "BC_min_path.rpt").write_text("unparseable native report\n")
            report = m0.check_observations(folder)
            self.assertFalse(report["passed"])
            self.assertFalse(report["all_groups_compared"])
            self.assertEqual(len(report["unmatched_points"]), 1)

    def test_new_and_deleted_protected_files_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first = folder / "input.def"
            first.write_text("baseline")
            before = m0.input_fingerprint([folder])
            extra = folder / "new.sdc"
            extra.write_text("extra")
            self.assertNotEqual(before, m0.input_fingerprint([folder]))
            extra.unlink()
            self.assertEqual(before, m0.input_fingerprint([folder]))
            first.unlink()
            self.assertNotEqual(before, m0.input_fingerprint([folder]))

    def test_output_cannot_be_nested_inside_baseline(self):
        baseline = m0.ROOT / ".runs" / "baseline"
        with self.assertRaisesRegex(ValueError, "protected baseline"):
            m0.validate_output(baseline, baseline / "probe")
        with self.assertRaisesRegex(ValueError, "protected baseline"):
            m0.validate_output(baseline, baseline)
        m0.validate_output(baseline, m0.ROOT / ".runs" / "probe")


class ClosureTests(unittest.TestCase):
    def test_native_lookup_is_independently_reconstructed_and_mismatch_is_visible(self):
        text = '''------- constrained_pin_transition = 5.000000000
|       related_pin_transition = 10.000000000
|        0.000000000     20.000000000
v      --------------------
0.000000000 | 2.000000000     4.000000000
10.000000000 | 4.000000000     6.000000000
Table value = 4.000000000
PVT scale factor = 1.000000000
Check = 4.000000000
'''
        table = dict(axes=['constrained_pin_transition', 'related_pin_transition'],
                     index_1=[0, 10], index_2=[0, 20], values=[[2, 4], [4, 6]])
        result = m0.compare_lookup(text, table, 'Check', 0.05)
        self.assertTrue(result['passed'])
        self.assertEqual(result['reconstructed_ps'], 4)
        table['values'][1][1] = 8
        self.assertFalse(m0.compare_lookup(text, table, 'Check', 0.05)['passed'])

    def test_missing_native_relations_and_samples_cannot_pass_interface_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ('samples', 'endpoints', 'states', 'relations', 'boundary', 'load_points', 'net_slews'):
                (folder / ('closure_' + name + '.tsv')).write_text('placeholder\n')
            (folder / 'closure_interfaces.txt').write_text('No compatibility oracle\n')
            plan = json.loads(Path(__file__).with_name('closure_plan.json').read_text())
            result = m0.evaluate_closure(folder, plan)
            self.assertFalse(result['passed'])
            self.assertEqual([False, False, False], [c['passed'] for c in result['checks'].values()])

    def test_preregistration_retains_exact_samples_and_rejects_unidentified_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source = Path(__file__).with_name('closure_plan.json')
            plan, digest, env = m0.prepare_plan(source, Path('quad-m0-baseline-20261018-01'), output)
            self.assertEqual((output / 'closure_plan.json').read_bytes(), source.read_bytes())
            self.assertEqual(env['QG_CLOSURE_ENDPOINT_0'], 'sa23_sr[5]$_DFF_P_/D')
            self.assertEqual(plan['tolerance_ps'], 0.05)
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, 'baseline identity'):
                m0.prepare_plan(source, Path('different-baseline'), output)

    def test_pi_and_connected_caps_remain_distinct_model_inputs(self):
        # Net connectedCap reduces transitions; DMP's per-transition Pi may differ.
        result = m0.cap_correspondence([5.0, 4.0, 1.0], [0.1, 4.7, 4.5], [4.5, 4.5], 0.00001)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['pi_minus_connected_fF'], -0.2)
        self.assertAlmostEqual(result['effective_minus_connected_fF'], -0.5)
        self.assertFalse(m0.cap_correspondence([5, 3, 1], [0.1, 4.7, 4.5], [4.5], 0.00001)['passed'])
        self.assertFalse(m0.cap_correspondence([5, 4, 1], [0.1, 4.7, 4.5], [5.0], 0.00001)['passed'])

    def test_duplicate_setup_sample_cannot_replace_missing_transition(self):
        plan = dict(endpoints=['ff/D'], transitions=['rise', 'fall'])
        endpoints = [dict(endpoint='ff/D', data_transition=rf) for rf in plan['transitions']]
        samples = [dict(kind='setup', to_pin='ff/D', to_transition=rf) for rf in plan['transitions']]
        self.assertTrue(m0.setup_coverage(endpoints, samples, plan)['passed'])
        self.assertFalse(m0.setup_coverage(endpoints, [samples[0], samples[0]], plan)['passed'])
        self.assertFalse(m0.setup_coverage([endpoints[0], endpoints[0]], samples, plan)['passed'])

    def test_report_cap_requires_selected_driver_transition_and_data_branch(self):
        report = '''  2 4.250000000 20.000000000 3.000000000 15.000000000 ^ u/Y (X)
 data arrival time
  2 8.000000000 20.000000000 3.000000000 15.000000000 ^ u/Y (X)
'''
        self.assertEqual(m0.native_report_cap(report, 'u/Y', 'rise'), 4.25)
        with self.assertRaisesRegex(ValueError, 'report Cap'):
            m0.native_report_cap(report, 'u/Y', 'fall')
        with self.assertRaisesRegex(ValueError, 'report Cap'):
            m0.native_report_cap(report, 'other/Y', 'rise')


if __name__ == "__main__":
    unittest.main()
