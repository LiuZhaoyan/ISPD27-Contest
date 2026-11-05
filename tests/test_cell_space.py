"""Checks for invalid moves, semantic mismatches and file-format boundaries."""
import tempfile
import unittest
from pathlib import Path
from cell_space import boolean_key, nominal_state, action_rows, def_components, netlist_instances, entries


def cell(i, k, t, frozen=None):
    return dict(id=i, name=f'c{i}', family_id=0, official_group=0, kappa=k, theta=t,
                frozen_reason=frozen, liberty_data={'behavior_sha256': 'same'},
                physical={'width_um': k + 1, 'height_um': 1, 'site': 'core'})


class CellSpaceTests(unittest.TestCase):
    def test_vendor_missing_area_semicolon(self):
        self.assertEqual(list(entries('area : 0.69984\n clock_gating_integrated_cell : latch_posedge_precontrol;')),
                         [('area', '0.69984'), ('clock_gating_integrated_cell', 'latch_posedge_precontrol')])

    def test_boolean_equivalence_and_non_equivalence(self):
        self.assertEqual(boolean_key('!(A * B)'), boolean_key("A' + !B"))
        self.assertNotEqual(boolean_key('A * B'), boolean_key('A + B'))
        self.assertEqual(boolean_key('(A ^ B) ^ C'), boolean_key('A ^ (B ^ C)'))

    def test_unknown_expression_rejected(self):
        with self.assertRaises(ValueError):
            boolean_key('A ? B : C')

    def test_fractional_strength_and_variant(self):
        self.assertEqual(nominal_state('NAND2xp33_ASAP7_75t_SL')['nominal_strength'], .33)
        self.assertEqual(nominal_state('ICGx2p67DC_ASAP7_75t_R')['nominal_strength'], 2.67)
        self.assertNotEqual(nominal_state('BUFx12_ASAP7_75t_L')['series'], nominal_state('BUFx12f_ASAP7_75t_L')['series'])
        self.assertEqual(nominal_state('INVx1_ASAP7_75t_R')['theta'], 2)

    def test_sparse_grid_available_neighbors_and_boundary(self):
        cells = [cell(0, 0, 0), cell(1, 1, 0), cell(2, 2, 0), cell(3, 0, 2)]
        rows = action_rows(cells, [{'cell_ids': list(range(4))}], available={0, 2, 3})
        self.assertEqual([r['target_id'] for r in rows[:4]], [2, -1, 3, -1])
        self.assertEqual(rows[0]['delta_kappa'], 2)
        self.assertTrue(rows[0]['requires_legalization'])
        self.assertTrue(all(not r['valid'] for r in rows[4:8]))

    def test_dont_use_is_neither_source_nor_target(self):
        cells = [cell(0, 0, 0), cell(1, 1, 0, 'dont_use')]
        self.assertTrue(all(not r['valid'] for r in action_rows(cells, [{'cell_ids': [0, 1]}])))

    def test_behavior_mismatch_rejected(self):
        cells = [cell(0, 0, 0), cell(1, 1, 0)]
        cells[1]['liberty_data']['behavior_sha256'] = 'different'
        with self.assertRaises(AssertionError):
            action_rows(cells, [{'cell_ids': [0, 1]}])

    def test_def_multiline_fixed_and_escaped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.def'
            p.write_text('COMPONENTS 1 ;\n - a\\[0\\] INV\n + FIXED ( 10 20 ) N ;\nEND COMPONENTS\n')
            self.assertEqual(list(def_components(p)), [('a[0]', 'INV', 'FIXED', 10, 20, 'N')])
            p.write_text('COMPONENTS 2 ;\nEND COMPONENTS\n')
            with self.assertRaises(ValueError):
                list(def_components(p))

    def test_netlist_escaped_name_and_unknown_master(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.v'
            p.write_text('module top (a,y);\n INV \\u/a[0] (.A(a),\n .Y(y));\nendmodule\n')
            self.assertEqual(list(netlist_instances(p, {'INV'})), [('u/a[0]', 'INV', ['A', 'Y'])])
            with self.assertRaises(ValueError):
                list(netlist_instances(p, {'OTHER'}))
            p.write_text('TAP physical_only ();\n')
            self.assertEqual(list(netlist_instances(p, {'TAP'})), [('physical_only', 'TAP', [])])


if __name__ == '__main__':
    unittest.main()
