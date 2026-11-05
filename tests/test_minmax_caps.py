from pathlib import Path
import tempfile
import unittest
from minmax_caps import read_ranges


class CapacitanceRanges(unittest.TestCase):
    def test_ranges_and_nominal_fallback_are_distinct_for_early_and_late(self):
        text='''library (L) {
 capacitive_load_unit (1,ff);
 cell (G) {
  pin (A) {direction : input;capacitance : 5;
   rise_capacitance : 9;rise_capacitance_range (3,9);
   fall_capacitance : 8;fall_capacitance_range (4,8);}
  pin (B) {direction : input;capacitance : 2;}
  pin (Y) {direction : output;}
 }
}'''
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.lib';path.write_text(text);result=read_ranges(path)['G']
        self.assertEqual(result['A'],{'min':[3.,4.],'max':[9.,8.]})
        self.assertEqual(result['B'],{'min':[2.,2.],'max':[2.,2.]})
        self.assertEqual(result['Y'],{'min':[0.,0.],'max':[0.,0.]})


if __name__=='__main__':unittest.main()
