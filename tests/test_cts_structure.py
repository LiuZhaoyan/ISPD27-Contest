"""Adversarial checks for the CTS gate-pin equivalence certificate."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from verify_cts_structure import verify


class CTSStructureTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.original=[('u','AND'),('v','REG')]
        self.before=[('','in','INPUT','a'),('u','A','INPUT','a'),('u','Y','OUTPUT','b'),('v','D','INPUT','b')]
        self.library={'BUF':{'A':dict(direction='input',buffer='1',inverter='0'),
                             'Y':dict(direction='output',buffer='1',inverter='0')},
                      'INV':{'A':dict(direction='input',buffer='0',inverter='1'),
                             'Y':dict(direction='output',buffer='0',inverter='1')}}

    def tearDown(self):self.tmp.cleanup()

    def snapshot(self,label,cells,pins):
        prefix=self.root/label
        with Path(str(prefix)+'.cells.tsv').open('w') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(('instance','master','physical','macro','x','y','orientation','status'))
            w.writerows((n,m,'0','0','0','0','R0','PLACED') for n,m in cells)
        with Path(str(prefix)+'.pins.tsv').open('w') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(('instance','pin','direction','signal','net'))
            w.writerows((i,p,d,'SIGNAL',n) for i,p,d,n in pins)
        return prefix

    def check(self,post_cells,post_pins):
        return verify(self.snapshot('source',self.original,self.before),
                      self.snapshot('post',post_cells,post_pins),self.library)

    def test_identity_buffer_and_dummy_pass(self):
        pins=[('','in','INPUT','a'),('u','A','INPUT','a'),('u','Y','OUTPUT','b'),('v','D','INPUT','c'),
              ('buf','A','INPUT','b'),('buf','Y','OUTPUT','c'),('dummy','A','INPUT','c'),('dummy','Y','OUTPUT','')]
        r=self.check(self.original+[('buf','BUF'),('dummy','INV')],pins)
        self.assertEqual(r['added_identity_buffers'],1)
        self.assertEqual(r['added_unloaded_dummies'],1)

    def test_existing_net_split_rejected(self):
        with self.assertRaisesRegex(ValueError,'split'):
            self.check(self.original,self.before[:-1]+[('v','D','INPUT','c')])

    def test_inserting_buffer_between_distinct_old_nets_rejected(self):
        with self.assertRaisesRegex(ValueError,'shorted'):
            self.check(self.original+[('buf','BUF')],self.before+[('buf','A','INPUT','a'),('buf','Y','OUTPUT','b')])

    def test_inverter_driving_original_logic_rejected(self):
        with self.assertRaisesRegex(ValueError,'Non-identity'):
            self.check(self.original+[('inv','INV')],self.before+[('inv','A','INPUT','a'),('inv','Y','OUTPUT','b')])

    def test_original_master_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'removed or changed'):
            self.check([('u','OR'),('v','REG')],self.before)

    def test_top_port_removal_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing original pins'):
            self.check(self.original,self.before[1:])

if __name__=='__main__':unittest.main()
