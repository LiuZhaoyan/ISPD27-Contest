import unittest
import tempfile
from pathlib import Path
import numpy as np
from rc_tree import transfer_resistance,spef_networks

class RCTree(unittest.TestCase):
    def test_shared_path_resistance(self):
        r,kind=transfer_resistance('d',['a','b'],[('d','x',2.),('x','a',3.),('x','b',7.)])
        np.testing.assert_allclose(r,[[5,2],[2,9]]);self.assertEqual(kind,'tree')
        # Increasing b's cap changes a by 2*C, not by a's total resistance 5.
        np.testing.assert_allclose(r@np.array([0.,4.]),[8,36])
    def test_mesh_matches_independent_network_equations(self):
        r,kind=transfer_resistance('d',['a','b'],[('d','a',1.),('d','b',1.),('a','b',1.)])
        np.testing.assert_allclose(r,np.linalg.inv([[2.,-1.],[-1.,2.]]));self.assertEqual(kind,'mesh')
    def test_ideal_connections_are_contracted(self):
        r,_=transfer_resistance('d',['a','b'],[('d','x',0.),('x','a',2.),('a','b',0.)])
        np.testing.assert_allclose(r,np.full((2,2),2.))
    def test_disconnected_sink_rejected(self):
        with self.assertRaises(ValueError):transfer_resistance('d',['b'],[('d','a',1.)])
    def test_equivalent_spef_units_have_equal_physical_rc(self):
        # One explicit physical fixture: 3 fF and 2 kohm, encoded two ways.
        body='*D_NET n 3\n*CONN\n*P d O\n*I g:A I\n*CAP\n1 g:A {cap}\n*RES\n1 d g:A {res}\n*END\n'
        with tempfile.TemporaryDirectory() as directory:
            results=[]
            for i,(cu,ru,cap,res) in enumerate([('1 FF','1 KOHM',3,2),('1 PF','1 OHM',.003,2000)]):
                p=Path(directory)/str(i)
                p.write_text('*C_UNIT '+cu+'\n*R_UNIT '+ru+'\n'+body.format(cap=cap,res=res))
                results.append(list(spef_networks(p)))
            self.assertEqual(results[0],results[1])
            self.assertEqual(results[0][0][3]['g:A'],3.)
    def test_spef_escape_decoding_preserves_rc_node_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'network.spef'
            p.write_text('*C_UNIT 1 FF\n*R_UNIT 1 KOHM\n*D_NET n\\[0\\] 1\n*CONN\n*P d O\n*I g\\[0\\]\\$:A I\n*CAP\n1 g\\[0\\]\\$:A 1\n*RES\n1 d g\\[0\\]\\$:A 2\n*END\n')
            net,conn,res,wire=next(spef_networks(p))
            self.assertEqual(net,'n[0]');self.assertEqual(conn['g\\[0\\]\\$:A'],'g[0]$/A')
            resistance,_=transfer_resistance('d',['g\\[0\\]\\$:A'],res)
            np.testing.assert_allclose(resistance,[[2.]])
if __name__=='__main__':unittest.main()
