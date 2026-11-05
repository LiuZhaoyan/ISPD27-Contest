import tempfile
from pathlib import Path
import unittest
import numpy as np
from placement_rc import Pin,Layer,Technology,PlacementRC,Network,Flute,DemandGrid
from rc_tree import transfer_resistance,spef_networks


class Manhattan:
    def edges(self, points):
        points=sorted(set(points))
        return list(zip(points,points[1:]))


def technology():
    return Technology([Layer(1,'V',0.,0.,.1),Layer(2,'H',2.,3.,.1),
                       Layer(3,'V',4.,5.,.1)],{1:.2,2:.4},signal=(2,3),clock=(2,3))


class PlacementRCTest(unittest.TestCase):
    def test_native_capacity_adjustment_is_removed_track_fraction(self):
        grid=DemandGrid(technology(),tile=2.,adjustment=.25)
        # Twenty physical tracks retain fifteen after a 25% reduction.
        self.assertAlmostEqual(grid.capacity(2),15.)
        self.assertEqual(DemandGrid(technology(),adjustment=0.).capacity(2),10.)
        self.assertEqual(DemandGrid(technology(),adjustment=-.5).capacity(2),15.)
        with self.assertRaises(ValueError):DemandGrid(technology(),adjustment=1.)

    def test_pin_branch_reuse_refreshes_after_nine_updates(self):
        class Counted(Manhattan):
            def __init__(self):self.calls=0
            def edges(self,points):self.calls+=1;return super().edges(points)
        flute=Counted();est=PlacementRC([Pin('d','n',0.,0.,'INPUT',port=True),
            Pin('a/A','n',2.,1.,'INPUT')],technology(),flute,congestion=False)
        for i in range(1,10):
            update=est.update({'a/A':(2.+i*.01,1.)})
            self.assertEqual(update['reused_topology_nets'],1)
            self.assertAlmostEqual(est.networks['n'].wirelength,3.+i*.01)
        self.assertEqual(flute.calls,1)
        self.assertEqual(est.update({'a/A':(2.1,1.)})['reused_topology_nets'],0)
        self.assertEqual(flute.calls,2)

    def test_pin_coordinate_rank_or_tie_change_forces_reconstruction(self):
        class Counted(Manhattan):
            def __init__(self):self.calls=0
            def edges(self,points):self.calls+=1;return super().edges(points)
        flute=Counted();est=PlacementRC([Pin('d','n',0.,0.,'INPUT',port=True),
            Pin('a/A','n',2.,1.,'INPUT'),Pin('b/A','n',2.,2.,'INPUT')],technology(),flute,congestion=False)
        self.assertEqual(est.update({'a/A':(2.1,1.)})['reused_topology_nets'],0)
        self.assertEqual(flute.calls,2)
        self.assertEqual(est.update({'a/A':(-1.,1.)})['reused_topology_nets'],0)
        self.assertEqual(flute.calls,3)

    def test_first_moment_against_conductance_equations(self):
        n=Network('n',(),('d','x','a','b'),((0,1,2.),(1,2,3.),(1,3,7.)),
                  np.array([1.,2.,3.,4.]),{'d':0,'a':2,'b':3},'d',0,0,{})
        moments,load=n.moments({'a':[5.,6.],'b':[7.,8.]})
        # Solve G*tau=C directly, rather than duplicating tree traversal.
        g=np.array([[1/2+1/3+1/7,-1/3,-1/7],[-1/3,1/3,0],[-1/7,0,1/7]])
        expected=np.linalg.solve(g,np.array([[2,2],[8,9],[11,12]]))
        np.testing.assert_allclose(moments['a'],expected[1]);np.testing.assert_allclose(moments['b'],expected[2])
        np.testing.assert_allclose(load,[22,24])

    def test_pin_capacitance_sensitivity_exact(self):
        est=PlacementRC([Pin('d','n',0.,0.,'INPUT',port=True),
                         Pin('u/A','n',2.,1.,'INPUT')],technology(),Manhattan(),congestion=False)
        n=est.networks['n'];first,_=n.moments({'u/A':[1,2]});second,_=n.moments({'u/A':[4,6]})
        r,_=transfer_resistance(n.pin_nodes['d'],[n.pin_nodes['u/A']],n.resistors)
        np.testing.assert_allclose(second['u/A']-first['u/A'],r[0,0]*np.array([3,4]))

    def test_geometry_move_changes_rc_and_roundtrips_spef(self):
        est=PlacementRC([Pin('d','n',0.,0.,'INPUT',port=True),
                         Pin('u/A','n',2.,1.,'INPUT')],technology(),Manhattan(),congestion=False)
        before=est.networks['n'].wirelength
        record=est.update({'u/A':(3,2)})
        self.assertEqual(record['direct'],['n']);self.assertGreater(est.networks['n'].wirelength,before)
        self.assertEqual(est.update({'u/A':(3,2)})['direct'],[])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'n.spef';est.write_spef(path)
            name,conn,res,wire=next(spef_networks(path))
            self.assertEqual(name,'n');self.assertEqual(set(conn.values()),{'d','u/A'})
            self.assertAlmostEqual(sum(wire.values()),est.networks['n'].wire_caps.sum())
            inv={v:k for k,v in conn.items()}
            matrix,_=transfer_resistance(inv['d'],[inv['u/A']],res)
            self.assertGreater(matrix[0,0],0)

    def test_fixed_clock_pin_rejects_move_without_mutation(self):
        est=PlacementRC([Pin('clk','n',0.,0.,'INPUT',clock=True,port=True),
                         Pin('u/CLK','n',2.,1.,'INPUT',clock=True)],technology(),Manhattan())
        with self.assertRaises(ValueError):est.update({'u/CLK':(4,2)})
        self.assertEqual(est.pins['u/CLK'].x,2.)

    def test_shared_wire_capacitance_is_counted_once(self):
        class Overlap:
            def edges(self,points):return [((0.,0.),(2.,0.)),((0.,0.),(3.,0.))]
        est=PlacementRC([Pin('d','n',0.,0.,'INPUT',port=True),
                         Pin('a/A','n',2.,0.,'INPUT'),Pin('b/A','n',3.,0.,'INPUT')],
                        technology(),Overlap(),congestion=False)
        n=est.networks['n'];self.assertEqual(n.wirelength,3.)
        self.assertEqual(n.wire_caps.sum(),9.)
        self.assertEqual(n.via_count,3)
        n.moments({'a/A':[1,1],'b/A':[2,2]})

    def test_neighbor_clock_is_not_invalidated_when_only_uncongested_extension_changes(self):
        est=PlacementRC([Pin('clk','c',0.,0.,'INPUT',clock=True,port=True),
                         Pin('ff/CLK','c',2.,0.,'INPUT',clock=True),
                         Pin('in','s',0.,0.,'INPUT',port=True),Pin('u/A','s',2.,0.,'INPUT')],
                        technology(),Manhattan(),tile=1.)
        record=est.update({'u/A':(3.,0.)})
        self.assertNotIn('c',record['congestion_induced'])
        self.assertEqual(est.pins['ff/CLK'].x,2.)

    def test_fixed_clock_is_invalidated_when_a_shared_bottleneck_is_relieved(self):
        est=PlacementRC([Pin('clk','c',0.,0.,'INPUT',clock=True,port=True),
                         Pin('ff/CLK','c',2.,0.,'INPUT',clock=True),
                         Pin('in','s',0.,0.,'INPUT',port=True),Pin('u/A','s',2.,0.,'INPUT')],
                        technology(),Manhattan(),tile=1.,adjustment=.9)
        record=est.update({'u/A':(2.,1.)})
        self.assertIn('c',record['congestion_induced'])
        self.assertGreater(record['changed_congestion_bins'],0)

    def test_real_flute_handles_duplicate_and_high_fanout_coordinates(self):
        flute=Flute()
        points=np.array([[0,0],[1,2],[3,1],[1,2]],dtype=float)
        edges=flute.edges(points)
        self.assertTrue(edges)
        terminals={tuple(p) for p in points}
        self.assertTrue(terminals.issubset({v for e in edges for v in e}))
        high=flute.edges(np.column_stack((np.arange(1001),np.zeros(1001))))
        self.assertTrue({(float(i),0.) for i in range(1001)}.issubset({v for e in high for v in e}))

    def test_two_terminal_fast_path_matches_native_flute(self):
        flute=Flute()
        for points in [[(1.,2.),(3.,4.)],[(3.,1.),(1.,4.)],
                       [(1.,2.),(1.,2.)],[(.0005,.0015),(.0015,.0025)],
                       [(-.1005,.9005),(.1005,-.9005)]]:
            self.assertEqual(flute.edges(points),flute.edges(np.asarray(points)))


if __name__=='__main__':unittest.main()
