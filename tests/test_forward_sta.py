"""Analytical tests independent of the real-design calibration data."""
import unittest
import numpy as np
from forward_sta import Table, ForwardModel, condition


class ForwardTests(unittest.TestCase):
    def test_bilinear_and_extrapolation(self):
        x=np.array([2.,4.,9.]);y=np.array([1.,5.,10.])
        f=lambda a,b:3+2*a+5*b+0.7*a*b
        t=Table(x,y,f(x[:,None],y[None,:]))
        for a,b in [(2,1),(3,7),(0,0),(20,30),(9,10)]: self.assertAlmostEqual(t(a,b),f(a,b))

    def test_bad_axis_rejected(self):
        with self.assertRaises(ValueError):Table([1,1],[1,2],np.ones((2,2)))

    def test_condition_equivalence(self):
        self.assertEqual(condition('!A * B'),condition('B & ~A'))
        self.assertNotEqual(condition('A + B'),condition('A * B'))

    def toy(self):
        # Two inputs reconverge: one inversion swaps rise/fall; the other
        # positive arc dominates rise. Both terminal transitions count once.
        m=ForwardModel.__new__(ForwardModel)
        m.n=3;m.masters={};m.cells={};m.base_cap=np.zeros(3)
        m.pin_parts=[(None,str(i)) for i in range(3)]
        m.ref_at=np.array([[10.,20.],[15.,7.],[0.,0.]])
        m.ref_slew=np.ones((3,2));m.order=np.arange(3)
        m.arcs=[dict(s=0,t=2,fr=1,to=0,delay=2.,role='combinational'),dict(s=0,t=2,fr=0,to=1,delay=4.,role='combinational'),dict(s=1,t=2,fr=0,to=0,delay=10.,role='combinational'),dict(s=1,t=2,fr=1,to=1,delay=2.,role='combinational')]
        m.incoming=[[],[],list(range(4))];m.checks=[];m.check_incoming=[[],[],[]]
        m.index={'out':2};m.endpoints=[dict(pin='out',rf=rf,required_ps='12',role='output setup') for rf in ('rise','fall')]
        return m

    def test_reconvergence_inversion_and_endpoint_counting(self):
        r=self.toy().run(mode='oracle')
        np.testing.assert_equal(r['at'][2],[25,14])
        self.assertEqual(r['tns_ps'],-13);self.assertEqual(r['wns_ps'],-13)
        np.testing.assert_equal(r['predecessor'][2],[2,1])

    def test_no_arrival_is_not_zero(self):
        m=self.toy();m.ref_at[:2]=-np.inf
        r=m.run(mode='oracle');self.assertTrue(np.isneginf(r['at'][2]).all())
        self.assertEqual(r['tns_ps'],0)

    def test_slew_dominance_is_independent_of_arrival(self):
        m=self.toy()
        m.arc_value=lambda a,s,c,masters:(a['delay'],99. if a['s']==0 else 2.)
        r=m.run(mode='raw')
        np.testing.assert_equal(r['at'][2],[25,14])
        np.testing.assert_equal(r['slew'][2],[99,99])

    def test_frozen_correction_responds_to_new_arc_values(self):
        m=self.toy();m.delay_correction=np.ones(4);m.slew_correction=np.zeros((3,2))
        m.arc_value=lambda a,s,c,masters:(a['delay']*2,1.)
        r=m.run()
        np.testing.assert_equal(r['at'][2],[36,19])

    def test_load_delta_updates_all_sinks_and_driver(self):
        m=self.toy();m.masters={'g':'small'}
        m.pin_parts=[('g','A'),(None,'other'),(None,'driver')]
        m.pins=[dict(net='n')]*3;m.by_net={'n':[0,1,2]};m.base_cap=np.full(3,5.)
        m.cells={name:dict(liberty_data=dict(pins={'A':{'capacitance_ff':cap}})) for name,cap in [('small',1.),('large',3.)]}
        np.testing.assert_equal(m.caps({'g':'large'}),[7,7,7])

    def test_setup_uses_data_and_clock_slew(self):
        m=self.toy();m.tables={('FF','CLK','D','setup','null'):{'rise_constraint':Table([0,10],[0,10],[[0,20],[10,30]])}}
        m.pin_parts=[('ff','CLK'),(None,'unused'),('ff','D')]
        a=dict(s=0,t=2,fr=0,to=0,role='setup',cond='null')
        s=np.array([[3,0],[0,0],[7,0]])
        self.assertAlmostEqual(m.check_value(a,s,{'ff':'FF'}),13)


if __name__=='__main__':unittest.main()
