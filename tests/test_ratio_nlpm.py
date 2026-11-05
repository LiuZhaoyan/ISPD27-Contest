import unittest
import numpy as np
from forward_sta import Table
from ratio_sta import RatioForwardModel,RatioBackwardModel,fit_ratio
from nlpm import PowerModel,probability,sensitivity
from cell_space import boolean_key

class Ratios(unittest.TestCase):
    def test_zero_signed_and_nonfinite_calibration(self):
        p=np.array([2.,0.,-3.,-2.,np.inf]);r=np.array([6.,4.,-6.,2.,np.inf])
        scale,offset,fallback=fit_ratio(p,r)
        np.testing.assert_allclose(p[:4]*scale[:4]+offset[:4],r[:4])
        np.testing.assert_equal(fallback,[False,True,False,True,False])

    def test_chain_rule_scales_delay_and_slew(self):
        # Two nonlinear stages, two RF endpoints counted once. Perturb complete
        # LUT values independently; the finite differences bypass the VJP code.
        m=RatioForwardModel.__new__(RatioForwardModel);m.n=3;m.masters={};m.cells={};m.base_cap=np.array([0.,2.,3.]);m.load_ratio=np.ones(3)
        m.pins=[dict(net=str(i)) for i in range(3)];m.pin_parts=[(None,str(i)) for i in range(3)];m.by_net={str(i):[i] for i in range(3)}
        m.arcs=[dict(s=s,t=s+1,fr=rf,to=rf,role='combinational',delay=0.) for s in (0,1) for rf in (0,1)]
        m.incoming=[[],[0,1],[2,3]];m.outgoing=[[0,1],[2,3],[]];m.checks=[];m.check_incoming=[[],[],[]];m.order=np.arange(3)
        m.ref_at=np.array([[2.,1.],[0.,0.],[0.,0.]]);m.ref_slew=np.ones((3,2));m.index={'out':2};m.endpoints=[dict(pin='out',rf=rf,role='output setup',required_ps='0') for rf in ('rise','fall')]
        m.at_ratio=np.array([[1,1],[.7,.7],[1.3,1.3]]);m.at_offset=np.zeros((3,2));m.slew_ratio=np.array([[1,1],[.4,.4],[1.2,1.2]]);m.slew_offset=np.zeros((3,2));m.check_ratio=np.empty(0);m.check_offset=np.empty(0)
        x=np.array([0.,100.]);delay=Table(x,x,1+2*x[:,None]+3*x[None,:]);slew=Table(x,x,2+.8*x[:,None]+.2*x[None,:]);tabs={'cell_rise':delay,'cell_fall':delay,'rise_transition':slew,'fall_transition':slew}
        m.table=lambda a,masters:tabs;m.arc_value=lambda a,s,c,masters:(delay(s[a['s'],a['fr']],c[a['t']]),slew(s[a['s'],a['fr']],c[a['t']]))
        r=m.run();b=RatioBackwardModel(m,r);g=b.backward('hard');h=1e-4
        for table,key in [(delay,'delay'),(slew,'local_slew')]:
            original=table.values.copy();table.values=original+h;plus=-m.run()['tns_ps'];table.values=original-h;minus=-m.run()['tns_ps'];table.values=original
            self.assertAlmostEqual((plus-minus)/(2*h),g[key].sum(),places=7)
        cap=m.base_cap.copy();perturb=np.array([0.,h,h]);m.base_cap=cap+perturb;plus=-m.run()['tns_ps'];m.base_cap=cap-perturb;minus=-m.run()['tns_ps'];m.base_cap=cap
        self.assertAlmostEqual((plus-minus)/(2*h),g['load'].sum(),places=7)

class PowerProbability(unittest.TestCase):
    def test_leakage_only_uses_conditional_duty_and_supply_groups(self):
        p=PowerModel.__new__(PowerModel);p._leakage={};p.state={'g':{'A':(1e8,.25)}}
        # Deliberately no internal-power/pin metadata: leakage admission must
        # not build unrelated LUT descriptors for every broad-search target.
        p.tables={'cell':dict(default_leakage=5.,leakage=[
            dict(pg='VDD',when=boolean_key('A'),value=10.),
            dict(pg='VDD',when=boolean_key('!A'),value=2.),
            dict(pg='AUX',when=None,value=3.)])}
        self.assertEqual(p.leakage('g','cell'),7.)
        self.assertEqual(p.leakage('g','cell'),7.)
    def test_conditional_probability(self):
        self.assertAlmostEqual(probability(boolean_key('A * !B'),{'A':.2,'B':.3}),.14)
        self.assertAlmostEqual(probability(boolean_key('A + B'),{'A':.2,'B':.3}),.44)
    def test_boolean_derivative(self):
        self.assertAlmostEqual(sensitivity(boolean_key('A * B'),'A',{'A':.2,'B':.3}),.3)
        self.assertAlmostEqual(sensitivity(boolean_key('A ^ B'),'A',{'A':.2,'B':.3}),1.)
    def test_reconvergent_variable_is_not_counted_twice(self):
        self.assertAlmostEqual(probability(boolean_key('(A * B) + (A * !B)'),{'A':.2,'B':.3}),.2)
if __name__=='__main__':unittest.main()
