import unittest
import numpy as np
from backward_sta import (table_partials,max_weights,endpoint_seeds,bundle_weights,
                          BackwardModel,SIGNS)
from forward_sta import Table
from test_forward_sta import ForwardTests


class BackwardTests(unittest.TestCase):
    def test_table_partials_inside_and_extrapolated(self):
        x=np.array([2.,4.,9.]);y=np.array([1.,5.,10.])
        f=lambda a,b:3+2*a+5*b+.7*a*b
        table=Table(x,y,f(x[:,None],y[None,:]))
        for a,b in [(3,7),(0,0),(20,30),(2,5)]:
            dx,dy=table_partials(table,a,b)
            self.assertAlmostEqual(dx,2+.7*b);self.assertAlmostEqual(dy,5+.7*a)
            h=1e-5
            self.assertAlmostEqual(dx,(table(a+h,b)-table(a-h,b))/(2*h),places=7)

    def test_ave_neighborhood(self):
        np.testing.assert_equal(max_weights([100,99,98],'ave'),[.5,.5,0])
        np.testing.assert_equal(max_weights([-100,-101,-102],'ave'),[.5,.5,0])
        np.testing.assert_equal(max_weights([-np.inf,-np.inf],'ave'),[0,0])

    def test_lse_stability_and_temperature(self):
        a=max_weights([1e8,1e8-10],'lse',gamma_ps=1)
        b=max_weights([1e8,1e8-10],'lse',gamma_ps=100)
        self.assertTrue(np.isfinite(a).all());self.assertAlmostEqual(a.sum(),1)
        self.assertGreater(b[1],a[1]);self.assertGreater(a[1],0)
        with self.assertRaises(ValueError):max_weights([1,2],'lse',gamma_ps=0)

    def test_endpoint_seeds_tns_not_double_counted(self):
        r=dict(endpoints=np.array([[0,0,0,-3],[0,1,0,-3],[1,0,0,-2],[1,1,0,1],[2,0,0,0]]),tns_ps=-5,wns_ps=-3)
        np.testing.assert_equal(endpoint_seeds(r,alpha=2,beta=4),[3,3,2,0,0])
        with self.assertRaises(ValueError):endpoint_seeds(r,alpha=-1)

    def test_direction_signs(self):
        # An actual improvement has delta_loss < 0 in every direction.
        delta=np.array([-3.,-2.,-1.,-4.])
        coordinate_derivative=delta*SIGNS
        paper_oriented_benefit=-SIGNS*coordinate_derivative
        np.testing.assert_equal(paper_oriented_benefit,-delta)
        self.assertTrue(np.all(paper_oriented_benefit>0))

    def test_conditional_arc_multiplicity_does_not_bias_lse(self):
        values=np.array([10.,10.,9.])
        weights=bundle_weights(values,[np.array([0,1]),np.array([2])],'lse')
        self.assertAlmostEqual(weights[0][1].sum(),max_weights([10,9],'lse')[0])
        self.assertAlmostEqual(sum(w.sum() for _,w in weights),1.)

    def test_delay_load_slew_setup_components(self):
        g=dict(delay=np.array([2.]),local_slew=np.array([3.]),setup=np.array([4.]),net_load={'n':5.})
        d=dict(arcs=np.array([0]),delay=np.array([-1.]),slew=np.array([-2.]),checks=np.array([0]),setup=np.array([.5]),cap={'n':.2})
        np.testing.assert_equal(BackwardModel.score_delta(g,d),[-2,1,-6,2])

    def toy(self):
        m=ForwardTests().toy()
        m.pins=[dict(net=str(i)) for i in range(3)];m.by_net={str(i):[i] for i in range(3)}
        m.delay_correction=np.zeros(4);m.slew_correction=np.zeros((3,2));m.check_correction=np.empty(0)
        table=Table([0,100],[0,100],[[1,1],[1,1]])
        m.table=lambda a,masters:{'cell_rise':table,'cell_fall':table,'rise_transition':table,'fall_transition':table}
        m.arc_value=lambda a,s,c,masters:(a['delay'],1.)
        return m

    def test_inverted_reconvergence_backward_and_custom_fd(self):
        m=self.toy();r=m.run();bm=BackwardModel(m,r)
        # Only the rise endpoint (-13 slack) seeds TNS. Its dominant path is
        # source 1 rise -> out rise, but LSE also reaches source 0 fall.
        for branch in ('hard','ave','lse'):
            g=bm.backward(branch,gamma_ps=10)
            self.assertAlmostEqual(g['delay'].sum(),1.)
            self.assertEqual(g['at'][0,0],0.)
            if branch=='lse':self.assertGreater(g['at'][0,1],0.)
            delta=dict(arcs=np.array([0,2]),delay=np.array([.7,-.2]),slew=np.zeros(2),checks=np.array([],dtype=int),setup=np.empty(0),cap={})
            predicted=bm.score_delta(g,delta).sum();h=1e-4
            numeric=(bm.replay(delta,h,g)-bm.replay(delta,-h,g))/(2*h)
            self.assertAlmostEqual(predicted,numeric,places=7)


if __name__=='__main__':unittest.main()
