import copy
import unittest
from model_runtime import feasible_constraints


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(scenarios={
            'BC': dict(hold_tns_tolerance_ns=1., hold_wns_tolerance_ns=.01),
            'TC': dict(hold_tns_tolerance_ns=.1, hold_wns_tolerance_ns=.005)},
            epsilon=dict(slew_ps=.001, cap_ff=1e-6, fanout=1e-6))
        self.rows = {c:dict(hold_tns_ns=0., hold_wns_ns=0.,
                           slew_violation_sum_ps=0., cap_violation_sum_ff=0.,
                           fanout_violation_sum=0., global_routing_overflow=0.,
                           max_global_routing_overflow=0.) for c in self.config['scenarios']}

    def test_hold_uses_each_corner_tolerance(self):
        self.rows['BC']['hold_tns_ns']=-.5
        self.rows['BC']['hold_wns_ns']=-.009
        self.assertTrue(feasible_constraints(self.rows,self.config))
        self.rows['TC']['hold_tns_ns']=-.5
        self.assertFalse(feasible_constraints(self.rows,self.config))

    def test_wns_cannot_hide_behind_small_tns(self):
        self.rows['TC']['hold_tns_ns']=-.02
        self.rows['TC']['hold_wns_ns']=-.006
        self.assertFalse(feasible_constraints(self.rows,self.config))

    def test_erc_and_per_layer_overflow_are_hard_gates(self):
        for key,value in [('slew_violation_sum_ps',.01),('cap_violation_sum_ff',.001),
                          ('fanout_violation_sum',1.),('global_routing_overflow',1.),
                          ('max_global_routing_overflow',1.)]:
            rows=copy.deepcopy(self.rows)
            rows['BC'][key]=value
            self.assertFalse(feasible_constraints(rows,self.config),key)

    def test_roundoff_and_positive_hold_slack(self):
        self.rows['TC']['hold_wns_ns']=.03
        self.rows['BC']['slew_violation_sum_ps']=.0001
        self.rows['BC']['cap_violation_sum_ff']=1e-7
        original=copy.deepcopy(self.rows)
        self.assertTrue(feasible_constraints(self.rows,self.config))
        self.assertEqual(self.rows,original)
    def test_deferred_erc_stage_still_checks_hold_and_routing(self):
        self.rows['BC']['slew_violation_sum_ps']=10.
        self.assertTrue(feasible_constraints(self.rows,self.config,check_erc=False))
        self.rows['BC']['max_global_routing_overflow']=1.
        self.assertFalse(feasible_constraints(self.rows,self.config,check_erc=False))
    def test_deferred_hold_is_explicit_and_keeps_routing_gate(self):
        self.rows['BC']['hold_tns_ns']=-2.
        self.rows['BC']['hold_wns_ns']=-.02
        self.assertFalse(feasible_constraints(self.rows,self.config))
        self.assertTrue(feasible_constraints(self.rows,self.config,check_hold=False))
        self.rows['BC']['global_routing_overflow']=1.
        self.assertFalse(feasible_constraints(self.rows,self.config,check_hold=False))


if __name__=='__main__':unittest.main()
