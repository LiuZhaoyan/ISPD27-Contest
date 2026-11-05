import unittest
from types import SimpleNamespace
import numpy as np
from forward_sta import Table,ForwardModel
from ratio_sta import RatioForwardModel,RatioBackwardModel

class RecoveryChecks(unittest.TestCase):
    def test_negative_margin_data_clock_axes_and_reverse_sign(self):
        # A recovery constraint q=-4+.5*(data_slew-2)+.25*(clock_slew-3).
        # With fixed clock boundary 96, R=96-q and loss=A-R, so dL/dq=+1.
        axes=np.array([0.,10.]);tab=Table(axes,axes,-5.75+.5*axes[:,None]+.25*axes[None,:])
        m=RatioForwardModel.__new__(RatioForwardModel);m.n=2;m.arcs=[];m.pin_parts=[('ff','CLK'),('ff','RESETN')]
        m.masters={'ff':'FF'};m.cells={};m.by_net={'clock':[0],'reset':[1]};m.table=lambda a,masters:{'rise_constraint':tab}
        m.checks=[dict(s=0,t=1,fr=0,to=0,role='recovery')];m.check_incoming=[[],[0]]
        m.index={'ff/RESETN':1};m.endpoints=[dict(pin='ff/RESETN',rf='rise',role='recovery',required_ps='100',margin_ps='-4')]
        m.order=np.arange(2);m.at_ratio=np.ones((2,2));m.slew_ratio=np.ones((2,2));m.load_ratio=np.ones(2);m.check_ratio=np.ones(1)
        at=np.array([[0.,0.],[120.,120.]]);slew=np.array([[3.,3.],[2.,2.]])
        q=ForwardModel.check_value(m,m.checks[0],slew,m.masters)
        self.assertAlmostEqual(q,-4.)
        result=m.finish(at,slew,np.zeros(2),np.empty(0),np.array([q]))
        self.assertAlmostEqual(result['tns_ps'],-20.)
        backward=RatioBackwardModel(m,result);g=backward.backward('hard')
        self.assertAlmostEqual(g['setup'][0],1.)
        self.assertAlmostEqual(g['slew'][1,0],.5)
        self.assertAlmostEqual(g['slew'][0,0],.25)
        self.assertAlmostEqual(g['at'][1,0],1.)
        for node,expected in ((1,.5),(0,.25)):
            h=1e-4;loss=[]
            for sign in (1,-1):
                changed=slew.copy();changed[node,0]+=sign*h
                check=ForwardModel.check_value(m,m.checks[0],changed,m.masters)
                loss.append(-m.finish(at,changed,np.zeros(2),np.empty(0),np.array([check]))['tns_ps'])
            self.assertAlmostEqual((loss[0]-loss[1])/(2*h),expected,places=8)

if __name__=='__main__':unittest.main()
