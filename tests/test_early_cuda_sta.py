import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_autograd_sta import model
from resident_cuda_sta import ResidentCudaSTA
from early_cuda_sta import EarlyCudaSTA
from source_group_sta import SourceGroupPlan
from placement_rc import Pin


def minimum_model():
    m=model();m.analysis='min';m.cap_ranges={}
    for a in m.checks:a['role']='hold'
    m.endpoints=[dict(pin='h/Y',rf=rf,role='hold',required_ps='8',margin_ps='1',
        capture_pin='clk',capture_rf=rf) for rf in ('rise','fall')]
    m.index['clk']=0
    return m


def independent(m):
    at=np.full((m.n,2),np.inf);sl=np.zeros((m.n,2));delays=[]
    for pin in np.argsort(m.level,kind='stable'):
        incoming=m.incoming[pin]
        if not incoming:at[pin]=m.ref_at[pin];sl[pin]=m.ref_slew[pin];continue
        for rf in (0,1):
            values=[];slews=[]
            for k in incoming:
                a=m.arcs[k]
                if a['to']!=rf:continue
                if a['role']=='wire':
                    d=m.rc_base_tau[k];s=np.hypot(sl[a['s'],a['fr']],2.197224577*d)
                else:
                    t=m.table(a,m.masters);label=('rise','fall')[rf]
                    x,y=sl[a['s'],a['fr']],m.rf_base_load[pin,rf]
                    d=t['cell_'+label](x,y);s=max(0.,t[label+'_transition'](x,y))
                values.append(at[a['s'],a['fr']]+d);slews.append(s)
            if values:at[pin,rf]=min(values)*m.at_ratio[pin,rf]+m.at_offset[pin,rf]
            sl[pin,rf]=max(0.,(min(slews) if slews else 0.)*m.slew_ratio[pin,rf]+m.slew_offset[pin,rf])
    return at,sl


class EarlyTests(unittest.TestCase):
    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_early_source_class_installation_preserves_anchor_and_updates(self):
        m=minimum_model();m.calibration_mode='arc_offset';m.at_ratio[:]=1.
        names=['clk','a','b','g/Y','h/A','h/Y']
        m.pins=[dict(name=n) for n in names];m.index={n:i for i,n in enumerate(names)}
        e=EarlyCudaSTA(m,device='cpu');before=e.forward();ends=[]
        for i,row in enumerate(m.endpoints):
            ends.append(dict(row,source_group='input',is_path_delay='0',
                at_ps=str(before['at'][5,i]),required_ps=str(before['endpoints'][i,2]),
                margin_ps=str(before['check'][i])))
        geometry=[Pin('clk','clk',0,0,'INPUT',clock=True,port=True),
                  Pin('a','a',0,0,'INPUT',port=True),Pin('b','b',0,0,'INPUT',port=True)]
        e.enable_source_groups(SourceGroupPlan(m,geometry,ends))
        np.testing.assert_allclose(e.forward()['endpoints'],before['endpoints'],atol=1e-12)
        e.tau0[6]+=1.
        after=e.forward()
        expected=after['at'][5,0]-after['endpoints'][0,2]
        self.assertAlmostEqual(after['endpoints'][0,3],expected)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_reconvergent_early_arrival_and_slew_match_independent_scalar_oracle(self):
        m=minimum_model();e=EarlyCudaSTA(m,device='cpu');r=e.forward()
        at,sl=independent(m)
        np.testing.assert_allclose(r['at'],at,rtol=1e-12)
        np.testing.assert_allclose(r['slew'],sl,rtol=1e-12)
        maximum=ResidentCudaSTA(m,device='cpu').forward()
        self.assertGreater(np.max(maximum['at']-r['at']),.5)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_hold_uses_late_capture_slew_and_arrival_change(self):
        m=minimum_model();e=EarlyCudaSTA(m,device='cpu')
        late_at=e.tensor(m.ref_at);late_slew=e.tensor(m.ref_slew)
        e.bind_late_clock(late_at,late_slew,late_at.clone())
        before=e.forward()
        late_at[0,0]+=3.;late_slew[0,0]+=5.
        after=e.forward()
        # Only the rise hold endpoint changes: +3ps capture arrival plus
        # 5ps * .03ps/ps related-clock slope * .8 check correction.
        self.assertAlmostEqual(after['endpoints'][0,2]-before['endpoints'][0,2],3.+5.*.03*.8)
        self.assertAlmostEqual(after['endpoints'][0,3]-before['endpoints'][0,3],-3.-5.*.03*.8)
        np.testing.assert_allclose(after['endpoints'][1],before['endpoints'][1])

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_binding_mixed_clock_slew_preserves_initial_hold_anchor(self):
        m=minimum_model();e=EarlyCudaSTA(m,device='cpu');before=e.forward()
        m.ref_slew=before['slew'].copy()
        late_at=e.tensor(m.ref_at.copy());late_slew=e.tensor(m.ref_slew.copy());late_slew[0]+=7.
        e.bind_late_clock(late_at,late_slew,late_at.clone())
        after=e.forward()
        np.testing.assert_allclose(after['endpoints'],before['endpoints'],atol=1e-12)
        late_at[0,0]+=2.
        with self.assertRaisesRegex(ValueError,'initial reference'):
            e.bind_late_clock(late_at,late_slew,e.tensor(m.ref_at))

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_inherited_max_reverse_and_source_tags_are_rejected(self):
        e=EarlyCudaSTA(minimum_model(),device='cpu')
        for f in (e.prepare,e.backward):
            with self.assertRaises(NotImplementedError):f()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_replay_preserves_min_propagation_and_updated_rc(self):
        e=EarlyCudaSTA(minimum_model());before=e.fast_forward();graph=e._replay_graph
        e.tau0[6]+=2.;after=e.fast_forward();reference=e.forward()
        self.assertIs(e._replay_graph,graph)
        np.testing.assert_allclose(after['at'],reference['at'],atol=1e-12)
        self.assertGreater(after['at'][4,0],before['at'][4,0])


if __name__=='__main__':unittest.main()
