import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.sparse import csr_matrix
from forward_sta import Table
from resident_cuda_sta import ResidentCudaSTA
from autograd_cuda_sta import AutogradCudaSTA
from power_cuda import PowerCuda
from placement_rc import Pin,Network
from tree_rc_cuda import TreeRCOperator
from source_group_sta import SourceGroupPlan


def model():
    x=np.array([0.,100.])
    def tables(d,s):
        return dict(cell_rise=Table(x,x,d+.1*x[:,None]+.2*x[None,:]),
                    cell_fall=Table(x,x,d+.2*x[:,None]+.1*x[None,:]),
                    rise_transition=Table(x,x,s+.3*x[:,None]+.2*x[None,:]),
                    fall_transition=Table(x,x,s+.2*x[:,None]+.3*x[None,:]),
                    rise_constraint=Table(x,x,1.+.02*x[:,None]+.03*x[None,:]),
                    fall_constraint=Table(x,x,2.+.03*x[:,None]+.02*x[None,:]))
    bank={'A':tables(2.,1.),'B':tables(3.,2.)}
    arcs=[dict(s=s,t=3,fr=rf,to=rf,role='combinational',variant=v)
          for rf in (0,1) for s,v in ((1,'A'),(1,'B'),(2,'A'))]
    arcs += [dict(s=3,t=4,fr=rf,to=rf,role='wire') for rf in (0,1)]
    arcs += [dict(s=4,t=5,fr=rf,to=rf,role='combinational',variant='A') for rf in (0,1)]
    checks=[dict(s=0,t=5,fr=rf,to=rf,role='setup',variant='A') for rf in (0,1)]
    rc=np.zeros((len(arcs),12));rc[6,8]=.2;rc[7,9]=.3
    m=SimpleNamespace(n=6,level=np.array([0,0,0,1,2,3]),
        incoming=[[],[],[],list(range(6)),[6,7],[8,9]],arcs=arcs,checks=checks,
        pin_parts=[(None,str(i)) for i in range(3)]+[('g','Y'),('h','A'),('h','Y')],
        by_net={'clk':[0],'a':[1],'b':[2],'net':[3,4],'out':[5]},tables=bank,
        rc_matrix=csr_matrix(rc),rf_sink=np.array([False]*4+[True,False]),
        index={'g/Y':3,'h/Y':5},
        endpoints=[dict(pin=pin,rf=rf,role='setup',required_ps='1',margin_ps='0')
                   for pin in ('g/Y','h/Y') for rf in ('rise','fall')],
        at_ratio=np.array([[1.,1.]]*3+[[.9,1.1],[1.2,.8],[1.1,.9]]),
        at_offset=np.zeros((6,2)),slew_ratio=np.array([[1.,1.]]*3+[[1.1,.9],[.8,1.2],[1.3,.7]]),
        slew_offset=np.zeros((6,2)),check_ratio=np.array([.8,1.2]),check_offset=np.array([.2,-.1]),
        ref_at=np.array([[0.,0.],[2.,3.],[2.5,3.5],[0.,0.],[0.,0.],[0.,0.]]),
        ref_slew=np.array([[1.,2.],[2.,3.],[3.,2.],[0.,0.],[0.,0.],[0.,0.]]),
        rf_base_load=np.array([[1.,2.],[2.,3.],[3.,2.],[4.,5.],[4.,5.],[6.,7.]]),
        rc_base_tau=np.array([0.]*6+[1.,1.5]+[0.,0.]),masters={'g':'A','h':'A'})
    m.table=lambda a,masters:bank[a.get('variant','A')]
    return m


class AutogradSTA(unittest.TestCase):
    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_family_width_scores_equal_separate_quad_batches(self):
        m=model();m.changed_groups=lambda a,masters:None
        m.cells={name:dict(name=name,liberty_data={'pins':{'A':{'capacitance_ff':cap},'Y':{'capacitance_ff':0.}}})
                 for name,cap in [('A',1.),('B',2.)]}
        m.catalog={'cells':[m.cells['A'],m.cells['B']]}
        bank=m.tables
        m.table=lambda a,masters:bank[masters.get(m.pin_parts[a['t']][0],'A')]
        e=ResidentCudaSTA(m,device='cpu');e.prepare();g=e.backward('hard')
        instances=[dict(instance_id=i,name=n,master='A') for i,n in enumerate(['g','h'])]
        targets=np.array([[1,0,1,1,0,1,0],[0,1,0,1,1,0,1]])
        mask=np.ones_like(targets,dtype=bool);mask[0,5]=False
        wide=e.score_all(instances,dict(action_target_id=targets,action_mask=mask),dict(timing=g,lse=g))['delta_loss']['timing']
        expected=np.empty_like(wide)
        for lo in (0,4):
            width=min(4,targets.shape[1]-lo);t=np.zeros((2,4),dtype=int);valid=np.zeros((2,4),dtype=bool)
            t[:,:width]=targets[:,lo:lo+width];valid[:,:width]=mask[:,lo:lo+width]
            small=e.score_all(instances,dict(action_target_id=t,action_mask=valid),dict(timing=g,lse=g))
            expected[:,lo:lo+width]=small['delta_loss']['timing'][:,:width]
        np.testing.assert_allclose(wide,expected,rtol=1e-13,equal_nan=True)
        self.assertTrue(np.any(abs(wide[np.isfinite(wide)])>0.))

    def test_power_family_width_keeps_distinct_gate_destinations(self):
        p=object.__new__(PowerCuda)
        p.engine=SimpleNamespace(m=SimpleNamespace(catalog={'cells':[{'name':str(i)} for i in range(7)]}),
                                 tensor=lambda x:torch.tensor(x,dtype=torch.float64),numpy=lambda x:x.numpy())
        p.descriptor=lambda instance,target:([],100*int(instance)+int(target))
        targets=np.array([list(range(7)),list(reversed(range(7)))])
        mask=np.ones_like(targets,dtype=bool);mask[0,6]=False
        result=p.score([{'name':'0'},{'name':'1'}],targets,mask)
        expected=targets.astype(float)+np.array([[0],[100]]);expected[~mask]=np.nan
        np.testing.assert_array_equal(result,expected)

    def engines(self,m):
        return ResidentCudaSTA(m,device='cpu'),AutogradCudaSTA(m,device='cpu')

    def assert_reverse(self,a,b):
        for key in ('at','slew','pin_load','delay','local_slew','setup'):
            np.testing.assert_allclose(a[key],b[key],rtol=2e-11,atol=2e-12,err_msg=key)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_source_group_prediction_and_all_reverse_branches(self):
        m=model();m.calibration_mode='arc_offset';m.arc_delay_offset=np.linspace(-.1,.2,len(m.arcs))
        names=['clk','a','b','g/Y','h/A','h/Y'];m.pins=[dict(name=x) for x in names]
        m.index={x:i for i,x in enumerate(names)}
        geo=[Pin('clk','clk',0,0,'INPUT',clock=True,port=True),
             Pin('a','a',0,0,'INPUT',port=True),Pin('b','b',0,0,'INPUT',port=True)]
        manual,auto=self.engines(m);baseline=manual.forward();ends=[]
        for i,endpoint in enumerate(m.endpoints):
            row=dict(endpoint,source_group='input',is_path_delay=str(i%2),is_check='0',
                     at_ps=str(baseline['at'][m.index[endpoint['pin']],i%2]),
                     capture_pin='g/Y',capture_rf='rise')
            ends.append(row)
        plan=SourceGroupPlan(m,geo,ends)
        for engine in (manual,auto):engine.enable_source_groups(plan)
        mf=manual.prepare();af=auto.prepare()
        np.testing.assert_allclose(mf['endpoints'],af['endpoints'])
        np.testing.assert_array_equal(manual.capture_valid.cpu().numpy(),[True,False,True,False])
        for branch in ('hard','ave','lse'):
            self.assert_reverse(manual.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3),
                                auto.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3))
        # Anchoring an impulse model before installing its moment state must
        # fail before changing endpoint bindings or fitting a wrong residual.
        raw=model();raw.calibration_mode='arc_offset';raw.wire_slew_model='impulse'
        engine=ResidentCudaSTA(raw,device='cpu')
        with self.assertRaisesRegex(ValueError,'Refresh second-moment'):
            engine.enable_source_groups(plan)
        self.assertFalse(hasattr(engine,'source_groups'))

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_all_branches_with_rc_checks_calibration_and_conditional_bundles(self):
        manual,auto=self.engines(model());manual.prepare();auto.prepare()
        for branch in ('hard','ave','lse'):
            self.assert_reverse(manual.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3),
                                auto.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3))

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_capture_clock_delta_forward_and_signed_reverse(self):
        m=model()
        # Use a propagated capture node so its changes affect required time.
        for endpoint in m.endpoints:
            endpoint.update(capture_pin='g/Y',capture_rf='rise')
        manual,auto=self.engines(m)
        frozen=manual.forward()
        manual.enable_capture_clock_delta();auto.enable_capture_clock_delta()
        current=manual.prepare();auto.prepare()
        expected_delta=current['at'][3,0]-m.ref_at[3,0]
        finite=np.isfinite(frozen['endpoints'][:,2])
        np.testing.assert_allclose(current['endpoints'][finite,2]-frozen['endpoints'][finite,2],expected_delta)
        np.testing.assert_allclose(current['endpoints'],auto.b.forward['endpoints'])
        for branch in ('hard','ave','lse'):
            self.assert_reverse(manual.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3),
                                auto.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3))

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_signed_power_vjp(self):
        manual,auto=self.engines(model());manual.prepare();auto.prepare()
        slew=np.linspace(-.3,.7,12).reshape(6,2)
        load=np.linspace(.6,-.2,12).reshape(6,2)
        a=manual.backward_power(slew,load);b=auto.backward_power(slew,load)
        for key in ('at','pin_load','delay','local_slew','setup'):
            np.testing.assert_allclose(a[key],b[key],rtol=2e-11,atol=2e-12,err_msg=key)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_arc_offsets_keep_physical_wire_slew_and_reverse(self):
        m=model();m.calibration_mode='arc_offset'
        m.arc_delay_offset=np.array([.1]*6+[-1.2,-.3]+[.2,.2])
        manual,auto=self.engines(m);mf=manual.prepare();af=auto.prepare()
        np.testing.assert_allclose(mf['at'],af['at'])
        self.assertEqual(mf['delay'][6],0.)
        self.assertAlmostEqual(mf['delay'][7],1.2)
        # Wire slew uses the physical Elmore moment, not its calibrated delay.
        self.assertAlmostEqual(manual.b.local_slew[6],np.hypot(mf['slew'][3,0],2.197224577))
        for branch in ('hard','ave','lse'):
            self.assert_reverse(manual.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3),
                                auto.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3))

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_second_moment_timing_and_power_reverse_match_autograd(self):
        m=model();m.index.update({'h/A':4});m.rf_base_pin=np.zeros((6,2));m.rf_base_pin[4]=[4.,5.]
        m.calibration_mode='arc_offset';m.arc_delay_offset=np.zeros(len(m.arcs));m.arc_delay_offset[6]=-3.
        manual,auto=self.engines(m)
        network=Network('net',(Pin('g/Y','net',0.,0.,'OUTPUT'),Pin('h/A','net',1.,0.,'INPUT')),
            ((0.,0.,2),(.5,0.,2),(1.,0.,2)),((0,1,.2),(1,2,.3)),
            np.array([.1,.2,.3]),{'g/Y':0,'h/A':2},'g/Y',1.,0,{})
        op=TreeRCOperator(m,{'net':network},device='cpu')
        load,tau,state=op.absolute_moments(m.rf_base_pin,{})
        for engine in [manual,auto]:engine.update_tree_parasitics(op,load,tau,moment_state=state)
        mf=manual.prepare();af=auto.prepare();np.testing.assert_allclose(mf['at'],af['at'])
        self.assertEqual(mf['delay'][6],0.)
        for branch in ['hard','ave','lse']:
            self.assert_reverse(manual.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3),
                                auto.backward(branch,gamma_ps=1000.,alpha=.7,beta=.3))
        seed=np.linspace(-.3,.7,12).reshape(6,2);cap=np.linspace(.6,-.2,12).reshape(6,2)
        a=manual.backward_power(seed,cap);b=auto.backward_power(seed,cap)
        for key in ['pin_load','local_slew']:
            np.testing.assert_allclose(a[key],b[key],rtol=2e-11,atol=2e-12)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_committed_master_change_requires_fresh_capacitance_moments(self):
        m=model();m.index.update({'h/A':4});m.rf_base_pin=np.zeros((6,2));m.rf_base_pin[4]=[4.,5.]
        network=Network('net',(Pin('g/Y','net',0.,0.,'OUTPUT'),Pin('h/A','net',1.,0.,'INPUT')),
            ((0.,0.,2),(1.,0.,2)),((0,1,.5),),np.array([.1,.3]),
            {'g/Y':0,'h/A':1},'g/Y',1.,0,{})
        for engine in self.engines(m):
            engine.set_model(m)
            op=TreeRCOperator(m,{'net':network},device='cpu')
            load,tau,state=op.absolute_moments(m.rf_base_pin,{})
            engine.update_tree_parasitics(op,load,tau,moment_state=state)
            engine.prepare()
            m.masters['g']='B' if m.masters['g']=='A' else 'A'
            engine.set_model(m)
            with self.assertRaisesRegex(RuntimeError,'refresh tree parasitics'):
                engine.prepare()
            # Rebinding again cannot legitimize the old capacitance baseline.
            engine.set_model(m)
            with self.assertRaisesRegex(RuntimeError,'refresh tree parasitics'):
                engine.candidate_wire_moments(engine.zeros(m.n*2))
            load,tau,state=op.absolute_moments(m.rf_base_pin,{})
            engine.update_tree_parasitics(op,load,tau,moment_state=state)
            engine.prepare()

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_group_ties_and_negative_arrivals(self):
        e=AutogradCudaSTA(model(),device='cpu')
        bundle=e.tensor([0,0,1,2],True);dest=e.tensor([0,0,0],True)
        for vals in ([100.,100.,99.5,90.],[-100.,-100.,-100.5,-110.]):
            for branch in ('hard','ave','lse'):
                x=e.tensor(vals).requires_grad_()
                y=e._merge(x,bundle,dest,3,1,branch,.01,1000.)
                self.assertEqual(y.item(),max(vals))
                grad=torch.autograd.grad(y.sum(),x)[0].numpy()
                peaks=np.array([vals[0],vals[2],vals[3]])
                if branch=='lse':w=np.exp((peaks-peaks.max())/1000.)
                else:w=(peaks>=peaks.max()-(.01*abs(peaks.max()) if branch=='ave' else 0.)).astype(float)
                w/=w.sum()
                np.testing.assert_allclose(grad,[w[0]/2,w[0]/2,w[1],w[2]],rtol=1e-13)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_zero_slew_and_zero_wire_delay(self):
        m=model();m.ref_slew[:]=0.;m.slew_ratio[:]=0.;m.rc_base_tau[:]=0.
        manual,auto=self.engines(m);manual.prepare();auto.prepare()
        self.assert_reverse(manual.backward('lse',gamma_ps=1000.),auto.backward('lse',gamma_ps=1000.))
        a=manual.backward_power(np.ones((6,2)),np.zeros((6,2)))
        b=auto.backward_power(np.ones((6,2)),np.zeros((6,2)))
        for key in ('pin_load','local_slew'):
            np.testing.assert_allclose(a[key],b[key],atol=1e-12)
            self.assertTrue(np.isfinite(b[key]).all())

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_power_lut_autograd_matches_manual_with_extrapolation_and_load_ties(self):
        m=model();manual,auto=self.engines(m);manual.prepare();auto.prepare()
        a=Table([0.,1.],[0.,2.,5.],np.array([[1.,3.,6.],[21.,33.,51.]])*1e-15)
        b=Table([0.,3.,10.],[0.,5.],np.array([[2.,7.],[11.,31.],[32.,92.]])*1e-15)
        tabs={'rise':a,'fall':b}
        p=SimpleNamespace(tables={'A':{'pins':{'Y':{'powers':[{'tables':tabs}]}}}},
                          inst_pins={'g':{}},model=m,
                          factors=lambda i,master:([(1,3,False,2e9,tabs)],[(3,4e-6)],1e-7))
        for engine in (manual,auto):engine._load[3,:]=4.
        gm=PowerCuda(p,manual).internal_partials();ga=PowerCuda(p,auto).internal_partials()
        for x,y in zip(gm,ga):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-15)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_parallel_endpoint_ties_are_counted_like_manual(self):
        m=model();m.at_ratio[:]=1.;m.ref_at[:]=0.;m.ref_slew[:]=0.
        # Output endpoints have identical RF values and no setup check.
        m.endpoints=[dict(pin='g/Y',rf=rf,role='output',required_ps='0') for rf in ('rise','fall')]
        m.checks=[];m.check_ratio=np.empty(0);m.check_offset=np.empty(0)
        m.slew_ratio[:]=0.
        manual,auto=self.engines(m);manual.prepare();auto.prepare()
        self.assert_reverse(manual.backward('ave',alpha=1.,beta=1.),auto.backward('ave',alpha=1.,beta=1.))

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_missing_rf_arc_keeps_zero_slew_before_calibration(self):
        m=model()
        m.arcs=[a for a in m.arcs if a['to']==0]
        m.incoming=[[] for _ in range(m.n)]
        for k,a in enumerate(m.arcs):m.incoming[a['t']].append(k)
        m.rc_matrix=csr_matrix((len(m.arcs),m.n*2))
        m.rc_base_tau=np.array([1. if a['role']=='wire' else 0. for a in m.arcs])
        m.slew_offset[:,1]=2.
        manual,auto=self.engines(m);f=manual.prepare();auto.prepare()
        for branch in ('hard','ave','lse'):
            graph=auto._graph(branch,.01,1000.)
            np.testing.assert_allclose(f['slew'],auto.numpy(graph['slew']).reshape(-1,2),atol=1e-12)
            self.assert_reverse(manual.backward(branch,gamma_ps=1000.),auto.backward(branch,gamma_ps=1000.))

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_updated_working_point_preserves_forward_reverse_consistency(self):
        m=model();manual,auto=self.engines(m)
        for scale in (1.,1.3,.8):
            m.at_ratio[5]=scale;m.slew_offset[4]=scale
            for e in (manual,auto):e.set_model(m);e.prepare()
            self.assert_reverse(manual.backward('ave'),auto.backward('ave'))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA graph replay requires a GPU')
    def test_cuda_rebind_updates_captured_values_without_recapture(self):
        m=model();engine=ResidentCudaSTA(m)
        before=engine.fast_forward();graph=engine._replay_graph
        m.ref_at[1]+=.5;m.at_ratio[5]*=1.2;m.slew_offset[4]+=.3
        engine.set_model(m)
        self.assertIs(engine._replay_graph,graph)
        replay=engine.fast_forward();full=engine.forward()
        for key in ['at','slew','delay','load_rf','check','endpoints']:
            np.testing.assert_allclose(replay[key],full[key],rtol=1e-12,atol=1e-12)
        self.assertFalse(np.allclose(before['at'],replay['at']))
        m.calibration_mode='arc_offset';m.arc_delay_offset=np.zeros(len(m.arcs))
        engine.set_model(m)
        self.assertIsNone(engine._replay_graph)

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_reverse_does_not_call_manual_sweep_or_lut_partials(self):
        e=AutogradCudaSTA(model(),device='cpu')
        with (patch.object(ResidentCudaSTA,'backward',side_effect=AssertionError('manual reverse')),
              patch.object(ResidentCudaSTA,'lut_parts',side_effect=AssertionError('analytic LUT derivative'))):
            e.prepare();e.backward('ave');e.backward('lse',gamma_ps=1000.)
            e.backward_power(np.ones((6,2)),np.ones((6,2)))


if __name__=='__main__':unittest.main()
