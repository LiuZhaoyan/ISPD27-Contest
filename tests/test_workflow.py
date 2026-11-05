import sys
from pathlib import Path
import unittest
from unittest.mock import patch,Mock
import copy
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quad'))
from workflow_policy import WorkflowConfig,select_batch,select_buffer_batch,buffering_due,accept,accuracy_pass,tighten_budget,reference_due,erc_revert_actions
from tagged_arrival import TaggedArrival
from workflow_buffering import library as dp_library
from concurrent.futures import ThreadPoolExecutor


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TaggedArrivalTests(unittest.TestCase):
    def tensor(self,x,grad=False):
        return torch.tensor(x,dtype=torch.float64,device='cuda',requires_grad=grad)

    def test_startpoint_pairing_and_reverse(self):
        # Source 0 arrives later, but source 1 has the tighter constraint.
        r=TaggedArrival([0,1,2],[2,2,3],[0,0,1,2],[-1]*3,[-1]*3,[0,1,-1,-1],k=2,groups=1)
        d=self.tensor([4.,2.,3.],True);p=self.tensor([0.,0.,0.,0.],True)
        at,tags,overflow=r(d,p)
        required=self.tensor([10.,4.])
        slack=required[tags[-1,0].long()]-at[-1,0]
        self.assertEqual(tags[-1,0,slack.argmin()].item(),1)
        loss=-slack.min();loss.backward()
        np.testing.assert_array_equal(d.grad.cpu(),[0.,1.,1.])
        np.testing.assert_array_equal(p.grad.cpu(),[0.,1.,0.,0.])
        self.assertEqual(overflow.sum().item(),0)

    def test_same_source_reconvergence_and_finite_difference(self):
        r=TaggedArrival([0,0,1,2],[1,2,3,3],[0,1,1,2],[-1]*4,[-1]*4,[0,-1,-1,-1],k=2,groups=1)
        d=self.tensor([1.,2.,4.,1.],True);p=self.tensor([0.]*4)
        at,tags,_=r(d,p);self.assertEqual(tags[-1,0].tolist(),[0,-1])
        at[-1,0,0].backward()
        for i in range(4):
            plus=d.detach().clone();minus=plus.clone();plus[i]+=.0001;minus[i]-=.0001
            fd=(r(plus,p)[0][-1,0,0]-r(minus,p)[0][-1,0,0])/.0002
            self.assertAlmostEqual(fd.item(),d.grad[i].item(),places=8)

    def test_min_and_overflow(self):
        r=TaggedArrival([0,1],[2,2],[0,0,1],[-1]*2,[-1]*2,[0,1,-1],k=1,groups=1,analysis='min')
        d=self.tensor([4.,2.],True);a,t,o=r(d,self.tensor([0.,0.,0.]))
        self.assertEqual(t[-1,0,0].item(),1);self.assertEqual(o[-1,0].item(),1)
        a[-1,0,0].backward();np.testing.assert_array_equal(d.grad.cpu(),[0.,1.])

    def test_launch_and_normalized_constraint_classes(self):
        r=TaggedArrival([0],[1],[0,1],[9],[1],[-1,-1],k=2,groups=5)
        d=self.tensor([3.],True);p=self.tensor([100.,0.],True)
        at,tags,_=r(d,p)
        self.assertEqual(at[1,1,0].item(),103.)
        self.assertEqual(at[1,3,0].item(),3.)
        (at[1,1,0]+at[1,3,0]).backward()
        np.testing.assert_array_equal(d.grad.cpu(),[2.]);np.testing.assert_array_equal(p.grad.cpu(),[1.,0.])


class WorkflowPolicyTests(unittest.TestCase):
    def test_buffer_admission_keeps_chain_and_tries_cheaper_same_net_plan(self):
        def plan(net,cost,count,x):
            return dict(net=net,cost=cost,actions=[dict(instance=f'{net}{i}',location=[x,0.]) for i in range(count)])
        proposals=[plan('n',3.,2,0),plan('n',1.,2,0),plan('m',.1,2,4)]
        stats={};batch=select_buffer_batch(proposals,3,{'leak':2.},lambda p:{'leak':p['cost']},stats)
        self.assertEqual([a['instance'] for a in batch],['n0','n1'])
        self.assertEqual(stats['resource_budget'],1)
        self.assertEqual(stats['nets'],1)
    def test_erc_reversion_keeps_unrelated_epoch_changes(self):
        original=[dict(name=n,master='old') for n in ['driver','load','unrelated']]
        current=[dict(name=n,odb_name=n,master='old' if n=='driver' else 'new',status='PLACED',frozen='0')
                 for n in ['driver','load','unrelated']]
        geometry=[dict(pin='driver/Y',net='bad'),dict(pin='load/A',net='bad'),dict(pin='unrelated/A',net='good')]
        actions=erc_revert_actions(original,current,geometry,{'driver/Y'})
        self.assertEqual([a['instance'] for a in actions],['load'])
        self.assertEqual(actions[0]['target'],'old')
    def test_accuracy_budget_tightens_frequency_before_batch(self):
        c=WorkflowConfig();errors={'TC':{'max':dict(missing=0,extra=0,mae_ps=.9,p99_abs_ps=7.)}}
        self.assertFalse(accuracy_pass(errors,c))
        self.assertEqual(tighten_budget(4,12),(2,12))
        self.assertEqual(tighten_budget(1,12),(1,6))
        errors['TC']['max']['p99_abs_ps']=4.
        self.assertTrue(accuracy_pass(errors,c))
    def test_sizing_first_buffer_budget(self):
        c=WorkflowConfig()
        self.assertFalse(buffering_due(7,10,0,0,c))
        self.assertFalse(buffering_due(8,1,0,0,c))
        self.assertTrue(buffering_due(8,2,0,0,c))
        self.assertTrue(buffering_due(16,0,0,0,c))
        self.assertTrue(buffering_due(4,0,0,0,c,final=True))
        self.assertFalse(buffering_due(1,0,0,0,c,final=True))
        self.assertFalse(buffering_due(8,2,c.max_buffer_batches,0,c))
        self.assertFalse(buffering_due(8,2,0,c.max_new_buffers,c))

    def test_reference_budget_tracks_work_and_geometry(self):
        c=WorkflowConfig()
        self.assertIsNone(reference_due(2,256,20,10000,8,c))
        self.assertEqual(reference_due(4,512,20,10000,8,c),'accepted_move_budget')
        self.assertEqual(reference_due(1,20,500,10000,8,c),'changed_rc_budget')
        self.assertEqual(reference_due(8,100,20,10000,8,c),'accepted_batch_budget')
        self.assertIsNone(reference_due(0,1000,10000,10000,8,c))

    def test_batch_fills_distinct_gates_from_broad_alternatives(self):
        candidates=[dict(instance=f'g{i}',gain=float(1000-i-j),writes=[f'inst:g{i}'],reads=[])
                    for i in range(100) for j in range(5)]
        stats={};batch=select_batch(candidates,64,stats)
        self.assertEqual(len(batch),64);self.assertEqual(len({a['instance'] for a in batch}),64)
        self.assertGreater(stats['duplicate_gate'],0)

    def test_power_budget_keeps_smaller_alternative_for_same_gate(self):
        def a(instance,gain,leak,target):
            return dict(instance=instance,gain=gain,target=target,writes=['inst:'+instance],reads=[],resources={'leakage:TC':leak})
        pool=[a('a',10,7,'expensive'),a('a',9,2,'small'),a('b',8,3,'useful'),a('c',7,2,'over_budget')]
        stats={};batch=select_batch(pool,3,stats,{'leakage:TC':5})
        self.assertEqual([(a['instance'],a['target']) for a in batch],[('a','small'),('b','useful')])
        self.assertEqual(stats['resource_budget'],2)

    def test_symmetric_dependency_and_region_conflicts(self):
        def a(name,gain,writes,reads,regions=()):return dict(instance=name,gain=gain,writes=writes,reads=reads,regions=regions)
        candidates=[a('a',5,['net:a'],['inst:b']),a('b',4,['inst:b'],[]),
                    a('c',3,['net:c'],[],['row:1']),a('d',2,['net:d'],[],['row:1'])]
        self.assertEqual([x['instance'] for x in select_batch(candidates,10)],['a','c'])

    def test_bad_hold_cannot_buy_setup_gain(self):
        b={'TC':dict(setup_tns_ns=-10.,setup_wns_ns=-.2,hold_tns_ns=0.,hold_wns_ns=0.,dynamic_power_w=1.,leakage_power_w=.1)}
        a={'TC':dict(b['TC'],setup_tns_ns=-5.,hold_wns_ns=-.02)}
        ok,why=accept(b,a,b,WorkflowConfig());self.assertFalse(ok)
        self.assertIn('TC:hold_wns_ns',why)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):WorkflowConfig(refresh_every=0).validate()
        with self.assertRaises(ValueError):WorkflowConfig(tag_k=32,tag_max_k=16).validate()


class BufferDPTests(unittest.TestCase):
    def test_dp_filters_loss_and_ranks_improvement_against_unbuffered_tree(self):
        from workflow_buffering import ParallelBufferDP
        from placement_rc import Network,Pin
        from forward_sta import Table,condition
        pins=(Pin('d/Y','n',0,0,'OUTPUT'),Pin('a/A','n',1,0,'INPUT'),Pin('b/A','n',0,1,'INPUT'))
        network=Network('n',pins,('d','a','b'),((0,1,5.),(0,2,5.)),np.zeros(3),
                        {'d/Y':0,'a/A':1,'b/A':2},'d/Y',2.,0,{})
        cell=dict(name='BUFx1_ASAP7_75t_R',frozen_reason=None,liberty_data=dict(
            pins={'A':dict(direction='input',capacitance_ff=1.),'Y':dict(direction='output')},area=.1))
        model=SimpleNamespace(index={'d/Y':0,'a/A':1,'b/A':2},rf_base_pin=np.array([[0.,0.],[10.,10.],[10.,10.]]),incoming=[[],[],[]])
        forward={'slew':np.ones((3,2))};required=np.full((3,2),100.)
        for delay,expect in [(10000.,False),(1.,True)]:
            table=Table([0.,100.],[0.,100.],np.full((2,2),delay))
            dp=ParallelBufferDP({'cells':[cell]},
                {(cell['name'],'A','Y','combinational',condition('')):{'cell_rise':table,'cell_fall':table}},max_per_net=2)
            plans=dp.solve(network,SimpleNamespace(m=model),forward,required,0)
            self.assertEqual(bool(plans),expect)
            for plan in plans:self.assertGreater(plan['gain'],0.)

    def test_partial_native_insertion_error_restores_transaction(self):
        from post_cts_workflow import PostCTSWorkflow
        w=object.__new__(PostCTSWorkflow)
        w.config=WorkflowConfig();w.out=Path('/tmp/unused_workflow_test')
        w.catalogs={'TC':{}};w.tables={'TC':{}}
        w.pairs={'TC':SimpleNamespace(forest=None,late=None)}
        w.predictions={'TC':{'max':{}}};w.net_scores={}
        w.report={'optimization_buffers':0};w.candidates=lambda *args:None;w.timed=lambda name,fn:fn()
        w.batch_resource_limits=lambda:{};w.buffer_resources=lambda p:{}
        w.native=Mock();w.native.snapshot.return_value='original_def'
        w.native.apply.side_effect=RuntimeError('partial insertion')
        dp=Mock();dp.propose.return_value=[dict(net='n',actions=[dict(instance='b',location=[0.,0.])])]
        with patch('post_cts_workflow.ParallelBufferDP',return_value=dp):
            with self.assertRaisesRegex(RuntimeError,'partial insertion'):w.buffer_step(0)
        w.native.restore.assert_called_once_with('original_def')

    def test_rejected_buffer_plan_does_not_export_calibration_graph(self):
        from post_cts_workflow import PostCTSWorkflow
        w=object.__new__(PostCTSWorkflow);w.config=WorkflowConfig();w.out=Path('/tmp/unused_buffer_test')
        w.catalogs={'TC':{}};w.tables={'TC':{}};w.pairs={'TC':SimpleNamespace(forest=None,late=None)}
        w.predictions={'TC':{'max':{}}};w.net_scores={};w.checkpoint_id=0
        w.report={'optimization_buffers':0,'buffer_batches':0,'checkpoints':[],'steps':[]}
        w.candidates=lambda *args:None;w.timed=lambda name,fn:fn();w.save=Mock()
        w.batch_resource_limits=lambda:{};w.buffer_resources=lambda p:{}
        w.native=Mock();w.native.reference.return_value=({},{});w.native.snapshot.return_value='original_def'
        w.reference_metrics=w.baseline={};w.build_models=Mock()
        dp=Mock();dp.propose.return_value=[dict(net='n',actions=[dict(instance='b',location=[0.,0.])])]
        with patch('post_cts_workflow.ParallelBufferDP',return_value=dp),patch('post_cts_workflow.accept',return_value=(False,['TC:wns_guard'])):
            w.buffer_step(0)
        w.native.reference.assert_called_once_with(w.out/'reference_000',models=False)
        w.native.complete_reference.assert_not_called();w.build_models.assert_not_called()
        w.native.restore.assert_called_once_with('original_def')


class ReferenceSchedulingTests(unittest.TestCase):
    def test_final_reference_reuse_requires_fresh_oracle_and_identical_design(self):
        from workflow_native import NativeWorkflow
        n=object.__new__(NativeWorkflow);n.reference=Mock(return_value=({'new':True},{}))
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            files=('model.def','instances.tsv','pins_geometry.tsv')
            for name in files:(source/name).write_text('original')
            for name in ('gr','metrics'):(source/name).mkdir()
            metadata={'timing_reference':'fresh_mcmm_from_independent_global_routing_spef','metrics':{'old':True}}
            (source/'reference.json').write_text(json.dumps(metadata))
            def snapshot(path):
                path.mkdir()
                for name in files:(path/name).write_text('original')
                return path
            n.snapshot=snapshot
            metrics={'old':True};got,timing=n.final_reference(root/'same',source,metrics)
            self.assertIs(got,metrics);self.assertIn('reused_validated_reference',timing);n.reference.assert_not_called()
            (source/'model.def').write_text('different connectivity')
            self.assertEqual(n.final_reference(root/'changed',source,metrics)[0],{'new':True})
            n.reference.assert_called_once_with(root/'changed',models=False)
            n.reference.reset_mock();(source/'model.def').write_text('original')
            (source/'reference.json').write_text(json.dumps({'timing_reference':'old_resident_reference'}))
            n.final_reference(root/'legacy',source,metrics);n.reference.assert_called_once()

    def test_partial_native_export_is_not_a_completed_cache_entry(self):
        from workflow_native import NativeWorkflow
        n=object.__new__(NativeWorkflow);n.server=Mock();n.threads=8;n.benchmark='aes'
        with TemporaryDirectory() as folder:
            path=Path(folder)
            for c in ('BC','TC','WC'):
                p=path/c/'placement/max/pins.tsv';p.parent.mkdir(parents=True);p.write_text('partial export')
            with patch('workflow_native.Server',side_effect=RuntimeError('re-export required')):
                with self.assertRaisesRegex(RuntimeError,'re-export required'):n.complete_reference(path)
            (path/'endpoint_export_check.json').write_text('{}')
            with patch('workflow_native.Server') as server:
                self.assertTrue(n.complete_reference(path)['reused_export']);server.assert_not_called()

    def make_workflow(self,error=0.,regression=False):
        from post_cts_workflow import PostCTSWorkflow
        w=object.__new__(PostCTSWorkflow);w.out=Path('/tmp/mock_search_policy');w.checkpoint_id=1
        w.config=WorkflowConfig();w.refresh_interval=8;w.batch_limit=128;w.rounds_since_reference=2
        w.epoch_moves=256;w.epoch_changed_nets={'n'};w.stalls=0
        row=dict(setup_tns_ns=-10.,setup_wns_ns=-.2,hold_tns_ns=0.,hold_wns_ns=0.,dynamic_power_w=1.,leakage_power_w=.1,
                 slew_violation_sum_ps=0.,cap_violation_sum_ff=0.,fanout_violation_sum=0.,global_routing_overflow=0.,max_global_routing_overflow=0.)
        w.baseline=w.reference_metrics=w.current={'TC':row}
        w.reference_snapshot=Path('/tmp/last_validated_lightweight_reference')
        w.native=Mock();w.native.reference.return_value=({'TC':dict(row,setup_tns_ns=-9. if not regression else -11.)},{})
        w.predictions={};w.report={'checkpoints':[]};w.save=Mock();w.build_models=Mock()
        w.calibration_error=lambda *args:{'TC':{'max':dict(mae_ps=error,p99_abs_ps=error)}}
        w.timed=lambda name,fn:fn()
        return w

    def test_passing_native_audit_reuses_anchor_without_full_export(self):
        w=self.make_workflow();self.assertTrue(w.checkpoint('test'))
        w.native.reference.assert_called_once_with(w.out/'reference_001',models=False,timing_only=True)
        w.native.complete_reference.assert_not_called();w.build_models.assert_not_called()
        self.assertFalse(w.report['checkpoints'][-1]['recalibrated'])
        self.assertEqual(w.rounds_since_reference,0);self.assertEqual(w.refresh_interval,9)

    def test_a_validated_single_gate_batch_can_grow_again(self):
        w=self.make_workflow();w.batch_limit=1
        self.assertTrue(w.checkpoint('recovery'));self.assertEqual(w.batch_limit,2)

    def test_drift_uses_saved_gr_then_reanchors(self):
        w=self.make_workflow(error=6.);self.assertTrue(w.checkpoint('drift'))
        self.assertEqual(w.native.reference.call_count,1)
        w.native.complete_reference.assert_called_once_with(w.out/'reference_001')
        w.build_models.assert_called_once();self.assertEqual(w.refresh_interval,4)
        self.assertTrue(w.report['checkpoints'][-1]['recalibrated'])

    def test_rollback_materializes_last_lightweight_reference(self):
        w=self.make_workflow(regression=True);snapshot=w.reference_snapshot
        self.assertFalse(w.checkpoint('reject'))
        w.native.restore.assert_called_once_with(snapshot)
        w.native.complete_reference.assert_called_once_with(snapshot)
        w.build_models.assert_called_once_with(snapshot,w.reference_metrics)

    def test_successful_native_erc_salvage_rebuilds_without_epoch_rollback(self):
        w=self.make_workflow();good=w.native.reference.return_value[0]
        w.native.reference.return_value=({'TC':dict(good['TC'],slew_violation_sum_ps=1.)},{})
        retry=w.out/'reference_002';record=dict(path=str(retry),native=good,accepted=True)
        def salvage(*args):
            w.report['checkpoints'].append(record);return retry,good,record
        w.salvage_reference=Mock(side_effect=salvage)
        self.assertTrue(w.checkpoint('erc'))
        w.native.restore.assert_not_called()
        w.native.complete_reference.assert_called_once_with(retry)
        w.build_models.assert_called_once_with(retry,good,reuse=False)

    def test_parallel_calls_preserve_plan_and_budget(self):
        lib=dp_library()
        def solve(_):
            parent=np.array([-1,0,0],dtype=np.int32)
            data=[np.array(x,dtype=np.float64) for x in
                ([0.,5.,5.],[0.,10.,10.],[np.inf,100.,100.],[1.],[.1],[1e-9],[[0.,100.]],[[3.,3.]])]
            r,cap,q,ic,area,power,loads,delays=data
            scores=np.empty(8);summary=np.empty((8,4));counts=np.empty(8,dtype=np.int32)
            nodes=np.empty((8,1),dtype=np.int32);buffers=np.empty_like(nodes)
            n=lib.workflow_buffer_dp(3,parent.ctypes.data,r.ctypes.data,cap.ctypes.data,q.ctypes.data,0,1,2,
                ic.ctypes.data,area.ctypes.data,power.ctypes.data,loads.ctypes.data,delays.ctypes.data,
                8,1,10.,0.,0.,scores.ctypes.data,summary.ctypes.data,counts.ctypes.data,nodes.ctypes.data,buffers.ctypes.data)
            self.assertGreater(n,0);self.assertLessEqual(int(counts[:n].max()),1)
            self.assertEqual(counts[0],1);self.assertGreater(scores[0],-150.)
            return scores[:n].copy(),nodes[0,0]
        expected=solve(0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for scores,node in pool.map(solve,range(12)):
                np.testing.assert_array_equal(scores,expected[0]);self.assertEqual(node,expected[1])


class CalibrationAndRCTests(unittest.TestCase):
    def test_batched_lca_matches_clock_paths_in_two_disjoint_trees(self):
        from workflow_clock import clock_lca_table
        rng=np.random.default_rng(32);n=300;parent=np.full(n,-1,dtype=int)
        for i in range(2,n):parent[i]=rng.integers(i)
        capture=np.arange(n);launch=np.r_[-1,np.arange(n)]
        table=clock_lca_table(parent,capture,launch,np.arange(n),block_size=17)
        for i in capture:
            chain=set();node=i
            while node>=0:chain.add(node);node=parent[node]
            for j,node in enumerate(launch):
                while node>=0 and node not in chain:node=parent[node]
                self.assertEqual(table[i,j],node)
    def test_disabled_constant_net_may_be_absent_but_active_net_may_not(self):
        from tree_rc_cuda import TreePlacementBridge
        m=SimpleNamespace(pins=[dict(net='tie')],arcs=[])
        engine=SimpleNamespace(m=m);forest=SimpleNamespace(networks={})
        with patch('tree_rc_cuda.PlacementGPUBridge',return_value=SimpleNamespace(external={'tie':np.array([.1,.1])})):
            bridge=TreePlacementBridge(forest,engine)
            self.assertEqual(bridge.unrouted_inactive_nets,['tie'])
            m.arcs=[dict(role='wire',s=0)]
            with self.assertRaisesRegex(ValueError,'Active externally loaded'):TreePlacementBridge(forest,engine)
    def test_native_spef_self_resistor_is_ignored_and_tree_cached(self):
        from placement_rc import Pin
        from workflow_rc import PlacementForest
        from tree_rc_cuda import pack_tree_reference
        pins=[Pin('in','net',0,0,'INPUT',port=True),Pin('g/A','net',1,0,'INPUT')]
        record=('net',{'a':'in','b':'g/A'},[('a','a',1e-6),('a','b',2.)],{'a':1.,'b':3.})
        with patch('workflow_rc.read_pins',return_value=pins),patch('workflow_rc.spef_networks',return_value=[record]):
            forest=PlacementForest('unused','unused');tree=forest.networks['net']
            self.assertEqual(tree.resistors,((0,1,2.),))
            pack_tree_reference(tree)
            result=forest.refresh('unused','unused')
            self.assertEqual(result['changed_nets'],0);self.assertIs(tree,forest.networks['net'])

    def test_native_rc_reordering_does_not_trigger_dirty_net_budget(self):
        from placement_rc import Pin
        from workflow_rc import PlacementForest
        pins=[Pin('in','net',0,0,'INPUT',port=True),Pin('g/A','net',1,0,'INPUT')]
        first=('net',{'a':'in','b':'g/A'},[('a','b',2.)],{'a':1.,'b':3.})
        reordered=('net',{'b':'g/A','a':'in'},[('b','a',2.)],{'b':3.,'a':1.})
        with patch('workflow_rc.read_pins',return_value=pins),patch('workflow_rc.spef_networks',side_effect=[[first],[reordered]]):
            forest=PlacementForest('unused','unused');tree=forest.networks['net']
            self.assertEqual(forest.refresh('unused','unused')['changed_nets'],0)
            self.assertIs(forest.networks['net'],tree)

    @patch('resident_cuda_sta.ResidentCudaSTA.sync',lambda self:None)
    def test_reanchor_reuses_graph_without_reference_rc(self):
        from test_autograd_sta import model
        from resident_cuda_sta import ResidentCudaSTA
        from workflow_sta import reanchor_electrical
        m=model();names=['clk','a','b','g/Y','h/A','h/Y']
        m.pins=[dict(name=n) for n in names];m.index={n:i for i,n in enumerate(names)};m.analysis='max'
        m.calibration_mode='arc_offset';m.arc_delay_offset=np.zeros(len(m.arcs))
        m.at_ratio[:]=1.;m.at_offset[:]=0.
        for a in m.arcs+m.checks:a['cond']=a.get('variant','');a['delay']=0.
        engine=ResidentCudaSTA(m,device='cpu');plan=engine.plan
        changed=copy.deepcopy(m);changed.tables=m.tables
        changed.arc_delay_offset[:]=.3;changed.slew_ratio*=1.05
        changed.ref_at[:3]+=.5
        expected=ResidentCudaSTA(changed,device='cpu').forward()
        native=copy.deepcopy(changed);native.ref_at=expected['at'];native.ref_slew=expected['slew']
        for a,d in zip(native.arcs,expected['delay']):a['delay']=d
        for a,d in zip(native.checks,expected['check']):a['delay']=d
        order=np.array([0,2,1,3,4,5]);inverse=np.argsort(order)
        native.pins=[native.pins[i] for i in order];native.index={p['name']:i for i,p in enumerate(native.pins)}
        native.ref_at=native.ref_at[order];native.ref_slew=native.ref_slew[order];native.level=native.level[order]
        for a in native.arcs+native.checks:a['s']=int(inverse[a['s']]);a['t']=int(inverse[a['t']])
        native.arcs.reverse();native.checks.reverse()
        reanchor_electrical(engine,native);actual=engine.forward()
        self.assertIs(engine.plan,plan)
        for key in ('at','slew','delay','check'):np.testing.assert_allclose(actual[key],expected[key],atol=1e-11)


if __name__=='__main__':unittest.main()
