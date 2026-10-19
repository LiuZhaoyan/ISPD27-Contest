"""Public numerical/action/acceptance contracts from implementation_spec.md."""
import unittest
from pathlib import Path
import csv
import json
import tempfile
from unittest.mock import patch
import optimizer
from actions import coordinate, forbidden
from model import Adjoint, lookup, ave, tc_loss
from optimizer import acceptance_failures, guards


class NumericalContracts(unittest.TestCase):
    def test_loss_sign_clipping_and_inside_epsilon(self):
        self.assertAlmostEqual(tc_loss((-1.,-.1),(-2.,-.2),.01), 1/1.99+.1/.19)
        self.assertEqual(tc_loss((0.,.1),(-2.,-.2)),0.)
        self.assertLess(tc_loss((-1.,-.1),(-2.,-.2)),tc_loss((-2.,-.2),(-2.,-.2)))

    def test_lookup_and_extrapolation(self):
        # f(x,y)=2x+3y+xy, independent hand-worked values.
        table=(([0.,2.],[0.,4.]),[[0.,12.],[4.,24.]])
        self.assertEqual(lookup(table,1.,2.),(10.,4.,4.))
        self.assertEqual(lookup(table,-1.,2.),(2.,4.,2.))

    def test_negative_and_zero_arrival_neighbors(self):
        self.assertEqual(ave([-2.,-2.01,-3.]),[.5,.5,0.])
        self.assertEqual(ave([0.,-1.]),[1.,0.])

    def test_sparse_coordinates_do_not_merge_structural_families(self):
        self.assertNotEqual(coordinate('BUFx2_ASAP7_75t_R')[0],coordinate('BUF-fx2_ASAP7_75t_R')[0])
        self.assertEqual(coordinate('INVx1p5_ASAP7_75t_L'),('INV',1.5,1))
        self.assertTrue(forbidden('INVx1p5_ASAP7_75t_L'))
        self.assertTrue(forbidden('INVxp67_ASAP7_75t_L'))
        self.assertFalse(forbidden('INVx2_ASAP7_75t_L'))

    def test_lookup_partials_match_local_differences(self):
        axes=([0.,1.,3.],[0.,2.,5.])
        table=(axes,[[x*x+y*y+x*y for y in axes[1]] for x in axes[0]])
        for x,y in ((.4,.8),(1.,2.),(-1.,6.)):
            _,dx,dy=lookup(table,x,y)
            for axis,derivative in enumerate((dx,dy)):
                with self.subTest(point=(x,y),axis=axis):
                    plus=[x,y]; minus=[x,y]; h=1e-5
                    plus[axis]+=h
                    at_knot=minus[axis] in axes[axis]
                    if not at_knot: minus[axis]-=h
                    numerical=(lookup(table,*plus)[0]-lookup(table,*minus)[0])/(h if at_knot else 2*h)
                    self.assertAlmostEqual(derivative,numerical,places=6)

    def test_adjoint_setup_seed_and_three_contributions(self):
        class Library:
            def table(self,sample,kind,cell=None):
                if sample['kind']=='setup': c,x,y=0.,-.5,1.
                elif sample['role']=='Reg Clk to Q':
                    c,x,y=(1.,0.,2.) if kind=='cell_rise' else (1.,0.,3.)
                else: c,x,y=(2.,4.,5.) if kind=='cell_rise' else (3.,6.,7.)
                if cell=='candidate': c+=1. if kind=='cell_rise' else 2.
                return (([0.,2.],[0.,2.]),[[c,c+2*y],[c+2*x,c+2*x+2*y]])
            def cap(self,cell,pin,transition): return 1.5 if cell=='candidate' else 1.
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            def table(name,columns,records):
                with (folder/name).open('w') as stream:
                    writer=csv.writer(stream,delimiter='\t'); writer.writerow(columns.split()); writer.writerows(records)
            table('paths.tsv','path pin transition arrival_ps slew_ps slack_ps endpoint',
                  [(0,p,'rise',i,1.,-100.,'ff/D') for i,p in enumerate(['launch/Q','gate/A','gate/Y','ff/D'])])
            records=[]
            for i,(kind,a,b,role,delay,slew) in enumerate([
                ('load','launch/CLK','launch/Q','Reg Clk to Q',3.,4.),
                ('load','gate/A','gate/Y','combinational',11.,16.),
                ('setup','ff/CLK','ff/D','setup',.5,0.)]):
                report=f'{i}.rpt'
                (folder/report).write_text(f'input_net_transition = 1\ntotal_output_net_capacitance = 1\nconstrained_pin_transition = 1\nrelated_pin_transition = 1\nDelay = {delay}\nSlew = {slew}\nCheck = .5\n')
                records.append((kind,str(i),a,b,role,'rise','rise','current',delay*1e-12,report))
            table('arcs.tsv','kind sample_id from_pin to_pin role from_transition to_transition liberty_cell graph_delay_s report_file',records)
            table('snapshot.pins.tsv','instance pin direction net',
                  [('launch','Q','OUTPUT','net'),('gate','A','INPUT','net')])
            model=Adjoint(folder,Library(),(-1.,-.1))
            scored=model.score(dict(instance='gate',target='candidate',current='current'))
            seed=1/(1000*abs(-1.+1e-9))+1/(1000*abs(-.1+1e-9))
            self.assertAlmostEqual(scored['own_delay'],seed)
            self.assertAlmostEqual(scored['downstream_slew'],-seed)
            self.assertAlmostEqual(scored['upstream_load'],2.5*seed)
            self.assertAlmostEqual(scored['benefit'],-2.5*seed)

    def test_hold_penalty_not_raw_hold_is_guarded(self):
        row=dict(hold_tns_ns=-.2,hold_wns_ns=-.004,slew_violation_sum_ps=0.,
                 cap_violation_sum_ff=0.,fanout_violation_sum=0.,global_routing_overflow=0.,
                 max_global_routing_overflow=0.)
        base={s:dict(row) for s in ('BC','TC','WC')}
        candidate={s:dict(row) for s in base}
        config={'scenarios':{s:dict(hold_tns_tolerance_ns=1.,hold_wns_tolerance_ns=.01) for s in base}}
        ref=(-1.,-.1)
        for scene in base:
            base[scene].update(setup_tns_ns=-1.,setup_wns_ns=-.1)
            candidate[scene].update(setup_tns_ns=-.9,setup_wns_ns=-.09)
        loss,failures=acceptance_failures(base,candidate,config,tc_loss(ref,ref),ref,1e-7)
        self.assertFalse(failures)
        self.assertLess(loss,tc_loss(ref,ref))
        self.assertIn('fresh_TC_loss_not_improved',acceptance_failures(base,base,config,tc_loss(ref,ref),ref,1e-7)[1])
        candidate['TC']['hold_tns_ns']=-.8
        candidate['TC']['hold_wns_ns']=-.009
        self.assertEqual(guards(base,candidate,config),[])
        candidate['TC']['hold_wns_ns']=-.011
        self.assertIn('TC:hold_wns',guards(base,candidate,config))
        candidate['BC']['cap_violation_sum_ff']=.01
        candidate['WC']['global_routing_overflow']=1.
        self.assertIn('BC:cap_violation_sum_ff',guards(base,candidate,config))
        self.assertIn('WC:global_routing_overflow',guards(base,candidate,config))


class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.input=self.root/'benchmarks/aes'
        self.platform=self.root/'platform/asap7'
        for folder in (self.input,self.platform,self.root/'evaluator/config'):
            folder.mkdir(parents=True)
        self.config={'scenarios':{s:dict(hold_tns_tolerance_ns=1.,hold_wns_tolerance_ns=.01)
                                  for s in ('BC','TC','WC')}}
        (self.root/'evaluator/config/scoring_config.json').write_text(json.dumps(self.config))
        tool=self.root/'tool'
        tool.write_text('fixture')
        self.output=self.root/'output'
        self.scores={}
        self.outcomes={}
        self.summaries={}
        self.probes=[]
        self.evaluations=[]
        owner=self

        class Runner:
            def __init__(self,workspace,benchmark): self.workspace=workspace
            def command(self,*args,**kwargs): pass
            def evaluate(self,name,deffile,verilog):
                owner.evaluations.append(name)
                folder=self.workspace/name
                folder.mkdir()
                summary=(owner.summary() if name=='r0' else
                         owner.outcomes[json.loads(deffile.with_name('action.json').read_text())['target']])
                owner.summaries[folder]=summary
                for filename in ('evaluated.def','evaluated.v'):
                    (folder/filename).write_text(name+':'+filename)
                return folder
            def probe(self,baseline,name,action=None,graph=False):
                owner.probes.append((name,baseline))
                folder=self.workspace/name
                folder.mkdir()
                summary=owner.summary(*action['proxy']) if action else owner.summaries[baseline]
                with (folder/'native.tsv').open('w') as f:
                    writer=csv.writer(f,delimiter='\t')
                    writer.writerow(['scene','analysis','tns_ps','wns_ps'])
                    for scene,row in summary.items():
                        for analysis,prefix in (('max','setup'),('min','hold')):
                            writer.writerow([scene,analysis,row[prefix+'_tns_ns']*1000,row[prefix+'_wns_ns']*1000])
                if action:
                    optimizer.save(folder/'action.json',action)
                    for filename in ('candidate.def','candidate.v'): (folder/filename).write_text(name)
                return folder

        patches=patch.multiple(optimizer,ROOT=self.root,Runner=Runner,
                               analyze=lambda folder,*args:(self.scores[folder.name],{'approximations':[]}),
                               read_summary=lambda path:('fixture',self.summaries[path.parent]))
        patches.start()
        self.addCleanup(patches.stop)
        tools=patch.object(optimizer.shutil,'which',return_value=str(tool))
        tools.start()
        self.addCleanup(tools.stop)

    def summary(self,tns=-1.,wns=-.1,hold_wns=-.004):
        return {s:dict(setup_tns_ns=tns,setup_wns_ns=wns,hold_tns_ns=-.2,hold_wns_ns=hold_wns,
                       slew_violation_sum_ps=0.,cap_violation_sum_ff=0.,fanout_violation_sum=0.,
                       global_routing_overflow=0.,max_global_routing_overflow=0.) for s in ('BC','TC','WC')}

    def action(self,target,proxy=(-.9,-.09)):
        return dict(instance='u1',target=target,direction='vt_down',benefit=.1,proxy=proxy)

    def optimize(self,steps=1):
        optimizer.optimize(self.input,self.platform,self.output,'aes_cipher_top',steps)

    def test_runtime_only_accepts_fresh_guarded_checkpoints_and_refreshes_for_next_step(self):
        self.scores={'initial-graph':[self.action('bad'),self.action('good')],
                     'refresh-1':[self.action('better',(-.8,-.08))]}
        self.outcomes={'bad':self.summary(-.9,-.09,hold_wns=-.02),
                       'good':self.summary(-.9,-.09),'better':self.summary(-.8,-.08)}
        self.optimize(steps=2)
        self.assertEqual([name for name,_ in self.probes],
                         ['initial-graph','step-0-trial-0','step-0-trial-1','refresh-1','step-1-trial-0'])
        self.assertEqual(self.evaluations,['r0','step-0-trial-0-fresh','step-0-trial-1-fresh','step-1-trial-0-fresh'])
        self.assertEqual(self.probes[3][1].name,'step-0-trial-1-fresh')
        self.assertEqual((self.output/'aes.def').read_text(),'step-1-trial-0-fresh:evaluated.def')
        self.assertEqual((self.output/'aes.v').read_text(),'step-1-trial-0-fresh:evaluated.v')
        events=json.loads((self.output/'quad_gradient/updates.json').read_text())
        self.assertEqual([e['accepted'] for e in events],[False,True,True])
        self.assertIn('TC:hold_wns',events[0]['guard_failures'])
        self.assertEqual(events[0]['before_checkpoint'],events[0]['after_checkpoint'])
        manifest=json.loads((self.output/'quad_gradient/manifest.json').read_text())
        self.assertEqual(manifest['status'],'candidate_exported')
        self.assertNotIn('validation_passed',manifest)

    def test_proxy_gain_cannot_export_when_fresh_loss_does_not_improve(self):
        self.scores={'initial-graph':[self.action('bad')]}
        self.outcomes={'bad':self.summary()}
        with self.assertRaisesRegex(RuntimeError,'No physically accepted improvement'):
            self.optimize()
        events=json.loads((self.output/'quad_gradient/updates.json').read_text())
        self.assertIn('fresh_TC_loss_not_improved',events[0]['guard_failures'])
        self.assertEqual(events[0]['before_checkpoint'],events[0]['after_checkpoint'])
        self.assertFalse((self.output/'aes.def').exists())
        self.assertFalse((self.output/'aes.v').exists())


if __name__=='__main__': unittest.main()
