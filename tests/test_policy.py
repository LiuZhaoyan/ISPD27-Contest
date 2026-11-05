import unittest
import numpy as np
from policy import gpu_topk,contest_neighbors,contest_targets

class PaperPolicy(unittest.TestCase):
    def test_full_family_contains_long_and_joint_moves_once(self):
        cells=[]
        for strength in (1,2,4):
            for vt,theta in [('L',1),('R',2)]:
                cells.append(dict(id=len(cells),name=f'AND2x{strength}_ASAP7_75t_{vt}',
                    family_id=0,kappa=strength,theta=theta,frozen_reason=None,
                    liberty_data={'timing_arc_signatures':['A/Y']}))
        catalog=dict(cells=cells,families=[dict(cell_ids=list(range(6)))])
        targets=contest_targets(catalog)
        for i,row in enumerate(targets):
            values=[v for v in row if v>=0]
            self.assertEqual(set(values),set(range(6))-{i})
            self.assertEqual(len(values),len(set(values)))
        self.assertNotIn(5,contest_neighbors(catalog)[0])
        self.assertIn(5,targets[0])  # x1 LVT -> x4 RVT in one legal proposal.
        cells[5]['frozen_reason']='dont_use'
        self.assertNotIn(5,contest_targets(catalog)[0])
    def test_prohibited_targets_are_excluded_before_finding_neighbors(self):
        cells=[dict(id=i,name=n,family_id=0,kappa=k,theta=t,frozen_reason=None) for i,(n,k,t) in enumerate([
            ('NOR2x1_ASAP7_75t_R',0,2),('NOR2x1p5_ASAP7_75t_R',1,2),
            ('NOR2x2_ASAP7_75t_R',2,2),('NOR2x1_ASAP7_75t_L',0,1),
            ('NOR2xp5_ASAP7_75t_R',-1,2),('NOR2xp5_ASAP7_75t_L',-1,1)])]
        neighbors=contest_neighbors(dict(cells=cells,families=[dict(cell_ids=list(range(6)))]))
        self.assertEqual(neighbors[0].tolist(),[2,-1,-1,3])
        # Existing forbidden cells can be resized into an allowed master; VT
        # swaps to another forbidden cell remain forbidden.
        self.assertEqual(neighbors[4].tolist(),[0,-1,-1,-1])
    def test_topk_masks_invalid_actions_and_deduplicates_gates(self):
        scores=np.array([[100,3,-1,np.nan],[8,7,6,5],[4,2,1,0.]])
        mask=np.ones((3,4),dtype=bool);mask[0,0]=False
        # Algorithm 1 first selects actions, then removes duplicate gates.
        self.assertEqual(gpu_topk(scores,mask,4,device='cpu'),[(1,0)])
        self.assertEqual(gpu_topk(scores,mask,6,device='cpu'),[(1,0),(2,0),(0,1)])

if __name__=='__main__':unittest.main()
