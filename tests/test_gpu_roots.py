import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from forward_sta import Table
from resident_cuda_sta import ResidentCudaSTA

class DisabledArcRoots(unittest.TestCase):
    def test_native_nonzero_level_root_keeps_reference_without_double_offset(self):
        # Pin 1 becomes a root after disabled arcs are removed, but native
        # levelization retains level 1. A wire to pin 2 shares that level.
        # Root state is already calibrated and must never get its offset again.
        m=SimpleNamespace(n=3,level=np.array([0,1,1]),incoming=[[],[],[0,1]],
            arcs=[dict(s=0,t=2,fr=rf,to=rf,role='wire') for rf in (0,1)],checks=[],
            pin_parts=[(None,str(i)) for i in range(3)],by_net={str(i):[i] for i in range(3)},
            tables={'unused':{'unused':Table([0.,1.],[0.,1.],np.zeros((2,2)))}},
            rc_matrix=csr_matrix((2,6)),rf_sink=np.zeros(3,dtype=bool),index={'out':2},
            endpoints=[dict(pin='out',rf='rise',role='output setup',required_ps='10')],
            at_ratio=np.ones((3,2)),at_offset=np.array([[0.,0.],[3.,3.],[0.,0.]]),
            slew_ratio=np.ones((3,2)),slew_offset=np.array([[0.,0.],[4.,4.],[0.,0.]]),
            check_ratio=np.empty(0),check_offset=np.empty(0),
            ref_at=np.array([[1.,1.],[3.,3.],[3.,3.]]),
            ref_slew=np.array([[2.,2.],[4.,4.],[5.,5.]]),rf_base_load=np.zeros((3,2)),
            rc_base_tau=np.array([2.,2.]),masters={})
        # CPU torch runs the same vectorized plan; synchronization is the only
        # CUDA-only operation. Expected root values are independent constants.
        with patch.object(ResidentCudaSTA,'sync',lambda self:None):
            result=ResidentCudaSTA(m,device='cpu').forward()
        np.testing.assert_allclose(result['at'][1],[3.,3.])
        np.testing.assert_allclose(result['slew'][1],[4.,4.])
        np.testing.assert_allclose(result['at'][2],[3.,3.])

if __name__=='__main__':unittest.main()
