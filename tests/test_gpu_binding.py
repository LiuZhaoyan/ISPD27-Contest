import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from forward_sta import Table
from resident_cuda_sta import ResidentCudaSTA


class BindingUpdate(unittest.TestCase):
    def test_changed_master_and_unchanged_master_calibration(self):
        x=np.array([0.,100.])
        def tables(d0,dx,dc,s0,sx,sc,q):
            d=Table(x,x,d0+dx*x[:,None]+dc*x[None,:])
            s=Table(x,x,s0+sx*x[:,None]+sc*x[None,:])
            check=Table(x,x,np.full((2,2),q))
            return dict(cell_rise=d,cell_fall=d,rise_transition=s,fall_transition=s,
                        rise_constraint=check,fall_constraint=check)
        bank={'A':tables(1.,.1,.2,1.,.2,.1,1.),'B':tables(4.,.4,.5,2.,.3,.2,3.)}
        arcs=[dict(s=0,t=1,fr=rf,to=rf,role='wire') for rf in (0,1)]
        arcs += [dict(s=1,t=2,fr=rf,to=rf,role='combinational') for rf in (0,1)]
        checks=[dict(s=0,t=1,fr=rf,to=rf,role='setup') for rf in (0,1)]
        m=SimpleNamespace(n=3,level=np.array([0,1,2]),incoming=[[],[0,1],[2,3]],arcs=arcs,checks=checks,
            pin_parts=[(None,'x'),('g','A'),('g','Y')],by_net={'a':[0,1],'b':[2]},tables=bank,
            rc_matrix=csr_matrix((4,6)),rf_sink=np.array([False,True,False]),index={'out':2},
            endpoints=[dict(pin='out',rf='rise',role='output setup',required_ps='0')],
            at_ratio=np.ones((3,2)),at_offset=np.zeros((3,2)),slew_ratio=np.ones((3,2)),
            slew_offset=np.zeros((3,2)),check_ratio=np.array([.8,.9]),check_offset=np.array([.1,.2]),
            ref_at=np.zeros((3,2)),ref_slew=np.array([[2.,3.],[0.,0.],[0.,0.]]),
            rf_base_load=np.array([[4.,5.],[4.,5.],[6.,7.]]),rc_base_tau=np.array([1.,1.,0.,0.]),masters={'g':'A'})
        m.table=lambda a,masters:bank[masters['g']]
        with patch.object(ResidentCudaSTA,'sync',lambda self:None):
            e=ResidentCudaSTA(m,device='cpu')
            # Changing the same model object must not hide the previous master.
            m.masters['g']='B';e.set_model(m);f=e.forward()
            wire=np.hypot([2.,3.],2.197224577)
            arrival=1.+4.+.4*wire+.5*np.array([6.,7.])
            np.testing.assert_allclose(f['at'][2],arrival,rtol=1e-13)
            np.testing.assert_allclose(f['slew'][2],2.+.3*wire+.2*np.array([6.,7.]),rtol=1e-13)
            np.testing.assert_allclose(f['check'],[2.5,2.9],rtol=1e-13)
            m.at_ratio[2]=1.2;e.set_model(m)
            np.testing.assert_allclose(e.forward()['at'][2],arrival*1.2,rtol=1e-13)
            m.masters['g']='A';e.set_model(m);f=e.forward()
            np.testing.assert_allclose(f['at'][2],(1.+1.+.1*wire+.2*np.array([6.,7.]))*1.2,rtol=1e-13)
            np.testing.assert_allclose(f['check'],[.9,1.1],rtol=1e-13)
            # Geometry refresh must update both baseline delay and the sparse
            # pin-C derivative used by subsequent sizing and reverse passes.
            new_rc=csr_matrix(([.5,.7],([0,1],[2,3])),shape=(4,6))
            new_load=np.array([[5.,6.],[5.,6.],[8.,9.]])
            e.update_parasitics(new_load,np.array([2.,3.,0.,0.]),new_rc)
            refreshed=e.forward()
            wire=np.hypot([2.,3.],2.197224577*np.array([2.,3.]))
            np.testing.assert_allclose(refreshed['at'][2],
                (np.array([2.,3.])+1.+.1*wire+.2*np.array([8.,9.]))*1.2,rtol=1e-13)
            np.testing.assert_allclose(e.rc.to_dense().numpy(),new_rc.toarray())
            np.testing.assert_allclose(e.rc_t.to_dense().numpy(),new_rc.T.toarray())
            self.assertFalse(e.prepared)
            # Invalid refreshes must leave the prior valid working point intact.
            with self.assertRaises(ValueError):e.update_parasitics(new_load,np.array([-1.,3.,0.,0.]),new_rc)
            np.testing.assert_allclose(e.forward()['at'],refreshed['at'])


if __name__=='__main__':unittest.main()
