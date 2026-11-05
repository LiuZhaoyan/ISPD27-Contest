import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.sparse import csr_matrix
from forward_sta import Table
from resident_cuda_sta import ResidentCudaSTA
from power_cuda import PowerCuda


class PropagatedPower(unittest.TestCase):
    def test_internal_seeds_match_energy_finite_difference(self):
        # Different table dimensions exercise padding and extrapolation.
        a=Table([0.,10.],[0.,2.,5.],np.array([[1.,3.,6.],[21.,33.,51.]])*1e-15)
        b=Table([0.,3.,10.],[0.,5.],np.array([[2.,7.],[11.,31.],[32.,92.]])*1e-15)
        tabs={'rise':a,'fall':b}
        factors=[(0,1,False,2e9,tabs)]
        p=SimpleNamespace(tables={'c':{'pins':{'Y':{'powers':[{'tables':tabs}]}}}},
                          inst_pins={'g':{}},model=SimpleNamespace(masters={'g':'c'}),
                          factors=lambda i,m:(factors,[(1,4e-6)],1e-7))
        sl=np.array([[12.,4.],[2.,3.]]);load=np.array([[0.,0.],[3.,1.]])
        e=SimpleNamespace(device='cpu',m=SimpleNamespace(n=2),_sl=torch.tensor(sl.reshape(-1)),_load=torch.tensor(load),
                          tensor=lambda x,integer=False:torch.tensor(x,dtype=torch.int64 if integer else torch.float64),
                          zeros=lambda n:torch.zeros(n,dtype=torch.float64),numpy=lambda x:x.numpy())
        gpu=PowerCuda(p,e);gs,gc=gpu.internal_partials()
        def energy(s,l):return 1e9*(a(s[0,1],max(l[1]))+b(s[0,0],max(l[1])))
        total=gpu.total(dict(slew=sl,load=load.max(1)))
        self.assertAlmostEqual(total['internal_w'],energy(sl,load),places=12)
        self.assertAlmostEqual(total['switching_w'],4e-6*max(load[1]),places=12)
        self.assertAlmostEqual(total['total_w'],energy(sl,load)+4e-6*max(load[1])+1e-7,places=12)
        self.assertEqual(gpu.total(dict(slew=sl,load=load.max(1)),instances=['g']),total)
        self.assertEqual(gpu.total(dict(slew=sl,load=load.max(1)),instances=[])['total_w'],0.)
        h=1e-4
        for array,grad in ((sl,gs),(load,gc)):
            for ix in np.ndindex(array.shape):
                old=array[ix];array[ix]=old+h;plus=energy(sl,load)
                array[ix]=old-h;minus=energy(sl,load);array[ix]=old
                self.assertAlmostEqual((plus-minus)/(2*h),grad[ix],places=12)

    def test_propagated_slew_and_load_vjp_matches_full_forward(self):
        x=np.array([0.,100.]);delay=Table(x,x,np.ones((2,2)))
        transition=Table(x,x,2.+.4*x[:,None]+.3*x[None,:])
        tabs={'cell_rise':delay,'cell_fall':delay,'rise_transition':transition,'fall_transition':transition}
        # Input pin 1 is driven through RC; its input cap also loads cell output 0.
        arcs=[dict(s=0,t=1,fr=rf,to=rf,role='wire') for rf in (0,1)]
        arcs += [dict(s=1,t=2,fr=rf,to=rf,role='combinational') for rf in (0,1)]
        rc=np.zeros((4,6));rc[0,2]=.2;rc[1,3]=.2
        m=SimpleNamespace(n=3,level=np.array([0,1,2]),incoming=[[],[0,1],[2,3]],arcs=arcs,checks=[],
            pin_parts=[(None,str(i)) for i in range(3)],by_net={'a':[0,1],'b':[2]},tables={'t':tabs},
            rc_matrix=csr_matrix(rc),rf_sink=np.array([False,True,False]),index={'out':2},
            endpoints=[dict(pin='out',rf='rise',role='output setup',required_ps='10')],
            at_ratio=np.ones((3,2)),at_offset=np.zeros((3,2)),slew_ratio=np.array([[1.,1.],[.8,.9],[1.2,1.3]]),
            slew_offset=np.zeros((3,2)),check_ratio=np.empty(0),check_offset=np.empty(0),
            ref_at=np.zeros((3,2)),ref_slew=np.array([[2.,3.],[0.,0.],[0.,0.]]),
            rf_base_load=np.array([[4.,5.],[4.,5.],[6.,7.]]),rc_base_tau=np.array([1.,1.,0.,0.]),masters={})
        m.table=lambda a,masters:tabs
        with patch.object(ResidentCudaSTA,'sync',lambda self:None):
            e=ResidentCudaSTA(m,device='cpu');f=e.prepare()
            # Synthetic internal energy depends on all downstream slews/load.
            seed=np.array([[0.,0.],[.2,.3],[.7,.8]])
            loadseed=np.array([[.1,.2],[.3,.4],[.5,.6]])
            g=e.backward_power(seed,loadseed)
            def power():
                f=e.forward();return float((f['slew']*seed).sum()+(f['load_rf']*loadseed).sum())
            h=1e-4
            # GPU table bank is a copy; perturb the transition bank directly.
            tid=e.table_ids[id(transition)];bank=e.z[tid].clone()
            e.z[tid]=bank+h;plus=power();e.z[tid]=bank-h;minus=power();e.z[tid]=bank
            self.assertAlmostEqual((plus-minus)/(2*h),g['local_slew'][2:].sum(),places=7)
            load=e.load0.clone();tau=e.tau0.clone()
            # Increase both RF input capacitances by h with the RC mapping.
            dc=e.tensor([[0.,0.],[h,h],[0.,0.]])
            for sign in (1.,-1.):
                e.load0=load+sign*e.tensor([[h,h],[h,h],[0.,0.]])
                e.tau0=tau+sign*torch.sparse.mm(e.rc,dc.reshape(-1,1)).reshape(-1)
                if sign==1:plus=power()
                else:minus=power()
            e.load0=load;e.tau0=tau
            self.assertAlmostEqual((plus-minus)/(2*h),g['pin_load'][1].sum(),places=7)


if __name__=='__main__':unittest.main()
