"""Independent source-class and launch-latency examples, including CUDA VJP."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from placement_rc import Pin
from source_group_sta import SourceGroupPlan,CudaSourceGroups
from cuda_sta import CudaSTA


def example():
    arcs=[dict(s=0,t=2,fr=0,to=0,role='wire'),dict(s=0,t=2,fr=1,to=1,role='wire'),
          dict(s=2,t=3,fr=0,to=0,role='Reg Clk to Q'),dict(s=3,t=4,fr=0,to=0,role='wire'),
          dict(s=1,t=4,fr=0,to=0,role='combinational')]
    names=['clk','input','ff/CLK','ff/Q','out'];incoming=[[],[],[0,1],[2],[3,4]]
    m=SimpleNamespace(n=5,pins=[dict(name=n) for n in names],index={n:i for i,n in enumerate(names)},
        arcs=arcs,incoming=incoming,level=np.array([0,0,1,2,3]),ref_at=np.array([[0.,10.],[0.,0.],[5.,15.],[7.,-np.inf],[10.,-np.inf]]))
    geometry=[Pin('clk','clk',0,0,'INPUT',clock=True,port=True),Pin('input','in',0,0,'INPUT',port=True)]
    ends=[dict(pin='out',rf='rise',source_group='input',is_path_delay='0',capture_pin='ff/CLK'),
          dict(pin='out',rf='rise',source_group='clock_rise',is_path_delay='0',capture_pin='ff/CLK'),
          dict(pin='out',rf='rise',source_group='clock_rise',is_path_delay='1',capture_pin='')]
    return m,geometry,ends,np.array([5.,5.,2.,1.,10.])


class SourceGroupTests(unittest.TestCase):
    def test_minimum_classes_keep_clock_path_separate_from_faster_input(self):
        m,geo,ends,d=example();m.analysis='min'
        p=SourceGroupPlan(m,geo,ends);at=p.forward(d)
        # The same graph supports max and min semirings. A faster input
        # branch must not replace the independently selected launch class.
        d[4]=1.;after=p.forward(d)
        np.testing.assert_allclose(after[p.ep_group,p.ep_node],[1,8,3])
        d[0]+=4.;changed=p.forward(d)
        np.testing.assert_allclose(changed[p.ep_group,p.ep_node],[1,12,3])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_minimum_cuda_classes_match_scalar_and_reject_max_reverse(self):
        m,geo,ends,d=example();m.analysis='min';p=SourceGroupPlan(m,geo,ends)
        e=CudaSTA.__new__(CudaSTA);e.device='cuda:0';e.m=m
        runtime=CudaSourceGroups(p,e);at=runtime.forward(e.tensor(d))
        np.testing.assert_allclose(e.numpy(at),p.forward(d))
        with self.assertRaises(NotImplementedError):runtime.delay_vjp(at,e.tensor(d),e.tensor([1.,2.,3.]))

    def test_independent_input_register_and_normalized_arrivals(self):
        m,geo,ends,d=example();p=SourceGroupPlan(m,geo,ends);at=p.forward(d)
        np.testing.assert_allclose(at[p.ep_group,p.ep_node],[10,8,3])
        d[2]+=5;after=p.forward(d)
        np.testing.assert_allclose(after[p.ep_group,p.ep_node],[10,13,8])
        self.assertEqual(max(after[:,8])-max(at[:,8]),3)
        self.assertEqual(after[p.labels.index('clock_rise'),8]-at[p.labels.index('clock_rise'),8],5)

    def test_normalized_path_excludes_clock_wire_changes(self):
        m,geo,ends,d=example();p=SourceGroupPlan(m,geo,ends)
        d[0]=20;at=p.forward(d)
        np.testing.assert_allclose(at[p.ep_group,p.ep_node],[10,23,3])

    def test_multiple_clock_roots_rejected(self):
        m,geo,ends,d=example();geo[1]=Pin('input','in',0,0,'INPUT',clock=True,port=True)
        with self.assertRaisesRegex(ValueError,'exactly one'):SourceGroupPlan(m,geo,ends)

    def test_falling_launch_phase_and_latency_normalization(self):
        m,geo,ends,d=example();m.arcs[2]['fr']=1
        for row in ends:
            if row['source_group']=='clock_rise':row['source_group']='clock_fall'
        p=SourceGroupPlan(m,geo,ends);at=p.forward(d)
        np.testing.assert_allclose(at[p.ep_group,p.ep_node],[10,18,3])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_graph_replay_keeps_updated_roots_and_delays(self):
        m,geo,ends,d=example();p=SourceGroupPlan(m,geo,ends)
        e=CudaSTA.__new__(CudaSTA);e.device='cuda:0';e.m=m
        runtime=CudaSourceGroups(p,e);delay=e.tensor(d)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):runtime.forward(delay)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):at=runtime.forward(delay)
        delay[0]=7.;updated=m.ref_at.copy();updated[1]=4.;runtime.update_roots(updated)
        graph.replay();np.testing.assert_allclose(e.numpy(at[runtime.ep_group,runtime.ep_node]),[14,10,3])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_forward_vjp_and_autograd(self):
        m,geo,ends,d=example();p=SourceGroupPlan(m,geo,ends)
        e=CudaSTA.__new__(CudaSTA);e.device='cuda:0';e.m=m
        e.bundle=e.tensor(np.arange(len(d)),True);e.bt=e.tensor(p.t,True);e.nb=len(d)
        runtime=CudaSourceGroups(p,e);delay=e.tensor(d);at=runtime.forward(delay)
        np.testing.assert_allclose(e.numpy(at),p.forward(d))
        seeds=e.tensor([1.,2.,3.]);gradient=runtime.delay_vjp(at,delay,seeds)
        np.testing.assert_allclose(e.numpy(gradient),[2.,0.,5.,5.,1.])
        varying=delay.clone().requires_grad_();functional=runtime.forward(varying,functional=True)
        loss=(functional[runtime.ep_group,runtime.ep_node]*seeds).sum();loss.backward()
        np.testing.assert_allclose(e.numpy(gradient),e.numpy(varying.grad))
        eps=1e-5;finite=[]
        for k in range(len(d)):
            hi=d.copy();lo=d.copy();hi[k]+=eps;lo[k]-=eps
            delta=p.forward(hi)[p.ep_group,p.ep_node]-p.forward(lo)[p.ep_group,p.ep_node]
            finite.append(np.dot(delta,np.array([1.,2.,3.]))/(2*eps))
        np.testing.assert_allclose(e.numpy(gradient),finite,atol=1e-8)


if __name__=='__main__':unittest.main()
