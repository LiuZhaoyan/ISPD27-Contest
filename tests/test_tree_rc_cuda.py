from types import SimpleNamespace
import unittest
import numpy as np
import torch
from placement_rc import Pin,Network
from tree_rc_cuda import TreeRCOperator


def fixture():
    pins=[Pin('d','n',0,0,'INPUT',port=True),Pin('a/A','n',1,0,'INPUT'),
          Pin('b/A','n',2,0,'INPUT'),Pin('e','m',0,1,'INPUT',port=True),
          Pin('c/A','m',1,1,'INPUT')]
    arcs=[dict(s=0,t=t,fr=rf,to=rf,role='wire') for t in [1,2] for rf in [0,1]]
    arcs += [dict(s=3,t=4,fr=rf,to=rf,role='wire') for rf in [0,1]]
    arcs += [dict(s=1,t=2,fr=0,to=1,role='combinational')]
    model=SimpleNamespace(n=5,index={p.name:i for i,p in enumerate(pins)},arcs=arcs,
        rf_sink=np.array([False,True,True,False,True]),
        rf_base_load=np.zeros((5,2)),rc_base_tau=np.zeros(7),
        by_net={'n':[0,1,2],'m':[3,4]})
    networks={
        'n':Network('n',tuple(pins[:3]),('d','x','a','b'),((0,1,2.),(1,2,3.),(1,3,7.)),
                    np.array([1.,2.,3.,4.]),{'d':0,'a/A':2,'b/A':3},'d',0,0,{}),
        'm':Network('m',tuple(pins[3:]),('e','c'),((0,1,5.),),
                    np.array([.5,1.5]),{'e':0,'c/A':1},'e',0,0,{})}
    explicit=np.zeros((7,10))
    for rf in [0,1]:
        explicit[rf,2+rf]=5.;explicit[rf,4+rf]=2.
        explicit[2+rf,2+rf]=2.;explicit[2+rf,4+rf]=9.
        explicit[4+rf,8+rf]=5.
    return model,networks,explicit


class TreeRC(unittest.TestCase):
    def test_second_moment_impulse_against_direct_matrix_algebra(self):
        m,n,_=fixture();operator=TreeRCOperator(m,n,device='cpu')
        caps=np.array([[0,0],[5,6],[7,8],[0,0],[2,3]],dtype=float)
        load,tau,state=operator.absolute_moments(caps,{})
        green=np.linalg.inv([[1/2+1/3+1/7,-1/3,-1/7],[-1/3,1/3,0],[-1/7,0,1/7]])
        c=np.array([[2.,2.],[8.,9.],[11.,12.]])
        first=green@c;beta=green@(c*first)
        expected=np.sqrt(2*beta-first**2)
        np.testing.assert_allclose(state['arc_impulse'].numpy()[:4],expected[[1,2]].reshape(-1),atol=1e-12)
        # An isolated one-pole RC has impulse equal to its first moment.
        np.testing.assert_allclose(state['arc_impulse'].numpy()[4:6],tau.numpy()[4:6],atol=1e-12)

    def test_second_moment_vjp_matches_autograd_and_finite_difference(self):
        m,n,_=fixture();op=TreeRCOperator(m,n,device='cpu')
        caps=np.array([[0,0],[5,6],[7,8],[0,0],[2,3]],dtype=float)
        _,_,state=op.absolute_moments(caps,{})
        rng=np.random.default_rng(9);gd=torch.tensor(rng.normal(size=7));gi=torch.tensor(rng.normal(size=7))
        leaf=torch.zeros(10,dtype=torch.float64,requires_grad=True)
        delay,impulse=op.moments_delta(leaf,state)
        (torch.sum(delay*gd)+torch.sum(impulse*gi)).backward()
        manual=op.moments_transpose(gd,gi,state).numpy()
        np.testing.assert_allclose(manual,leaf.grad.numpy(),atol=1e-11)
        for j in [2,3,4,5,8,9]:
            epsilon=1e-5;plus=np.zeros(10);minus=plus.copy();plus[j]=epsilon;minus[j]=-epsilon
            dp,ip=op.moments_delta(torch.tensor(plus),state);dm,im=op.moments_delta(torch.tensor(minus),state)
            fd=((dp-dm)@gd+(ip-im)@gi).item()/(2*epsilon)
            self.assertAlmostEqual(manual[j],fd,places=7)

    def test_signed_operator_and_transpose_against_grounded_conductance(self):
        m,n,explicit=fixture();operator=TreeRCOperator(m,n,device='cpu')
        rng=np.random.default_rng(5);x=rng.normal(size=10);y=rng.normal(size=7)
        np.testing.assert_allclose(operator.apply(torch.tensor(x)).numpy(),explicit@x,atol=1e-13)
        np.testing.assert_allclose(operator.transpose(torch.tensor(y)).numpy(),explicit.T@y,atol=1e-13)
        linear=operator.as_linear_operator()
        np.testing.assert_allclose(linear@x,explicit@x,atol=1e-13)
        np.testing.assert_allclose(linear.T@y,explicit.T@y,atol=1e-13)
        leaf=torch.tensor(x,requires_grad=True)
        torch.sum(operator.apply(leaf)*torch.tensor(y)).backward()
        np.testing.assert_allclose(leaf.grad.numpy(),explicit.T@y,atol=1e-13)

    def test_absolute_moments_match_independent_network_equations(self):
        m,n,_=fixture();operator=TreeRCOperator(m,n,device='cpu')
        caps=np.array([[0,0],[5,6],[7,8],[0,0],[2,3]],dtype=float)
        load,tau=operator.absolute(caps,{})
        g=np.array([[1/2+1/3+1/7,-1/3,-1/7],[-1/3,1/3,0],[-1/7,0,1/7]])
        native=np.linalg.solve(g,[[2,2],[8,9],[11,12]])
        np.testing.assert_allclose(tau.numpy()[:4],native[[1,2]].reshape(-1),atol=1e-12)
        np.testing.assert_allclose(tau.numpy()[4:6],5*np.array([3.5,4.5]),atol=1e-12)
        np.testing.assert_allclose(load.numpy()[:3],[[22,24]]*3)
        np.testing.assert_allclose(load.numpy()[3:],[[4,5]]*2)

    def test_dirty_patch_changes_only_its_network(self):
        m,n,explicit=fixture();operator=TreeRCOperator(m,n,device='cpu')
        old=n['n'];n['n']=Network(old.name,old.pins,old.nodes,
            tuple((a,b,r*2) for a,b,r in old.resistors),old.wire_caps,
            old.pin_nodes,old.driver,0,0,{})
        status=operator.update(n);self.assertFalse(status['storage_rebuilt']);self.assertEqual(status['dirty_nets'],1)
        explicit[:4]*=2
        x=np.arange(10,dtype=float)/3
        np.testing.assert_allclose(operator.apply(torch.tensor(x)),explicit@x,atol=1e-12)

    def test_topology_growth_rebuild_preserves_other_network(self):
        m,n,_=fixture();operator=TreeRCOperator(m,n,device='cpu')
        old=n['n'];n['n']=Network(old.name,old.pins,tuple(range(9)),
            tuple((i,i+1,1.) for i in range(8)),np.zeros(9),
            {'d':0,'a/A':4,'b/A':8},'d',0,0,{})
        status=operator.update(n);self.assertTrue(status['storage_rebuilt'])
        x=np.ones(10);expected=np.array([8,8,12,12,5,5,0])
        np.testing.assert_allclose(operator.apply(torch.tensor(x)),expected,atol=1e-12)


if __name__=='__main__':unittest.main()
