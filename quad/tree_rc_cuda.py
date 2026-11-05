"""Linear-space GPU RC-tree operator using Euler intervals and prefix sums.

For DFS order i, subtree(i)=[i,end[i]). Subtree capacitance is a prefix-sum
difference; root-path Elmore is the prefix sum of interval-add events R_i*C_i.
The grounded tree Green function is symmetric, so its transpose uses the same
solve with exchanged pin/arc scatter and gather. No dense sink-pair matrix.

Wire geometry is packed on CPU; dirty nets patch preallocated node ranges.
Torch operations are original code and work on CPU too for independent tests.
"""
from collections import defaultdict
import numpy as np
from scipy.sparse.linalg import LinearOperator
import torch
from placement_gpu_rc import PlacementGPUBridge


def pack_tree(network):
    if hasattr(network,'packed_tree'):return network.packed_tree()
    return pack_tree_reference(network)


def pack_tree_reference(network):
    n=len(network.nodes);adj=[[] for _ in range(n)]
    for a,b,r in network.resistors:
        if r<0:raise ValueError('Negative tree resistance')
        adj[a].append((b,r));adj[b].append((a,r))
    if len(network.resistors)!=n-1:raise ValueError('RC topology is not a tree')
    root=network.pin_nodes[network.driver];order=[];end=[];resistance=[];where={}
    stack=[(root,-1,0.,False)]
    while stack:
        node,parent,r,leaving=stack.pop()
        if leaving:end[where[node]]=len(order);continue
        if node in where:raise ValueError('Cycle in RC tree')
        where[node]=len(order);order.append(node);end.append(0);resistance.append(r)
        stack.append((node,parent,r,True))
        for child,cr in reversed(adj[node]):
            if child!=parent:stack.append((child,node,cr,False))
    if len(order)!=n:raise ValueError('Disconnected RC tree')
    return (np.asarray(end,dtype=np.int64),np.asarray(resistance),
            network.wire_caps[order],{p:where[v] for p,v in network.pin_nodes.items()})


class TreeRCOperator:
    def __init__(self,model,networks,device='cuda:0'):
        self.model=model;self.device=device;self.slots={};self.cache={};self.capacity=0;self.next_free=0
        self._wire=np.array([a['role']=='wire' for a in model.arcs])
        self._dst=np.array([a['t'] for a in model.arcs],dtype=np.int64)
        self._rf=np.array([a['fr'] for a in model.arcs],dtype=np.int64)
        self.update(networks)

    def tensor(self,x,integer=False):
        return torch.as_tensor(np.asarray(x),device=self.device,dtype=torch.int64 if integer else torch.float64)

    def update(self,networks):
        dirty=[name for name,n in networks.items() if self.cache.get(name) is not n]
        if self.cache and self.cache.keys()!=networks.keys():raise ValueError('Netlist changed: rebuild tree RC operator')
        if not dirty:return dict(dirty_nets=0,allocated_nodes=self.capacity,storage_rebuilt=False)
        packed={name:pack_tree(networks[name]) for name in dirty}
        rebuild=not self.slots;patches=[]
        if rebuild:
            self.slots={};offset=0
            for name,n in sorted(networks.items()):
                needed=len(packed[name][0]);size=1<<(max(1,needed)-1).bit_length()
                self.slots[name]=(offset,size);offset+=size
            self.next_free=offset;self.capacity=max(offset,int(offset*1.25))
            self._ends=np.arange(1,self.capacity+1,dtype=np.int64)
            self._r=np.zeros(self.capacity);self._wire_c=np.zeros(self.capacity)
            self._node_root=np.arange(self.capacity,dtype=np.int64)
            self._pin_node=np.full(self.model.n,-1,dtype=np.int64)
            self._roots={};self._root_index=np.zeros(self.model.n,dtype=np.int64)
            self._root_valid=np.zeros(self.model.n,dtype=bool)
        else:
            for name in dirty:
                if len(packed[name][0])<=self.slots[name][1]:continue
                base,size=self.slots[name]
                # Retire this range and append a larger one. Other nets retain
                # their Euler positions; no all-design repacking is needed.
                self._ends[base:base+size]=np.arange(base+1,base+size+1)
                self._r[base:base+size]=0.;self._wire_c[base:base+size]=0.
                self._node_root[base:base+size]=np.arange(base,base+size)
                patches.append(np.arange(base,base+size,dtype=np.int64))
                new_size=1<<(len(packed[name][0])-1).bit_length()
                self.slots[name]=(self.next_free,new_size);self.next_free+=new_size
            if self.next_free>self.capacity:
                new_capacity=max(self.next_free,int(self.capacity*1.5));old=self.capacity
                ends=np.arange(1,new_capacity+1,dtype=np.int64);ends[:old]=self._ends;self._ends=ends
                self._r=np.pad(self._r,(0,new_capacity-old));self._wire_c=np.pad(self._wire_c,(0,new_capacity-old))
                self._node_root=np.concatenate((self._node_root,np.arange(old,new_capacity,dtype=np.int64)))
                self.capacity=new_capacity;rebuild=True
        for name in dirty:
            ends,r,wire,pins=packed[name];base,size=self.slots[name];n=len(r)
            self._ends[base:base+size]=np.arange(base+1,base+size+1)
            self._r[base:base+size]=0.;self._wire_c[base:base+size]=0.
            self._node_root[base:base+size]=base
            self._ends[base:base+n]=base+ends;self._r[base:base+n]=r;self._wire_c[base:base+n]=wire
            for pin,where in pins.items():
                if pin in self.model.index:self._pin_node[self.model.index[pin]]=base+where
            self._roots[name]=base;self.cache[name]=networks[name]
            if name in self.model.by_net:
                pins_in_net=self.model.by_net[name]
                self._root_index[pins_in_net]=base;self._root_valid[pins_in_net]=True
            patches.append(np.arange(base,base+size,dtype=np.int64))
        if rebuild:
            self.ends=self.tensor(self._ends,True);self.r=self.tensor(self._r)
            self.wire_c=self.tensor(self._wire_c)
        elif patches:
            ids=np.concatenate(patches);index=self.tensor(ids,True)
            self.ends.index_copy_(0,index,self.tensor(self._ends[ids],True))
            self.r.index_copy_(0,index,self.tensor(self._r[ids]));self.wire_c.index_copy_(0,index,self.tensor(self._wire_c[ids]))
        self.pin_node=self.tensor(np.maximum(self._pin_node,0),True)
        self.pin_valid=self.tensor((self._pin_node>=0)&self.model.rf_sink,True).bool()
        self.arc_node=self.pin_node[self.tensor(self._dst,True)]
        valid=self._wire&(self._pin_node[self._dst]>=0)
        missing=np.flatnonzero(self._wire&~valid&(np.abs(self.model.rc_base_tau)>1e-8))
        if len(missing):raise ValueError('Nonideal wire arcs absent from packed trees')
        self.arc_valid=self.tensor(valid,True).bool();self.rf=self.tensor(self._rf,True)
        self.root_index=self.tensor(self._root_index,True);self.root_valid=self.tensor(self._root_valid,True).bool()
        self.node_root=self.tensor(self._node_root,True)
        return dict(dirty_nets=len(dirty),allocated_nodes=self.capacity,storage_rebuilt=rebuild)

    def solve(self,node_cap):
        # A scan of shape [nodes,2] along dim=0 can select PyTorch's serial
        # outer-dimension kernel. Flatten RF-major storage to use a parallel
        # one-dimensional scan instead. Interval differences cancel RF offsets.
        columns=node_cap.shape[1];n=self.capacity
        flat=node_cap.transpose(0,1).contiguous().reshape(-1)
        zero=torch.zeros(1,dtype=torch.float64,device=self.device)
        prefix=torch.cat((zero,torch.cumsum(flat,dim=0)))
        offset=torch.arange(columns,device=self.device)[:,None]*n
        start=torch.arange(n,device=self.device)[None,:]+offset
        subtree=prefix[self.ends[None,:]+offset]-prefix[start]
        weighted=subtree*self.r[None,:]
        difference=torch.cat((weighted,torch.zeros((columns,1),dtype=torch.float64,device=self.device)),dim=1)
        difference=difference.index_add(1,self.ends,-weighted)
        cumulative=torch.cumsum(difference.reshape(-1),dim=0).reshape(columns,n+1)
        baseline=torch.cat((zero,cumulative[:-1,-1]))[:,None]
        delay=cumulative[:,:-1]-baseline
        delay=delay.transpose(0,1)
        # Remove floating-point carry between independent trees. Their ideal
        # roots must have exactly zero delay, especially before second moments.
        delay=delay-delay[self.node_root]
        return delay,subtree.transpose(0,1)

    def apply(self,pin_delta):
        cap=pin_delta.reshape(-1,2)*self.pin_valid[:,None]
        nodes=torch.zeros((self.capacity,2),dtype=torch.float64,device=self.device).index_add(0,self.pin_node,cap)
        delay,_=self.solve(nodes)
        return torch.where(self.arc_valid,delay[self.arc_node,self.rf],0.)

    def transpose(self,arc_gradient):
        flat_index=self.arc_node*2+self.rf
        nodes=torch.zeros(self.capacity*2,dtype=torch.float64,device=self.device).index_add(
            0,flat_index,arc_gradient*self.arc_valid).reshape(-1,2)
        result,_=self.solve(nodes)
        return (result[self.pin_node]*self.pin_valid[:,None]).reshape(-1)

    def absolute(self,pin_caps,external,reference_load=None):
        load,tau,_=self.absolute_moments(pin_caps,external,reference_load,second_moment=False)
        return load,tau

    def absolute_moments(self,pin_caps,external,reference_load=None,second_moment=True):
        caps=np.asarray(pin_caps).copy()*self.model.rf_sink[:,None]
        for name,load in external.items():caps[self.model.index[name]]=load
        nodes=self.wire_c[:,None].expand(-1,2).clone().index_add(0,self.pin_node,self.tensor(caps)*(self.tensor(self._pin_node>=0,True).bool()[:,None]))
        delay,subtree=self.solve(nodes)
        tau=torch.where(self.arc_valid,delay[self.arc_node,self.rf],0.)
        load=self.tensor(self.model.rf_base_load if reference_load is None else reference_load).clone()
        # Map each pin to its net root in one gather; unmapped ideal nets retain
        # their reference external load.
        load=torch.where(self.root_valid[:,None],subtree[self.root_index],load)
        state=self.moment_state(nodes,delay) if second_moment else None
        return load.clamp_min(0),tau.clamp_min(0),state

    def moment_state(self,node_cap,node_tau=None):
        if node_tau is None:node_tau,_=self.solve(node_cap)
        beta,_=self.solve(node_cap*node_tau)
        variance=2*beta-node_tau.square()
        # Use the zero subgradient at a zero-width impulse. Clamping sqrt alone
        # gives an infinite derivative at zero in autograd.
        positive=variance>0
        impulse=torch.where(positive,torch.sqrt(torch.where(positive,variance,1.)),0.)
        arc_tau=torch.where(self.arc_valid,node_tau[self.arc_node,self.rf],0.)
        arc_impulse=torch.where(self.arc_valid,impulse[self.arc_node,self.rf],0.)
        return dict(node_cap=node_cap,node_tau=node_tau,arc_tau=arc_tau,
                    arc_impulse=arc_impulse)

    def moments_delta(self,pin_delta,state):
        cap=pin_delta.reshape(-1,2)*self.pin_valid[:,None]
        nodes=state['node_cap'].index_add(0,self.pin_node,cap)
        current=self.moment_state(nodes)
        return current['arc_tau'],current['arc_impulse']

    def moments_transpose(self,delay_gradient,impulse_gradient,state):
        # beta=G(C*tau), tau=G*C and G is symmetric. Thus
        # dL/dC=tau*G(g_beta)+G(g_tau+C*G(g_beta)).
        impulse=state['arc_impulse'];safe=impulse.clamp_min(1e-30)
        gb=torch.where(impulse>0,impulse_gradient/safe,0.)*self.arc_valid
        tau=state['arc_tau'];gt=(delay_gradient-gb*tau)*self.arc_valid
        index=self.arc_node*2+self.rf
        def nodes(values):
            return torch.zeros(self.capacity*2,dtype=torch.float64,device=self.device).index_add(0,index,values).reshape(-1,2)
        beta_adjoint,_=self.solve(nodes(gb))
        cap_adjoint,_=self.solve(nodes(gt)+state['node_cap']*beta_adjoint)
        cap_adjoint=cap_adjoint+state['node_tau']*beta_adjoint
        return (cap_adjoint[self.pin_node]*self.pin_valid[:,None]).reshape(-1)

    def _cpu_solve(self,caps):
        prefix=np.concatenate((np.zeros((1,2)),np.cumsum(caps,axis=0)))
        sub=prefix[self._ends]-prefix[:-1];weighted=sub*self._r[:,None]
        diff=np.concatenate((weighted,np.zeros((1,2))));np.add.at(diff,self._ends,-weighted)
        result=np.cumsum(diff,axis=0)[:-1]
        return result-result[self._node_root]

    def as_linear_operator(self):
        m=self.model;valid=self._wire&(self._pin_node[self._dst]>=0)
        node=np.maximum(self._pin_node,0);pin_valid=(self._pin_node>=0)&m.rf_sink
        def matvec(x):
            caps=np.zeros((self.capacity,2));np.add.at(caps,node,np.asarray(x).reshape(-1,2)*pin_valid[:,None])
            solved=self._cpu_solve(caps)
            return np.where(valid,solved[node[self._dst],self._rf],0.)
        def rmatvec(x):
            caps=np.zeros((self.capacity,2))
            np.add.at(caps,(node[self._dst],self._rf),np.asarray(x).reshape(-1)*valid)
            return (self._cpu_solve(caps)[node]*pin_valid[:,None]).reshape(-1)
        return LinearOperator((len(m.arcs),2*m.n),matvec=matvec,rmatvec=rmatvec,dtype=np.float64)


class TreePlacementBridge:
    """Placement RC updates without building all sink-pair transfer entries."""
    def __init__(self,estimator,engine,shared=None):
        self.estimator=estimator;self.engine=engine
        self.owner=shared.owner if shared is not None else self
        self.last_generation=None;self.last_status=None
        if shared is not None:
            other=shared.engine.m;m=engine.m
            signature=lambda model:[(a['s'],a['t'],a['fr'],a['to'],a['role']) for a in model.arcs]
            if m.index!=other.index or signature(m)!=signature(other) or not np.array_equal(m.rf_sink,other.rf_sink):
                raise ValueError('Cannot share tree RC across different timing graphs')
        self.external=PlacementGPUBridge(estimator,engine).external
        self.operator=None;self.external_pins={};self.unrouted_inactive_nets=[]
        active_wire_nets={engine.m.pins[a['s']]['net'] for a in engine.m.arcs if a['role']=='wire'}
        for net,load in self.external.items():
            if not np.any(load):continue
            if net not in estimator.networks:
                # OpenSTA omits disabled constant nets from SPEF while its
                # pin load report can retain their placement capacitance.
                # No distributed timing solve exists for these nets; the
                # unmapped pins keep their native baseline load below.
                if net in active_wire_nets:
                    raise ValueError('Active externally loaded net absent from RC trees: '+net)
                self.unrouted_inactive_nets.append(net)
                continue
            ports=[p for p in estimator.networks[net].pins if p.port and p.name!=estimator.networks[net].driver]
            if len(ports)!=1:raise ValueError('Ambiguous external port load')
            self.external_pins[ports[0].name]=load

    def refresh(self):
        engine=self.engine
        owner=self.owner
        if owner.operator is None:
            owner.operator=TreeRCOperator(engine.m,self.estimator.networks,device=engine.device)
            status=dict(dirty_nets=len(self.estimator.networks),allocated_nodes=owner.operator.capacity,storage_rebuilt=True)
        elif owner.last_generation!=self.estimator.generation:status=owner.operator.update(self.estimator.networks)
        else:status=dict(dirty_nets=0,allocated_nodes=owner.operator.capacity,storage_rebuilt=False,metadata_reused=True)
        owner.last_generation=self.estimator.generation;owner.last_status=status;self.operator=owner.operator
        moments=getattr(engine.m,'wire_slew_model','elmore')=='impulse'
        load,tau,state=self.operator.absolute_moments(engine.m.rf_base_pin,self.external_pins,
                                                     engine.m.rf_base_load,second_moment=moments)
        engine.update_tree_parasitics(self.operator,load,tau,moment_state=state)
        return status
