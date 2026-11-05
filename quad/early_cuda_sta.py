"""Float64 GPU early-arrival/slew propagation with shared physical tree RC.

Minimum input capacitance ranges, min arrival/slew reductions, and early wire
moments are distinct from max analysis. Frozen initial local corrections are
never refitted to changed designs. Hold checks currently retain reference
capture-clock slew/latency unless an explicit late state is provided. General
exception tags and dynamic CRPR are still outside this prototype's scope.

The forward-only class deliberately rejects inherited max-delay derivatives.
"""
import numpy as np
import copy
import torch
from resident_cuda_sta import ResidentCudaSTA
from forward_sta import MIN_CHECKS


class EarlyCudaSTA(ResidentCudaSTA):
    def __init__(self, model, device='cuda:0'):
        if getattr(model,'analysis',None)!='min':raise ValueError('Early engine requires a min-analysis reference')
        if not hasattr(model,'cap_ranges'):raise ValueError('Early engine requires Liberty min-capacitance ranges')
        super().__init__(model,device)
        self.hold_endpoint=self.tensor([e['role'] in MIN_CHECKS for e in model.endpoints],True).bool()
        self.ep_req.sub_(self.tensor([float(e['margin_ps']) if e['role'] in MIN_CHECKS else 0. for e in model.endpoints]))
        self.capture_late_slew=self.tensor(model.ref_slew.reshape(-1))[self.cs]

    def peak(self,x,index,n):
        z=torch.full((n,),torch.inf,dtype=torch.float64,device=self.device)
        return z.scatter_reduce_(0,index,x,reduce='amin',include_self=True)

    def bind_late_clock(self,late_at,late_slew,reference_late_at,reference_late_slew=None):
        """Bind persistent GPU tensors from the paired max engine.

        Caller runs max forward first. No host transfers are required. The
        hold constraint uses late clock slew and its arrival change; early
        data slew comes from this engine. General CRPR changes remain frozen.
        """
        expected=(2*self.m.n,)
        arrays=[x.reshape(-1) for x in (late_at,late_slew,reference_late_at)]
        if any(x.shape!=expected or x.device!=self.s.device for x in arrays):
            raise ValueError('Late-clock tensors do not match early physical pins/device')
        if any(not torch.isfinite(x).all() for x in (arrays[1][self.cs],arrays[2][self.cs])):
            raise ValueError('Non-finite late capture state')
        if not torch.allclose(arrays[0][self.cs],arrays[2][self.cs],atol=1e-3,rtol=0):
            raise ValueError('Bind late clocks at the initial reference before changing clock RC')
        reference_slew=(arrays[1].clone() if reference_late_slew is None else reference_late_slew.reshape(-1))
        if reference_slew.shape!=expected or reference_slew.device!=self.s.device:
            raise ValueError('Reference late-clock slew does not match early physical pins/device')
        if not torch.isfinite(reference_slew[self.cs]).all():raise ValueError('Non-finite reference capture slew')
        self.late_at,self.late_slew=arrays[:2]
        self.late_reference=arrays[2].clone()
        self.late_reference_slew=reference_slew.clone()
        # Transport the frozen initial check correction to the mixed-slew
        # reference. Without this constant, binding a different late slew
        # would change the initial hold margin despite unchanged RC.
        ref_data=self.tensor(self.m.ref_slew.reshape(-1))[self.ct]
        self.late_check_reference_delta=(
            self.lut(self.check_tab,ref_data,self.late_reference_slew[self.cs])-
            self.lut(self.check_tab,ref_data,self.capture_late_slew))
        nodes=[];valid=[]
        for e in self.m.endpoints:
            p=e.get('capture_pin','');rf=e.get('capture_rf','')
            ok=p in self.m.index and rf in ('rise','fall')
            nodes.append(2*self.m.index[p]+('rise','fall').index(rf) if ok else 0)
            valid.append(ok)
        self.late_capture_nodes=self.tensor(nodes,True)
        self.late_capture_valid=self.tensor(valid,True).bool()
        self._replay_graph=None

    def forward(self,replacements=None,capture=False,_gpu_only=False):
        if capture:raise NotImplementedError('Early reverse derivatives are not implemented')
        data=self._candidate(replacements or {})
        if data is None:raise ValueError('Changed conditional graph requires rebuilding the early graph')
        dt,st,qt,load,tau,impulse=data;m=self.m;n=2*m.n
        at=torch.full((n,),torch.inf,dtype=torch.float64,device=self.device)
        sl=torch.full_like(at,torch.inf)
        at[self.roots]=self.root_at;sl[self.roots]=self.root_sl
        delay=self.zeros(len(m.arcs));local=self.zeros(len(m.arcs));cap=load.reshape(-1)
        for ids,cell,wire,nodes in self.plan:
            delay[cell]=self.lut(dt[cell],sl[self.s[cell]],cap[self.t[cell]])
            local[cell]=self.lut(st[cell],sl[self.s[cell]],cap[self.t[cell]]).clamp_min(0)
            delay[wire],local[wire]=self.wire_values(tau[wire],impulse[wire],sl[self.s[wire]],wire)
            if self.arc_calibrated:
                delay[cell]+=self.arc_offset[cell]
                delay[wire]=(delay[wire]+self.arc_offset[wire]).clamp_min(0)
            at.scatter_reduce_(0,self.t[ids],at[self.s[ids]]+delay[ids],reduce='amin',include_self=True)
            sl.scatter_reduce_(0,self.t[ids],local[ids],reduce='amin',include_self=True)
            at[nodes]=at[nodes]*self.ar[nodes]+self.ao[nodes]
            value=torch.where(torch.isfinite(sl[nodes]),sl[nodes],0.)
            sl[nodes]=(value*self.sr[nodes]+self.so[nodes]).clamp_min(0)
        clock_slew=self.late_slew[self.cs] if hasattr(self,'late_slew') else self.capture_late_slew
        raw_check=self.lut(qt,sl[self.ct],clock_slew)
        if hasattr(self,'late_check_reference_delta'):raw_check=raw_check-self.late_check_reference_delta
        check=raw_check*self.qr+self.qo
        margin=torch.full((n,),-torch.inf,dtype=torch.float64,device=self.device)
        margin.scatter_reduce_(0,self.ct,check,reduce='amax',include_self=True)
        req=self.ep_req+torch.where(self.hold_endpoint,margin[self.ep_node],0.)
        if hasattr(self,'late_at'):
            ids=self.late_capture_nodes
            req=req+torch.where(self.late_capture_valid,self.late_at[ids]-self.late_reference[ids],0.)
        data_at=at[self.ep_node]
        if hasattr(self,'source_groups'):
            group_at=self.source_groups.forward(delay)
            data_at=group_at[self.source_groups.ep_group,self.ep_node]+self.group_endpoint_offset
        slack=data_at-req
        worst=torch.full((m.n,),torch.inf,dtype=torch.float64,device=self.device)
        worst.scatter_reduce_(0,self.ep_pin,slack,reduce='amin',include_self=True)
        tensors=dict(at=at.reshape(-1,2),slew=sl.reshape(-1,2),load=load.min(1).values,
                     load_rf=load,delay=delay,check=check,
                     endpoints=torch.stack((self.ep_pin,self.ep_node%2,req,slack),dim=1),
                     tns_ps=worst.clamp_max(0).sum(),wns_ps=worst.min().clamp_max(0))
        if _gpu_only:return tensors
        return self._export_tensors(tensors)

    def prepare(self,*args,**kwargs):
        raise NotImplementedError('Early reverse derivatives are not implemented')

    def backward(self,*args,**kwargs):
        raise NotImplementedError('Early reverse derivatives are not implemented')

    def enable_source_groups(self,plan):
        from source_group_sta import CudaSourceGroups
        if plan.analysis!='min':raise ValueError('Early engine requires min-analysis source classes')
        if not self.arc_calibrated:raise ValueError('Early source groups require local arc calibration')
        if hasattr(self,'late_at'):raise ValueError('Install early source classes before binding late clocks')
        if hasattr(self,'source_groups'):raise ValueError('Early source classes already installed')
        if getattr(self.m,'wire_slew_model',None)=='impulse' and not self.wire_moment_model:
            raise ValueError('Refresh second moments before early source-class initialization')
        np.testing.assert_array_equal(plan.s,self.numpy(self.s));np.testing.assert_array_equal(plan.t,self.numpy(self.t))
        initial=self.forward();runtime=CudaSourceGroups(plan,self)
        group_at=runtime.forward(self.tensor(initial['delay']))
        native=self.tensor([float(e['at_ps']) for e in plan.endpoints])
        anchor=group_at[runtime.ep_group,runtime.ep_node]
        if not torch.isfinite(anchor).all():raise ValueError('Initial early endpoint source class is unreachable')
        self.group_endpoint_offset=native-anchor
        m=copy.copy(self.m);m.endpoints=[dict(e) for e in plan.endpoints];self.m=m;self.b.model=m
        self.ep_node=runtime.ep_node;self.ep_pin=self.ep_node//2
        checks={(a['t'],a['to']) for a in m.checks}
        mask=[(m.index[e['pin']],('rise','fall').index(e['rf'])) in checks for e in m.endpoints]
        self.hold_endpoint=self.tensor(mask,True).bool()
        self.ep_req=self.tensor([float(e['required_ps'])-(float(e['margin_ps']) if use else 0.)
                                for e,use in zip(m.endpoints,mask)])
        self.source_groups=runtime;self._replay_graph=None
