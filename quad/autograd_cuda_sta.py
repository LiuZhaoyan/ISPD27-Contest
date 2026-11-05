"""Hard-forward STA with Ave/LSE proxy expressions and PyTorch autograd.

The discrete library differences/ranking remain in ResidentCudaSTA.score_all.
No analytic LUT derivatives or handwritten reverse level sweep are used here.
Virtual additive inputs expose the arc/pin sensitivities needed by that scorer.
"""
import numpy as np
import torch
from resident_cuda_sta import ResidentCudaSTA


class AutogradCudaSTA(ResidentCudaSTA):
    gradient_backend = 'autograd'

    def __init__(self, model, device='cuda:0'):
        super().__init__(model, device)
        self.autograd_plan = []
        # All grouping/index metadata is static; never rebuild it during AD.
        raw_bundle = self.bundle.detach().cpu().numpy()
        for ids, cell, wire, nodes in self.plan:
            ai = ids.cpu().numpy()
            nd = nodes.cpu().numpy()
            _, inverse = np.unique(raw_bundle[ai], return_inverse=True)
            groups = int(inverse.max()) + 1
            group_dst = np.empty(groups, dtype=np.int64)
            node_index = {int(v): i for i, v in enumerate(nd)}
            for k, arc in enumerate(ai):
                a = model.arcs[int(arc)]
                group_dst[inverse[k]] = node_index[a['t']*2+a['to']]
            positions = {int(v): i for i, v in enumerate(ai)}
            cp = self.tensor([positions[int(v)] for v in cell.cpu().numpy()], True)
            wp = self.tensor([positions[int(v)] for v in wire.cpu().numpy()], True)
            self.autograd_plan.append((ids, cell, wire, nodes, cp, wp,
                                       self.tensor(inverse, True),
                                       self.tensor(group_dst, True), groups))

    def prepare(self):
        # Keep the original hard forward for numerical prediction/scoring.
        # AD builds its own functional graph when a reverse pass is requested.
        result = self.forward(capture=True)
        self.b.forward = result
        self.b.local_slew = self.numpy(self.local_slew)
        self.prepared = True
        self.sync()
        return result

    def _merge(self, values, bundle, destinations, groups, size,
               branch, epsilon, gamma, floor=None):
        # A conditional-arc bundle uses a hard max (equal gradient at ties).
        # Smoothing is applied between bundles, not between their alternatives.
        peaks = self.peak(values, bundle, groups)
        hard = self.peak(peaks, destinations, size)
        if floor is not None:
            hard = hard.clamp_min(floor)
        if branch == 'hard':
            return hard
        finite = torch.isfinite(peaks)
        safe = torch.where(finite, peaks, 0.)
        if branch == 'ave':
            top = hard.detach()[destinations]
            active = finite & (peaks.detach() >= top-epsilon*top.abs())
            weight = active.to(values.dtype)
            count = self.zeros(size).index_add(0, destinations, weight)
            weight = (weight/count[destinations].clamp_min(1.)).detach()
            proxy = self.zeros(size).index_add(0, destinations, weight*safe)
        else:
            shift = torch.where(torch.isfinite(hard), hard, 0.).detach()
            exponent = torch.where(finite, (safe-shift[destinations])/gamma,
                                   -torch.inf).exp()
            total = self.zeros(size).index_add(0, destinations, exponent)
            proxy = shift+gamma*total.clamp_min(1e-300).log()
        # Parentheses make the zero-valued proxy correction explicit.
        return hard.detach()+(proxy-proxy.detach())

    def _graph(self, branch, epsilon, gamma, power=False):
        n = self.m.n*2
        na = len(self.m.arcs)
        nq = len(self.m.checks)
        leaves = [self.zeros(k).requires_grad_() for k in (n,n,n,na,na,nq)]
        arrival_bias, slew_bias, pin_bias, delay_bias, local_bias, check_bias = leaves
        dc = pin_bias.view(-1, 2)
        net = self.zeros(self.nn*2).view(-1,2).index_add(
            0, self.netid, dc*self.sink[:,None])
        load = self.load0+net[self.netid]
        tau,impulse = self.candidate_wire_moments(pin_bias)
        cap = load.reshape(-1)
        at = torch.full((n,),-torch.inf,dtype=torch.float64,device=self.device)
        at = at.index_copy(0,self.roots,self.root_at+arrival_bias[self.roots])
        sl = self.zeros(n).index_copy(0,self.roots,self.root_sl+slew_bias[self.roots])
        grouped=hasattr(self,'source_groups') and not power
        collect_delay = grouped or (getattr(self,'export_local_delays',False) and not power)
        if collect_delay:group_delay=self.zeros(na)
        for ids,cell,wire,nodes,cp,wp,bundle,dst,groups in self.autograd_plan:
            d = self.zeros(len(ids))
            local = self.zeros(len(ids))
            if len(cell):
                source = sl[self.s[cell]]
                c = cap[self.t[cell]]
                if not power:
                    d = d.index_copy(0,cp,self.lut(self.delay_tab[cell],source,c))
                local = local.index_copy(0,cp,torch.relu(self.lut(self.slew_tab[cell],source,c)))
            if len(wire):
                port=self.port_wire[wire]
                wire_delay=torch.where(port,tau[wire]*self.port_delay_factor[wire],tau[wire])
                d = d.index_copy(0,wp,wire_delay)
                # vector_norm has the zero subgradient at (0,0), unlike hypot.
                ws = torch.linalg.vector_norm(torch.stack(
                    (sl[self.s[wire]],2.197224577*impulse[wire])),dim=0)
                ws=torch.where(port,sl[self.s[wire]]+tau[wire]*self.port_slew_factor[wire],ws)
                local = local.index_copy(0,wp,ws)
            if self.arc_calibrated and not power:
                if len(cell):d=d.index_copy(0,cp,d[cp]+self.arc_offset[cell])
                if len(wire):d=d.index_copy(0,wp,torch.relu(d[wp]+self.arc_offset[wire]))
            local = local+local_bias[ids]
            if not power:
                if collect_delay:group_delay=group_delay.index_copy(0,ids,d+delay_bias[ids])
                candidate = at[self.s[ids]]+d+delay_bias[ids]
                av = self._merge(candidate,bundle,dst,groups,len(nodes),branch,epsilon,gamma)
                at = at.index_copy(0,nodes,av*self.ar[nodes]+self.ao[nodes]+arrival_bias[nodes])
            sv = self._merge(local,bundle,dst,groups,len(nodes),branch,epsilon,gamma,floor=0.)
            sl = sl.index_copy(0,nodes,torch.relu(sv*self.sr[nodes]+self.so[nodes])+slew_bias[nodes])
        graph = dict(at=at,slew=sl,load=load,leaves=leaves)
        if collect_delay:graph['delay']=group_delay
        if not power:
            raw = self.lut(self.check_tab,sl[self.ct],sl[self.cs])+check_bias
            check = raw*self.qr+self.qo
            qmax = self.peak(check,self.ct,n)
            req = self.ep_req-torch.where(self.ep_setup,qmax[self.ep_node],0.)+self.capture_clock_delta(at)
            if grouped:
                runtime=self.source_groups;ga=runtime.seed.clone();ng=runtime.plan.g
                for ids,cell,wire,nodes,cp,wp,bundle,dst,groups in self.autograd_plan:
                    candidates=torch.where(runtime.reset[:,ids],group_delay[ids][None,:],
                        ga[:,self.s[ids]]+group_delay[ids][None,:])
                    merged=torch.stack([self._merge(candidates[g],bundle,dst,groups,len(nodes),
                        branch,epsilon,gamma) for g in range(ng)])
                    ga=ga.index_copy(1,nodes,merged)
                data_at=ga[runtime.ep_group,self.ep_node]+self.group_endpoint_offset
            else:data_at=at[self.ep_node]
            slack = req-data_at
            worst = torch.full((self.m.n,),torch.inf,dtype=torch.float64,device=self.device)
            worst = worst.scatter_reduce(0,self.ep_pin,slack,reduce='amin',include_self=True)
            graph.update(check=check,slack=slack,
                         tns=torch.relu(-worst).sum(),
                         # WNS seeds are equal over endpoint ties, not pin ties.
                         wns=torch.relu(-torch.amin(slack)))
        return graph

    def _differentiate(self, graph, loss):
        if loss.requires_grad:
            gradients = torch.autograd.grad(loss,graph['leaves'],allow_unused=True)
        else:
            gradients = [None]*len(graph['leaves'])
        return [self.numpy(torch.zeros_like(x) if g is None else g)
                for x,g in zip(graph['leaves'],gradients)]

    def backward(self, branch='ave', epsilon=.01, gamma_ps=1., alpha=1., beta=0.):
        if not self.prepared:
            raise ValueError('prepare() required')
        if branch not in ('hard','ave','lse') or not 0<=epsilon<1 or gamma_ps<=0 or min(alpha,beta)<0 or alpha+beta<=0:
            raise ValueError('Invalid proxy VJP')
        with torch.enable_grad():
            graph = self._graph(branch,epsilon,gamma_ps)
            ga,gs,gc,gd,gsl,gq = self._differentiate(graph,alpha*graph['tns']+beta*graph['wns'])
        return dict(at=ga.reshape(-1,2),slew=gs.reshape(-1,2),
                    load=np.zeros(self.m.n),net_load={net:0. for net in self.netnames},
                    pin_load=gc.reshape(-1,2),delay=gd,local_slew=gsl,setup=gq,
                    branch=branch,epsilon=epsilon,gamma_ps=gamma_ps,alpha=alpha,beta=beta)

    def backward_power(self, slew_seed, load_seed):
        if not self.prepared:
            raise ValueError('prepare() required')
        with torch.enable_grad():
            graph = self._graph('hard',.01,1.,power=True)
            loss = ((graph['slew']*self.tensor(np.asarray(slew_seed).reshape(-1))).sum()
                    +(graph['load']*self.tensor(load_seed)).sum())
            ga,gs,gc,gd,gsl,gq = self._differentiate(graph,loss)
        return dict(at=ga.reshape(-1,2),slew=gs.reshape(-1,2),pin_load=gc.reshape(-1,2),
                    delay=gd,local_slew=gsl,setup=gq)
