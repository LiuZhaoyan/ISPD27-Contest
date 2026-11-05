"""Bind placement-derived physical RC to the resident GPU timing engine.

Reference AT/slew calibration is preserved and never refitted to candidate GR.
The estimator may be shared across PVT corners, while pin loads are corner
specific. This bridge rebuilds sparse sensitivities for changed geometry;
the common sizing-only inner loop continues using the resident GPU matrix.
"""
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from rc_tree import transfer_resistance, spef_networks


class PlacementGPUBridge:
    def __init__(self, estimator, engine):
        self.estimator=estimator;self.engine=engine;self.m=engine.m
        self.cache={};self.last_recomputed=[]
        self.arcs_by_net=defaultdict(list)
        for k,a in enumerate(self.m.arcs):
            if a['role']=='wire':self.arcs_by_net[self.m.pins[a['s']]['net']].append(k)
        if hasattr(engine,'_placement_external_loads'):
            self.external={k:v.copy() for k,v in engine._placement_external_loads.items()}
            return
        # Preserve external port loads, not the placement estimator's wire C.
        wire_caps={net:sum(wire.values()) for net,_,_,wire in
                   spef_networks(self.m.reference/'network.spef')}
        self.external={}
        for net,pins in self.m.by_net.items():
            pc=self.m.rf_base_pin[pins][self.m.rf_sink[pins]].sum(0)
            external=self.m.rf_base_load[pins[0]]-pc-wire_caps.get(net,0.)
            if np.min(external)<-1e-4:
                raise ValueError(f'Negative external load on {net}: {external}')
            tolerance=max(1e-5,wire_caps.get(net,0.)*1e-5)
            external[np.abs(external)<tolerance]=0.
            self.external[net]=np.maximum(external,0.)
        engine._placement_external_loads={k:v.copy() for k,v in self.external.items()}

    def arrays(self):
        m=self.engine.m;self.m=m
        pin_caps={p['name']:m.rf_base_pin[i] for i,p in enumerate(m.pins)
                           if m.rf_sink[i]}
        initial=not hasattr(self,'_matrix')
        dirty=set(self.estimator.networks) if initial else {
            net for net,n in self.estimator.networks.items()
            if net not in self.cache or self.cache[net][0] is not n}
        if not initial:
            changed=np.flatnonzero(np.any(m.rf_base_pin!=self._pin_caps,axis=1))
            dirty.update(m.pins[i]['net'] for i in changed)
        load=m.rf_base_load.copy() if initial else self._load.copy()
        tau=np.zeros(len(m.arcs)) if initial else self._tau.copy()
        rr=[];cc=[];vv=[]
        covered=set();self.last_recomputed=[]
        for net in sorted(dirty):
            if net not in self.estimator.networks:continue
            n=self.estimator.networks[net]
            ids=self.arcs_by_net.get(net,[])
            if not ids:continue
            signature=tuple((p.name,tuple(pin_caps.get(p.name,(0.,0.)))) for p in n.pins)
            self.last_recomputed.append(net);cr=[];ci=[];cv=[];arc_tau=[]
            # External port C belongs at the physical port node and affects
            # distributed RC, rather than just the driver's capacitive load.
            caps=pin_caps.copy() if np.any(self.external.get(net,0.)) else pin_caps
            ports=[p for p in n.pins if p.port and p.name!=n.driver]
            if np.any(self.external.get(net,0.)):
                if len(ports)!=1:raise ValueError(f'Ambiguous external load: {net}')
                caps[ports[0].name]=self.external[net]
            delays,total=n.moments(caps)
            if net in m.by_net:load[m.by_net[net]]=total
            sinks=[p for p in n.pins if p.name!=n.driver and p.name in m.index]
            r,_=transfer_resistance(n.pin_nodes[n.driver],
                [n.pin_nodes[p.name] for p in sinks],n.resistors)
            sink_index={p.name:i for i,p in enumerate(sinks)}
            for k in ids:
                a=m.arcs[k];dest=m.pins[a['t']]['name'];source=m.pins[a['s']]['name']
                if source!=n.driver or dest not in delays:
                    raise ValueError(f'Estimated/native wire topology mismatch: {net}')
                tau[k]=delays[dest][a['fr']];covered.add(k)
                arc_tau.append((k,tau[k]))
                row=r[sink_index[dest]]
                for j,value in enumerate(row):
                    p=m.index[sinks[j].name]
                    if value>0 and m.rf_sink[p]:
                        cr.append(k);ci.append(2*p+a['fr']);cv.append(value)
            rr.extend(cr);cc.extend(ci);vv.extend(cv)
            self.cache[net]=(n,signature,total.copy(),arc_tau,cr,ci,cv)
        missing=[k for net in dirty for k in self.arcs_by_net.get(net,[])
                 if k not in covered and abs(m.rc_base_tau[k])>1e-8]
        if missing:raise ValueError(f'Uncovered nonideal wire arcs: {len(missing)}')
        rc=coo_matrix((vv,(rr,cc)),shape=(len(m.arcs),2*m.n)).tocsr()
        if not initial:
            keep=np.ones(len(m.arcs))
            for net in dirty:keep[self.arcs_by_net.get(net,[])]=0.
            previous=self._matrix.multiply(keep[:,None]).tocsr();previous.eliminate_zeros()
            rc=previous+rc
        self._load=load.copy();self._tau=tau.copy();self._matrix=rc
        self._pin_caps=m.rf_base_pin.copy()
        return load,tau,rc

    def refresh(self):
        load,tau,rc=self.arrays()
        if self.last_recomputed:self.engine.update_parasitics(load,tau,rc)
        return dict(estimated_nets=len(self.estimator.networks),
                    sensitivity_nonzeros=rc.nnz,generation=self.estimator.generation,
                    recomputed_nets=len(self.last_recomputed))
