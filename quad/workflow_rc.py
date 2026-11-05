"""Native placement trees, cached by net, with GPU RC solves and derivatives."""
import hashlib
from pathlib import Path
import numpy as np
from placement_rc import Network, read_pins
from rc_tree import spef_networks
from tree_rc_cuda import TreePlacementBridge


class PlacementForest:
    def __init__(self,spef,geometry):
        self.networks={};self.signatures={};self.generation=0
        self.refresh(spef,geometry)

    def refresh(self,spef,geometry):
        pins={p.name:p for p in read_pins(geometry)}
        networks={};signatures={};changed=[]
        for net,conn,res,wire in spef_networks(spef):
            physical=tuple(pins[p] for p in conn.values() if p in pins)
            drivers=[p for p in physical if (p.port and p.direction=='INPUT') or (not p.port and p.direction=='OUTPUT')]
            if not physical:continue
            if len(drivers)!=1:raise ValueError('Placement RC requires one driver: '+net)
            # DEF reloads can reorder SPEF connections without changing any
            # geometry or RC. Cache by electrical content, not serialization.
            signature=(tuple(sorted(conn.items())),
                       tuple(sorted((min(a,b),max(a,b),r) for a,b,r in res)),
                       tuple(sorted(wire.items())),tuple(sorted(physical,key=lambda p:p.name)))
            signatures[net]=signature
            if self.signatures.get(net)==signature:
                networks[net]=self.networks[net];continue
            nodes=list(dict.fromkeys([*conn,*wire,*[v for a,b,r in res for v in (a,b)]]))
            index={x:i for i,x in enumerate(nodes)}
            networks[net]=Network(net,physical,tuple(nodes),
                tuple((index[a],index[b],r) for a,b,r in res if a!=b),
                np.array([wire.get(v,0.) for v in nodes]),
                {name:index[node] for node,name in conn.items()},drivers[0].name,0.,0,{})
            changed.append(net)
        if self.networks and self.networks.keys()!=networks.keys():
            raise ValueError('Changed net topology requires a new RC forest')
        self.networks=networks;self.signatures=signatures;self.generation+=1
        self.changed_nets=changed
        return dict(nets=len(networks),changed_nets=len(changed),generation=self.generation)


def attach_forest(pair,forest,shared=None):
    pair.forest=forest
    pair.late_bridge=TreePlacementBridge(forest,pair.late,shared=shared)
    pair.early_bridge=TreePlacementBridge(forest,pair.early,shared=pair.late_bridge)
    pair.late_bridge.refresh();pair.early_bridge.refresh()


def commit_electrical(engine,replacements,prediction):
    import copy
    m=copy.copy(engine.m);m.masters={**m.masters,**replacements}
    m.rf_base_pin=m.pin_caps(m.masters);m.rc_pin_caps=m.rf_base_pin.copy()
    # Root/clock seeds and calibration anchors stay at the epoch reference.
    # Only cell tables and pin caps change; the RC bridge refreshes moments.
    engine.set_model(m)


def commit_sizing_pairs(pairs,replacements,predictions):
    for c,pair in pairs.items():
        for a,e in [('max',pair.late),('min',pair.early)]:
            commit_electrical(e,replacements,predictions[c][a])
        pair.late_bridge.refresh();pair.early_bridge.refresh()
