"""Deterministic placement-to-RC estimation, with dirty-net incremental updates.

No trained parameters and no global-route input. FLUTE supplies the initial
topology. Geometry changes rebuild only incident nets; sizing at unchanged pin
locations reuses wire RC. All lengths are um, R kohm, C fF, delays ps.

Layer-aware pattern routing is an approximation, not a legal global router.
Its GCell occupancy includes all estimated nets, and changed cells invalidate
other nets traversing the affected region (including fixed clock nets).
"""
from collections import defaultdict
from dataclasses import dataclass, replace
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np


class Flute:
    """External FLUTE, built once; no OpenROAD process in the iteration loop."""
    _lock = threading.Lock()

    def __init__(self, source=None, cache=None):
        root = Path(__file__).resolve().parents[1]
        source = Path(source or os.environ.get('QUAD_FLUTE_SOURCE',
                                               root/'third_party/flute'))
        cache = Path(cache or root/'cache/placement_rc')
        wrapper = Path(__file__).with_name('flute_bridge.cpp')
        dependencies = [wrapper, source/'flute.cpp', source/'flute.hpp', source/'LICENSE']
        if any(not p.is_file() for p in dependencies):
            raise FileNotFoundError('Optional FLUTE sources missing; set QUAD_FLUTE_SOURCE to a FLUTE source directory')
        fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in dependencies)).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache/f'flute_{fingerprint[:16]}.so'
        if not binary.exists():
            temporary = binary.with_suffix(f'.{os.getpid()}.tmp.so')
            subprocess.run(['g++', '-std=c++17', '-O3', '-fPIC', '-shared',
                            '-I'+str(source), str(wrapper), str(source/'flute.cpp'),
                            '-o', str(temporary)], check=True, capture_output=True)
            temporary.replace(binary)
        self.lib = ctypes.CDLL(str(binary))
        self.lib.quad_flute_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        ptr = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        self.lib.quad_flute_tree.argtypes = [ctypes.c_int, ptr, ptr, ptr]
        lut = source/'lut.ICCAD2015'
        for name in ('POWV9.dat', 'POST9.dat'):
            if not (lut/name).is_file(): raise FileNotFoundError(lut/name)
        print('FLUTE: Dr. Chris C. N. Chu, Iowa State University, '
              'http://class.ee.iastate.edu/cnchu/', flush=True)
        with self._lock:
            self.lib.quad_flute_init(os.fsencode(lut/'POWV9.dat'), os.fsencode(lut/'POST9.dat'))
        self.provenance = dict(source=str(source), fingerprint=fingerprint,
                               license=str(source/'LICENSE'), binary=str(binary))

    def edges(self, points):
        """Rectilinear-tree branches as endpoint pairs, on a 1 nm grid.

        Large nets are spatially partitioned, connected through representatives;
        this is a documented heuristic rather than passing MAXD to FLUTE.
        """
        # Two-terminal nets need no LUT or NumPy sort. They are common in
        # local optimization, including after duplicate GCell snapping.
        if isinstance(points,(list,tuple)) and len(points)<=2:
            import math
            unique=set()
            for point in points:
                if len(point)!=2 or any(not math.isfinite(float(v)) or abs(float(v))>2e6 for v in point):
                    raise ValueError('Invalid or out-of-range pin coordinates')
                unique.add(tuple(round(float(v)*1000)/1000 for v in point))
            return [tuple(sorted(unique))] if len(unique)==2 else []
        points=np.asarray(points,dtype=float)
        if points.size==0:return []
        if points.ndim!=2 or points.shape[1]!=2 or not np.all(np.isfinite(points)) or np.any(np.abs(points)>2e6):
            raise ValueError('Invalid or out-of-range pin coordinates')
        p = np.unique(np.rint(points*1000).astype(np.int32), axis=0)
        if len(p) <= 1: return []
        if len(p) >= 1000:
            groups = []
            def partition(q):
                if len(q) < 128: groups.append(q); return
                axis = np.argmax(np.ptp(q.astype(np.int64), axis=0))
                q = q[np.argsort(q[:,axis], kind='stable')]; mid = len(q)//2
                partition(q[:mid]); partition(q[mid:])
            partition(p)
            out = self.edges(np.array([g[0] for g in groups])/1000)
            for g in groups: out.extend(self.edges(g/1000))
            return out
        x = np.ascontiguousarray(p[:,0]); y = np.ascontiguousarray(p[:,1])
        out = np.empty((2*len(p)-2, 3), dtype=np.int32)
        with self._lock: n = self.lib.quad_flute_tree(len(p), x, y, out)
        if n != len(out): raise RuntimeError('FLUTE failed')
        edges = set()
        for x, y, j in out:
            a = (int(x), int(y)); b = tuple(int(v) for v in out[j,:2])
            if a != b: edges.add(tuple(sorted((a,b))))
        return [(tuple(v/1000 for v in a), tuple(v/1000 for v in b)) for a,b in sorted(edges)]


@dataclass(frozen=True)
class Pin:
    name: str
    net: str
    x: float
    y: float
    direction: str
    layer: int = 1
    clock: bool = False
    port: bool = False


@dataclass(frozen=True)
class Layer:
    number: int
    direction: str
    resistance: float
    capacitance: float
    pitch: float


class Technology:
    def __init__(self, layers, vias, signal=(2,7), clock=(4,7)):
        self.layers = {l.number:l for l in layers}; self.vias = dict(vias)
        self.signal = signal; self.clock = clock

    @classmethod
    def asap7(cls, root=None):
        """Read evaluator RC and LEF; do not fit against benchmark GR labels."""
        import re
        root = Path(root or Path(__file__).resolve().parents[1]/'platform/asap7')
        text = (root/'util/setRC.tcl').read_text()
        rc = {int(n):(float(r),float(c)) for n,r,c in re.findall(
            r'set_layer_rc -layer M(\d+) -resistance (\S+) -capacitance (\S+)', text)}
        vias = {int(n):float(r) for n,r in re.findall(
            r'set_layer_rc -via V(\d+) -resistance (\S+)', text)}
        lef = (root/'lef/asap7_tech_1x_201209.lef').read_text()
        layers = []
        for n,(r,c) in rc.items():
            block = re.search(r'\bLAYER M'+str(n)+r'\b(.*?)\bEND M'+str(n)+r'\b', lef, re.S)[1]
            direction = re.search(r'DIRECTION\s+(HORIZONTAL|VERTICAL)', block)[1][0]
            pitch = float(re.search(r'\bPITCH\s+([\d.]+)', block)[1])
            layers.append(Layer(n,direction,r,c,pitch))
        return cls(layers,vias)


class DemandGrid:
    def __init__(self, technology, tile=1.0, adjustment=.25):
        # Match OpenROAD: adjustment is the fraction removed, not retained.
        # E.g. set_global_routing_layer_adjustment * .25 retains 75%.
        if tile <= 0 or not np.isfinite(adjustment) or adjustment >= 1: raise ValueError('Invalid grid')
        self.tech=technology; self.tile=tile; self.adjustment=adjustment
        self.demand=defaultdict(float); self.nets=defaultdict(set)
        self.observers=defaultdict(set);self.watch=None

    def bins(self, a, b, layer):
        axis = 0 if a[1] == b[1] else 1
        if a[axis] == b[axis]: return []
        if a[1-axis] != b[1-axis]: raise ValueError('Non-Manhattan segment')
        lo,hi=sorted((a[axis],b[axis])); fixed=int(np.floor(a[1-axis]/self.tile))
        result=[]
        for i in range(int(np.floor(lo/self.tile)),int(np.ceil(hi/self.tile))):
            key=(layer,i,fixed) if axis==0 else (layer,fixed,i)
            fraction=(min(hi,(i+1)*self.tile)-max(lo,i*self.tile))/self.tile
            if fraction>0: result.append((key,fraction))
        return result

    def cost(self, a, b, layer):
        capacity=self.capacity(layer)
        bins=self.bins(a,b,layer)
        if self.watch is not None:self.watch.update(k for k,_ in bins)
        return sum(f*(1.+4.*max(0.,(self.demand[k]+f)/capacity-.8)**2)
                   for k,f in bins)*self.tile

    def add(self, net, occupancy, sign=1,watch_bins=()):
        for k,v in occupancy.items():
            self.demand[k]+=sign*v
            if sign>0: self.nets[k].add(net)
            else: self.nets[k].discard(net)
        for k in watch_bins:
            if sign>0:self.observers[k].add(net)
            else:self.observers[k].discard(net)

    def penalty(self,key,demand):
        capacity=self.capacity(key[0])
        return 1.+4.*max(0.,demand/capacity-.8)**2

    def capacity(self,layer):
        return self.tile/self.tech.layers[layer].pitch*(1.-self.adjustment)


@dataclass
class Network:
    name: str
    pins: tuple
    nodes: tuple
    resistors: tuple
    wire_caps: np.ndarray
    pin_nodes: dict
    driver: str
    wirelength: float
    via_count: int
    occupancy: dict
    watch_bins: frozenset = frozenset()

    def moments(self, pin_caps):
        """Exact first RC moment, independently for rise/fall pin loads."""
        count=len(self.nodes); adj=[[] for _ in range(count)]
        for a,b,r in self.resistors: adj[a].append((b,r)); adj[b].append((a,r))
        root=self.pin_nodes[self.driver]; parent={root:(-1,0.)}; order=[root]
        for a in order:
            for b,r in adj[a]:
                if b not in parent: parent[b]=(a,r); order.append(b)
        if len(order)!=count or len(self.resistors)!=count-1:
            raise ValueError('RC network must be a connected tree')
        caps=np.repeat(self.wire_caps[:,None],2,axis=1)
        for name,node in self.pin_nodes.items():
            if name!=self.driver: caps[node]+=np.asarray(pin_caps.get(name,(0.,0.)))
        load=caps.copy(); delay=np.zeros_like(load)
        for b in reversed(order[1:]): load[parent[b][0]]+=load[b]
        for b in order[1:]:
            a,r=parent[b]; delay[b]=delay[a]+r*load[b]
        return {name:delay[node].copy() for name,node in self.pin_nodes.items()},load[root]


class PlacementRC:
    def __init__(self, pins, technology, flute=None, tile=1., adjustment=.25,
                 congestion=True,congestion_threshold=.02,reuse_topology=True,fast_patterns=True):
        pins=tuple(pins)
        self.pins={p.name:p for p in pins}
        if len(self.pins)!=len(pins): raise ValueError('Duplicate pin')
        self.by_net=defaultdict(list)
        for p in pins: self.by_net[p.net].append(p.name)
        self.tech=technology; self.flute=flute or Flute(); self.congestion=congestion
        self.fast_patterns=bool(fast_patterns)
        if congestion_threshold<0:raise ValueError('Invalid congestion cost threshold')
        self.congestion_threshold=congestion_threshold
        self.reuse_topology=reuse_topology;self._tree_age=defaultdict(int)
        if not pins:raise ValueError('No physical pins')
        self.bounds=(min(p.x for p in pins),min(p.y for p in pins),
                     max(p.x for p in pins),max(p.y for p in pins))
        self.grid=DemandGrid(technology,tile,adjustment); self.networks={}
        self.generation=0
        def priority(net):
            group=[self.pins[n] for n in self.by_net[net]]
            non_leaf=any(p.clock for p in group) and not any(p.name.endswith('/CLK') for p in group)
            return (not non_leaf,net)
        for net in sorted(self.by_net,key=priority): self._rebuild(net)

    def _rebuild(self, net):
        self._tree_age[net]=0
        old=self.networks.pop(net,None)
        if old: self.grid.add(net,old.occupancy,-1,old.watch_bins)
        pins=tuple(self.pins[n] for n in self.by_net[net])
        drivers=[p for p in pins if p.direction==('INPUT' if p.port else 'OUTPUT')]
        if not drivers: return  # Constant/physical-only nets without a timing source.
        if len(drivers)!=1: raise ValueError(f'Multiple drivers: {net}')
        network=self._route(net,pins,drivers[0])
        self.networks[net]=network; self.grid.add(net,network.occupancy,watch_bins=network.watch_bins)

    def _follow_pin_branches(self,net):
        """Reuse a tree between FLUTE refreshes, as in DAC'22.

        Under unchanged pin coordinate ordering, all rectilinear branch
        coordinates remain associated with their original pin coordinates.
        Rank/tie changes force a complete reconstruction. RSMT optimality can
        still change at fixed ranks, so refresh after at most nine reuses.
        """
        old=self.networks.get(net)
        if old is None or self._tree_age[net]>=9:return False
        maps=[]
        for axis in ['x','y']:
            mapping={}
            for p in old.pins:
                before=getattr(p,axis);after=getattr(self.pins[p.name],axis)
                if before in mapping and mapping[before]!=after:return False
                mapping[before]=after
            keys=sorted(mapping);values=[mapping[k] for k in keys]
            if any(a>=b for a,b in zip(values,values[1:])):return False
            maps.append(mapping)
        if any(x not in maps[0] or y not in maps[1] for x,y,_ in old.nodes):return False
        nodes=tuple((maps[0][x],maps[1][y],l) for x,y,l in old.nodes)
        caps=np.zeros(len(nodes));res=[];length=0.
        for a,b,r in old.resistors:
            u,v=nodes[a],nodes[b]
            if u[2]==v[2]:
                distance=abs(u[0]-v[0])+abs(u[1]-v[1]);layer=self.tech.layers[u[2]]
                r=distance*layer.resistance;caps[a]+=distance*layer.capacitance/2;caps[b]+=distance*layer.capacitance/2
                length+=distance
            res.append((a,b,r))
        # Non-congestion mode does not use grid demand. Retire stale occupancy;
        # the next full reconstruction restores it if requested.
        self.grid.add(net,old.occupancy,-1,old.watch_bins)
        self.networks[net]=Network(net,tuple(self.pins[n] for n in self.by_net[net]),
            nodes,tuple(res),caps,old.pin_nodes.copy(),old.driver,length,old.via_count,{})
        self._tree_age[net]+=1;return True

    def tree_edges(self,net,pins,driver):
        return self.flute.edges([(p.x,p.y) for p in pins])

    def _route(self, net, pins, driver):
        self.grid.watch=set() if self.congestion else None
        # OpenROAD applies the clock range only to non-leaf clock nets.
        # Leaf nets feeding register clock pins use the signal layer range.
        leaf_clock=any(p.clock and p.name.endswith('/CLK') for p in pins)
        lo,hi=self.tech.clock if driver.clock and not leaf_clock else self.tech.signal
        if not self.congestion and self.fast_patterns:
            # Both L orientations have identical length and layer-change cost
            # without congestion. The original enumerator chooses X first on
            # that tie, and its layer cost chooses the lowest preferred layer.
            layers={direction:next((l for l in range(lo,hi+1)
                if self.tech.layers[l].direction==direction),None) for direction in ('H','V')}
            segments=[];via_stacks=[]
            for a,b in self.tree_edges(net,pins,driver):
                corner=(b[0],a[1]);parts=[]
                for u,v,direction in ((a,corner,'H'),(corner,b,'V')):
                    if u==v:continue
                    layer=layers[direction]
                    if layer is None:raise ValueError('No preferred routing layer')
                    parts.append((u,v,layer))
                segments.extend(parts)
                if len(parts)==2:via_stacks.append((corner,parts[0][2],parts[1][2]))
            return self._network(net,pins,driver,segments,via_stacks)
        # Use the lowest legal preferred-direction layer. Congestion can promote
        # a segment to a higher layer; via resistance is included in path cost.
        def layer_for(a,b):
            direction='H' if a[1]==b[1] else 'V'
            choices=[l for l in range(lo,hi+1) if self.tech.layers[l].direction==direction]
            return min(choices,key=lambda l:(self.grid.cost(a,b,l) if self.congestion
                                             else abs(a[0]-b[0])+abs(a[1]-b[1]))+.02*(l-lo))
        segments=[]; via_stacks=[]
        for a,b in self.tree_edges(net,pins,driver):
            candidates=[[a,(b[0],a[1]),b],[a,(a[0],b[1]),b]]
            if self.congestion:
                # A bounded Z search, with two possible detours in each axis.
                mid=((a[0]+b[0])/2,(a[1]+b[1])/2); step=self.grid.tile
                for offset in (-step,step):
                    x=round((mid[0]+offset)/step)*step
                    y=round((mid[1]+offset)/step)*step
                    x=min(max(x,self.bounds[0]),self.bounds[2])
                    y=min(max(y,self.bounds[1]),self.bounds[3])
                    candidates.extend([[a,(x,a[1]),(x,b[1]),b],
                                       [a,(a[0],y),(b[0],y),b]])
            def evaluate(path):
                parts=[(u,v,layer_for(u,v)) for u,v in zip(path,path[1:]) if u!=v]
                cost=sum(self.grid.cost(u,v,l) if self.congestion else
                         abs(u[0]-v[0])+abs(u[1]-v[1]) for u,v,l in parts)
                cost+=.02*sum(abs(parts[i][2]-parts[i-1][2]) for i in range(1,len(parts)))
                return cost,parts
            _,parts=min((evaluate(p) for p in candidates),key=lambda p:p[0])
            segments.extend(parts)
            for (_,v,l),(_,_,m) in zip(parts,parts[1:]): via_stacks.append((v,l,m))
        return self._network(net,pins,driver,segments,via_stacks)

    def _network(self,net,pins,driver,segments,via_stacks):
        """Build the physical RC tree from independently chosen route segments."""
        # Union collinear intervals and split at all original segment endpoints.
        # This counts shared wire once even when two pattern routes overlap.
        lines=defaultdict(list)
        for a,b,l in segments:
            axis=0 if a[1]==b[1] else 1
            lines[(l,axis,a[1-axis])].append(tuple(sorted((a[axis],b[axis]))))
        segments=[]
        for (l,axis,fixed),intervals in sorted(lines.items()):
            cuts=sorted({v for interval in intervals for v in interval})
            for x,y in zip(cuts,cuts[1:]):
                if not any(a<=x and y<=b for a,b in intervals): continue
                a=(x,fixed) if axis==0 else (fixed,x)
                b=(y,fixed) if axis==0 else (fixed,y)
                segments.append((a,b,l))
        layers_at=defaultdict(set)
        for a,b,l in segments: layers_at[a].add(l); layers_at[b].add(l)
        for p in pins: layers_at[(p.x,p.y)].add(p.layer)
        # FLUTE uses the same 1 nm rounding as the native export. Reject missing
        # pin junctions rather than silently disconnecting non-grid coordinates.
        for xy,layers in layers_at.items():
            for l in range(min(layers),max(layers)): via_stacks.append((xy,l,l+1))
        edges={}; cap=defaultdict(float)
        def edge(a,b,r):
            if a!=b: edges[tuple(sorted((a,b)))]=r
        for a,b,l in segments:
            length=abs(a[0]-b[0])+abs(a[1]-b[1]); tech=self.tech.layers[l]
            u=(*a,l); v=(*b,l); edge(u,v,length*tech.resistance)
            cap[u]+=length*tech.capacitance/2; cap[v]+=length*tech.capacitance/2
        for xy,l,m in via_stacks:
            for k in range(min(l,m),max(l,m)): edge((*xy,k),(*xy,k+1),self.tech.vias[k])
        for p in pins: cap[(p.x,p.y,p.layer)]+=0.
        nodes=tuple(sorted(set(cap)|{n for pair in edges for n in pair})); index={n:i for i,n in enumerate(nodes)}
        # Pattern overlaps can form loops. Prune by resistance to a tree, then
        # remove capacitance/occupancy of removed wire; never double-count it.
        parent=list(range(len(nodes)))
        def find(i):
            while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
            return i
        selected=[]; wire_caps=np.zeros(len(nodes)); actual_occupancy=defaultdict(float)
        length=0.; vias=0
        for (a,b),r in sorted(edges.items(),key=lambda item:(item[1],item[0])):
            i,j=index[a],index[b]; u,v=find(i),find(j)
            if u==v: continue
            parent[u]=v; selected.append((i,j,r))
            if a[2]==b[2]:
                distance=abs(a[0]-b[0])+abs(a[1]-b[1]); length+=distance
                c=distance*self.tech.layers[a[2]].capacitance/2
                wire_caps[i]+=c; wire_caps[j]+=c
                if self.congestion:
                    for k,f in self.grid.bins(a[:2],b[:2],a[2]): actual_occupancy[k]+=f
            else: vias+=1
        if len(selected)!=len(nodes)-1: raise ValueError(f'Disconnected geometry: {net}')
        watch=frozenset(self.grid.watch or ());self.grid.watch=None
        return Network(net,pins,nodes,tuple(selected),wire_caps,
                       {p.name:index[(p.x,p.y,p.layer)] for p in pins},driver.name,
                       length,vias,dict(actual_occupancy),watch)

    def update(self, positions):
        """Update named pins; return direct and congestion-induced dirty nets.

        Fixed clock pins are rejected. Connectivity changes require rebuilding
        this object and the STA graph; they are never accepted as a pin move.
        """
        started=time.perf_counter(); direct=set(); old_bins=set()
        changed={}
        for name,xy in positions.items():
            p=self.pins[name]
            if len(xy)!=2 or not np.all(np.isfinite(xy)) or np.any(np.abs(xy)>2e6):
                raise ValueError('Invalid or out-of-range pin coordinates')
            xy=tuple(round(float(v),3) for v in xy)
            if (p.x,p.y)==xy: continue
            if p.clock: raise ValueError('Clock pin positions are fixed')
            changed[name]=replace(p,x=xy[0],y=xy[1]); direct.add(p.net)
            if p.net in self.networks: old_bins.update(self.networks[p.net].occupancy)
        before=dict(self.grid.demand) if self.congestion and changed else {}
        self.pins.update(changed)
        reused=0
        for net in sorted(direct):
            if not self.congestion and self.reuse_topology and self._follow_pin_branches(net):reused+=1
            else:self._rebuild(net)
        for net in direct:
            if net in self.networks:
                old_bins.update(self.networks[net].occupancy)
        affected=set();changed_bins=0;cost_delta={}
        if self.congestion:
            for k in old_bins:
                old=self.grid.penalty(k,before.get(k,0.));new=self.grid.penalty(k,self.grid.demand[k])
                if abs(new-old)<=1e-12:continue
                changed_bins+=1
                cost_delta[k]=abs(new-old)
                # Watch alternative candidates too: a net can benefit when a
                # previously avoided region becomes less congested.
                affected.update(self.grid.observers[k]);affected.update(self.grid.nets[k])
        induced=set()
        if self.congestion:
            changed_keys=cost_delta.keys()
            for net in affected-direct:
                n=self.networks[net]
                # Bound how much any watched alternative's cost can change.
                # A small change in one bin of a long net must not invalidate
                # every neighbour as though its entire route had changed.
                bound=sum(cost_delta[k] for k in n.watch_bins&changed_keys)*self.grid.tile
                current=sum(f*self.grid.penalty(k,self.grid.demand[k]) for k,f in n.occupancy.items())*self.grid.tile
                if bound>self.congestion_threshold*max(current,self.grid.tile):induced.add(net)
        for net in sorted(induced): self._rebuild(net)
        self.generation+=bool(changed)
        return dict(direct=sorted(direct),congestion_induced=sorted(induced),
                    reused_topology_nets=reused,
                    changed_congestion_bins=changed_bins,
                    generation=self.generation,seconds=time.perf_counter()-started)

    def write_spef(self, path, design='placement_estimate'):
        """Explicit named SPEF, wire-only capacitance; Liberty provides pin C."""
        def escape(name):
            return ''.join('\\'+c if c in '[]\\:$' else c for c in name)
        port_names={p.name for p in self.pins.values() if p.port}
        # OpenSTA preserves the bus spelling of top-level ports. Internal
        # identifiers use SPEF escaping, matching its native SPEF writer.
        def net_name(name):return name if name in port_names else escape(name)
        with Path(path).open('w') as f:
            f.write(f'*SPEF "ieee 1481-1999"\n*DESIGN "{design}"\n'
                    '*DATE "deterministic placement estimate"\n'
                    '*VENDOR "Quad placement RC"\n*PROGRAM "placement_rc"\n'
                    '*VERSION "1.0"\n*DESIGN_FLOW "NAME_SCOPE LOCAL" "PIN_CAP NONE"\n'
                    '*DIVIDER /\n*DELIMITER :\n*BUS_DELIMITER []\n'
                    '*T_UNIT 1 PS\n*C_UNIT 1 FF\n*R_UNIT 1 KOHM\n*L_UNIT 1 HENRY\n')
            ports=[p for p in self.pins.values() if p.port]
            if ports:
                f.write('\n*PORTS\n')
                for p in sorted(ports,key=lambda p:p.name):
                    f.write(f'{p.name} {"I" if p.direction=="INPUT" else "O"}\n')
            for net,n in sorted(self.networks.items()):
                names={i:f'{net_name(net)}:{i+1}' for i in range(len(n.nodes))}
                # Separate coincident logical pins with an ideal connection.
                conns=[]; resistors=[(names[a],names[b],r) for a,b,r in n.resistors]
                for p in n.pins:
                    name=p.name if p.port else ':'.join(map(escape,p.name.rsplit('/',1)))
                    direction=('I' if p.direction=='INPUT' else 'O')
                    conns.append(f'{"*P" if p.port else "*I"} {name} {direction}')
                    resistors.append((name,names[n.pin_nodes[p.name]],0.))
                f.write(f'\n*D_NET {net_name(net)} {n.wire_caps.sum():.12g}\n*CONN\n')
                f.write('\n'.join(conns)+'\n*CAP\n')
                for i,c in enumerate(n.wire_caps): f.write(f'{i+1} {names[i]} {c:.12g}\n')
                f.write('*RES\n')
                for i,(a,b,r) in enumerate(resistors): f.write(f'{i+1} {a} {b} {r:.12g}\n')
                f.write('*END\n')


def read_pins(path):
    import csv
    with Path(path).open() as f:
        return [Pin(r['pin'],r['net'],round(float(r['x_um']),3),round(float(r['y_um']),3),
                    r['direction'],int(r['layer']),bool(int(r['clock'])),bool(int(r['port'])))
                for r in csv.DictReader(f,delimiter='\t')]
