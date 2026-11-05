"""Batched clock-tree LCA queries for startpoint-aware CRPR corrections."""
from collections import OrderedDict
import hashlib
import numpy as np
import torch

_CACHE=OrderedDict()


def clock_lca_table(parent,capture,launch,levels,block_size=64):
    parent=np.asarray(parent,dtype=np.int64);n=len(parent)
    capture=np.asarray(capture,dtype=np.int64);launch=np.asarray(launch,dtype=np.int64)
    depth=np.zeros(n+1,dtype=np.int32)
    active=np.flatnonzero(parent>=0)
    order=active[np.argsort(np.asarray(levels)[active],kind='stable')]
    for node in order:depth[node]=depth[parent[node]]+1
    up=[np.r_[np.where(parent>=0,parent,n),n]]
    for _ in range(max(1,int(depth.max()).bit_length())-1):up.append(up[-1][up[-1]])
    result=np.full((len(capture),len(launch)),-1,dtype=np.int32)
    launch=np.where(launch>=0,launch,n)
    for start in range(0,len(capture),block_size):
        c=capture[start:start+block_size]
        a=np.broadcast_to(c[:,None],(len(c),len(launch))).copy()
        b=np.broadcast_to(launch[None,:],a.shape).copy()
        swap=depth[a]<depth[b];a,b=np.where(swap,b,a),np.where(swap,a,b)
        difference=depth[a]-depth[b]
        for bit,ancestors in enumerate(up):a=np.where((difference&(1<<bit))!=0,ancestors[a],a)
        for ancestors in reversed(up):
            different=ancestors[a]!=ancestors[b]
            a,b=np.where(different,ancestors[a],a),np.where(different,ancestors[b],b)
        lca=np.where(a==b,a,up[0][a])
        result[start:start+len(c)]=np.where(lca<n,lca,-1)
    return result


def shared_clock_lca(parent,capture,launch,levels,device):
    digest=hashlib.sha256()
    for values in (parent,capture,launch,levels):
        a=np.asarray(values,dtype=np.int64);digest.update(str(a.shape).encode());digest.update(a.tobytes())
    key=(str(device),digest.hexdigest())
    if key not in _CACHE:
        array=clock_lca_table(parent,capture,launch,levels)
        _CACHE[key]=torch.as_tensor(array,dtype=torch.int64,device=device)
        while len(_CACHE)>4:_CACHE.popitem(last=False)
    _CACHE.move_to_end(key)
    return _CACHE[key]
