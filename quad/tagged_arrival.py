"""GPU unique-startpoint queues with hard-forward automatic differentiation.

K is per constraint class, not a global queue shared by incompatible classes.
The custom reverse traverses the retained predecessor DAG; it differentiates
both electrical arc delays and the launch clock arrival. Top-K pruning is an
explicit approximation, and all overflows are exposed to the caller.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch


def library():
    source = Path(__file__).with_suffix('.cu')
    compiler = os.environ.get('NVCC') or shutil.which('nvcc')
    if not compiler:
        for p in ('/usr/local/cuda/bin/nvcc',):
            if Path(p).exists(): compiler = p; break
    if not compiler: raise RuntimeError('nvcc is required for unique-origin GPU propagation')
    arch = ''.join(map(str, torch.cuda.get_device_capability()))
    digest = hashlib.sha256(source.read_bytes()+arch.encode()).hexdigest()[:16]
    target = source.parents[1]/'cache/workflow'/digest/'tagged_arrival.so'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(f'.{os.getpid()}.so')
        subprocess.run([compiler, '-O3', '-shared', '-Xcompiler', '-fPIC',
                        '-arch=sm_'+arch, str(source), '-o', str(temp)], check=True)
        temp.replace(target)
    lib = ctypes.CDLL(str(target))
    p, i = ctypes.c_void_p, ctypes.c_int
    lib.tagged_forward.argtypes = [p,i,p,p,i,i,ctypes.c_double]+[p]*14
    lib.tagged_forward.restype = i
    lib.tagged_backward.argtypes = [p,i,p,p,i,i]+[p]*6
    lib.tagged_backward.restype = i
    return lib


class _Propagate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delay, physical_at, runtime):
        if delay.dtype != torch.float64 or physical_at.dtype != torch.float64:
            raise ValueError('Tagged arrival requires FP64 electrical inputs')
        shape = (runtime.n, runtime.groups, runtime.k)
        at = torch.empty(shape, dtype=delay.dtype, device=delay.device)
        integers = [torch.empty(shape, dtype=torch.int32, device=delay.device) for _ in range(4)]
        tag, pred, arc, physical = integers
        overflow = torch.empty(shape[:2], dtype=torch.int32, device=delay.device)
        delay, physical_at = delay.contiguous(), physical_at.contiguous().view(-1)
        args = [x.data_ptr() for x in runtime.arrays]
        args += [x.data_ptr() for x in (delay, physical_at, at, tag, pred, arc, physical, overflow)]
        status = runtime.lib.tagged_forward(torch.cuda.current_stream(delay.device).cuda_stream,
            len(runtime.starts)-1, runtime.starts.ctypes.data, runtime.nodes.data_ptr(),
            runtime.groups, runtime.k, runtime.sign, *args)
        if status: raise RuntimeError('Tagged CUDA forward failed: '+str(status))
        ctx.runtime = runtime
        ctx.save_for_backward(pred, arc, physical)
        ctx.arc_count = delay.numel()
        ctx.mark_non_differentiable(tag, overflow)
        return at, tag, overflow

    @staticmethod
    def backward(ctx, grad_at, grad_tag, grad_overflow):
        runtime = ctx.runtime
        pred, arc, physical = ctx.saved_tensors
        gat = grad_at.contiguous().clone()
        gd = torch.zeros(ctx.arc_count, dtype=gat.dtype, device=gat.device)
        gp = torch.zeros(runtime.n, dtype=gat.dtype, device=gat.device)
        status = runtime.lib.tagged_backward(torch.cuda.current_stream(gat.device).cuda_stream,
            len(runtime.starts)-1, runtime.starts.ctypes.data, runtime.nodes.data_ptr(),
            runtime.groups, runtime.k, *[x.data_ptr() for x in (pred,arc,physical,gat,gd,gp)])
        if status: raise RuntimeError('Tagged CUDA reverse failed: '+str(status))
        return gd, gp, None


class TaggedArrival:
    def __init__(self, source, target, levels, launch, launch_group, root_tag,
                 k=16, groups=5, analysis='max', device='cuda:0'):
        self.n = len(levels); self.k = int(k); self.groups = int(groups)
        if not 1 <= self.k <= 256 or analysis not in ('min','max'):
            raise ValueError('Invalid unique-origin queue configuration')
        self.sign = 1. if analysis == 'max' else -1.
        source, target = np.asarray(source), np.asarray(target)
        levels = np.asarray(levels)
        if len(source) != len(target) or np.any(levels[source] >= levels[target]):
            raise ValueError('Tagged graph must be a levelized DAG')
        order = np.argsort(target, kind='stable')
        offsets = np.r_[0, np.cumsum(np.bincount(target, minlength=self.n))]
        nodes = np.argsort(levels, kind='stable')
        _, counts = np.unique(levels[nodes], return_counts=True)
        self.starts = np.asarray(np.r_[0,np.cumsum(counts)], dtype=np.int32)
        tensor = lambda x: torch.as_tensor(np.asarray(x),dtype=torch.int32,device=device)
        self.nodes = tensor(nodes)
        self.arrays = [tensor(x) for x in (offsets,order,source,launch,launch_group,root_tag)]
        self.lib = library()

    def __call__(self, delay, physical_at):
        return _Propagate.apply(delay, physical_at.view(-1), self)
