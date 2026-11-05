"""Shared calibrated models and final constraint checks for the current solver."""
import numpy as np
from forward_sta import ForwardModel
from gpu_calibration import calibrate_gpu
from sta_server import CORNERS

def feasible_constraints(candidate,config,check_erc=True,check_hold=True):
    """Preserve contest hold tolerances and zero electrical/routing violations."""
    for c,row in candidate.items():
        limits=config['scenarios'][c]
        if check_hold:
            if -row['hold_tns_ns']>limits['hold_tns_tolerance_ns']+1e-6:return False
            if -row['hold_wns_ns']>limits['hold_wns_tolerance_ns']+1e-6:return False
        if check_erc:
            if row['slew_violation_sum_ps']>config['epsilon']['slew_ps']:return False
            if row['cap_violation_sum_ff']>config['epsilon']['cap_ff']:return False
            if row['fanout_violation_sum']>config['epsilon']['fanout']:return False
        if row['global_routing_overflow']>0 or row['max_global_routing_overflow']>0:return False
    return True

def load_models(reference,masters,catalogs,tables,gradient_backend='manual'):
    engines={};shared=None
    for c in CORNERS:
        model,engine=calibrate_gpu(ForwardModel(reference/c,catalogs[c],masters,tables[c]),shared_rc=shared)
        if gradient_backend=='autograd':
            from autograd_cuda_sta import AutogradCudaSTA
            engine=AutogradCudaSTA(model)
        elif gradient_backend!='manual':
            raise ValueError('Unknown gradient backend')
        engines[c]=engine;shared=model
    return engines

def verify_working_point(engines,native):
    for c,engine in engines.items():
        forward=engine.prepare();m=engine.m
        for key,reference in [('at',m.ref_at),('slew',m.ref_slew)]:
            actual=forward[key];finite=np.isfinite(reference)
            if not np.array_equal(np.isfinite(actual),finite) or np.max(abs(actual[finite]-reference[finite]))>1e-3:raise ValueError('Invalid calibrated '+c+' '+key)
        target_tns=native[c]['setup_tns_ns']*1000;target_wns=native[c]['setup_wns_ns']*1000
        tolerance=max(.001,4*abs(float(np.spacing(np.float32(target_tns*1e-12))))*1e12)
        if abs(forward['tns_ps']-target_tns)>tolerance or abs(forward['wns_ps']-target_wns)>1e-3:raise ValueError('Native working point mismatch '+c)
