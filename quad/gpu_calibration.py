"""Same frozen RF ratio calibration, with parallel GPU LUT evaluation.

Graph construction, SPEF parsing and sparse transfer resistances remain on CPU.
The native reference pin inputs already represent the calibrated previous layer,
so all local calibration LUT queries can be evaluated simultaneously.
"""
import numpy as np
import hashlib
from rf_paper_sta import RFPaperModel
from rc_tree import build_sensitivity
from ratio_sta import fit_ratio
from resident_cuda_sta import ResidentCudaSTA

def calibrate_gpu(base,shared_rc=None,mode='ratio'):
    if mode not in ('ratio','arc_offset'):raise ValueError('Unknown GPU calibration mode')
    from wire_thresholds import input_port_parameters
    base.port_wire_parameters=input_port_parameters(base)
    m=RFPaperModel.wrap(base);repaired=0
    # A CRPR-aware worst-slack PathEnd need not use the maximum-arrival tag
    # at that pin. Preserve its reference clock/path boundary explicitly.
    # This frozen translation is a local surrogate, not tagged-path STA;
    # candidate validation must still use native worst-slack selection.
    for e in m.endpoints:
        rf={'rise':0,'fall':1}[e['rf']]
        e['path_at_shift_ps']=float(m.ref_at[m.index[e['pin']],rf]-float(e['at_ps']))
    for a in m.arcs:
        if a['role']!='wire':continue
        for empty,known in ((a['s'],a['t']),(a['t'],a['s'])):
            if not m.pins[empty]['net'] and m.pins[known]['net']:
                m.pins[empty]=dict(m.pins[empty],net=m.pins[known]['net'],wire_ff=m.pins[known].get('wire_ff',0));m.base_cap[empty]=m.base_cap[known];repaired+=1
    m.by_net={}
    for i,p in enumerate(m.pins):m.by_net.setdefault(p['net'],[]).append(i)
    m.spef_sha256=hashlib.sha256((m.reference/'network.spef').read_bytes()).hexdigest()
    signature=lambda model:[(a['s'],a['t'],a['fr'],a['to'],a['role'],a['cond']) for a in model.arcs]
    same=(shared_rc is not None and m.spef_sha256==shared_rc.spef_sha256
          and [p['name'] for p in m.pins]==[p['name'] for p in shared_rc.pins]
          and signature(m)==signature(shared_rc))
    if same:
        for k,arc in enumerate(m.arcs):arc['arc_index']=k
        m.rc_matrix=shared_rc.rc_matrix;m.rc_statistics=dict(shared_rc.rc_statistics,shared_identical_spef=True)
    else:m.rc_matrix,m.rc_statistics=build_sensitivity(m,m.reference/'network.spef')
    m.rc_statistics['repaired_legacy_port_nets']=repaired
    m.rc_base_tau=np.array([a['elmore'] for a in m.arcs]);m.rc_pin_caps=m.pin_caps(m.masters);m.rc_current_tau=m.rc_base_tau.copy();m.rf_base_pin=m.rc_pin_caps.copy();m.rf_base_load=np.zeros((m.n,2));m.rf_sink=np.zeros(m.n,dtype=bool)
    for i,(inst,pin) in enumerate(m.pin_parts):
        if inst is not None:m.rf_sink[i]=m.cells[m.masters[inst]].get('liberty_data',{}).get('pins',{}).get(pin,{}).get('direction')=='input'
    for pins in m.by_net.values():
        pc=m.rf_base_pin[pins][m.rf_sink[pins]].sum(0)
        selected=pc.min() if getattr(m,'analysis','max')=='min' else pc.max()
        fixed=float(np.max(m.base_cap[pins]))-float(selected)
        if fixed<-(1e-6+float(selected)*1e-5):raise ValueError('Negative native fixed capacitance beyond rounding tolerance')
        m.rf_base_load[pins]=np.maximum(0.,pc+fixed)
    m.rf_current_load=m.rf_base_load.copy();m.rf_ready=True;m.at_ratio=np.ones((m.n,2));m.at_offset=np.zeros((m.n,2));m.slew_ratio=np.ones((m.n,2));m.slew_offset=np.zeros((m.n,2));m.check_ratio=np.ones(len(m.checks));m.check_offset=np.zeros(len(m.checks));m.load_ratio=np.ones(m.n);m.load_offset=np.zeros(m.n);m.raw_base_cap=m.base_cap.copy()
    m.calibration_mode=mode
    m.delay_correction[:]=0;m.slew_correction[:]=0;m.check_correction[:]=0
    engine=ResidentCudaSTA
    if getattr(m,'analysis','max')=='min':
        from early_cuda_sta import EarlyCudaSTA
        engine=EarlyCudaSTA
    return refit_gpu_at_reference(engine(m),mode=mode)


def refit_gpu_at_reference(gpu,mode=None,verify_physical=False):
    """One-time calibration after choosing the reference's physical slew model.

    This must only be called at the initial native reference, never on candidate
    or GR labels. Verification rejects calibrating geometry that no longer
    matches that reference's Elmore moments and connected loads.
    """
    m=gpu.m;mode=mode or getattr(m,'calibration_mode','ratio')
    if mode not in ('ratio','arc_offset'):raise ValueError('Unknown GPU calibration mode')
    if verify_physical:
        wires=[i for i,a in enumerate(m.arcs) if a['role']=='wire']
        if np.max(abs(m.rc_base_tau[wires]-np.array([m.arcs[i]['elmore'] for i in wires])))>.001:
            raise ValueError('Physical RC differs from native reference; calibration would be invalid')
        physical_load=m.rf_base_load.min(1) if getattr(m,'analysis','max')=='min' else m.rf_base_load.max(1)
        if np.max(abs(physical_load-m.base_cap))>.001:
            raise ValueError('Physical loads differ from native reference; calibration would be invalid')
    m.calibration_mode=mode
    sl=gpu.tensor(m.ref_slew.reshape(-1));at=gpu.tensor(m.ref_at.reshape(-1));load=gpu.load0.reshape(-1);delay=gpu.zeros(len(m.arcs));transition=gpu.zeros(len(m.arcs));cell,wire=gpu.cells,gpu.wires
    delay[cell]=gpu.lut(gpu.delay_tab[cell],sl[gpu.s[cell]],load[gpu.t[cell]]);transition[cell]=gpu.lut(gpu.slew_tab[cell],sl[gpu.s[cell]],load[gpu.t[cell]]).clamp_min(0);delay[wire]=gpu.tau0[wire]
    import torch
    impulse=gpu.impulse0 if gpu.wire_moment_model else gpu.tau0
    if hasattr(gpu,'wire_values'):
        delay[wire],transition[wire]=gpu.wire_values(gpu.tau0[wire],impulse[wire],sl[gpu.s[wire]],wire)
    else:transition[wire]=torch.hypot(sl[gpu.s[wire]],2.197224577*impulse[wire])
    candidate=gpu.numpy(gpu.peak(at[gpu.s]+delay,gpu.t,m.n*2)).reshape(-1,2);trans=gpu.numpy(gpu.peak(transition,gpu.t,m.n*2).clamp_min(0)).reshape(-1,2)
    m.at_ratio,m.at_offset,af=fit_ratio(candidate,m.ref_at)
    if mode=='arc_offset':
        native=np.array([a['delay'] for a in m.arcs])
        m.arc_delay_offset=native-gpu.numpy(delay)
        native_candidate=gpu.numpy(gpu.peak(at[gpu.s]+gpu.tensor(native),gpu.t,m.n*2)).reshape(-1,2)
        valid=np.isfinite(native_candidate)&np.isfinite(m.ref_at)
        m.at_ratio=np.ones_like(m.ref_at);m.at_offset=np.zeros_like(m.ref_at)
        m.at_offset[valid]=m.ref_at[valid]-native_candidate[valid]
        af=valid&(np.abs(m.at_offset)>1e-6)
    m.slew_ratio,m.slew_offset,sf=fit_ratio(trans,m.ref_slew);rawcheck=gpu.numpy(gpu.lut(gpu.check_tab,sl[gpu.ct],sl[gpu.cs]));m.check_ratio,m.check_offset,qf=fit_ratio(rawcheck,np.array([a['delay'] for a in m.checks]));m.calibration_fallbacks=dict(load=0,at=int(af.sum()),slew=int(sf.sum()),setup=int(qf.sum()));gpu.set_model(m);gpu.sync();return m,gpu
