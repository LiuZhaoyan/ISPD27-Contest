"""Contest candidate legality and paper Top-K action ranking."""
import torch
import fnmatch
import numpy as np

PROHIBITED_MASTERS = ('*x1p*_ASAP7*', '*xp*_ASAP7*', 'SDF*', 'ICG*')


def contest_neighbors(catalog):
    """Nearest permitted target on each axis, including exits from old blocked cells.

    Liberty dont_use alone is insufficient: the organizer additionally prohibits
    these name patterns for every changed master. An existing prohibited master
    may remain unchanged or move to an allowed master.
    """
    cells=catalog['cells'];result=np.full((len(cells),4),-1,dtype=np.int64)
    for source in cells:
        if source['family_id']<0 or source.get('frozen_reason'):continue
        family=catalog['families'][source['family_id']]
        allowed=[cells[i] for i in family['cell_ids'] if not cells[i].get('frozen_reason')
                 and not any(fnmatch.fnmatchcase(cells[i]['name'],p) for p in PROHIBITED_MASTERS)]
        for d,(axis,other,sign) in enumerate((('kappa','theta',1),('kappa','theta',-1),('theta','kappa',1),('theta','kappa',-1))):
            candidates=[c for c in allowed if c[other]==source[other] and sign*(c[axis]-source[axis])>0]
            if candidates:result[source['id'],d]=min(candidates,key=lambda c:abs(c[axis]-source[axis]))['id']
    return result


def contest_targets(catalog,space='family'):
    """Legal equivalent-cell alternatives, including multi-step and joint moves.

    The first four columns keep the quad-gradient directions. Additional
    columns cover the rest of the functional family exactly once. They are
    scored with actual Liberty differences, not a linear step extrapolation.
    """
    near=contest_neighbors(catalog)
    if space=='neighbors':return near
    if space!='family':raise ValueError('Unknown cell search space: '+space)
    alternatives=[];cells=catalog['cells']
    for source,adjacent in zip(cells,near):
        used=set(adjacent)|{source['id']};extra=[]
        if source['family_id']>=0 and not source.get('frozen_reason'):
            for i in catalog['families'][source['family_id']]['cell_ids']:
                cell=cells[i]
                if i in used or cell.get('frozen_reason'):continue
                if any(fnmatch.fnmatchcase(cell['name'],p) for p in PROHIBITED_MASTERS):continue
                if sorted(map(str,source['liberty_data']['timing_arc_signatures']))!=sorted(map(str,cell['liberty_data']['timing_arc_signatures'])):continue
                extra.append(i)
        alternatives.append(extra)
    result=np.full((len(cells),4+max(map(len,alternatives),default=0)),-1,dtype=np.int64)
    result[:,:4]=near
    for i,extra in enumerate(alternatives):result[i,4:4+len(extra)]=extra
    return result


def gpu_topk(scores,mask,budget,device='cuda:0'):
    if budget<1:raise ValueError('Positive action budget required')
    values=torch.as_tensor(scores,dtype=torch.float64,device=device)
    valid=torch.as_tensor(mask,dtype=torch.bool,device=device)&torch.isfinite(values)&(values>0)
    count=int(valid.sum().item())
    if not count:return []
    ranked=values.masked_fill(~valid,-torch.inf).flatten()
    indices=torch.topk(ranked,min(budget,count),sorted=True).indices.cpu().tolist()
    result=[];used=set()
    for index in indices:
        gate,direction=divmod(index,4)
        if gate not in used:used.add(gate);result.append((gate,direction))
    return result
