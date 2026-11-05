"""Mirror the evaluator's TC-linked graph and OpenSTA scene-arc fallback.

Loading each corner in isolation is not equivalent to the official MCMM flow:
TC is loaded first and defines timing arc identities. When a corresponding arc
is absent in another corner, TimingArc::sceneArc returns the linked TC arc.
Pin capacitances and power data still belong to the requested corner.
"""
import copy,json,pickle
import hashlib
from pathlib import Path

def recovery_tables(cache_root,corner,catalog):
    from forward_sta import load_tables
    cells=[c for c in catalog['cells'] if c.get('liberty_data') and any(s[2].startswith('recovery_') for s in c['liberty_data']['timing_arc_signatures'])]
    files=sorted({c['liberty_data']['liberty'] for c in cells})
    h=hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(__file__).with_name('forward_sta.py').read_bytes())
    for path in files:h.update(Path(path).read_bytes())
    dest=Path(cache_root)/corner/'recovery.pkl';stamp=dest.with_suffix('.sha256')
    if dest.exists() and stamp.exists() and stamp.read_text()==h.hexdigest():
        with dest.open('rb') as f:return pickle.load(f)
    tables=load_tables(dict(cells=cells),selected_roles={'recovery'})
    with dest.open('wb') as f:pickle.dump(tables,f)
    stamp.write_text(h.hexdigest());return tables

def minimum_check_tables(cache_root,corner,catalog):
    from forward_sta import load_tables
    cells=[c for c in catalog['cells'] if c.get('liberty_data') and any(
        s[2].startswith(('hold_','removal_')) for s in c['liberty_data']['timing_arc_signatures'])]
    h=hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(__file__).with_name('forward_sta.py').read_bytes())
    for path in sorted({c['liberty_data']['liberty'] for c in cells}):h.update(Path(path).read_bytes())
    dest=Path(cache_root)/corner/'minimum_checks.pkl';stamp=dest.with_suffix('.sha256')
    if dest.exists() and stamp.exists() and stamp.read_text()==h.hexdigest():
        with dest.open('rb') as f:return pickle.load(f)
    tables=load_tables(dict(cells=cells),selected_roles={'hold','removal'})
    with dest.open('wb') as f:pickle.dump(tables,f)
    stamp.write_text(h.hexdigest());return tables


def load_corner(cache_root,corner,include_min_checks=False):
    cache_root=Path(cache_root)
    catalog=json.loads((cache_root/corner/'catalog.json').read_text())
    linked=json.loads((cache_root/'TC/catalog.json').read_text())
    with (cache_root/corner/'timing.pkl').open('rb') as f:physical=pickle.load(f)
    physical.update(recovery_tables(cache_root,corner,catalog))
    if include_min_checks:physical.update(minimum_check_tables(cache_root,corner,catalog))
    if corner=='TC':return catalog,physical,dict(linked_corner='TC',fallback_groups=[],fallback_tables=0)
    with (cache_root/'TC/timing.pkl').open('rb') as f:canonical=pickle.load(f)
    canonical.update(recovery_tables(cache_root,'TC',linked))
    if include_min_checks:canonical.update(minimum_check_tables(cache_root,'TC',linked))
    linked_cells={c['name']:c for c in linked['cells']}
    effective={};fallback=[];count=0
    for key,group in canonical.items():
        target=physical.get(key,{})
        effective[key]={kind:target.get(kind,table) for kind,table in group.items()}
        missing=[kind for kind in group if kind not in target]
        if missing:fallback.append(dict(cell=key[0],source_pin=key[1],target_pin=key[2],role=key[3],condition=key[4],tables=missing));count+=len(missing)
    catalog=copy.deepcopy(catalog)
    for cell in catalog['cells']:
        if cell.get('liberty_data'):
            cell['liberty_data']['timing_arc_signatures']=linked_cells[cell['name']]['liberty_data']['timing_arc_signatures']
    return catalog,effective,dict(linked_corner='TC',fallback_groups=fallback,fallback_tables=count)
