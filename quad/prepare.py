"""Parse official ISPD27 libraries and build discrete drive x VT actions."""
import argparse, csv, gzip, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
from cell_space import parse_standard_lib, parse_lef, nominal_state, action_rows
from forward_sta import load_tables
from nlpm import load_power_tables

ROOT = Path(__file__).resolve().parents[1]
CONTEST = ROOT
CORNERS = {'BC':'FF', 'TC':'TT', 'WC':'SS'}

def prepare():
    start = time.perf_counter()
    cache = ROOT / 'cache'; cache.mkdir(exist_ok=True)
    lefs = {}
    for path in sorted((CONTEST/'platform/asap7/lef').glob('*.lef')):
        if path.name in ('asap7sc7p5t_28_R_1x_220121a.lef', 'asap7sc7p5t_28_L_1x_220121a.lef', 'asap7sc7p5t_28_SL_1x_220121a.lef'):
            lefs.update(parse_lef(path))
    groupfile=CONTEST/'evaluator/third_party/displacement/asap7_equivalent_cell_list.csv'
    groups={cell:i for i,row in enumerate(csv.reader(groupfile.open())) for cell in row if cell}
    signatures={}
    for corner,process in CORNERS.items():
        folder=cache/corner;folder.mkdir(exist_ok=True)
        libs={};paths=[]
        for src in sorted((CONTEST/'platform/asap7/lib/NLDM').glob(f'*_{process}_*')):
            path=folder/src.name.removesuffix('.gz')
            data=gzip.decompress(src.read_bytes()) if src.suffix=='.gz' else src.read_bytes()
            if not path.exists() or path.read_bytes()!=data:path.write_bytes(data)
            paths.append(path)
        fingerprint=hashlib.sha256(groupfile.read_bytes())
        for path in paths:fingerprint.update(path.read_bytes())
        for name in ('prepare.py','cell_space.py','forward_sta.py','nlpm.py'):
            fingerprint.update(Path(__file__).with_name(name).read_bytes())
        stamp=folder/'fingerprint.txt'
        if stamp.exists() and stamp.read_text()==fingerprint.hexdigest():
            print('CACHE',corner,flush=True);continue
        for path in paths:
            print('PARSE',corner,path.name,flush=True)
            parsed=parse_standard_lib(path)
            if set(parsed)&set(libs):raise ValueError('Duplicate cell master')
            libs.update(parsed)
        cells=[]
        for name in sorted(set(libs)|set(lefs)):
            lib=libs.get(name);lef=lefs.get(name)
            if lib and not lef:raise ValueError('Missing cell LEF '+name)
            c=dict(id=len(cells),name=name,official_group=groups.get(name),liberty_data=lib,physical=lef,family_id=-1,kappa=-1,theta=-1)
            state=nominal_state(name) if lib else None
            if state:c.update(state)
            c['frozen_reason']='physical_only' if not lib else 'tie_cell' if lib['cell_type']=='tie' else 'dont_use' if lib['dont_use'] else None
            cells.append(c)
        grouped={}
        for c in cells:
            if c.get('series'):
                key=(c['official_group'],c['series'],c['liberty_data']['behavior_sha256'],c['physical']['site'],c['physical']['height_um'])
                grouped.setdefault(key,[]).append(c)
        families=[]
        for key,members in sorted(grouped.items(),key=lambda x:str(x[0])):
            strengths=sorted({c['nominal_strength'] for c in members})
            f=dict(id=len(families),official_group=key[0],series=key[1],behavior_sha256=key[2],strengths=strengths,vt_order=['SLVT','LVT','RVT'],cell_ids=[],grid={})
            for c in members:
                c['family_id']=f['id'];c['kappa']=strengths.index(c['nominal_strength'])
                coordinate=f"{c['kappa']},{c['theta']}"
                if coordinate in f['grid']:raise ValueError('Duplicate size coordinate')
                f['grid'][coordinate]=c['id'];f['cell_ids'].append(c['id'])
            families.append(f)
        catalog=dict(cells=cells,families=families,corner=corner)
        actions=action_rows(cells,families)
        targets=np.array([r['target_id'] for r in actions],dtype=np.int32).reshape(-1,4)
        (folder/'catalog.json').write_text(json.dumps(catalog)+'\n')
        np.save(folder/'targets.npy',targets)
        tables=load_tables(catalog)
        with (folder/'timing.pkl').open('wb') as f:pickle.dump(tables,f)
        power=load_power_tables(catalog)
        with (folder/'power.pkl').open('wb') as f:pickle.dump(power,f)
        signatures[corner]=[(c['name'],c['family_id'],c['kappa'],c['theta']) for c in cells]
        stamp.write_text(fingerprint.hexdigest())
        print('PREPARED',corner,'cells',len(cells),'actions',int((targets>=0).sum()),flush=True)
    cats=[json.loads((cache/c/'catalog.json').read_text()) for c in CORNERS]
    axes=[[(c['name'],c['family_id'],c['kappa'],c['theta']) for c in cat['cells']] for cat in cats]
    if any(a!=axes[0] for a in axes):raise ValueError('Corner action spaces differ')
    print('PREPARATION_SECONDS',time.perf_counter()-start,flush=True)

if __name__=='__main__':prepare()
