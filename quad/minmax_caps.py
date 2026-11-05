"""Liberty min/max input capacitances, including transition-specific ranges.

The existing catalog stores nominal/maximum values. Early delay analysis must
honour rise/fall_capacitance_range instead of reusing the maximum pin load.
This reads technology data directly; no fitted or trained parameters.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from cell_space import entries,clean,attributes,unquote


def read_ranges(path):
    lib=next(value for key,value in entries(clean(Path(path).read_text())) if key=='library')
    parts=list(entries(lib.body));meta=dict(parts)
    if meta['capacitive_load_unit'][1].replace(' ','')!='1,ff':
        raise ValueError('Min/max capacitance reader requires ASAP7 fF units')
    result={}
    for key,cell in parts:
        if key!='cell':continue
        pins={}
        for pk,pin in entries(cell.body):
            if pk!='pin':continue
            values=list(entries(pin.body));pa=attributes(values);fields=dict(values)
            pair=[]
            for rf in ['rise','fall']:
                nominal=float(pa.get(rf+'_capacitance',pa.get('capacitance',0.)))
                value=fields.get(rf+'_capacitance_range')
                if value is None:limits=[nominal,nominal]
                else:limits=[float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',value[1])]
                if len(limits)!=2 or limits[0]<0 or limits[1]<limits[0]:
                    raise ValueError('Invalid Liberty pin capacitance range')
                pair.append(limits)
            pins[unquote(pin.args)]=dict(min=[p[0] for p in pair],max=[p[1] for p in pair])
        result[unquote(cell.args)]=pins
    return result


def load_ranges(catalog,cache=None):
    paths=sorted({c['liberty_data']['liberty'] for c in catalog['cells'] if c.get('liberty_data')})
    h=hashlib.sha256(Path(__file__).read_bytes())
    for path in paths:h.update(Path(path).read_bytes())
    destination=Path(cache)/f'{h.hexdigest()}.json' if cache else None
    if destination is not None and destination.exists():return json.loads(destination.read_text())
    result={}
    for path in paths:
        value=read_ranges(path)
        if result.keys()&value.keys():raise ValueError('Duplicate Liberty cell')
        result.update(value)
    if destination is not None:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(result,separators=(',',':'))+'\n')
    return result


def pin_capacitances(model,ranges,analysis,masters=None):
    if analysis not in ('min','max'):raise ValueError('Expected min/max analysis')
    masters=masters or model.masters;caps=np.zeros((model.n,2))
    for i,(inst,pin) in enumerate(model.pin_parts):
        if inst is not None:caps[i]=ranges[masters[inst]][pin][analysis]
    return caps
