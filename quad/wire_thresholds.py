"""Library-threshold exponential wire response at primary input ports.

OpenSTA's DSPF input-port model uses Elmore as an exponential time constant,
so wire delay is -tau*ln(1-vth), rather than tau plus a frozen correction.
This physical formula has no timing-label fit or trained parameters.
"""
from pathlib import Path
import re
import numpy as np


def read_thresholds(path):
    text=Path(path).read_text()
    def number(name,default):
        value=re.search(r'\b'+name+r'\s*:\s*([-+\d.eE]+)\s*;',text)
        return float(value[1]) if value else default
    derate=number('slew_derate_from_library',1.)
    result=[]
    for rf in ('rise','fall'):
        vth=number('input_threshold_pct_'+rf,50.)/100
        vl=number('slew_lower_threshold_pct_'+rf,20.)/100
        vh=number('slew_upper_threshold_pct_'+rf,80.)/100
        if not (0<vth<1 and 0<vl<vh<1 and derate>0):raise ValueError('Invalid Liberty voltage/slew thresholds')
        result.append((-np.log1p(-vth),np.log((1-vl)/(1-vh))/derate))
    return np.array(result)


def input_port_parameters(model):
    parameters=np.zeros((len(model.arcs),2));libraries={}
    for k,a in enumerate(model.arcs):
        if a['role']!='wire' or model.pin_parts[a['s']][0] is not None:continue
        inst,_=model.pin_parts[a['t']]
        if inst is None:
            # Same fallback thresholds as the native DSPF routine when no
            # load library exists. Normal benchmark clock sinks have a cell.
            threshold=np.array([[np.log(2.),np.log(4.)]]*2)
        else:
            path=model.cells[model.masters[inst]]['liberty_data']['liberty']
            if path not in libraries:libraries[path]=read_thresholds(path)
            threshold=libraries[path]
        parameters[k]=threshold[a['fr']]
    return parameters
