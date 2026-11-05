import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from wire_thresholds import read_thresholds
from test_early_cuda_sta import minimum_model
from resident_cuda_sta import ResidentCudaSTA
from early_cuda_sta import EarlyCudaSTA
from autograd_cuda_sta import AutogradCudaSTA
from test_autograd_sta import model


class WireThresholdTests(unittest.TestCase):
    def test_library_percentages_and_slew_derate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x.lib'
            path.write_text('''library(x) {
input_threshold_pct_rise : 50;
input_threshold_pct_fall : 25;
slew_lower_threshold_pct_rise : 10;
slew_upper_threshold_pct_rise : 90;
slew_lower_threshold_pct_fall : 20;
slew_upper_threshold_pct_fall : 80;
slew_derate_from_library : 0.8;
}''')
            p=read_thresholds(path)
        np.testing.assert_allclose(p,[[np.log(2),np.log(9)/.8],[-np.log(.75),np.log(4)/.8]])

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_port_wire_uses_exponential_tau_not_second_moment_or_additive_slew(self):
        m=minimum_model();m.port_wire_parameters=np.zeros((len(m.arcs),2))
        m.port_wire_parameters[6]=[np.log(2),np.log(9)]
        e=EarlyCudaSTA(m,device='cpu')
        d,s=e.wire_values(e.tensor([2.,3.]),e.tensor([20.,30.]),e.tensor([4.,5.]),e.tensor([6,7],True))
        np.testing.assert_allclose(d.numpy(),[2*np.log(2),3.])
        np.testing.assert_allclose(s.numpy(),[4+2*np.log(9),np.hypot(5,30*2.197224577)])

    @patch.object(ResidentCudaSTA,'sync',lambda self:None)
    def test_threshold_wire_manual_reverse_matches_independent_autograd(self):
        m=model();m.calibration_mode='arc_offset';m.arc_delay_offset=np.zeros(len(m.arcs))
        m.port_wire_parameters=np.zeros((len(m.arcs),2))
        m.port_wire_parameters[6:8]=[np.log(2),np.log(9)]
        manual=ResidentCudaSTA(m,device='cpu');auto=AutogradCudaSTA(m,device='cpu')
        first=manual.prepare();second=auto.prepare()
        np.testing.assert_allclose(first['at'],second['at'],atol=1e-12)
        for branch in ('hard','ave','lse'):
            a=manual.backward(branch=branch);b=auto.backward(branch=branch)
            for key in ('slew','pin_load','delay','local_slew'):
                np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-11,err_msg=key)
        slew_seed=np.linspace(-.3,.7,12).reshape(6,2)
        load_seed=np.linspace(.5,-.2,12).reshape(6,2)
        a=manual.backward_power(slew_seed,load_seed);b=auto.backward_power(slew_seed,load_seed)
        for key in ('pin_load','local_slew'):
            np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-11,err_msg=key)


if __name__=='__main__':unittest.main()
