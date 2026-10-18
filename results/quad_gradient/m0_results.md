# Quad-Gradient M0 — Results

## Outcome

**M0 accepted on 2026-10-19 with limitations.** Read-only STA data extraction and bounded setup/load checks are sufficient to proceed with optimizer-prototype development. Complete Timing Tag predecessor compatibility remains unverified and is deferred; it is not a blocker for M1–M4. The historical checker still returns `status=blocked`, `m0_passed=false`, exit 2 for that missing proof.

This does not establish full-graph gradient correctness, network Jacobians, optimization gains, or contest success.

## Verified

- Six BC/TC/WC × setup/hold endpoint WNS/TNS reconstructions matched native statistics within **0.05 ps**; 691 endpoints per group, maximum discrepancy about **0.002147 ps**.
- **15,990/15,990** STA/OpenDB master pairs matched; 153 connected-capacitance decomposition checks passed.
- Of 153 exported path points, 65 arrival/slew observations matched native reports; the remainder are unverified.
- Bounded `functional_TC/max`: setup LUT/check binding **4/4**; delay/slew/load-model samples **20/20**; waveform-slew aggregation **4/4**. Maximum observed errors were below **0.000019 ps**.
- Merged Tcl probe and closure branches ran on 2026-10-19 (49.526 s / 90.504 s). Focused tests **10/10**, evaluator tests **50/50**, and Python compilation passed. Both real probes still exit 2 for the explicitly deferred Tag-compatibility check.

- Merged Python runner and test file were verified remotely on 2026-10-19: **10/10** tests passed; fresh plain/closure OpenROAD runs took **49.960 s / 91.575 s**. Aggregation, native observations, baseline timing and complete closure-check results exactly matched the pre-Python-merge manifests. Protected inputs and probe sources were unchanged; both runs retained the expected exit 2, with only Tag compatibility deferred in the closure run.

## Limitations

- Full legal Tag-to-predecessor/edge/check compatibility and raw required-state validity are not verified.
- Connected capacitance, report Cap, effective capacitance (`Ceff`) and Pi RC parameters are distinct; sampled lookup agreement does not establish network derivatives.
- No cell-sizing optimizer, full gradient propagation, or contest-gain experiment has been completed.

## Reproduce

The commands below use only repository files plus the prescribed `gpu_402` / `ispd27-contest` environment. They create a new baseline and fresh outputs; no pre-existing `.runs/` artifacts are required. Run from `/workspace/ISPD27-Contest`.

```bash
BASELINE=/tmp/qg-aes-r0-$(date +%Y%m%d-%H%M%S)
PROBE=/tmp/qg-m0-probe-$(date +%Y%m%d-%H%M%S)
CLOSURE=/tmp/qg-m0-closure-$(date +%Y%m%d-%H%M%S)

bash evaluator/eval.sh aes "$BASELINE"
python3 -m unittest discover -s tools/quad_gradient -p 'test_*.py'
python3 tools/quad_gradient/m0.py "$BASELINE" "$PROBE"
python3 tools/quad_gradient/m0.py "$BASELINE" "$CLOSURE" \
  --closure-plan tools/quad_gradient/closure_plan.json
```

The probe is read-only. Historical manifest hashes are intentionally omitted: they refer to private, ignored runtime artifacts and are not required to reproduce the workflow.
