# Quad-gradient M1–M4 — AES results

## Outcome (2026-10-19)

| Claim | Result |
| --- | --- |
| Implementation complete | **Yes**, bounded AES / Functional-TC prototype |
| Frozen prototype validation passed | **Yes**, M1–M4 and public submission certification |
| Runtime-adjusted contest net gain | **No**: provisional score **−2.0053485866**, versus packaged Resizer numerical reference **0.5913433713** |
| Full legal tagged-graph proof | **Unverified / deferred**, unchanged from accepted M0 |

This is a successful *prototype validation*, not evidence of a superior sizing
algorithm. Gradient and delay-only selected the **same three actions**, with
identical measured aggregate benefit. Seven independent actions are a small
bounded sample; one predicted-positive drive action actually worsened TC loss.

[Machine-readable evidence](m1_m4_evidence.json) contains the frozen plan,
per-action predictions/observations/guards, persistence checks, update and rollback
hashes, all-scene metrics, measured runtime, source/tool hashes and score breakdown.
Full raw artifacts remain on `gpu_402`, container `ispd27-contest`, under:

```
/workspace/ISPD27-Contest/.runs/qg-aes-m1-m4-20261019-03/
/workspace/ISPD27-Contest/.runs/quad-gradient-source-20261019-03/
```

The second directory is the immutable tool-source snapshot used by the public
launcher, with read-only-use links to the unchanged benchmark/platform/evaluator.
Its implementation matches this delivery except a final trailing-whitespace
cleanup at `tools/quad_gradient/optimizer.py:319` (separately recorded in evidence).
No semantic change followed execution. Source and protected-input fingerprints
were unchanged throughout the run. OpenROAD reports
`26Q3-2904-g8443f6ff39`; executable SHA-256:
`d964e1d0d62982e9571e2a0534772c26457d9ff7499b4583d4a7e75666470893`.
Kepler Formal SHA-256:
`80682c0f7d9b669f8a4c86924f59b7b5ba32892fd97ff6f528d11467abf47dc4`.

## M1 — legal action coverage

- 15,990 instances; 63,960 four-direction decisions; **16,518 valid** actions.
- Invalid: 5,132 protected, 42,154 without a legal neighbor, 156 unsupported
  structural coordinates. Every decision and rejected target retains its reason.
- Candidates intersect official equivalence groups, loaded BC/TC/WC Liberty and
  LEF views, forbidden-target policy and functional pin/protection contracts.
  Existing forbidden masters can remain; they are never newly introduced.
- Nearest drive/VT coordinates are sparse; special structural prefixes remain
  distinct. Fractional `xp*` coordinates are recognized but forbidden targets
  remain excluded. No fabricated intermediate cell or physical step divisor.

## M2 — physical persistence

Both actions passed STA/OpenDB agreement, paired DEF/Verilog export, complete
fresh-route evaluation including formal/placement/clock/protection checks, and
reloaded target-master identity:

| Instance | Replacement | Footprint, DBU | Legalization | Moved instances | Fresh guards |
| --- | --- | --- | --- | --- | --- |
| `_25470_` | `OAI21x1_ASAP7_75t_R` → `OAI21x1_ASAP7_75t_L` | 432×270 → 432×270 | Not needed; dimensions checked before swap | 0 | Pass |
| `_19650_` | `XNOR2x2_ASAP7_75t_SL` → `XNOR2x1_ASAP7_75t_SL` | 594×270 → 648×270 | Executed, placement checked | 0 | Pass |

Numeric drive order does not imply monotonic footprint size. Drive persistence
is not claimed as an improving action: its fixed-RC TC benefit was negative.
Proxy displacement/hold/ERC and later fresh physical evaluation are separate
artifacts. Old SPEF does not establish new-geometry timing validity.

## M3 — adjoint and independent prediction

Native forward timing is retained; the model differentiates own delay, upstream
pin-cap load and downstream slew once at the current environment. The endpoint
seed includes the signed setup-table data-slew derivative, with fixed clock
slew until refresh. No candidate-environment recomputation or direction-specific
sign flip is added. `benefit = -predicted_delta_loss`.

Initial graph: 7,522 pooled states, 8,598 edges, **10,143 local checks**, no
model-rejected legal candidates. Maximum native LUT-value error:
**0.000016774 ps**; maximum analytical/finite-difference derivative error:
**4.10×10⁻¹⁰**. Initial, accepted-state refresh and final-state checks passed.
Final graph: 7,536 states, 8,612 edges, 10,157 checks.

Debug repeat loss difference was **0**. Before independent trials, the plan froze:
loss epsilon `1e-9 ns` inside the absolute denominator, improvement threshold
`1e-7`, native-value tolerance `0.001 ps`, derivative tolerance `1e-5`, direction
hit rate ≥0.75, Kendall tau-a ≥0.3, and budget **3 real STA validations per ranking**.
Overlapping lists share identical observations, not extra validation budget.
Disjoint controls cover adverse actions and lower-criticality measured path
slacks (+129.528 ps drive, +207.701 ps VT).

| Instance / action | Predicted benefit | Actual fixed-RC benefit |
| --- | ---: | ---: |
| `_25470_` VT down | 0.0804512 | 0.0282977 |
| `_19650_` drive down | 0.0402319 | **−0.00464206** |
| `_25345_` VT down | 0.0263443 | 0.00147150 |
| `_25481_` drive up, adverse | −0.0129800 | −0.0156010 |
| `_30571_` drive up, lower criticality | −4.42948e−7 | −4.78538e−7 |
| `_19650_` VT up, adverse | −0.0289291 | −0.0289418 |
| `_16000_` VT down, lower criticality | 0 | 0 |

Direction hit rate **6/7 = 0.857143**; Kendall tau-a **0.714286**. Each ranking
executed 3/3, hit 2/3 positives, total benefit **0.02512717**, best **0.02829773**.
Frozen thresholds passed, but **no ranking advantage over delay-only** is shown.
Both VT controls worsened BC hold-TNS penalty; they were measurement controls,
not accepted optimization actions. The XNOR drive sign error remains a concrete
model limitation, not hidden by forward calibration or threshold changes.

## M4 — real K=1 acceptance, refresh and rollback

Two fresh-MCMM-accepted updates:

1. `_25470_`: OAI21 R → L, TC loss benefit **0.02829773**.
2. After re-extraction/re-scoring, `_19820_`: AND2 R → L, benefit **0.01330975**.

The first reused M2's fresh evaluation **only after exact candidate DEF/Verilog
hash equality**, from the same physical checkpoint/action. All evaluation and
validation time remains inside measured tool runtime. The second was separately
routed/evaluated. Both passed all guards. The final accepted checkpoint was
re-extracted and locally validated too; the previous Jacobian is not called final.

Neither positive optimization trial was rejected. A separate adverse safety
control (`_20780_`, NOR2 drive up) therefore exercised rejection: predicted benefit
**−0.02088339**, actual **−0.02050036**. It was never eligible as a positive
optimization candidate. Reloading the last accepted checkpoint reproduced all
six native timing summaries exactly; its paired design and all three SPEF hashes
were unchanged. This verifies rollback by whole-checkpoint reference, not an
inverse master-only edit.

Fresh public-launcher output metrics (ns):

| Scene | Setup TNS, R0 → final | Setup WNS, R0 → final | Hold TNS, unchanged | Hold WNS, unchanged |
| --- | --- | --- | ---: | ---: |
| BC | −32.3519630 → −32.2471402 | −0.198650708 → −0.198650708 | −1.37974643 | −0.0185681332 |
| TC | −44.5446348 → −44.3617660 | −0.282912722 → −0.272302876 | −0.226772033 | −0.00773359158 |
| WC | −65.4867452 → −65.1789381 | −0.444280392 → −0.412392566 | −0.0681130016 | −0.00385519396 |

All hold **scoring penalties** are no worse than R0, including existing BC
violations. BC/TC ERC remains zero; WC slew sum remains **1565.05436325 ps** (79
violators); cap/fanout violation sums and total/max overflow are zero. Final mean
and max displacement are zero. Reported per-scene power is unchanged at report
precision; exact values are in the evidence JSON. Formal, clock/protection,
placement, pair consistency, SPEF and runtime certification all passed.

Output hashes:

- DEF: `5b8b7fffa5f1850ad7a49060689078c26b286086fe6cb98db87da8895b4254f8`
- Verilog: `bc17ce0f5f31b9fe0fb6dbb9a07624c77cc9832634aab8886d123b29f085fab3`
- Final summary: `6a3f9bc907b127785deb667023dc556413fd67176933029de5e6bff3e75c50b5`

### Runtime and provisional score

Public measured **tool 2175.061620 s / flow 2447.699469 s**, below 18,000 s.
Packaged Resizer reference times: 83.357 s / 255.523344 s. Costs include baseline,
probes, independent controls, persistence evaluations, updates and rollback
validation; validation overhead is not subtracted.

| Component | Value |
| --- | ---: |
| PPA contribution | +0.0404407504 |
| Hold penalty | 0.4121579010 |
| ERC penalty | 9.69e−18 |
| Runtime penalty | **1.6836245385** |
| Displacement penalty | −0.0499931025 |
| Overflow penalty | 0 |
| **Certified candidate score** | **−2.0053485866** |

The packaged Resizer numerical comparison is **0.5913433713**, a difference of
**−2.5966919580**. The candidate's official score uses certified measured runtime;
the comparison uses the packaged reference metrics/times, not a fabricated new
certified reference run. No runtime-adjusted net-gain claim is justified.

## Verification, review and reproduction

- Quad-gradient tests: **19** (10 unchanged M0 + 9 optimizer); focused final tests
  **9/9**. Hand-worked fixtures cover loss sign/epsilon/clipping, LUT derivatives,
  negative/zero arrivals, setup seed and all three gradient contributions,
  controls, missing/nonpositive axes, hold penalties, acceptance and export gates.
- Evaluator full suite: **50/50**. Python compilation passed; no configured static
  Python typechecker exists. All execution was on the required host/container.
- Parallel Standards/Spec review found unsafe failed-gate export, positive-only
  persistence selection, inadequate control coverage and stale documentation;
  these were fixed. Follow-up found missing final-checkpoint refresh; fixed with
  regression coverage. Final focused review: **no issues found**.
- Revision 01 is a pre-review diagnostic, not the claimed delivery; revision 02
  was explicitly cancelled after review found the final-state refresh gap.
  Revision 03 alone supports the results above. Historical M0 remains accepted
  with its checker/tag limitation; it was not rewritten as passing.

Fresh reproduction from `/workspace/ISPD27-Contest`, using the environment in
`AGENTS.md` (every output directory/file must be new):

```bash
python3 -m unittest discover -s tools/quad_gradient -p 'test_*.py'
python3 -m compileall -q tools/quad_gradient
python3 -m unittest discover -s evaluator/tests
python3 evaluator/run_submission.py aes .runs/qg-aes-NEW -- \
  bash /workspace/ISPD27-Contest/tools/quad_gradient/run.sh
python3 evaluator/score.py \
  --baseline evaluator/reference_results/aes/baseline.csv \
  --candidate .runs/qg-aes-NEW/evaluation/summary.csv \
  --reference evaluator/reference_results/aes/resizer.csv \
  --runtimes .runs/qg-aes-NEW/evaluation/runtimes.json \
  --output .runs/qg-aes-NEW/score.json
```

## Remaining approximation and route judgment

The prototype pools native path-union tags and does not prove complete legal
predecessor/check compatibility or raw required-state validity. Wire delay is
fixed; wire slew slope is 1 with native offset; Ceff load slope is 1; waveform
correction slope is 0. These network derivatives are assumptions, **not** made
exact by matching local LUT values. Clock slew stays fixed only within a
reference and is reread after acceptance. AveSmooth uses a 1% relative absolute
neighborhood for zero/negative arrivals; endpoint ties split seeds; native
additive offsets, caps, driver models and Jacobians are rebuilt on refresh.

The bounded frozen test passes, but sign/magnitude errors and identical top-three
rankings do not establish broad gradient validity or an advantage. Runtime cost
currently overwhelms the modest setup benefit. No kernel/backend change or
out-of-scope algorithm expansion was made; this result motivates future model
and cost studies, not a claim that the first prototype wins the contest.
