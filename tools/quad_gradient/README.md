# Quad-gradient: AES / Functional-TC prototype

`run.sh` implements the M1–M4 sizing prototype from
`research/quad_gradient/implementation_spec.md`. Implementation, validation and
runtime-adjusted contest benefit are separate claims: consult the latest results
report, not the existence of an output file. Deferred tag compatibility is **not
proven**. The historical `m0.py` remains a separate read-only interface probe.

## Optimizer

Run inside the prescribed container, after activating the environment:

```bash
python3 evaluator/run_submission.py aes .runs/quad-gradient-NEW -- \
  bash /workspace/ISPD27-Contest/tools/quad_gradient/run.sh
```

The launcher appends `<input_dir> <platform_dir> <output_dir> <top_module>`.
Only AES/ASAP7 is supported. The tool creates a fresh routed R0 and runs at most
two accepted K=1 iterations (three attempts per iteration).
`optimizer.py --help` exposes a 1–4 step bound. The normalized-loss improvement
tolerance remains `1e-7`, the floor used by the completed development validation.
There is no repeated noise measurement or independent-validation CLI option.

Evidence lives under `tool/quad_gradient`: `manifest.json`, `commands.json`,
`frozen_plan.json` (optimization parameters and assumptions), `updates.json`,
raw native arc/path/SPEF/snapshot reports, and complete certified physical
checkpoints. Every candidate still needs real improvement and fresh-route MCMM
guards. Rejected trials retain the current checkpoint reference. No accepted
improvement also fails rather than copying R0. Final public launcher evaluation
must certify runtime and the exported pair; scoring is then separate:

```bash
python3 evaluator/score.py --help
```

Approximation: native path-union states pooled across tags, wire delay fixed,
wire slew slope 1 with native offset, Ceff load slope 1, waveform correction
slope 0; clock slew is fixed until each accepted fresh-route refresh. NLDM
Jacobians use the current environment once; candidate cap is a true discrete
difference. Negative/zero arrivals use an absolute relative AveSmooth
neighborhood. Setup data-slew derivatives retain their sign. These assumptions
remain model limitations despite the completed development validation.
Drive trials legalize but retain old SPEF for the proxy: displacement and proxy
ERC/hold are recorded separately, **not** claimed as fresh-route validation.
Actual acceptance requires fresh route, parasitics, formal, placement,
clock/protection checks, all-scene hold penalties and ERC, and zero overflow.
Rollback retains the complete last accepted checkpoint, never only the master.

One-time persistence/reload experiments, independent ranking comparisons,
deliberate rollback controls, local finite-difference self-checks and a final
gradient rebuild have been removed from formal optimization. Numerical and
runtime acceptance/rollback regression checks live in `test_optimizer.py`.
The retained M1–M4 results describe the historical validation run, including its
validation overhead; they are not performance measurements of this shorter flow.

## Historical M0 probe

The following describes `m0.py`, not the optimizer.
No evaluator, scoring, constraints, libraries, or container configuration is changed.
The probe runs a separate OpenROAD process, loads a certified unchanged AES R0
DEF and its three routed SPEFs, and preserves the original functional modes,
Liberty views, RC settings, and propagated clocks.

## Run (gpu_402, udxs/ispd27:v1 / ispd27-contest)

From `/workspace/ISPD27-Contest`, after the environment activation in `AGENTS.md`:

```bash
# Both directories must be new. Do not use a candidate as R0.
bash evaluator/eval.sh aes .runs/quad-m0-baseline-NEW
python3 tools/quad_gradient/m0.py \
  .runs/quad-m0-baseline-NEW .runs/quad-m0-probe-NEW
```

Exit 2 means **blocked** in the original M0 checker (baseline observations agree, but at least one
required interface check is unpassed); exit 1 means failure. Exit 0 requires
all explicit required checks to pass. **Project policy:** Timing Tag compatibility proof is deferred and no longer blocks M1–M4 prototype development; this does not change the historical checker status or imply verified tag relations. A manifest is saved for
probe failures after output creation; an existing output directory is rejected
without modification. The probe timeout is 1800 seconds. No rerouting, master
replacement, legalization, timing-backend patch, or gradient is performed.

## Evidence contract

- `manifest.json`: commands, version, executable/source/input SHA-256,
  scoring policy snapshot, runtime, baseline identity checks, error/gap state,
  aggregate and operating-point checks, and before/after input identity.
  Every benchmark/platform/evaluator/baseline file is fingerprinted (excluding
  Python bytecode). Hash checks detect accidental mutation, not a malicious
  concurrent process; avoid concurrent writers to inputs/baseline.
- `native.tsv` versus `endpoints.tsv`: all BC/TC/WC max/min native WNS/TNS
  reconstructed by taking worst slack per endpoint once, not summing paths.
  Time is ps here; divide by 1000 for ns. The tolerance is frozen at 0.05 ps
  before the first probe (report rounding / float accumulation), not fitted to
  held-out optimizer performance. This is an interface test, not M3 calibration.
- `points.tsv`: selected paths retain scene, functional mode, max/min,
  transition, timing tag, arrival/required/slack and explicitly scene-selected
  vertex slew. No arbitrary-pin all-scene arrival property is relabelled.
- Native full-clock path reports independently check the exported arrivals and
  slews where the native report prints the same point/transition. Connected
  total/pin/wire capacitance is in **SI F**, report Cap is in **fF**; sum checks
  and connected-minus-native-Cap differences are retained. Equality with
  effective load is deliberately **not** asserted.
- `masters.tsv`: joined STA/OpenDB instance masters, checked for agreement.
- `fanin.tsv`, `states.tsv`, `liberty_arcs.tsv`: bounded structural fan-in along
  the selected TC setup path, incoming arc roles/senses/transitions, vertex
  tags, and cell arc-set conditions. Local arc ordinals are meaningful only
  within their edge/cell. Vertex states are capped at 32 per
  vertex/transition/analysis; explicit truncation rows prevent a completeness
  claim. TSV fields escape backslash/tab/newline/carriage return as
  `\\`, `\t`, `\n`, `\r`.
- Setup/hold endpoint check roles, margins, launch/capture clocks, and data
  arrival/required are exported. Low-level methods with `_s` columns return
  SI seconds; public property/report values with `_ps` columns are ps.
  Check margins are not a setup LUT derivative or a selected-table identity.

## Preregistered closure experiments

The updated specification's M0 closure protocol is implemented with:

```bash
python3 tools/quad_gradient/m0.py \
  .runs/quad-m0-baseline-20261018-01 .runs/quad-m0-closure-NEW \
  --closure-plan tools/quad_gradient/closure_plan.json
```

The checked-in plan targets the existing certified R0 by name. Its full input
hashes are retained by the runner. To use another certified baseline, first
create a new plan with that baseline identity, fixed samples, assumptions and
tolerances; do not tune a plan after measuring its samples. The runner retains
its exact bytes and hash **before** OpenROAD starts. Timing tolerance remains
0.05 ps; the new independent cap decomposition/model-input tolerance is
0.00001 fF. Neither tolerance is for M3 derivatives or benefit ranking.

- `closure_boundary.tsv`, `closure_relations.tsv`, `closure_states.tsv`: the
  specified `_19650_/Y` fan-in, all immediate cell inputs and their wire
  drivers, including competing transitions, conditions, disabled flags and
  max tags. `probe.log` contains native `report_arrival` / `report_required`
  blocks after explicitly computing requireds. `closure_interfaces.txt`
  inventories the installed Tcl surface. These do **not** provide a native
  tag-compatibility oracle: rejected relation pairs, full predecessor sets
  and validity of raw required states remain unverified. That check stays
  unpassed rather than guessing relationships from structural edges.
- `closure_endpoints.tsv`, `setup_*_dcalc.rpt`, `endpoint_*_path.rpt`: both
  data transitions at two fixed setup endpoints. The checker independently
  reads the unchanged TT Liberty files, binds the unique related-pin,
  condition, timing-type and transition table, and performs bilinear lookup
  (linear extrapolation at boundaries). It compares the pure native Check,
  graph arc delay, derated endpoint margin, data/clock slews, capture clock
  and recognized CRPR/uncertainty adjustments to reconstruct RAT/slack.
  Conditional selected endpoint ambiguity is rejected, not resolved by value
  coincidence. Only explicit 2-D ps/fF NLDM tables and unit PVT scaling are
  supported; unknown contracts remain unmatched.
- `load_*_dcalc.rpt`, `closure_load_points.tsv`, `closure_net_slews.tsv`: every
  combinational arc/transition at two fixed real drivers. Checks independently
  enumerate Liberty arc coverage and bind delay/slew tables. Native DMP Pi
  C1/C2/Rpi and **Ceff** are retained separately from connected pin/wire/total
  cap and path-report Cap. Connected-net caps reduce rise/fall via the chosen
  min/max (installed `Sta::connectedCap(Net)`); per-transition Pi C1+C2 is
  **not** asserted equal to that aggregate. The difference is evidence, not
  an error to hide by widening tolerance. Independent LUT values are checked against native
  driver delay/slew lookups and graph delay; waveform slew is retained and
  its maximum is compared with the native output vertex slew. DMP waveform
  slew need not equal the Liberty transition-table result. This does not
  implement or validate the network Jacobian; per-sink native slews are
  observations only, not independently reconstructed network responses.

`manifest.json` includes per-sample table identity, inputs, units, errors,
coverage/missing items and reasons. `m0_passed` is derived from required
baseline, setup, load, tagged-relation and input/source-identity checks, not
an unconditional flag. Omitting the closure plan leaves all three new checks
unpassed. Native interface sources are hashed alongside the executable.
The legal tagged-relation check is still unpassed; a different timer, backend
patch or broader algorithm requires the separate route decision in the spec.
No M1–M4 work is performed by `m0.py`; use the optimizer above for those stages.

## Tests

Run in the same container:

```bash
python3 -m unittest discover -s tools/quad_gradient -p 'test_*.py'
python3 -m compileall -q tools/quad_gradient
python3 -m unittest discover -s evaluator/tests
```

The supporting evidence checker tests use hand-worked endpoint and report
fixtures; actual OpenROAD execution is necessary to validate the installed
interfaces and the benchmark observations. Python has no configured static
typechecker in this repository; compilation is only a syntax check.
