#!/usr/bin/env python3
"""Generate deterministic evaluation references from R0, never from a candidate."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/quad_gradient'), str(ROOT / 'evaluator')]
from objective import tc_loss
from score import read_summary, validate_config

# Match the prototype's fixed policy, not scoring_config.epsilon.time_ns.
# check_algorithm.py also checks these against the actual loss and frozen plan.
EPSILON_NS = 1e-9
IMPROVEMENT_THRESHOLD = 1e-7
DEFAULT_CONFIG = ROOT / 'evaluator/config/scoring_config.json'


def generate(baseline, config=DEFAULT_CONFIG):
    baseline, config = Path(baseline), Path(config)
    baseline_bytes, config_bytes = baseline.read_bytes(), config.read_bytes()
    policy = validate_config(json.loads(config_bytes))
    try:
        (design, configuration), scenes = read_summary(baseline)
    except (KeyError, csv.Error) as error:
        raise ValueError('Expected evaluator summary.csv columns, with timing in ns') from error
    if set(scenes) != {'BC', 'TC', 'WC'} or set(scenes) != set(policy['scenarios']):
        raise ValueError('Baseline and config must both cover exactly BC/TC/WC')
    reference = [scenes['TC']['setup_tns_ns'], scenes['TC']['setup_wns_ns']]
    denominators = [abs(value + EPSILON_NS) for value in reference]
    if any(value <= 0 for value in denominators):
        raise ValueError('TC normalization denominators must be positive')
    if baseline.read_bytes() != baseline_bytes or config.read_bytes() != config_bytes:
        raise ValueError('Inputs changed during parameter generation')
    return dict(
        schema='quad-gradient-eval-params-v1', design=design, configuration=configuration,
        eligibility='not_certified',
        units=dict(summary_time='ns', model_time='ps', model_capacitance='fF'),
        algorithm=dict(reference_tc_ns=reference, epsilon_ns=EPSILON_NS,
                       normalization_denominators_ns=denominators,
                       baseline_loss=tc_loss(reference, reference),
                       improvement_threshold=IMPROVEMENT_THRESHOLD),
        scoring_config=policy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Unoptimized R0 summary.csv')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, help='NEW JSON file; otherwise print JSON')
    args = parser.parse_args(argv)
    try:
        text = json.dumps(generate(args.baseline, args.config), sort_keys=True,
                          indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                stream.write(text)
        else:
            print(text, end='')
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
