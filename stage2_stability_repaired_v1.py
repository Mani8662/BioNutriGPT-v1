"""Prepared engineering stability study. No inference without explicit --run."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Saved Model/stage2_stability_repaired_v1'
SOURCE = ROOT / 'Saved Model/stage2_ablation_v2'
SCENARIOS = {'baseline': (1., 1.), 'repeat': (1., 1.),
             'diet_minus1': (.99, 1.), 'diet_plus1': (1.01, 1.),
             'diet_minus5': (.95, 1.), 'diet_plus5': (1.05, 1.),
             'lipid_minus5': (1., .95), 'lipid_plus5': (1., 1.05)}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)

def need(ok, message):
    if not ok:
        raise ValueError(message)

def prepare():
    need(not (OUT/'manifest.json').exists(), 'Already frozen; do not overwrite')
    dev = cases(True); ev = cases(False)
    need(not ({c['SEQN'] for c in dev} & {c['SEQN'] for c in ev}), 'Development/evaluation overlap')
    paths = {ROOT/k for k in read(SOURCE/'input_manifest.json')}
    for name, digest in read(SOURCE/'input_manifest.json').items():
        need(sha(ROOT/name) == digest, 'Source binding changed: '+name)
    for name in ['development_participants_LOCAL.json', 'evaluation_participants_LOCAL.json', 'variant_manifest.json']:
        paths.add(SOURCE/name)
    paths.update([Path(__file__), ROOT/'verify_stage2_stability_repaired_v1.py'])
    pending = list(paths); seen = set()
    while pending:
        path = pending.pop()
        if path in seen or path.suffix != '.py':
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                local = ROOT/(name.replace('.', '/')+'.py')
                if local.exists() and local not in paths:
                    paths.add(local); pending.append(local)
    protocol = {'status': 'prepared_not_run', 'scenarios': SCENARIOS,
        'cohort': 'Reuse the frozen ablation 100 evaluation / 3 development profiles; not a new independent cohort.',
        'counts': {'pilot': 24, 'full': 800}, 'bootstrap': {'resamples': 10000, 'seed': 20261124},
        'scope': 'Engineering perturbations, not clinically validated error distributions. Anthropometry unchanged. All raw dietary nutrients scaled together; both measured lipids scaled together; derived features recomputed.',
        'thresholds': 'Actual frozen production cutoffs. Report naturally observed crossings separately; no forced probability overrides. No crossing does not demonstrate boundary robustness.',
        'repeat': 'Same inputs and seed, fresh optimizer; assess numerical repeatability, not training-seed variation.',
        'statistics': 'Per-scenario paired participant bootstrap, exploratory pointwise 95% intervals. No universal stability cutoff or clinical efficacy claim.',
        'limitations': 'Does not cover anthropometric error, missingness, forced near-boundary profiles, or individual nutrient noise.'}
    write(OUT/'protocol.json', protocol); paths.add(OUT/'protocol.json')
    write(OUT/'manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)})

def check_manifest():
    for name, digest in read(OUT/'manifest.json').items():
        need(sha(ROOT/name) == digest, 'Frozen file changed: '+name)

def cases(pilot):
    data = read(SOURCE/('development_participants_LOCAL.json' if pilot else 'evaluation_participants_LOCAL.json'))
    need(len(data) == (3 if pilot else 100), 'Wrong participant count')
    need(len({c['SEQN'] for c in data}) == len(data), 'Duplicate participants')
    return data

def perturb(case, scenario):
    import stage2_counterfactual_engine as s2
    out = copy.deepcopy(case)
    raw, _ = s2.canonicalize_dietary_ratios(out['vector'])
    diet, lipid = SCENARIOS[scenario]
    for name in s2.DIET_RAW_COLS:
        raw[s2.FEATURE_COLS.index(name)] *= diet
    for name in ['LBDHDD', 'LBXTC']:
        raw[s2.FEATURE_COLS.index(name)] *= lipid
    out['vector'] = s2.canonicalize_dietary_ratios(raw)[0].tolist()
    return out

def config():
    return read(SOURCE/'variant_manifest.json')['variant_0_reference']

def identity(case, scenario):
    return {'case': case, 'scenario': scenario, 'manifest_sha256': sha(OUT/'manifest.json')}

def validate(record, case, scenario, context):
    import verify_stage2_ablation_v1 as audit
    need(record['study_identity'] == identity(case, scenario), 'Checkpoint identity/input mismatch')
    need(record['variant_config'] == config(), 'Wrong engine configuration')
    audit.audit_checkpoint_record(record, {case['SEQN']: case}, *context)

def load_records(pilot):
    folder = OUT/('pilot' if pilot else 'full')
    return {(c['SEQN'], s): read(folder/f"{c['SEQN']}_{s}.json") for c in cases(pilot) for s in SCENARIOS}

def summarize(records, cohort):
    import numpy as np
    import stage2_counterfactual_engine as s2
    scales = np.array([s2.DAILY_REFERENCE_SCALES[n] for n in s2.DIET_RAW_COLS])
    cutoffs = dict(zip(['diabetes', 'obesity', 'cvd'], [s2.CUTOFF_DIABETES, s2.CUTOFF_OBESITY, s2.CUTOFF_CVD]))
    result = {}
    for scenario in SCENARIOS:
        values = []; crossing = []; near = []
        for case in cohort:
            b = records[case['SEQN'], 'baseline']; r = records[case['SEQN'], scenario]
            delta = (np.array(r['vectors']['final_target'])-b['vectors']['final_target'])/scales
            values.append([float(np.abs(delta).sum()), float(np.abs(delta).max()),
                r['prescribed_energy']['target_calories']-b['prescribed_energy']['target_calories'],
                int(r['evaluation']['numerical_pass'])-int(b['evaluation']['numerical_pass']),
                int(r['evaluation']['overall_accepted'])-int(b['evaluation']['overall_accepted'])])
            crossing.append(any((r['baseline_probabilities'][k] >= t) != (b['baseline_probabilities'][k] >= t) for k,t in cutoffs.items()))
            near.append(any(abs(b['baseline_probabilities'][k]-t) <= .02 for k,t in cutoffs.items()))
        x = np.array(values); rng = np.random.default_rng(20261124)
        boot = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
        result[scenario] = {'n': len(x), 'metric_order': ['target_scaled_L1', 'target_scaled_Linf', 'energy_delta_kcal', 'numerical_pass_delta', 'overall_acceptance_delta'],
            'mean': x.mean(axis=0).tolist(), 'ci95': np.quantile(boot, [.025, .975], axis=0).tolist(),
            'median': np.median(x, axis=0).tolist(), 'p95': np.quantile(x,.95,axis=0).tolist(),
            'max': x.max(axis=0).tolist(), 'routing_crossings': sum(crossing), 'near_threshold_baselines': sum(near),
            'crossing_mean_L1': float(x[np.array(crossing),0].mean()) if any(crossing) else None,
            'noncrossing_mean_L1': float(x[~np.array(crossing),0].mean()) if not all(crossing) else None,
            'numerical_pass_count': sum(bool(records[c['SEQN'],scenario]['evaluation']['numerical_pass']) for c in cohort),
            'overall_accepted_count': sum(bool(records[c['SEQN'],scenario]['evaluation']['overall_accepted']) for c in cohort)}
    return result

def run(pilot):
    import torch
    import stage2_ablation_runner_v2 as base
    import stage2_ablation_engine_v1 as engine
    check_manifest()
    if not pilot:
        receipt = read(OUT/'pilot/verification.json')
        need(receipt['status'] == 'passed' and receipt['manifest_sha256'] == sha(OUT/'manifest.json'), 'Verified pilot required')
        need(receipt['analysis_sha256'] == sha(OUT/'pilot/analysis.json'), 'Pilot analysis changed')
    folder = OUT/('pilot' if pilot else 'full'); folder.mkdir(parents=True,exist_ok=True)
    need(not (folder/'execution.json').exists(), 'Completed execution exists; use verifier, not rerun')
    context = base.model_context()
    for i, original in enumerate(cases(pilot)):
        order = list(SCENARIOS); offset = i % len(order); order = order[offset:]+order[:offset]
        for scenario in order:
            case = perturb(original, scenario); path = folder/f"{case['SEQN']}_{scenario}.json"
            error_path = path.with_suffix('.error')
            need(not error_path.exists(), 'Prior execution error requires review: '+str(error_path))
            if path.exists():
                validate(read(path), case, scenario, context); continue
            torch.manual_seed(20261124)
            try:
                record = engine.run_ablation_variant(case=case, variant_config=config(), model=context[0], artifacts=context[1], device=context[2])
                record['study_identity'] = identity(case, scenario)
                validate(record, case, scenario, context); write(path, record)
            except Exception as exc:
                write(error_path, {'study_identity':identity(case,scenario),'error':repr(exc)})
                raise
            print(case['SEQN'], scenario, flush=True)
    write(folder/'analysis.json', summarize(load_records(pilot), cases(pilot)))
    write(folder/'execution.json', {'status': 'executed_pending_verification', 'manifest_sha256': sha(OUT/'manifest.json'),
        'hashes': {p.name: sha(p) for p in sorted(folder.glob('*.json')) if p.name not in ['verification.json','execution.json']}})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true'); group.add_argument('--run', choices=['pilot','full'])
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.run: run(args.run == 'pilot')
    else: print('No inference performed. Use --prepare, then --run pilot, verifier, --run full, verifier.')
