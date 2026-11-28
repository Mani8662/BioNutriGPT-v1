import json
from pathlib import Path

ROOT = Path('.')
analysis = json.loads((ROOT / 'Saved Model/stage2_cross_model_v1/analysis.json').read_text(encoding='utf-8'))

print("=== PRIMARY ANALYSIS: Reference Variant across all 100 participants ===")
for model in ['bionutrigpt', 'catboost', 'ft_transformer']:
    ref_all = analysis[model]['variant_0_reference']['all']
    print(f"\nModel: {model}")
    print(f"  Total n: {ref_all['n']}, Gate Pass (all <= 0.0015): {ref_all['gate_pass']}/100")
    for d in ['diabetes', 'obesity', 'cvd']:
        out = ref_all['outcomes'][d]
        print(f"  {d:8s}: mean = {out['mean']:+.6f}, 95% CI = [{out['ci95'][0]:+.6f}, {out['ci95'][1]:+.6f}], dec = {out['decrease_count']}, inc = {out['increase_count']}")

print("\n=== STRATA ANALYSIS: Reference Variant by Strata ===")
for model in ['bionutrigpt', 'catboost', 'ft_transformer']:
    print(f"\nModel: {model}")
    for stratum in ['all', 'numerically_valid', 'source_accepted']:
        st = analysis[model]['variant_0_reference'][stratum]
        print(f"  Stratum: {stratum} (n={st['n']}) | Gate Pass: {st['gate_pass']}/{st['n']}")
        for d in ['diabetes', 'obesity', 'cvd']:
            out = st['outcomes'][d]
            print(f"    {d:8s}: mean = {out['mean']:+.6f} [{out['ci95'][0]:+.6f}, {out['ci95'][1]:+.6f}], dec = {out['decrease_count']}, inc = {out['increase_count']}")

print("\n=== FULL MINUS NO-MODEL-OBJECTIVE CONTRASTS ===")
for model in ['bionutrigpt', 'catboost', 'ft_transformer']:
    diff = analysis[model]['full_minus_no_model_objective']
    print(f"\nModel: {model} (Full minus No-Model-Objective paired diff across 100 participants):")
    for d in ['diabetes', 'obesity', 'cvd']:
        out = diff[d]
        print(f"  {d:8s}: mean = {out['mean']:+.6f} [{out['ci95'][0]:+.6f}, {out['ci95'][1]:+.6f}]")

print("\n=== ALL SIX VARIANTS (Stratum: all, n=100) ===")
for model in ['bionutrigpt', 'catboost', 'ft_transformer']:
    print(f"\n--- Model: {model} ---")
    for v in sorted(analysis[model].keys()):
        if v == 'full_minus_no_model_objective': continue
        st = analysis[model][v]['all']
        print(f"  Variant: {v} | Gate Pass: {st['gate_pass']}/100")
        for d in ['diabetes', 'obesity', 'cvd']:
            out = st['outcomes'][d]
            print(f"    {d:8s}: mean = {out['mean']:+.6f} [{out['ci95'][0]:+.6f}, {out['ci95'][1]:+.6f}] (dec: {out['decrease_count']}, inc: {out['increase_count']})")
