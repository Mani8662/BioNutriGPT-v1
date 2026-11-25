"""Gemini-run verification: frozen inference replay and separate mean/CI formulas."""
import argparse
import copy
import numpy as np
import stage2_stability_repaired_v1 as study

def compare_summary(saved, expected):
    study.need(saved.keys() == expected.keys(), 'Summary scenario mismatch')
    for scenario in expected:
        study.need(saved[scenario].keys() == expected[scenario].keys(), 'Summary fields mismatch')
        for key, value in expected[scenario].items():
            actual = saved[scenario][key]
            if isinstance(value, (float, int, list)) and key != 'metric_order':
                study.need(np.allclose(actual, value, atol=1e-10, rtol=0), 'Summary mismatch: '+scenario+'/'+key)
            else:
                study.need(actual == value, 'Summary mismatch: '+scenario+'/'+key)

def verify(pilot):
    import stage2_ablation_runner_v2 as base
    import stage2_counterfactual_engine as s2
    study.check_manifest()
    folder = study.OUT/('pilot' if pilot else 'full')
    receipt = study.read(folder/'execution.json')
    study.need(receipt['manifest_sha256'] == study.sha(study.OUT/'manifest.json'), 'Wrong manifest')
    records = study.load_records(pilot); cohort = study.cases(pilot)
    required = {f"{c['SEQN']}_{s}.json" for c in cohort for s in study.SCENARIOS} | {'analysis.json'}
    study.need(set(receipt['hashes']) == required, 'Incomplete execution hashes')
    for name, digest in receipt['hashes'].items():
        study.need(study.sha(folder/name) == digest, 'Output hash changed: '+name)
    context = base.model_context()
    for c in cohort:
        for scenario in study.SCENARIOS:
            case = study.perturb(c,scenario)
            study.validate(records[c['SEQN'],scenario], case, scenario, context)
    expected = study.summarize(records, cohort)
    compare_summary(study.read(folder/'analysis.json'), expected)
    # Rebuild all five paired endpoints separately, without calling summarize.
    scale = np.asarray([s2.DAILY_REFERENCE_SCALES[n] for n in s2.DIET_RAW_COLS])
    intervals = 0
    for scenario in study.SCENARIOS:
        columns = [[] for _ in range(5)]
        for case in cohort:
            baseline = records[case['SEQN'],'baseline']; other = records[case['SEQN'],scenario]
            diff = [(abs(a-b)/d) for a,b,d in zip(other['vectors']['final_target'],baseline['vectors']['final_target'],scale)]
            vals = [sum(diff), max(diff), other['prescribed_energy']['target_calories']-baseline['prescribed_energy']['target_calories'],
                float(other['evaluation']['numerical_pass'])-float(baseline['evaluation']['numerical_pass']),
                float(other['evaluation']['overall_accepted'])-float(baseline['evaluation']['overall_accepted'])]
            for col,value in zip(columns,vals): col.append(value)
        indices = np.random.default_rng(20261124).integers(0,len(cohort),(10000,len(cohort)))
        for j,col in enumerate(columns):
            sample = np.asarray(col)
            means = np.sum(sample[indices],axis=1)/len(cohort)
            bounds = np.percentile(means,[2.5,97.5])
            study.need(np.allclose(bounds,np.array(expected[scenario]['ci95'])[:,j],atol=1e-10,rtol=0), 'Independent CI failed')
            study.need(abs(sum(col)/len(col)-expected[scenario]['mean'][j])<1e-10,'Independent mean failed')
            intervals += 1
    # Actual validator rejection, not a caught self-raised success sentinel.
    bad = copy.deepcopy(expected); bad['repeat']['ci95'][0][0] += .5
    rejected = False
    try: compare_summary(bad,expected)
    except ValueError: rejected = True
    study.need(rejected,'Corrupt CI accepted')
    c = study.perturb(cohort[0],'baseline'); bad_record = copy.deepcopy(records[c['SEQN'],'baseline'])
    bad_record['study_identity']['scenario'] = 'corrupted'
    identity_rejected = False
    try: study.validate(bad_record,c,'baseline',context)
    except ValueError: identity_rejected = True
    study.need(identity_rejected,'Corrupt identity accepted')
    # Repeat discrepancy is a study finding, never silently erased or turned into success.
    study.write(folder/'verification.json', {'status':'passed', 'participants':len(cohort), 'records':len(records),
        'independent_intervals':intervals, 'changed_ci_rejected':rejected,'changed_identity_rejected':identity_rejected,
        'repeat_max_scaled_Linf':expected['repeat']['max'][1],
        'manifest_sha256':study.sha(study.OUT/'manifest.json'),'analysis_sha256':study.sha(folder/'analysis.json'),
        'scope':'Shared input transform, numerical auditor and model loader; separate mean/CI formulas. Does not rerun Adam or independently implement production rules.'})

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',required=True,choices=['pilot','full'])
    verify(parser.parse_args().verify == 'pilot')
