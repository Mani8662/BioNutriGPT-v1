"""Copy verified frozen research assets into an immutable application release."""
from pathlib import Path
import json, shutil
import stage1_deployment as deployment
import run_stage1_comparison_v2 as study

def main():
    folder=study.RESULTS/'bionutrigpt'
    study.verify_completion(folder,'completed.json')
    choice=json.loads((folder/'selection.json').read_text())['selected_trial']
    trial=folder/f'trial_{choice:02d}'
    study.verify_completion(trial,'completed.json')
    repair=json.loads((study.DATA/'repair_report.json').read_text())
    source=study.DATA/'preprocessing_artifacts.joblib'
    assert study.sha(source)==repair['output_hashes'][source.name]
    out=deployment.BUNDLE
    if out.exists():
        deployment.verify_bundle()
        print('Existing verified release retained');return
    out.mkdir(parents=True)
    assets={'model.pt':trial/'model.pt','preprocessing_artifacts.joblib':source,
            'calibrators.joblib':folder/'calibrators.joblib','selection.json':folder/'selection.json'}
    for name,p in assets.items():shutil.copy2(p,out/name)
    manifest={'release':'repaired_v1','model':'bionutrigpt','selected_trial':choice,
      'selection_reason':'Migrate the existing architecture and counterfactual interface; no claim of superiority over other models.',
      'feature_order':repair['feature_order'],'targets':study.TARGETS,'thresholds':deployment.THRESHOLDS,
      'threshold_scope':'Fixed descriptive research threshold, not clinically validated; legacy cutoffs not reused.',
      'input_scope':'Complete finite 87-feature raw-unit encoded vectors. Missing-input deployment is unsupported; no sample-default imputation.',
      'source_paths':{k:str(v.relative_to(study.ROOT)) for k,v in assets.items()},
      'hashes':{k:study.sha(out/k) for k in assets},'created':'2026-11-22'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    deployment.verify_bundle();print('Verified repaired release created:',out)

if __name__=='__main__':main()
