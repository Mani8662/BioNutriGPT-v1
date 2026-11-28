"""Versioned correction; original experiment and production engine stay unchanged."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent
O=R/'Saved Model/stage2_comparison_study_v2'
O.mkdir(exist_ok=True)
old=R/'Saved Model/stage2_comparison_study_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,s): (R/name).write_text(s,encoding='utf-8')
checker='''"""Versioned numerical checker: 1e-6 absolute tolerance in nutrient units."""
import numpy as np
from verify_stage2_constraint_audit_repaired_v1 import check as historical_check
def check(values,lower,upper,keys,baseline,metadata,cvd):
    v=np.asarray(values,dtype=float)
    if v.shape != (39,) or not np.isfinite(v).all():
        raise ValueError('Nonfinite or invalid nutrient vector')
    result=historical_check(v,lower,upper,keys,baseline,metadata,cvd)
    n=dict(zip(keys,v)); w=float(metadata['weight_kg']); tol=1e-6
    result['filter2_ok']=bool(n['DR1TKCAL']>=1200-tol and n['DR1TSODI']<=(1500 if cvd>=.5 else 2300)+tol and n['DR1TPOTA']>=(3500 if cvd>=.5 else 2600)-tol and .8*w-tol<=n['DR1TPROT']<=2*w+tol)
    result['joint']=bool(not result['bound_names'] and result['energy_ok'] and result['ratios_ok'] and result['filter2_ok'])
    return result
'''
write('stage2_comparison_checks_v2.py',checker)
s=(R/'stage2_comparison_arms.py').read_text()
s=s.replace('from verify_stage2_constraint_audit_repaired_v1 import check','from stage2_comparison_checks_v2 import check')
s=s.replace('"adam_steps": 50,','"adam_steps": 50,\n        "pre_projection_candidate": n_adam_clipped.tolist(),')
write('stage2_comparison_arms_v2.py',s)
s=(R/'run_stage2_comparison_study_v1.py').read_text().replace('import stage2_comparison_arms as arms','import stage2_comparison_arms_v2 as arms').replace('stage2_comparison_study_v1','stage2_comparison_study_v2')
start=s.index('            # For logging, record')
end=s.index('            target_39, audit = arms.run_model_guided_arm(',start)
s=s[:start]+s[end:]
needle='            solver_audit = audit\n        else:'
s=s.replace(needle,'            pre_projection_candidate = audit["pre_projection_candidate"]\n'+needle)
s=s.replace('    t0 = time.perf_counter()\n    raw,', '    if device.type == "cuda": torch.cuda.synchronize()\n    t0 = time.perf_counter()\n    raw,')
s=s.replace('        dt = time.perf_counter() - t0','        if device.type == "cuda": torch.cuda.synchronize()\n        dt = time.perf_counter() - t0')
s=s.replace('    t_start = time.perf_counter()','    # Untimed warm-up; all arms include common baseline inference and evaluation.\n    warm = torch.as_tensor(cases[0]["vector"],dtype=torch.float32,device=device).unsqueeze(0)\n    s2._predict_probabilities(model,s2._normalize_torch(warm,artifacts),no_grad=True)\n    if device.type == "cuda": torch.cuda.synchronize()\n    protocol_hash = sha(protocol_path)\n    t_start = time.perf_counter()')
s=s.replace('                assert res["arm"] == arm_name','                assert res["arm"] == arm_name\n                assert res["protocol_sha256"] == protocol_hash')
s=s.replace('                write_json_atomic(ckpt_path, res)','                res["protocol_sha256"] = protocol_hash\n                write_json_atomic(ckpt_path, res)')
s=s.replace('        "device": str(device),','        "device": str(device),\n        "torch_threads": torch.get_num_threads(),\n        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,')
write('run_stage2_comparison_study_v2.py',s)
shutil.copy2(old/'cases_LOCAL.json',O/'cases_LOCAL.json')
p=json.loads((old/'protocol.json').read_text())
p['version']='stage2_comparison_study_v2'
p['amendment']='Post-result correction: one Adam invocation, consistent Filter 2 tolerance, exact paired binary tests, paired mean-distance bootstrap. Same participants/targets construction; not a new independent validation sample.'
p['statistical_analysis_plan']['p_values']='Exact two-sided binomial test of discordant paired binary outcomes; primary alpha .05; two secondary acceptance contrasts Bonferroni alpha .025. Other endpoints exploratory; pointwise percentile bootstrap intervals.'
p['timing']='Single search, CUDA synchronized at boundaries, common inference/evaluation included, model loading excluded, untimed inference warmup; descriptive single-run timings with fixed A/B/C order.'
p['hardware']['torch_threads']='os.cpu_count(); actual recorded in summary'
p['hashes'].update({f:sha(R/f) for f in ['stage2_comparison_checks_v2.py','stage2_comparison_arms_v2.py','run_stage2_comparison_study_v2.py','verify_stage2_constraint_audit_repaired_v1.py']})
p['original_protocol_sha256']=sha(old/'protocol.json')
(O/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
archive=O/'original_document_copies'; archive.mkdir(exist_ok=True)
for name in ['Progress.txt','Current_Model_Explanation.txt','Research Paper/Readiness_and_Evidence.md','Research Paper/BioNutriGPT_Manuscript_v2.md','Research Paper/Stage2_Comparison_Study_v1.md']:
    dest=archive/Path(name).name
    if not dest.exists(): shutil.copy2(R/name,dest)
print('Prepared v2 without overwriting v1 outputs or production sources.')
