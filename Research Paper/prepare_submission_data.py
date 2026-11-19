"""Prepare a local, journal-neutral evidence bundle; no training or LLM calls."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nutrient_rule_policy as policy

PACK = ROOT / "Research Paper" / "Submission_Pack"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ("Figures", "Reproducibility/source_code", "Evidence/Historical", "Evidence/Current", "Sources", "Author_Completion", "Reporting", "_QA"):
        (PACK / name).mkdir(parents=True, exist_ok=True)
    snap = json.loads((ROOT / "Research Paper/evidence_snapshot.json").read_text(encoding="utf-8"))
    catalog = json.loads((ROOT / "food_composition_cooked_sourced.json").read_text(encoding="utf-8"))
    codes = list(catalog[0]["nutrients_per_serving"])
    rules = []
    for code in codes:
        meta = policy.reviewed_rule_metadata(code, 0, policy.NUMERICAL_SEARCH_CAP)
        rules.append({k: meta[k] for k in ("nutrient", "unit", "planning_role", "source_url", "stage3_enforcement", "review_note")})
    features = [{"position": i, "modality": modality.replace("_COLS", "").lower(), "feature": feature}
                for i, (modality, feature) in enumerate(((m, f) for m, fs in snap["features"].items() for f in fs), 1)]
    # This is a feature-order contract, not a newly invented patient input file.
    schema = {"schema_version": "nhanes87_v2", "feature_count": len(features), "features": features,
              "scope": "Model feature order. A patient file also needs the application's validated input metadata; this is not an upload template."}
    (PACK / "Reproducibility/feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (PACK / "Reproducibility/nutrient_rule_roles.json").write_text(json.dumps({"rule_version":policy.RULESET_VERSION,
        "note":"Roles and contextual references only; numeric patient-specific bounds are not inferred here.", "rules": rules}, indent=2), encoding="utf-8")
    observed = {"python":sys.version,"platform":platform.platform(),"logical_cpu_count":os.cpu_count(),"packages":{}}
    for name in ("numpy","pandas","scipy","torch","scikit-learn","joblib","fastapi","uvicorn","pydantic","transformers","bitsandbytes","accelerate","safetensors"):
        try: observed["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: observed["packages"][name] = None
    (PACK / "Reproducibility/environment_observed.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
    required = ["numpy","pandas","scipy","torch","scikit-learn","joblib","fastapi","uvicorn","pydantic","python-multipart"]
    (PACK / "Reproducibility/requirements-core.txt").write_text("# Observed versions where available; not a clean-environment lockfile.\n" + "\n".join(
        name + ("=="+importlib.metadata.version(name) if importlib.util.find_spec(name.replace("scikit-learn","sklearn").replace("python-multipart","multipart")) and name not in ("uvicorn",) else "") for name in required) + "\n", encoding="utf-8")
    (PACK / "Reproducibility/requirements-llm.txt").write_text("# Optional; installation must match the operating system, Python and CUDA.\n-r requirements-core.txt\ntransformers\naccelerate\nbitsandbytes\nsafetensors\n", encoding="utf-8")
    whitelist = ["train_bionutrigpt_stage1.py","stage2_counterfactual_engine.py","nutrient_rule_policy.py","stage3_milp_solver.py",
                 "infer_bionutrigpt_pipeline_Stage_1_and_2.py","infer_bionutrigpt_pipeline_Stage_1_to_4.py","stage4_clinical_compiler.py",
                 "verify_stage3_production.py","verify_stage4_chat_repairs.py","prepare_research_evidence.py",
                 "benchmark_matched_meal_planners.py","benchmark_stage4_conversational_revisions.py",
                 "benchmark_optimizer_comparison.py", "audit_stage2_rules.py", "build_indian_food_database.py", "repair_b12_data.py",
                 "extract_lasi_indian_cohort.py", "benchmark_lasi_stage3_indian_cohort.py", "benchmark_stage4_lasi_conversational.py", "analyse_lasi_evaluation.py",
                 "benchmark_stage4_lasi_conversational_v3.py", "benchmark_stage4_lasi_conversational_v4.py",
                 "stage4_benchmark_analysis.py", "stage4_benchmark_protocol.py", "prepare_stage4_v4_runner.py",
                 "verify_stage4_benchmark_corrections.py", "verify_stage4_v4_results.py", "analyse_stage4_v3_corrections.py",
                 "build_sourced_cooked_catalog.py","enrich_recipe_active_gaps.py",
                 "backend/chat_service.py","backend/plan_store.py","backend/pipeline_service.py","backend/schemas.py","backend/main.py","backend/requirements.txt",
                 "Data/download_nhanes_2017_2020.py","Data/merge_raw_datasets.py","Data/preprocess_for_training.py","Data/preprocess_raw_genuinely.py"]
    for name in whitelist:
        dest = PACK / "Reproducibility/source_code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    evidence = {"Evidence/Current":["Research Paper/evidence_snapshot.json","Stage4_Chatbot_Repair_Report.md",
                                    "Saved Model/stage1_test_evaluation_report.json","Data/preprocessing_report.json","Data/cleaning_report.json",
                                    "Saved Model/stage4_lasi_v4_verified_summary.json", "Saved Model/stage4_lasi_v4_verified_summary.md",
                                    "Saved Model/stage4_lasi_v4_run_provenance.json", "Saved Model/lasi_cohort_extraction_summary.json"],
                "Evidence/Historical":["Saved Model/matched_milp_lp_qwen_benchmark_report.json","Saved Model/stage4_conversational_benchmark_report.json",
                                       "Saved Model/tripod_baseline_evaluation_report.json",
                                       "Saved Model/LASI_Evaluation_Audit.json", "Saved Model/LASI_Evaluation_Audit.md"]}
    for folder, names in evidence.items():
        for name in names: shutil.copy2(ROOT / name, PACK / folder / Path(name).name)
    configured = subprocess.run(["git","remote","get-url","origin"], cwd=ROOT,capture_output=True,text=True).stdout.strip()
    head = subprocess.run(["git","rev-parse","HEAD"], cwd=ROOT,capture_output=True,text=True).stdout.strip()
    artifacts = ["food_composition_cooked_sourced.json","Saved Model/bionutrigpt_stage1_best.pt", "Data/preprocessing_artifacts.joblib"]
    artifacts += ["Data/lasi_w1_indian_cohort_100.json", "Saved Model/stage4_lasi_conversational_benchmark_v4_report.json", "Saved Model/stage4_lasi_v4_frozen_scenarios.json"]
    artifacts += [str(p.relative_to(ROOT)) for p in (ROOT / "Saved Model").glob("*.joblib")]
    external = {name:{"sha256":sha(ROOT/name),"bytes":(ROOT/name).stat().st_size} for name in artifacts if (ROOT/name).exists()}
    manifest = {"configured_git_remote_not_verified_release":configured,"base_git_head_not_clean_release":head,
                "source_snapshot":"Curated local file snapshot; do not claim this equals HEAD.",
                "external_artifacts_not_bundled":external,
                "excluded":"Patient records, plan_sessions, source food workbooks, full NHANES tables, model weights and assigned-count 100-case summaries are not bundled.",
                "source_files":{name:sha(ROOT/name) for name in whitelist}}
    (PACK / "Reproducibility/artifact_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    data={"snapshot":snap,"features":features,"rules":rules,"catalog":[{k:r.get(k) for k in ("food_id","name","category","dietary_type","serving_weight_g","data_source","unknown_nutrients","allergen_info")} for r in catalog],
          "stage1":json.loads((ROOT/"Saved Model/stage1_test_evaluation_report.json").read_text(encoding="utf-8")),
          "stage4_v4":json.loads((ROOT/"Saved Model/stage4_lasi_v4_verified_summary.json").read_text(encoding="utf-8")),
          "prepared_without_new_experiments":True}
    (PACK / "Sources/submission_data.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    shutil.copy2(ROOT / "Research Paper/BioNutriGPT_Manuscript_v2.md", PACK / "Sources/Main_Manuscript.md")
    print("Prepared metadata, curated source snapshot, environment inventory and version-separated evidence.")


if __name__ == "__main__":
    main()
