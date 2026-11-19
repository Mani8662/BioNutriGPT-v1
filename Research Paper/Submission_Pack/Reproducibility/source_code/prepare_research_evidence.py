"""Snapshot existing research evidence without training or changing source artifacts."""
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nutrient_rule_policy as policy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Research Paper"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    features = {}
    for node in ast.parse((ROOT / "train_bionutrigpt_stage1.py").read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"BIO_COLS", "DIET_COLS", "LIFE_COLS", "DEMO_COLS"}:
                    features[target.id] = ast.literal_eval(node.value)
    all_features = sum(features.values(), [])
    cohorts = {}
    ids = {}
    for split, name in (("development", "BioNutriGPT_train_2007_2016_ml_ready.csv"),
                        ("temporal_holdout", "BioNutriGPT_test_2017_2020_ml_ready.csv")):
        path = ROOT / "Data" / name
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = list(reader)
        cohorts[split] = {"file": str(path.relative_to(ROOT)), "sha256": digest(path), "n": len(rows),
                          "positive_cases": {t: sum(float(r[t]) == 1 for r in rows) for t in
                                             ("TARGET_DIABETES", "TARGET_OBESITY", "TARGET_CVD")},
                          "all_87_features_present": all(f in columns for f in all_features)}
        ids[split] = {r["SEQN"] for r in rows}
    catalog_path = ROOT / "food_composition_cooked_sourced.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    nonactive = policy.REPORT_ONLY | policy.EXPOSURE_REPORTING
    active = set(catalog[0]["nutrients_per_serving"]) - nonactive
    complete = [r for r in catalog if not active.intersection(r.get("unknown_nutrients", []))]
    artifacts = {}
    names = ["Saved Model/stage1_test_evaluation_report.json", "Saved Model/tripod_baseline_evaluation_report.json",
             "Saved Model/matched_milp_lp_qwen_benchmark_report.json", "Saved Model/stage4_conversational_benchmark_report.json",
             "backend/chat_service.py", "backend/plan_store.py", "stage3_milp_solver.py", "nutrient_rule_policy.py",
             "verify_stage4_chat_repairs.py", "verify_stage3_production.py", "Stage4_Chatbot_Repair_Report.md"]
    for name in names:
        p = ROOT / name
        artifacts[name] = {"sha256": digest(p), "bytes": p.stat().st_size}
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Descriptive snapshot of current files; does not rerun model or benchmark evaluation.",
              "cohorts": cohorts, "participant_id_overlap": len(ids["development"] & ids["temporal_holdout"]),
              "features": features, "feature_count": len(all_features),
              "catalog": {"file": catalog_path.name, "sha256": digest(catalog_path), "count": len(catalog),
                          "versions": dict(Counter(r["catalog_version"] for r in catalog)),
                          "source_families": dict(Counter("INDB_recipe" if r["data_source"].startswith("INDB") else "USDA_single_food_proxy" for r in catalog)),
                          "categories": dict(Counter(r["category"] for r in catalog)),
                          "dietary_types": dict(Counter(r.get("dietary_type", "unspecified") for r in catalog)),
                          "active_nutrient_count": len(active), "report_only": sorted(policy.REPORT_ONLY),
                          "exposure_reporting": sorted(policy.EXPOSURE_REPORTING),
                          "complete_active_records_before_diet_or_preference_filters": len(complete),
                          "unknown_active_counts": dict(Counter(n for r in catalog for n in r.get("unknown_nutrients", []) if n in active)),
                          "records_with_known_allergen_tags": sum(bool(r.get("allergen_info", {}).get("known_allergens")) for r in catalog),
                          "records_with_unknown_allergen_risk": sum(r.get("allergen_info", {}).get("unknown_allergens_possible", True) for r in catalog)},
              "rule_version": policy.RULESET_VERSION, "artifacts": artifacts}
    lasi_path=ROOT / "Saved Model/LASI_Evaluation_Audit.json"
    if lasi_path.exists():
        lasi=json.loads(lasi_path.read_text(encoding="utf-8"))
        result["lasi_evaluation"]={"cohort":lasi["cohort"],"stage3":lasi["stage3"],
                                   "audit_file":str(lasi_path.relative_to(ROOT)),"audit_sha256":digest(lasi_path),
                                   "interpretation":"Selected-profile computational pilot; no Stage 1 external or clinical validation. Read audit limitations."}
    v4_path=ROOT / "Saved Model/stage4_lasi_v4_verified_summary.json"
    if v4_path.exists():
        v4=json.loads(v4_path.read_text(encoding="utf-8"))
        result["stage4_v4"]={"verified_summary_file":str(v4_path.relative_to(ROOT)),"sha256":digest(v4_path),
                            "systems":v4['systems'],"statistics":v4['statistics'],"scope":v4['scope']}
    (OUT / "evidence_snapshot.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("cohorts", "participant_id_overlap", "feature_count", "catalog", "rule_version")}, indent=2))


if __name__ == "__main__":
    main()
