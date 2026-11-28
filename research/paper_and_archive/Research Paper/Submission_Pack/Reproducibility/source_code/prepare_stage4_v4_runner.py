"""Create the v4 runner from the preserved v3 implementation, with scoped repairs."""
from pathlib import Path


def main():
    path=Path(__file__).resolve().parent
    t=(path/'benchmark_stage4_lasi_conversational_v3.py').read_text(encoding='utf-8').replace('v3','v4')
    def change(old,new):
        nonlocal t
        if t.count(old)!=1:raise ValueError((old[:80],t.count(old)))
        t=t.replace(old,new,1)
    change('import stage3_milp_solver as s3','import stage3_milp_solver as s3\nfrom stage4_benchmark_protocol import eligible_foods, prompt_post_checks, load_frozen_scenarios, snapshot, checkpoint_result\nfrom stage4_benchmark_analysis import revision_accounting, protocol_uncertainty, paired_ablation as analyse_paired_ablation, paired_protocol_difference')
    a=t.index('def _fmt_num(');b=t.index('\n\ndef _active_nutrient_codes',a)
    t=t[:a]+'def _fmt_num(val: float) -> str:\n    """Keep round-trip coefficient precision."""\n    return repr(float(val))\n'+t[b:]
    a=t.index('    eligible_foods = [',t.index('def _build_equal_info_qwen_prompt'));b=t.index('    short_codes =',a)
    t=t[:a]+'    prompt_foods = eligible_foods(catalog, active_codes, dietary_mode)\n'+t[b:]
    change('    for f in eligible_foods:\n','    for f in prompt_foods:\n')
    a=t.index('        "post_checks": {',t.index('def _build_equal_info_qwen_prompt'));b=t.index('        "revision_objective":',a)
    t=t[:a]+'        "post_checks": prompt_post_checks(dossier, catalog, s3),\n'+t[b:]
    change('"sum(|q_new - q_prev|) and calorie deviation from target_calories."','"0.35 * sum(|q_new - q_prev|) + (150 / target_calories) * abs(delivered_kcal - target_calories) + 0.10 * sum(selection_cost for selected foods)."')
    change('header = "id|name|cat|diet|wt_g|known_alg|"','header = "id|name|cat|diet|wt_g|selection_cost|known_alg|"')
    change('{int(round(f.serving_weight_g))}|{alg}|{vals}', '{repr(float(f.serving_weight_g))}|{repr(float(f.selection_cost))}|{alg}|{vals}')
    change('    prompt_tokens = int(inputs["input_ids"].shape[1])','    prompt_tokens = int(inputs["input_ids"].shape[1])\n    if prompt_tokens + cfg.max_new_tokens > model.config.max_position_embeddings:\n        raise ValueError("Context exceeded; do not silently truncate prompt.")\n    if input_device.type != "cuda":\n        raise RuntimeError("GPU benchmark must not silently run on CPU.")')
    change('    t0 = time.perf_counter()\n    with torch.inference_mode():','    torch.cuda.synchronize(input_device)\n    t0 = time.perf_counter()\n    with torch.inference_mode():')
    change('    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)\n    out_ids', '    torch.cuda.synchronize(input_device)\n    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)\n    out_ids')
    change('        "raw_llm_output_full": raw,','        "raw_llm_output_full": raw,\n        "prompt_full": rendered,\n        "actual_model_metadata": {"model_revision": getattr(model.config, "_commit_hash", None), "input_device": str(input_device), "gpu_name": torch.cuda.get_device_name(input_device), "quantization_config": str(getattr(model.config, "quantization_config", None)), "max_new_tokens": cfg.max_new_tokens, "do_sample": False, "repetition_penalty": 1.03},')
    change('                    msg = turn["message"]  # Strictly frozen; NEVER rewritten at runtime!','                    input_state = {"basket": sorted((x["food_id"], int(x["quantity"])) for x in cur_basket), "excluded_food_ids": sorted(accepted_excludes), "locked_food_quantities": dict(accepted_locks), "max_active_dishes": accepted_dish_cap}\n                    msg = turn["message"]  # Strictly frozen; NEVER rewritten at runtime!')
    change('                        "turn": t_idx,','                        "input_state": input_state,\n                        "turn": t_idx,')
    change('                        turn_entry["raw_llm_output_full"] = raw_llm_full','                        turn_entry["raw_llm_output_full"] = raw_llm_full\n                        turn_entry["prompt_full"] = raw_resp["prompt_full"]\n                        turn_entry["actual_model_metadata"] = raw_resp["actual_model_metadata"]')
    change('                    t_logs.append(turn_entry)', '                    journal_dir = ROOT / "Saved Model/stage4_lasi_v4_checkpoints"\n                    journal_dir.mkdir(parents=True, exist_ok=True)\n                    with (journal_dir / (system_name + ".turns.jsonl")).open("a", encoding="utf-8") as journal:\n                        journal.write(json.dumps({"scenario_id": sc["id"], "initial_v1_basket": v1_basket, "turn_record": turn_entry}, ensure_ascii=False) + "\\n")\n                        journal.flush()\n                    t_logs.append(turn_entry)')
    change('    if run_qwen:\n        summary["qwen_json_parse_count"]','    summary["revision_outcomes"] = revision_accounting(scenario_logs)\n    summary["uncertainty"] = protocol_uncertainty(scenario_logs)\n    if run_qwen:\n        summary["qwen_json_parse_count"]')
    change('    scenarios, dossiers, manifest = _select_balanced_lasi_scenarios(catalog)','    scenarios, dossiers, manifest = load_frozen_scenarios(ROOT, catalog, s3, build_stage2_from_lasi_patient)')
    change('    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")','    if manifest_path.exists():\n        raise FileExistsError("v4 manifest exists; archive an earlier run explicitly before replacing it.")\n    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")\n    run_provenance = snapshot(ROOT, manifest_path)\n    (ROOT / "Saved Model/stage4_lasi_v4_run_provenance.json").write_text(json.dumps(run_provenance, indent=2), encoding="utf-8")')
    change('        "benchmark_version": "stage4-lasi-indian-conversational-v4",\n        "generated_utc":','        "run_snapshot": run_provenance,\n        "source_hashes_unchanged_at_end": run_provenance["hashes"] == snapshot(ROOT, manifest_path)["hashes"],\n        "repaired_statistical_analysis": {"ablation": analyse_paired_ablation(full_guarded["scenarios"], ablation_no_l1["scenarios"]), "matched_protocol_difference": paired_protocol_difference(guarded_subset["scenarios"], qwen_subset_res["scenarios"])},\n        "benchmark_version": "stage4-lasi-indian-conversational-v4",\n        "generated_utc":')
    change('def main() -> None:', '_run_eval_without_checkpoint = run_eval_v4\n\ndef run_eval_v4(system_name, *args, **kwargs):\n    result = _run_eval_without_checkpoint(system_name, *args, **kwargs)\n    checkpoint_result(ROOT, system_name, result)\n    return result\n\ndef main() -> None:')
    t=t.replace('Strict Protocol Pass Rate (excludes timeouts)','Strict Protocol Pass Rate (timeouts count as failures)')
    # Replace the ambiguous old loop accounting too, although summaries are
    # independently reconstructed from logs above.
    t=t.replace('                                counts["missed_feasible_other"] += 1','                                counts["missed_feasible_other"] += int(oracle_status == "feasible")')
    (path/'benchmark_stage4_lasi_conversational_v4.py').write_text(t,encoding='utf-8')
    print('Created corrected v4 runner. No inference executed; v3 preserved.')


if __name__=='__main__':main()
