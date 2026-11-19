"""Regression checks for benchmark correctness, no optimization or LLM generation."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import contextlib
import io

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
from verify_stage3_production import fixture
import stage3_milp_solver as s3
import nutrient_rule_policy as nrp
from stage4_benchmark_protocol import eligible_foods,prompt_post_checks,load_frozen_scenarios
from stage4_benchmark_analysis import revision_accounting,paired_ablation,protocol_uncertainty
import benchmark_stage4_lasi_conversational_v4 as v4


class BenchmarkCorrections(unittest.TestCase):
    def setUp(self):
        self.foods,self.stage2,q=fixture()
        for r in self.stage2['nutrient_constraint_envelope']:
            r['rule_provenance']=nrp.reviewed_rule_metadata(r['nutrient'],r['min_bound'],r['max_bound'])
        out=s3._build_output_from_quantities(q,self.foods,s3._parse_stage2_input(self.stage2),{})
        out.accepted=True
        self.d={'stage2_counterfactual_prescription':self.stage2,'stage3_discrete_meal_synthesis':out.to_dict()}

    def test_missing_data_excluded_but_known_zeros_retained(self):
        broken=replace(self.foods[0],unknown_nutrients=('DR1TVB12',))
        known=replace(self.foods[1],nutrients_per_serving={**self.foods[1].nutrients_per_serving,'DR1TVB12':0.0})
        got=eligible_foods([broken,known],['DR1TVB12'],'pure_vegetarian')
        self.assertEqual([x.food_id for x in got],[known.food_id])

    def test_prompt_uses_canonical_checks_and_complete_catalog(self):
        prompt=v4._build_equal_info_qwen_prompt('How much protein?',self.d,self.foods,'pure_vegetarian',[],{},None,[])
        rules=json.loads(prompt.split('PLATE_AND_SAFETY_RULES:\n')[1].split('\n\n')[0])
        self.assertEqual(rules['post_checks'],prompt_post_checks(self.d,self.foods,s3))
        self.assertEqual(rules['post_checks']['calories_within_pct_of_target'],5.0)
        self.assertEqual(rules['post_checks']['carb_pct_kcal'],[45.0,65.0])
        self.assertEqual(rules['post_checks']['fat_pct_kcal'],[20.0,35.0])
        self.assertIn('selection_cost',prompt)

    def test_real_catalog_has_identical_mode_and_completeness_eligibility(self):
        cat=s3.load_food_database();codes=v4._active_nutrient_codes(self.stage2)
        for mode,expected in [('pure_vegetarian',37),('veg_nonveg_combo',43),('pure_non_vegetarian',43)]:
            self.assertEqual(len(eligible_foods(cat,codes,mode)),expected)

    def test_saved_log_accounting_does_not_call_infeasible_case_missed_feasible(self):
        report=json.loads((ROOT/'Saved Model/stage4_lasi_conversational_benchmark_v3_report.json').read_text(encoding='utf-8'))
        ss=report['detailed_logs']['prompt_only_qwen_equal_info_subset'];got=revision_accounting(ss)
        self.assertEqual(got['missed_feasible_other'],7)
        self.assertEqual(got['malformed_json_no_proposal'],2)
        self.assertEqual(got['unexplained_null_on_proven_infeasible'],1)
        self.assertEqual(got['invalid_proposals_on_feasible'],2)

    def test_same_state_ablation_and_cluster_intervals_reproduce(self):
        report=json.loads((ROOT/'Saved Model/stage4_lasi_conversational_benchmark_v3_report.json').read_text(encoding='utf-8'))
        full=report['detailed_logs']['guarded_stage4_full_l1'];other=report['detailed_logs']['guarded_stage4_no_l1_ablation']
        got=paired_ablation(full,other,draws=50)
        self.assertEqual(got['same_input_state_pairs']['paired_turns'],25)
        self.assertAlmostEqual(got['same_input_state_pairs']['full_mean'],7.72)
        self.assertAlmostEqual(got['same_input_state_pairs']['no_penalty_mean'],17.28)
        subset=report['detailed_logs']['guarded_stage4_matched_subset']
        u=protocol_uncertainty(subset,draws=50)
        self.assertEqual(u,protocol_uncertainty(subset,draws=50))
        self.assertLess(u['all_turns_passed_scenario_wilson_95ci'][0],1)

    def test_frozen_v3_inputs_validate_without_resampling(self):
        scenarios,dossiers,manifest=load_frozen_scenarios(ROOT,s3.load_food_database(),s3,v4.build_stage2_from_lasi_patient)
        original=json.loads((ROOT/'Saved Model/stage4_lasi_v3_frozen_scenarios.json').read_text(encoding='utf-8'))
        self.assertEqual(len(scenarios),36)
        self.assertEqual([s['turns'] for s in scenarios],[s['turns'] for s in original['scenarios']])
        self.assertEqual(len(dossiers),36)
        self.assertIn('source_v3_frozen_manifest_sha256',manifest)

    def test_checkpoint_written_before_report_assembly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result={'summary':{'measured':True},'scenarios':[{'raw':'preserved'}]}
            with patch.object(v4,'ROOT',Path(tmp)), patch.object(v4,'_run_eval_without_checkpoint',return_value=result):
                self.assertEqual(v4.run_eval_v4('test_system'),result)
            saved=Path(tmp)/'Saved Model/stage4_lasi_v4_checkpoints/test_system.json'
            self.assertEqual(json.loads(saved.read_text(encoding='utf-8')),result)

    def test_complete_main_report_export_without_inference(self):
        old=json.loads((ROOT/'Saved Model/stage4_lasi_conversational_benchmark_v3_report.json').read_text(encoding='utf-8'))
        manifest=json.loads((ROOT/'Saved Model/stage4_lasi_v3_frozen_scenarios.json').read_text(encoding='utf-8'))
        scenarios=copy.deepcopy(manifest['scenarios'])
        def fake_run(name,*args,**kwargs):
            if name.startswith('guarded_stage4_full_l1'):key='guarded_stage4_full_l1';group='full_36_scenario_ablation'
            elif name.startswith('guarded_stage4_no_l1'):key='guarded_stage4_no_l1_ablation';group='full_36_scenario_ablation'
            elif name.startswith('guarded_stage4_matched'):key='guarded_stage4_matched_subset';group='matched_12_scenario_equal_info_comparison'
            else:key='prompt_only_qwen_equal_info_subset';group='matched_12_scenario_equal_info_comparison'
            logs=old['detailed_logs'][key]
            return {'summary':old[group][key],'scenarios':logs,
                    'paired_deltas':{f"{s['scenario_id']}::T{t['turn']}":t['serving_delta'] for s in logs for t in s['turns'] if t['serving_delta'] is not None}}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'Saved Model').mkdir()
            with patch.object(v4,'ROOT',Path(tmp)),patch.object(v4,'load_frozen_scenarios',return_value=(scenarios,{},manifest)), \
                 patch.object(v4,'snapshot',return_value={'hashes':{}}),patch.object(v4,'run_eval_v4',side_effect=fake_run),contextlib.redirect_stdout(io.StringIO()):
                v4.main()
            result=Path(tmp)/'Saved Model/stage4_lasi_conversational_benchmark_v4_report.json'
            data=json.loads(result.read_text(encoding='utf-8'))
            self.assertIn('same_input_state_pairs',data['repaired_statistical_analysis']['ablation'])
            self.assertTrue(result.with_suffix('.md').exists())
            self.assertTrue(data['source_hashes_unchanged_at_end'])


if __name__=='__main__':unittest.main(verbosity=2)
