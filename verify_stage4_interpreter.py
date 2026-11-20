"""Mocked language-model checks; no model generation or patient-session writes."""
import json
import unittest
from unittest.mock import patch
from verify_stage4_conversation_v2 import ConversationChecks
import chat_interpreter as parser
import conversation_service as conversation
import chat_service as core
import plan_store
from types import SimpleNamespace


class InterpreterChecks(unittest.TestCase):
    setUp = ConversationChecks.setUp

    def payload(self, **kw):
        d=dict(intent='nutrient',food_id='0',replacement_id=None,servings=None,
               nutrients=['DR1TPROT'],meal=None,per_serving=False)
        d.update(kw); return json.dumps(d)

    def parse(self, **kw):
        return parser.canonicalize(self.payload(**kw), self.foods,
            self.plan['solved_food_basket'], {'DR1TPROT':'protein'})

    def ask(self,message):
        return conversation.answer(message,self.dossier,session_id=self.sid,
            store_dir=self.root,use_llm_interpretation=True)

    def test_schema_rejects_invented_fields_ids_and_nutrients(self):
        for kw in [dict(food_id='invented'),dict(nutrients=['fake']),dict(relax_rules=True),
                   dict(servings=True),dict(servings='2'),dict(nutrients=['DR1TPROT','DR1TPROT']),
                   dict(intent='replace',food_id='0',replacement_id='0',servings=2,nutrients=[]),
                   dict(intent='clarification'),dict(intent='unsupported',food_id=None)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):self.parse(**kw)

    def test_valid_read_request_uses_backend_arithmetic(self):
        req,command=self.parse()
        with patch.object(parser,'interpret',return_value=dict(status='parsed',request=req,command=command)):
            r=self.ask('Tell me the building-block nutrient contribution of the first dish')
        self.assertIn('20.00 g',r['answer'])
        self.assertEqual(r['interpretation_status'],'parsed')

    def test_compact_actions_allow_only_inapplicable_omissions(self):
        for data in [dict(intent='remove',food_id='0'),dict(intent='portion',food_id='0'),
                     dict(intent='replace',food_id='0',replacement_id='4',servings=2),dict(intent='plan')]:
            req,command=parser.canonicalize(json.dumps(data),self.foods,self.plan['solved_food_basket'],{'DR1TPROT':'protein'})
            self.assertEqual(req.intent,data['intent'])
            self.assertTrue(command)

    def test_compact_schema_still_rejects_missing_arguments_and_scope(self):
        cases=[dict(intent='remove'),dict(intent='replace',food_id='0',replacement_id='4'),
               dict(intent='nutrient',nutrients=['DR1TPROT']),
               dict(intent='remove',food_id='0',servings=2),
               dict(intent='remove',food_id='unknown'),dict(intent='plan',execute=True)]
        for data in cases:
            with self.subTest(data=data),self.assertRaises(ValueError):
                parser.canonicalize(json.dumps(data),self.foods,self.plan['solved_food_basket'],{'DR1TPROT':'protein'})

    def test_mutation_requires_confirmation(self):
        req,command=self.parse(intent='remove',nutrients=[])
        with patch.object(parser,'interpret',return_value=dict(status='parsed',request=req,command=command)), patch.object(core,'tool_propose_revision',return_value=(None,None,{})) as tool:
            first=self.ask('Please omit the first dish')
            tool.assert_not_called()
            self.assertIn('Check this interpreted request',first['clarification_options'])
            self.ask('Check this interpreted request')
            tool.assert_called_once()

    def test_stale_confirmation_does_not_execute(self):
        record=plan_store.get_session(self.sid,store_dir=self.root)
        record['conversation']={'plan_version':0,'pending_interpretation':{'command':'Remove Synthetic fixture 0'}}
        plan_store.save_session(record,store_dir=self.root)
        with patch.object(parser,'interpret',return_value={'status':'unavailable'}), patch.object(core,'tool_propose_revision') as tool:
            self.ask('Check this interpreted request')
            tool.assert_not_called()

    def test_unavailable_falls_back_and_reports_status(self):
        with patch.object(parser,'interpret',return_value={'status':'unavailable'}):
            r=self.ask('Tell me its nutritional contribution')
        self.assertEqual(r['intent'],'clarification')
        self.assertEqual(r['interpretation_status'],'unavailable')

    def test_clear_questions_bypass_even_wrong_model(self):
        with patch.object(parser,'interpret') as model:
            r=self.ask('How much protein in Synthetic fixture 0?')
            model.assert_not_called()
        self.assertIn('20.00 g',r['answer'])
        self.assertEqual(r['interpretation_status'],'standard_route')

    def test_information_cannot_become_removal(self):
        req,command=self.parse(intent='remove',nutrients=[])
        with patch.object(parser,'interpret',return_value=dict(status='parsed',request=req,command=command)), patch.object(core,'tool_propose_revision') as tool:
            r=self.ask('What can you tell me regarding this dish?')
            tool.assert_not_called()
        self.assertEqual(r['interpretation_status'],'semantic_rejected')
        self.assertNotIn('pending_interpretation',plan_store.get_session(self.sid,store_dir=self.root)['conversation'])

    def test_named_description_is_grounded_and_local(self):
        with patch.object(parser,'interpret') as model,patch.object(core,'tool_propose_revision') as solver:
            r=self.ask('Tell me about Synthetic fixture 0')
            model.assert_not_called();solver.assert_not_called()
            self.assertIn('200.00 kcal',r['answer'])
            self.assertIn('10.00 g',r['answer'])
            self.assertIn('catalog_food_description',r['tool_calls_used'])

    def test_ordinal_description_clarifies_then_answers(self):
        with patch.object(parser,'interpret') as model,patch.object(core,'tool_propose_revision') as solver:
            r=self.ask('Could you tell me about the first dish?')
            self.assertIn('Synthetic fixture 0',r['clarification_options'])
            r=self.ask('Synthetic fixture 0')
            self.assertIn('catalog_food_description',r['tool_calls_used'])
            model.assert_not_called();solver.assert_not_called()

    def test_negated_omission_stays_local(self):
        with patch.object(parser,'interpret') as model,patch.object(core,'tool_propose_revision') as solver:
            self.ask('Please do not omit Synthetic fixture 0')
            self.ask('Check this interpreted request')
            model.assert_not_called();solver.assert_not_called()

    def test_model_describe_uses_catalog(self):
        req,command=parser.canonicalize(json.dumps({'intent':'describe','food_id':'0'}),self.foods,self.plan['solved_food_basket'],{})
        with patch.object(parser,'interpret',return_value=dict(status='parsed',request=req,command=command)),patch.object(core,'tool_propose_revision') as solver:
            r=self.ask('Give me a short overview of this dish')
            self.assertIn('catalog_food_description',r['tool_calls_used'])
            solver.assert_not_called()

    def test_action_evidence_rejects_negation_and_extra_constraints(self):
        req,_=self.parse(intent='remove',nutrients=[])
        for message in ['How much protein is in dal?',"Don't leave the dal out",'Leave dal out but keep lunch']:
            with self.subTest(message=message),self.assertRaises(ValueError):parser.validate_action_evidence(req,message)

    def test_negated_explicit_action_bypasses_model_and_solver(self):
        with patch.object(parser,'interpret') as model,patch.object(core,'tool_propose_revision') as tool:
            r=self.ask('Do not remove Synthetic fixture 0')
            model.assert_not_called();tool.assert_not_called()
        self.assertEqual(r['intent'],'clarification')

    def test_allergy_bypasses_interpreter(self):
        with patch.object(parser,'interpret') as model:
            r=self.ask('I am allergic to dairy')
            model.assert_not_called()
        self.assertTrue(r['plan_flagged_unsuitable'])

    def test_rejected_generation_retains_pilot_diagnostics(self):
        import torch
        raw=self.payload(replacement_id='4')
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(core.ps.p14,'_run_transformers_backend',return_value=SimpleNamespace(text=raw,elapsed_ms=12)), \
             self.assertLogs(parser.LOGGER,level='WARNING'):
            result=parser.interpret('protein?',self.foods,self.plan['solved_food_basket'],{},[],{'DR1TPROT':'protein'},diagnostics=True)
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(result['raw_output'],raw)
        self.assertIn('Unexpected replacement',result['error'])
        self.assertGreaterEqual(result['elapsed_ms'],0)

if __name__=='__main__':unittest.main()

