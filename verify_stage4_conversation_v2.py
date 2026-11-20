"""Conversation tests with synthetic data and temporary sessions only."""
import unittest
from unittest.mock import patch
from verify_stage4_chat_repairs import ChatRepairs
import conversation_service as conversation
import chat_service as core
import plan_store


class ConversationChecks(unittest.TestCase):
    setUp = ChatRepairs.setUp

    def ask(self, text):
        return conversation.answer(text, self.dossier, session_id=self.sid, store_dir=self.root)

    def test_food_scope_and_followup(self):
        first = self.ask('How much protein in Synthetic fixture 0?')
        self.assertIn('20.00 g', first['answer'])
        second = self.ask('What about its calories?')
        self.assertIn('400.00 kcal', second['answer'])
        self.assertEqual(len(plan_store.get_session(self.sid,store_dir=self.root)['conversation_turns']),2)

    def test_per_serving(self):
        answer = self.ask('How much protein in one serving of Synthetic fixture 0?')
        self.assertIn('10.00 g',answer['answer'])

    def test_ambiguous_followup_and_choice(self):
        response = self.ask('How much protein does it have?')
        self.assertEqual(response['intent'],'clarification')
        self.assertIn('Synthetic fixture 0',response['clarification_options'])
        result = self.ask('Synthetic fixture 0')
        self.assertEqual(result['intent'],'nutrient')
        self.assertIn('20.00 g',result['answer'])

    def test_meal_scope_multiple_nutrients(self):
        with patch.object(core,'tool_get_plan',return_value={'meal_schedule':[{'meal':'Lunch','items':[{'food_id':'0','quantity':1}]}]}):
            result=self.ask('How much protein and fibre in lunch?')
        self.assertIn('10.00 g',result['answer'])
        self.assertIn('4.00 g',result['answer'])

    def test_exact_swap_spec(self):
        with patch.object(core,'tool_propose_revision',return_value=(None,None,{'summary':'fixture conflict'})) as tool:
            self.ask('Replace Synthetic fixture 0 with 2 servings of Synthetic fixture 4')
        spec=tool.call_args.args[3]
        self.assertEqual(spec['excluded_food_ids'],['0'])
        self.assertEqual(spec['locked_food_quantities'],{'4':2})

    def test_negation_never_calls_solver(self):
        with patch.object(core,'tool_propose_revision') as tool:
            result=self.ask("Don't replace Synthetic fixture 0 with 2 servings of Synthetic fixture 4")
        tool.assert_not_called()
        self.assertEqual(result['intent'],'clarification')

    def test_stale_focus_requires_clarification(self):
        self.ask('How much protein in Synthetic fixture 0?')
        record=plan_store.get_session(self.sid,store_dir=self.root)
        record['conversation']['plan_version']=0
        plan_store.save_session(record,store_dir=self.root)
        self.assertEqual(self.ask('What about its calories?')['intent'],'clarification')

    def test_allergy_guard_retained(self):
        result=self.ask('I am allergic to dairy')
        self.assertTrue(result['plan_flagged_unsuitable'])
        self.assertIn('dairy',plan_store.get_session(self.sid,store_dir=self.root)['preferences']['reported_allergies'])

    def test_replacement_paraphrases_preserve_both_foods_and_quantity(self):
        messages=[
            'I would rather have Synthetic fixture 4 instead of Synthetic fixture 0, two servings please.',
            "I'd rather have 2 servings of Synthetic fixture 4 instead of Synthetic fixture 0.",
            'Could I have two servings of Synthetic fixture 4 in place of Synthetic fixture 0?',
            'Please use 2 servings of Synthetic fixture 4 instead of Synthetic fixture 0',
        ]
        for message in messages:
            with self.subTest(message=message), patch.object(conversation.chat_interpreter,'interpret') as llm, patch.object(core,'tool_propose_revision',return_value=(None,None,{})) as tool:
                result=conversation.answer(message,self.dossier,session_id=self.sid,store_dir=self.root,use_llm_interpretation=True)
                llm.assert_not_called()
                self.assertEqual(result['intent'],'revision')
                spec=tool.call_args.args[3]
                self.assertEqual(spec['excluded_food_ids'],['0'])
                self.assertEqual(spec['locked_food_quantities'],{'4':2})
                self.assertNotIn('meal',spec)

    def test_quantity_followup_completes_replacement(self):
        with patch.object(core,'tool_propose_revision',return_value=(None,None,{})) as tool:
            first=self.ask('Replace Synthetic fixture 0 with Synthetic fixture 4')
            tool.assert_not_called()
            self.assertIn('2 servings',first['clarification_options'])
            self.ask('two servings please')
            self.assertEqual(tool.call_args.args[3]['locked_food_quantities'],{'4':2})

    def test_stale_quantity_followup_cannot_change_plan(self):
        self.ask('Replace Synthetic fixture 0 with Synthetic fixture 4')
        record=plan_store.get_session(self.sid,store_dir=self.root)
        record['conversation']['plan_version']=0
        plan_store.save_session(record,store_dir=self.root)
        with patch.object(core,'tool_propose_revision') as tool:
            self.ask('2 servings')
            tool.assert_not_called()

    def test_meal_extra_instruction_and_weight_are_not_dropped(self):
        for message in [
            'Replace Synthetic fixture 0 with 2 servings of Synthetic fixture 4 at lunch',
            'Replace Synthetic fixture 0 with 2 servings of Synthetic fixture 4. Keep dinner.',
            'Could I have 200 grams of Synthetic fixture 4 instead of Synthetic fixture 0?',
        ]:
            with self.subTest(message=message),patch.object(core,'tool_propose_revision') as tool:
                self.assertEqual(self.ask(message)['intent'],'clarification')
                tool.assert_not_called()

    def test_intervening_question_cancels_pending_quantity(self):
        self.ask('Replace Synthetic fixture 0 with Synthetic fixture 4')
        self.ask('How much protein in my plan?')
        with patch.object(core,'tool_propose_revision') as tool:
            self.ask('2 servings')
            tool.assert_not_called()

    def test_take_off_is_local_and_requires_confirmation(self):
        with patch.object(conversation.chat_interpreter,'interpret') as model, patch.object(core,'tool_propose_revision',return_value=(None,None,{})) as tool:
            r=conversation.answer('I am fed up with Synthetic fixture 0; take it off my plan.',self.dossier,session_id=self.sid,store_dir=self.root,use_llm_interpretation=True)
            model.assert_not_called();tool.assert_not_called()
            self.assertEqual(r['interpreted_request'],'Remove Synthetic fixture 0')
            self.ask('Check this interpreted request')
            self.assertEqual(tool.call_args.args[3]['excluded_food_ids'],['0'])

    def test_leave_out_uses_food_context(self):
        self.ask('How much protein in Synthetic fixture 0?')
        r=self.ask('Please leave it out')
        self.assertEqual(r['interpreted_request'],'Remove Synthetic fixture 0')

    def test_meal_placement_checks_actual_schedule(self):
        req={'recipe_boiled_egg':'Breakfast'}
        self.assertFalse(core._meal_requirements_met([{'meal':'Dinner','items':[{'food_id':'recipe_boiled_egg','quantity':1}]}],req))
        self.assertTrue(core._meal_requirements_met([{'meal':'Breakfast','items':[{'food_id':'recipe_boiled_egg','quantity':1}]}],req))

    def test_generic_food_meal_request(self):
        import dataclasses
        for food_name,meal in [('Boiled egg','Breakfast'),('Oat porridge','Dinner'),('Lentil dish','Lunch')]:
            food=dataclasses.replace(self.foods[0],food_id='generic_food',name=food_name)
            with patch.object(core.ps.s3,'load_food_database',return_value=self.foods+[food]),patch.object(core,'tool_propose_revision',return_value=(None,None,{})) as tool:
                self.ask('Include 2 servings of '+food_name+' at '+meal)
                tool.assert_not_called()
                self.ask('Check this interpreted request')
                self.assertEqual(tool.call_args.args[3]['meal_requirements'],{'generic_food':meal})
                self.assertEqual(tool.call_args.args[3]['locked_food_quantities'],{'generic_food':2})

    def test_generic_schedule_preserves_quantities(self):
        basket=self.plan['solved_food_basket']
        fid=basket[0]['food_id']
        schedule=core.build_requested_schedule(basket,{fid:'Dinner'})
        self.assertTrue(core._meal_requirements_met(schedule,{fid:'Dinner'}))
        actual={}
        for slot in schedule:
            for item in slot['items']:actual[item['food_id']]=actual.get(item['food_id'],0)+item['quantity']
        self.assertEqual(actual,{x['food_id']:x['quantity'] for x in basket})

    def test_compound_placements_are_atomic(self):
        with patch.object(core,'tool_propose_revision',return_value=(None,None,{})) as tool:
            self.ask('Include 1 serving of Synthetic fixture 0 at Breakfast and include two servings of Synthetic fixture 4 at Dinner')
            tool.assert_not_called()
            self.ask('Check this interpreted request')
            tool.assert_called_once()
            spec=tool.call_args.args[3]
            self.assertEqual(spec['locked_food_quantities'],{'0':1,'4':2})
            self.assertEqual(spec['meal_requirements'],{'0':'Breakfast','4':'Dinner'})

    def test_compound_unknown_clauses_never_partially_execute(self):
        for suffix in ['keep lunch unchanged','include 1 serving of Synthetic fixture 4 without salt at Dinner','include 1 serving of Synthetic fixture 0 at Dinner']:
            with self.subTest(suffix=suffix),patch.object(core,'tool_propose_revision') as tool:
                result=self.ask('Include 1 serving of Synthetic fixture 0 at Breakfast; '+suffix)
                self.assertNotIn('Check this interpreted request',result['clarification_options'])
                self.ask('Check this interpreted request')
                tool.assert_not_called()

    def test_compound_cancel_does_not_call_solver(self):
        with patch.object(core,'tool_propose_revision') as tool:
            self.ask('Include 1 serving of Synthetic fixture 0 at Breakfast; include 1 serving of Synthetic fixture 4 at Dinner')
            self.ask('Cancel interpreted request')
            self.ask('Check this interpreted request')
            tool.assert_not_called()

    def test_compound_group_exclusion_and_conflict(self):
        with patch.object(conversation.food_groups,'excluded_ids',return_value=['0']),patch.object(core,'tool_propose_revision',return_value=(None,None,{})) as tool:
            self.ask('Remove fish and include 1 serving of Synthetic fixture 4 at Dinner')
            tool.assert_not_called()
            self.ask('Check this interpreted request')
            self.assertEqual(tool.call_args.args[3]['excluded_food_groups'],['fish'])
            tool.reset_mock()
            result=self.ask('Remove fish and include 1 serving of Synthetic fixture 0 at Dinner')
            self.assertIn('conflict',result['answer'])
            tool.assert_not_called()

    def test_orphan_confirmation_and_cancel_stay_local(self):
        for message in ['Check this interpreted request','Cancel interpreted request']:
            with patch.object(conversation.chat_interpreter,'interpret') as model,patch.object(core,'tool_propose_revision') as tool:
                r=conversation.answer(message,self.dossier,session_id=self.sid,store_dir=self.root,use_llm_interpretation=True)
                model.assert_not_called();tool.assert_not_called()
                self.assertIn('no pending',r['answer'])

    def test_colloquial_removal_constraints_not_discarded(self):
        for message in ['Do not take Synthetic fixture 0 off my plan','Please leave Synthetic fixture 0 out at dinner','Take Synthetic fixture 0 off my plan and keep lunch','Keep lunch unchanged; take Synthetic fixture 0 off my plan']:
            with self.subTest(message=message),patch.object(core,'tool_propose_revision') as tool:
                r=self.ask(message)
                tool.assert_not_called()
                self.assertNotIn('Check this interpreted request',r.get('clarification_options',[]))

if __name__=='__main__':
    unittest.main()
