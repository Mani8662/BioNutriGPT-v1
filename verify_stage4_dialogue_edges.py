"""Adversarial routing regressions; synthetic fixtures, no LLM or clinical claims."""
import unittest
from unittest.mock import patch
from verify_stage4_conversation_v2 import ConversationChecks
import conversation_service as chat
import chat_service as core
import plan_store


class DialogueEdges(unittest.TestCase):
    setUp = ConversationChecks.setUp
    ask = ConversationChecks.ask

    def test_meal_modifiers_are_not_silently_dropped(self):
        for modifier in ['without salt', 'only if it costs less', 'with no oil']:
            with self.subTest(modifier=modifier), patch.object(core, 'tool_propose_revision') as solver:
                reply = self.ask('Include 2 servings of Synthetic fixture 0 at Dinner '+modifier)
                self.assertNotIn('Check this interpreted request', reply.get('clarification_options', []))
                self.ask('Check this interpreted request')
                solver.assert_not_called()

    def test_ambiguous_food_choice_preserves_quantity(self):
        with patch.object(core, 'tool_propose_revision', return_value=(None,None,{})) as solver:
            self.ask('Include 3 servings of Synthetic fixture at Dinner')
            self.ask('Synthetic fixture 0')
            self.ask('Check this interpreted request')
            self.assertEqual(solver.call_args.args[3]['locked_food_quantities'], {'0':3})

    def test_decimal_servings_are_not_read_as_integers(self):
        for amount in ['1.5', '-2', '0.2']:
            with self.subTest(amount=amount), patch.object(core, 'tool_propose_revision') as solver:
                result=self.ask('Include '+amount+' servings of Synthetic fixture 0 at Dinner')
                self.assertNotIn('Check this interpreted request', result.get('clarification_options', []))
                solver.assert_not_called()

    def test_cancel_compound_then_confirm_does_nothing(self):
        with patch.object(core, 'tool_propose_revision') as solver:
            self.ask('Include one serving of Synthetic fixture 0 at Lunch; include two servings of Synthetic fixture 4 at Dinner')
            self.ask('Cancel interpreted request')
            self.ask('Check this interpreted request')
            solver.assert_not_called()

    def test_question_invalidates_pending_confirmation(self):
        with patch.object(core, 'tool_propose_revision') as solver:
            self.ask('Include 2 servings of Synthetic fixture 0 at Lunch')
            self.ask('How much protein in my plan?')
            self.ask('Check this interpreted request')
            solver.assert_not_called()

    def test_version_change_invalidates_compound_confirmation(self):
        self.ask('Include 1 serving of Synthetic fixture 0 at Lunch; include 2 servings of Synthetic fixture 4 at Dinner')
        record=plan_store.get_session(self.sid,store_dir=self.root)
        record['conversation']['plan_version']=0
        plan_store.save_session(record,store_dir=self.root)
        with patch.object(core,'tool_propose_revision') as solver:
            self.ask('Check this interpreted request')
            solver.assert_not_called()

    def test_nutrient_followup_remains_read_only(self):
        with patch.object(core,'tool_propose_revision') as solver:
            self.ask('How much protein in Synthetic fixture 0?')
            reply=self.ask('And its calories per serving?')
            self.assertIn('200.00 kcal',reply['answer'])
            solver.assert_not_called()


if __name__ == '__main__':
    unittest.main()
