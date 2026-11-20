import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parent/'backend'))
import food_groups

class Groups(unittest.TestCase):
    def test_equivalent_exclusions(self):
        for message in ['remove fish','no fish','fish-free','I dont want to any fish on my meal plan',"I don't want any fish in my plan"]:
            self.assertEqual(food_groups.exclusion_request(message),'fish')
    def test_negation_and_named_food_are_not_group_exclusions(self):
        for message in ["Don't remove fish",'remove tuna','remove fish but keep salmon']:
            self.assertIsNone(food_groups.exclusion_request(message))
    def test_groups_are_catalog_data(self):
        import stage3_milp_solver as s3
        catalog=s3.load_food_database()
        ids=food_groups.excluded_ids(['fish'],catalog)
        self.assertEqual(len(ids),4)
        self.assertIn('recipe_light_tuna',ids)
        self.assertIn('recipe_pink_salmon_canned',ids)
        self.assertNotIn('recipe_boiled_egg',ids)

if __name__=='__main__':unittest.main()
