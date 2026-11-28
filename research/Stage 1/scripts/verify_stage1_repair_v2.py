"""Focused label, split and preprocessing checks for the isolated Stage 1 repair."""
import json
import unittest
import numpy as np
import pandas as pd
from prepare_stage1_repair_v2 import ROOT, PROTOCOL, harmonized_labels, make_splits, load

P=json.loads(PROTOCOL.read_text())


class RepairTests(unittest.TestCase):
    def example(self, **changes):
        row={'LBXGH':5.5,'DIQ010':2,'LBXGLU':90,'BMXBMI':24,
             **{c:2 for c in P['cvd_primary']['components']}}
        row.update(changes)
        return pd.DataFrame([row])

    def test_unobserved_negative_is_unknown(self):
        labels,_=harmonized_labels(self.example(DIQ010=9,MCQ160B=np.nan),P)
        self.assertTrue(labels[['TARGET_DIABETES','TARGET_CVD']].isna().all().all())

    def test_observed_positive_overrides_missing_other_components(self):
        labels,_=harmonized_labels(self.example(DIQ010=9,LBXGH=6.5,MCQ160B=1,MCQ160C=np.nan),P)
        self.assertEqual(labels.TARGET_DIABETES.iloc[0],1)
        self.assertEqual(labels.TARGET_CVD.iloc[0],1)

    def test_diagnosed_controlled_diabetes_remains_positive(self):
        labels,_=harmonized_labels(self.example(DIQ010=1,LBXGH=5),P)
        self.assertEqual(labels.TARGET_DIABETES.iloc[0],1)

    def test_borderline_is_explicit_sensitivity(self):
        labels,s=harmonized_labels(self.example(DIQ010=3),P)
        self.assertEqual(labels.TARGET_DIABETES.iloc[0],0)
        self.assertTrue(pd.isna(s.diabetes_exclude_borderline_negatives.iloc[0]))

    def test_glucose_does_not_change_primary(self):
        labels,s=harmonized_labels(self.example(LBXGLU=160),P)
        self.assertEqual(labels.TARGET_DIABETES.iloc[0],0)
        self.assertEqual(s.diabetes_add_available_glucose.iloc[0],1)

    def test_invalid_lab_and_bmi_not_assigned_negative(self):
        labels,_=harmonized_labels(self.example(LBXGH=999,BMXBMI=0),P)
        self.assertTrue(pd.isna(labels.TARGET_DIABETES.iloc[0]))
        self.assertTrue(pd.isna(labels.TARGET_OBESITY.iloc[0]))

    def test_labels_ignore_legacy_targets(self):
        df=self.example();a,_=harmonized_labels(df,P)
        for c in a:df[c]=1-a[c]
        b,_=harmonized_labels(df,P)
        pd.testing.assert_frame_equal(a,b)

    def test_split_repeatability_disjointness_and_coverage(self):
        df=pd.DataFrame([(i%2,(i//2)%2,(i//4)%2) for i in range(800)],columns=['d','o','c'])
        a=make_splits(df,list(df),P);b=make_splits(df,list(df),P)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(len(set(np.concatenate(list(a.values())))),800)
        self.assertEqual(sum(map(len,a.values())),800)


if __name__=='__main__':
    unittest.main(verbosity=2)
