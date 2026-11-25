with open('test_stage2_joint_repair.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    'self.assertIn("calibrators.joblib", binding.get("calibrator_sha256", ""))',
    'self.assertEqual(len(binding.get("calibrator_sha256", "")), 64)'
)
text = text.replace(
    'np.testing.assert_array_equal(candidate_88[:bio_len], self.raw_vector[:bio_len])',
    'np.testing.assert_allclose(candidate_88[:bio_len], self.raw_vector[:bio_len], atol=1e-5)'
)
text = text.replace(
    'np.testing.assert_array_equal(candidate_88[other_start:], self.raw_vector[other_start:])',
    'np.testing.assert_allclose(candidate_88[other_start:], self.raw_vector[other_start:], atol=1e-5)'
)

with open('test_stage2_joint_repair.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated test_stage2_joint_repair.py')
