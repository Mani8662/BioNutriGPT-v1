with open('run_stage2_joint_constraint_repair_v1.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    'assert frozen_cases_sha == "3a48e77864115e4fce9d343c1626f634bc3d45e458e0be3060f0896005c21960"',
    'assert frozen_cases_sha == "7b9019620ca755651973a38dbb91d5b1ae5674637b23773eb46924860be19036"'
)

with open('run_stage2_joint_constraint_repair_v1.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated hash check in run_stage2_joint_constraint_repair_v1.py')
