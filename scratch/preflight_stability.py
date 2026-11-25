import sys
import torch
import json
from pathlib import Path

ROOT = Path('.')
print("Python Executable:", sys.executable)
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Count:", torch.cuda.device_count())

dev = json.loads((ROOT / 'Saved Model/stage2_ablation_v2/development_participants_LOCAL.json').read_text(encoding='utf-8'))
ev = json.loads((ROOT / 'Saved Model/stage2_ablation_v2/evaluation_participants_LOCAL.json').read_text(encoding='utf-8'))

dev_seqns = [c['SEQN'] for c in dev]
ev_seqns = [c['SEQN'] for c in ev]

print("Dev Cohort Count:", len(dev), "Unique SEQN:", len(set(dev_seqns)))
print("Ev Cohort Count:", len(ev), "Unique SEQN:", len(set(ev_seqns)))
overlap = set(dev_seqns) & set(ev_seqns)
print("Cohort Overlap:", overlap)
assert len(dev) == 3 and len(set(dev_seqns)) == 3
assert len(ev) == 100 and len(set(ev_seqns)) == 100
assert len(overlap) == 0
print("Preflight assertions passed.")
