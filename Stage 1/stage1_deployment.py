"""Versioned Stage 1 release binding shared by screening and counterfactual scoring."""
from pathlib import Path
import hashlib
import json
import os
import joblib
import torch

ROOT = Path(__file__).resolve().parent
RELEASE = os.environ.get('BIONUTRIGPT_STAGE1_RELEASE', 'repaired_v1')
if RELEASE not in ('repaired_v1', 'legacy'):
    raise ValueError('BIONUTRIGPT_STAGE1_RELEASE must be repaired_v1 or legacy')
REPAIRED = RELEASE == 'repaired_v1'
BUNDLE = ROOT / 'saved_model/deployment/repaired_stage1_v1'
CHECKPOINT = BUNDLE/'model.pt' if REPAIRED else ROOT/'saved_model/bionutrigpt_stage1_best.pt'
PREPROCESSOR = BUNDLE/'preprocessing_artifacts.joblib' if REPAIRED else ROOT/'data/preprocessing_artifacts.joblib'
THRESHOLDS = dict.fromkeys(('diabetes','obesity','cvd'), .5) if REPAIRED else dict(diabetes=.617,obesity=.445,cvd=.603)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify_bundle(checkpoint=CHECKPOINT, preprocessor=PREPROCESSOR):
    if not REPAIRED:
        return {}
    if Path(checkpoint).resolve()!=CHECKPOINT.resolve() or Path(preprocessor).resolve()!=PREPROCESSOR.resolve():
        raise ValueError('Repaired release requires its complete bound bundle; individual artifact overrides are forbidden.')
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    from train_bionutrigpt_stage1 import FEATURE_COLS
    if manifest['feature_order']!=FEATURE_COLS or manifest['thresholds']!=THRESHOLDS:
        raise ValueError('Release feature order or threshold policy mismatch')
    for name,digest in manifest['hashes'].items():
        if sha(BUNDLE/name)!=digest:
            raise ValueError(f'Release artifact hash mismatch: {name}')
    return manifest

class CalibratedStage1(torch.nn.Module):
    """Expose calibrated logits while preserving the model's attention interface."""
    def __init__(self, base, calibrators):
        super().__init__()
        self.base=base
        self.register_buffer('slopes',torch.tensor([c.coef_[0,0] for c in calibrators],dtype=torch.float64))
        self.register_buffer('intercepts',torch.tensor([c.intercept_[0] for c in calibrators],dtype=torch.float64))

    def forward(self, x):
        logits,aux,attention=self.base(x)
        # Match the research evaluator: float32 sigmoid, float64 clip/logit/Platt.
        calibrated={}
        for i,key in enumerate(('diabetes','obesity','cvd')):
            p=torch.sigmoid(logits[key]).double().clamp(1e-7,1-1e-7)
            calibrated[key]=torch.logit(p)*self.slopes[i]+self.intercepts[i]
        return calibrated,aux,attention

def wrap_model(model,device):
    verify_bundle()
    return CalibratedStage1(model,joblib.load(BUNDLE/'calibrators.joblib')).to(device).eval()

def metadata():
    return {
        'model_version': 'BioNutriGPT-repaired-v1' if REPAIRED else 'BioNutriGPT-legacy',
        'calibration_status': 'reserved_NHANES_calibration_partition' if REPAIRED else 'uncalibrated_legacy',
        'intended_population': 'US NHANES adults age 20+; prevalent research phenotypes, not diagnosis or future disease risk. Indian external validation is not established.',
        'threshold_policy': 'Fixed 0.5 descriptive research routing; not a clinically selected or validated intervention threshold.' if REPAIRED else 'Legacy thresholds',
        'release': RELEASE,
        'attention_scope': 'Attention weights are not causal feature explanations.',
    }
