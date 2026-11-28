"""One actual Stage 1→2→3→4 orchestration check, with isolated output files."""
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'backend'))
import pipeline_service as ps
from schemas import FullPipelineResponse


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU sample requested; CUDA is unavailable.')
    old_path = os.environ.get('BIONUTRIGPT_STAGE2_JSON')
    os.environ['BIONUTRIGPT_STAGE2_JSON'] = str(ROOT / 'Saved Model/stage3_live_sample_stage2.json')
    started = time.perf_counter()
    try:
        result = ps.run_full_pipeline(ps.SAMPLE_PATIENT_VALUES, ps.SAMPLE_METADATA, backend='transformers')
        FullPipelineResponse(**result)
        report = {'device': torch.cuda.get_device_name(0), 'actual_model_execution': True,
                  'elapsed_seconds': time.perf_counter() - started,
                  'pipeline_status': result['pipeline_status'], 'pipeline_failure': result['pipeline_failure'],
                  'stage3_accepted': (result['meal_plan'] or {}).get('accepted', False),
                  'stage4_generated': result['clinical_report'] is not None,
                  'screening_checks_passed': result['screening']['certified_for_stage3'],
                  'stage3_failure': (result['meal_plan'] or {}).get('failure'),
                  'latency': result['pipeline_latency']}
        assert report['stage3_accepted'] or not report['stage4_generated']
        (ROOT / 'Saved Model/stage3_production_live_sample.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if old_path is None:
            os.environ.pop('BIONUTRIGPT_STAGE2_JSON', None)
        else:
            os.environ['BIONUTRIGPT_STAGE2_JSON'] = old_path


if __name__ == '__main__':
    main()
