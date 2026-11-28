"""Versioned numerical checker: 1e-6 absolute tolerance in nutrient units."""
import numpy as np
from verify_stage2_constraint_audit_repaired_v1 import check as historical_check
def check(values,lower,upper,keys,baseline,metadata,cvd):
    v=np.asarray(values,dtype=float)
    if v.shape != (39,) or not np.isfinite(v).all():
        raise ValueError('Nonfinite or invalid nutrient vector')
    result=historical_check(v,lower,upper,keys,baseline,metadata,cvd)
    n=dict(zip(keys,v)); w=float(metadata['weight_kg']); tol=1e-6
    result['filter2_ok']=bool(n['DR1TKCAL']>=1200-tol and n['DR1TSODI']<=(1500 if cvd>=.5 else 2300)+tol and n['DR1TPOTA']>=(3500 if cvd>=.5 else 2600)-tol and .8*w-tol<=n['DR1TPROT']<=2*w+tol)
    result['joint']=bool(not result['bound_names'] and result['energy_ok'] and result['ratios_ok'] and result['filter2_ok'])
    return result
