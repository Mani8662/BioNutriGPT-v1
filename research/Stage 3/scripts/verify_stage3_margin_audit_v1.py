"""Reproduce all-rule summaries and paired margin intervals from exported rule rows."""
import collections,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Saved Model/stage3_robustness_margins_v1'
def read(p):return json.loads(p.read_text())
def main():
 rows=read(OUT/'all_rule_records.json'); summary=read(OUT/'summary.json'); margins=read(OUT/'nominal_margin_comparisons.json')
 assert len(rows)==4200
 for s,variants in summary.items():
  for v,saved in variants.items():
   sub=[x for x in rows if x['identity']['scenario']==s and x['variant']==v]
   counts=collections.Counter()
   for x in sub:
    if x['audit']:
     failed=[k for k,z in x['audit']['rules'].items() if z['margin'] is None or z['margin'] < -z['tolerance']]
     assert failed==x['audit']['violations']
     assert (not failed)==x['audit']['valid']
     counts.update(failed)
   assert dict(counts)==saved['violations_all_rules']
   assert sum(bool(x['audit'] and x['audit']['valid']) for x in sub)==saved['valid']
 nominal={(x['identity']['case'],x['identity']['mode'],x['variant']):x for x in rows if x['identity']['scenario']=='nominal_baseline'}
 modes=sorted({k[1] for k in nominal}); checked=0
 for rule,saved in margins.items():
  people=[]; av=[];bv=[];cells=0
  for i in range(100):
   pairs=[]
   for mode in modes:
    a,b=nominal[i,mode,'a']['audit'],nominal[i,mode,'b']['audit']
    if not(a and b and a['valid'] and b['valid']):continue
    if rule not in a['rules'] or rule not in b['rules']:continue
    x,y=a['rules'][rule],b['rules'][rule]
    if x['margin'] is None or y['margin'] is None:continue
    pairs.append((x['margin']/x['scale'],y['margin']/y['scale']));cells+=1
   if pairs:
    pa,pb=np.asarray(pairs).mean(axis=0);av.append(pa);bv.append(pb);people.append(float(np.mean([y-x for x,y in pairs])))
  assert cells==saved['matched_cells']
  assert np.isclose(np.mean(av),saved['a_mean'],rtol=1e-12,atol=1e-12)
  assert np.isclose(np.mean(bv),saved['b_mean'],rtol=1e-12,atol=1e-12)
  rng=np.random.default_rng(20261124); a=np.asarray(people)
  ci=np.percentile(a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1),[2.5,97.5])
  assert np.allclose(ci,saved['b_minus_a']['ci95'],rtol=0,atol=1e-12), (rule, ci, saved['b_minus_a']['ci95'])
  assert np.isclose(a.mean(),saved['b_minus_a']['mean'],rtol=0,atol=1e-12);checked+=1
 for name,h in read(OUT/'input_manifest.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 receipt={'status':'passed','records':len(rows),'paired_margin_intervals_reproduced':checked,'input_checkpoint_hashes_verified':4200,'scope':'Independent aggregation/interval formulas from exported rule rows; does not independently reimplement all clinical-policy rules.'}
 (OUT/'statistics_verification.json').write_text(json.dumps(receipt,indent=2)); print(receipt)
if __name__=='__main__':main()
