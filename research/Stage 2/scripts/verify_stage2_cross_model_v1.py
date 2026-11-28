"""Reproduce inference and independently reconstruct cross-model summary intervals."""
import copy,gc
import numpy as np
import stage2_cross_model_v1 as r

def statistics(rows,probs,saved):
    variants=sorted({x['variant'] for x in rows}-{'baseline'});count=0
    def interval(values,entry):
        nonlocal count
        a=np.asarray(values,float);r.need(entry['n']==len(a),'Wrong denominator')
        if not len(a):r.need(entry['mean'] is None and entry['ci95'] is None,'Missing values imputed');return
        rng=np.random.default_rng(20261124);resamples=rng.integers(0,len(a),size=(10000,len(a)))
        ci=np.quantile(a[resamples].mean(axis=1),[.025,.975])
        r.need(np.isclose(entry['mean'],sum(a)/len(a),rtol=0,atol=1e-12),'Mean mismatch')
        r.need(np.allclose(entry['ci95'],ci,rtol=0,atol=1e-12),'CI mismatch');count+=1
    for name,p in probs.items():
        by_variant={}
        for i,v in enumerate(variants):
            d=np.array([p[100+i*100+j]-p[j] for j in range(100)]);by_variant[v]=d
            for label in ['all','numerically_valid','source_accepted']:
                ids=[j for j in range(100) if label=='all' or rows[100+i*100+j]['numerical_pass' if label=='numerically_valid' else 'source_accepted']]
                sub=d[ids];entry=saved[name][v][label]
                r.need(entry['n']==len(ids),'Stratum denominator mismatch')
                r.need(entry['gate_pass']==sum(all(x<=.0015 for x in delta) for delta in sub),'Gate count mismatch')
                for k,disease in enumerate(r.DISEASES):
                    e=entry['outcomes'][disease];interval(sub[:,k],e)
                    r.need(e['decrease_count']==sum(sub[:,k]<-1e-6) and e['increase_count']==sum(sub[:,k]>1e-6),'Direction counts mismatch')
        diff=by_variant['variant_0_reference']-by_variant['variant_1_no_model_objective']
        for j,disease in enumerate(r.DISEASES):interval(diff[:,j],saved[name]['full_minus_no_model_objective'][disease])
    return count
def main():
    r.check();done=r.read(r.OUT/'completed.json')
    for name,h in done['output_hashes'].items():r.need(r.sha(r.OUT/name)==h,'Output changed: '+name)
    rows,x,_=r.build_inputs();r.need(rows==r.read(r.OUT/'rows_LOCAL.json'),'Row identity mismatch')
    r.need(np.array_equal(x,np.load(r.OUT/'normalized_inputs_LOCAL.npy')),'Feature construction changed')
    probs={};errors={}
    for name in r.MODELS:
        archive=np.load(r.OUT/(name+'_predictions_LOCAL.npz'));model,cal,device=r.load_model(name)
        raw,p=r.score(name,model,cal,x,device)
        r.need(np.allclose(raw,archive['raw'],rtol=0,atol=1e-6) and np.allclose(p,archive['calibrated'],rtol=0,atol=1e-6),'Prediction reproduction failed: '+name)
        probs[name]=archive['calibrated'];errors[name]=float(abs(p-probs[name]).max());del model;gc.collect()
        if r.torch.cuda.is_available():r.torch.cuda.empty_cache()
    saved=r.read(r.OUT/'analysis.json');n=statistics(rows,probs,saved)
    corrupted=copy.deepcopy(saved);e=corrupted[r.MODELS[0]]['variant_0_reference']['all']['outcomes']['diabetes'];e['ci95']=[e['mean']-1,e['mean']+1]
    try:statistics(rows,probs,corrupted)
    except r.ValidationError:pass
    else:raise AssertionError('Corrupt interval accepted')
    r.check();r.write(r.OUT/'verification.json',{'status':'passed','participants':100,'proposals':600,'models':3,'summary_intervals_reproduced':n,'prediction_max_errors':errors,'changed_ci_rejected':True,'analysis_sha256':r.sha(r.OUT/'analysis.json'),'scope':'Repeated frozen inference and separate statistical formulas; shared preprocessing/model loaders. No independent training cohort or clinical intervention validation.'})
    print('Verification passed',n,'intervals')
if __name__=='__main__':main()
