"""Retrospective all-rule failure and margin audit; no solver/model calls."""
import collections, json, math
from pathlib import Path
import stage3_robustness_v3 as r

OUT=r.ROOT/'Saved Model/stage3_robustness_margins_v1'
def evaluate(basket,t,foods):
    rules={}; totals={}
    def bound(name,value,lo,hi,scale,tol=1e-4):
        for side,b in [('lower',lo),('upper',hi)]:
            if b is None or not math.isfinite(b): continue
            margin=None if value is None else value-b if side=='lower' else b-value
            rules[name+':'+side]={'value':value,'bound':b,'margin':margin,'normalized_margin':None if margin is None else margin/scale,'scale':scale,'tolerance':tol,'violated':margin is None or margin < -tol}
    ids=[x['food_id'] for x in basket]; counts=collections.Counter(); servings=collections.Counter()
    bound('nonempty',len(basket),1,None,1,0)
    bound('dish_count',len(set(ids)),r.legacy.s3.MIN_ACTIVE_DISHES,r.legacy.s3.MAX_ACTIVE_DISHES,1,0)
    bound('duplicates',len(ids)-len(set(ids)),None,0,1,0)
    bound('exclusions',sum(x in r.excluded(t) for x in ids),None,0,1,0)
    for item in basket:
        f=foods[item['food_id']]; q=item['quantity']
        bound('quantity_'+item['food_id'],q,1,r.legacy.s3.MAX_SERVINGS_PER_DISH,1,0)
        bound('integer_'+item['food_id'],abs(q-round(q)),None,0,1,0)
        counts[f['category']]+=1; servings[f['category']]+=q
    bound('diet_mode',sum(foods[i]['dietary_type']=='non_vegetarian' for i in ids),None,0 if t[3]=='pure_vegetarian' else None,1,0)
    cats,serv=r.legacy.s3._category_rules_for_mode(t[3],allow_zero_dairy=bool(t[5].get('allow_zero_dairy',False)))
    for k,(lo,hi) in cats.items(): bound('category_'+k,counts[k],lo,hi,1,0)
    for k,(lo,hi) in serv.items(): bound('servings_'+k,servings[k],lo,hi,1,0)
    for code in r.legacy.s3.NUTRIENT_CODES:
        vals=[None if code in foods[x['food_id']].get('unknown_nutrients',[]) else foods[x['food_id']]['nutrients_per_serving'].get(code) for x in basket]
        totals[code]=None if any(v is None for v in vals) else sum(v*x['quantity'] for v,x in zip(vals,basket))
    env={x['nutrient']:x for x in t[6]['nutrient_constraint_envelope']}
    for code,x in env.items():
        if x.get('rule_provenance',{}).get('planning_role') in ('report_only','exposure_reporting'):continue
        bound('nutrient_'+code,totals[code],x['min_bound'],x['max_bound'],r.legacy.tas.DAILY_REFERENCE_SCALES[code])
    energy=totals['DR1TKCAL']; target=t[6]['energy_prescription']['target_calories']
    bound('energy',energy,.95*target,1.05*target,target,target*1e-6)
    for code,label,lo,hi,mult in [('DR1TPROT','protein_pct',10,35,4),('DR1TCARB','carb_pct',45,65,4),('DR1TTFAT','fat_pct',20,35,9)]:
        value=None if not energy or totals[code] is None else 100*mult*totals[code]/energy
        bound(label,value,lo,hi,100)
    na,k=totals['DR1TSODI'],totals['DR1TPOTA']
    baseline=env['DR1TSODI']['baseline']/(env['DR1TPOTA']['baseline']+1)
    bound('sodium_potassium',None if na is None or k is None else na/(k+1),None,baseline,1)
    fiber=totals['DR1TFIBE']
    bound('fiber_density',None if not energy or fiber is None else fiber*1000/energy,14,None,14)
    violations=[k for k,x in rules.items() if x['violated']]
    return {'valid':not violations,'violations':violations,'rules':rules,'totals':totals}

def main():
    r.check_manifest(); OUT.mkdir(exist_ok=True)
    base=r.DIR/'full/checkpoints'; ts=r.tasks(); foods=r.catalog(); rows=[]
    failures=collections.defaultdict(collections.Counter); mismatches=[]; summary={}; bindings={}
    nominal_tasks={(t[0],t[3]):t for t in ts if t[4]=='nominal_baseline'}
    for t in ts:
        f=r.perturbed_foods(t) if t[5]['replay_evaluation_only'] else foods
        for variant in ['a','b']:
            p=r.path_for(base,t,variant); saved=r.read(p); bindings[str(p.relative_to(r.ROOT))]=r.sha(p)
            source=saved
            if t[5]['replay_evaluation_only']:
                source=r.read(r.path_for(base,nominal_tasks[t[0],t[3]],variant))
            available=bool(source['basket'])
            result=evaluate(source['basket'],t,f) if available else None
            observed=bool(result and result['valid'])
            if observed!=saved['accepted']:mismatches.append({'identity':r.identity(t),'variant':variant,'computed':observed,'saved':saved['accepted']})
            key=t[4]+'/'+variant
            if t[5]['solver_required'] and not saved['accepted']:
                failures[key][str((saved.get('failure') or {}).get('code','unspecified'))]+=1
            rows.append({'identity':r.identity(t),'variant':variant,'available':available,'audit':result})
    r.require(not mismatches,'All-rule audit disagrees: '+str(mismatches[:5]))
    for sid in sorted({x['identity']['scenario'] for x in rows}):
        summary[sid]={}
        for variant in ['a','b']:
            sub=[x for x in rows if x['identity']['scenario']==sid and x['variant']==variant]
            counter=collections.Counter(rule for x in sub if x['audit'] for rule in x['audit']['violations'])
            summary[sid][variant]={'records':len(sub),'plans_available':sum(x['available'] for x in sub),'valid':sum(bool(x['audit'] and x['audit']['valid']) for x in sub),'violations_all_rules':dict(counter.most_common()),'solver_failure_codes':dict(failures[sid+'/'+variant])}
    nominal={(x['identity']['case'],x['identity']['mode'],x['variant']):x for x in rows if x['identity']['scenario']=='nominal_baseline'}
    keys=sorted(set(k for x in nominal.values() if x['audit'] for k in x['audit']['rules'] if not k.startswith(('quantity_','integer_'))))
    margins={}
    for rule in keys:
        by_person=collections.defaultdict(list); aa=collections.defaultdict(list);bb=collections.defaultdict(list)
        for i in range(100):
            for mode in r.MODES:
                a,b=nominal[i,mode,'a']['audit'],nominal[i,mode,'b']['audit']
                if not(a and b and a['valid'] and b['valid']):continue
                if rule not in a['rules'] or rule not in b['rules']:continue
                x,y=a['rules'][rule]['normalized_margin'],b['rules'][rule]['normalized_margin']
                if x is None or y is None:continue
                by_person[i].append(y-x);aa[i].append(x);bb[i].append(y)
        if by_person:
            margins[rule]={'a_mean':sum(sum(v)/len(v) for v in aa.values())/len(aa),'b_mean':sum(sum(v)/len(v) for v in bb.values())/len(bb),'b_minus_a':r.bootstrap([sum(v)/len(v) for v in by_person.values()]),'matched_cells':sum(map(len,by_person.values()))}
    # Positive and negative controls for multiple simultaneous violations.
    t=next(t for t in ts if t[0]==0 and t[3]=='pure_vegetarian' and t[4]=='nominal_baseline')
    basket=r.read(r.path_for(base,t,'a'))['basket']; control=evaluate(basket,t,foods)
    r.require(control['valid'],'Clean control failed')
    import copy
    altered=copy.deepcopy(t); altered[6]['energy_prescription']['target_calories']=100
    for env in altered[6]['nutrient_constraint_envelope']:
        if env['nutrient'] in ['DR1TPROT','DR1TSODI']:env['max_bound']=0
    bad=evaluate(basket,altered,foods)
    r.require({'energy:upper','nutrient_DR1TPROT:upper','nutrient_DR1TSODI:upper'}.issubset(bad['violations']),'Failed simultaneous violations test')
    r.write(OUT/'all_rule_records.json',rows)
    r.write(OUT/'summary.json',summary);r.write(OUT/'nominal_margin_comparisons.json',margins)
    r.write(OUT/'input_manifest.json',bindings)
    for name,h in bindings.items():r.require(r.sha(r.ROOT/name)==h,'Input changed')
    r.write(OUT/'verification.json',{'status':'passed','records':len(rows),'classification_disagreements':len(mismatches),'simultaneous_violation_test':'passed','input_hashes_unchanged':True,'solver_calls':0,'scope':'New all-rule reconstruction; shared category policy and bootstrap helper. Exploratory paired margin intervals unadjusted.','script_sha256':r.sha(__file__)})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
