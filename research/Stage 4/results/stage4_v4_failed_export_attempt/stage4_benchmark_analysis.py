"""Version-independent benchmark accounting and conditional uncertainty estimates."""
from collections import Counter
import math
import random
import statistics


REVISION_TYPES = {'swap', 'dish_cap', 'locked_swap'}


def revision_accounting(scenarios):
    turns=[t for s in scenarios for t in s['turns'] if t['type'] in REVISION_TYPES]
    return {
        'revision_turns_total':len(turns),
        'oracle_feasible':sum(t['oracle_status']=='feasible' for t in turns),
        'oracle_proven_infeasible':sum(t['oracle_status']=='proven_infeasible' for t in turns),
        'oracle_timeout_unresolved':sum(t['oracle_status']=='timeout_unresolved' for t in turns),
        'validated_swap_proposals':sum(t['outcome_category']=='validated_swap_proposal' for t in turns),
        'valid_proven_infeasible_refusals':sum(t['outcome_category']=='valid_proven_infeasible_refusal' for t in turns),
        'timeout_no_proposal':sum(t['outcome_category'].startswith('timeout_') for t in turns),
        'missed_feasible_due_to_timeout':sum(t['outcome_category']=='timeout_missed_feasible' for t in turns),
        'missed_feasible_other':sum(t['oracle_status']=='feasible' and t['proposed_basket'] is None and not t['outcome_category'].startswith('timeout_') for t in turns),
        'invalid_proposals':sum(t['outcome_category']=='invalid_proposal' for t in turns),
        'malformed_json_no_proposal':sum(t['outcome_category']=='malformed_json_no_proposal' for t in turns),
        'unexplained_null_on_proven_infeasible':sum(t['oracle_status']=='proven_infeasible' and t['outcome_category']=='unexplained_null_on_infeasible' for t in turns),
        'invalid_proposals_on_feasible':sum(t['oracle_status']=='feasible' and t['outcome_category']=='invalid_proposal' for t in turns),
    }


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;den=1+z*z/n;p=k/n
    center=(p+z*z/(2*n))/den
    margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-margin),min(1,center+margin)]


def quantile(values,p):
    values=sorted(values);x=(len(values)-1)*p;a=int(x);b=min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(x-a)


def protocol_uncertainty(scenarios, draws=1000, seed=42):
    if len({s['lasi_prim_key'] for s in scenarios})!=len(scenarios):
        raise ValueError('Participant reuse requires participant-level clustering across scenarios.')
    rng=random.Random(seed);n=len(scenarios)
    vals=[]
    for _ in range(draws):
        sampled=[scenarios[rng.randrange(n)] for _ in range(n)]
        ts=[t for s in sampled for t in s['turns']]
        vals.append(sum(t['strict_protocol_passed'] for t in ts)/len(ts))
    allpass=sum(all(t['strict_protocol_passed'] for t in s['turns']) for s in scenarios)
    return {'bootstrap_draws':draws,'seed':seed,'resampling_unit':'participant/scenario',
            'turn_weighted_protocol_rate_cluster_bootstrap_95ci':[quantile(vals,.025),quantile(vals,.975)],
            'all_turns_passed_scenarios':allpass,'scenario_count':n,
            'all_turns_passed_scenario_wilson_95ci':wilson(allpass,n),
            'scope':'Exploratory uncertainty conditional on selected scripted cases, not design-weighted national inference. All-success bootstrap intervals can degenerate; Wilson bounds assume independent scenarios.'}


def trajectories(scenarios):
    records={}
    for s in scenarios:
        basket=s['initial_v1_basket'];excludes=set();locks={};cap=None
        for t in s['turns']:
            state=(tuple(sorted((r['food_id'],int(r['quantity'])) for r in basket)),tuple(sorted(excludes)),tuple(sorted(locks.items())),cap)
            records[(s['scenario_id'],t['turn'])]={'scenario_id':s['scenario_id'],'state':state,'delta':t['serving_delta']}
            if t['outcome_category']=='validated_swap_proposal':
                previous={r['food_id'] for r in basket}
                basket=t['proposed_basket']
                # Exact active restrictions are logged in v4; v3 reconstruction
                # conservatively compares baskets and prior request histories below.
            if t['type']=='rollback_check' and t['strict_protocol_passed']:
                basket=s['initial_v1_basket'];excludes=set();locks={};cap=None
    return records


def paired_ablation(full, other, draws=1000, seed=42):
    a=trajectories(full);b=trajectories(other)
    initial_a={s['scenario_id']:s['initial_v1_basket'] for s in full}
    initial_b={s['scenario_id']:s['initial_v1_basket'] for s in other}
    fa={s['scenario_id']:s for s in full};fb={s['scenario_id']:s for s in other}
    pairs=[]
    for key in sorted(a.keys() & b.keys()):
        x,y=a[key],b[key]
        if x['delta'] is None or y['delta'] is None:continue
        sid,turn=key
        # Matching frozen request sequence is necessary as well as basket equality.
        same_requests=[t['message'] for t in fa[sid]['turns'][:turn]]==[t['message'] for t in fb[sid]['turns'][:turn]]
        same_basket=x['state'][0]==y['state'][0]
        ta=fa[sid]['turns'][turn-1];tb=fb[sid]['turns'][turn-1]
        if 'input_state' in ta and 'input_state' in tb:
            same=ta['input_state']==tb['input_state']
        else:same=same_basket and same_requests
        pairs.append({'scenario_id':sid,'full':x['delta'],'no_penalty':y['delta'],'same_input_state':same})
    def summarize(rows):
        if not rows:return {'paired_turns':0}
        m=statistics.mean(x['full'] for x in rows);n=statistics.mean(x['no_penalty'] for x in rows)
        groups={sid:[r for r in rows if r['scenario_id']==sid] for sid in sorted({r['scenario_id'] for r in rows})}
        keys=list(groups);rng=random.Random(seed);diffs=[]
        for _ in range(draws):
            sample=[r for _ in keys for r in groups[keys[rng.randrange(len(keys))]]]
            diffs.append(statistics.mean(r['no_penalty']-r['full'] for r in sample))
        return {'paired_turns':len(rows),'scenarios':len(keys),'full_mean':m,'no_penalty_mean':n,
                'reduction_pct':100*(1-m/n) if n else None,
                'mean_difference_no_penalty_minus_full':n-m,
                'cluster_bootstrap_difference_95ci':[quantile(diffs,.025),quantile(diffs,.975)],
                'bootstrap_draws':draws,'seed':seed}
    return {'trajectory_pairs':summarize(pairs),
            'same_input_state_pairs':summarize([r for r in pairs if r['same_input_state']]),
            'different_input_state_pairs':sum(not r['same_input_state'] for r in pairs),
            'scope':'Successful-pair conditional descriptive analysis. v3 same-state reconstruction uses preceding basket and frozen history; v4 compares logged full input states. Not a randomized clinical effect.'}


def paired_protocol_difference(a,b,draws=1000,seed=42):
    aa={s['scenario_id']:s for s in a};bb={s['scenario_id']:s for s in b}
    if aa.keys()!=bb.keys():raise ValueError('Matched scenario IDs required.')
    keys=sorted(aa);rng=random.Random(seed);values=[]
    for _ in range(draws):
        sample=[keys[rng.randrange(len(keys))] for _ in keys]
        at=[t for k in sample for t in aa[k]['turns']];bt=[t for k in sample for t in bb[k]['turns']]
        values.append(sum(t['strict_protocol_passed'] for t in at)/len(at)-sum(t['strict_protocol_passed'] for t in bt)/len(bt))
    return {'cluster_bootstrap_difference_95ci':[quantile(values,.025),quantile(values,.975)],'bootstrap_draws':draws,'seed':seed,'paired_scenarios':len(keys),'scope':'Selected-case, scenario-paired exploratory comparison; no correction of historical prompt information.'}
