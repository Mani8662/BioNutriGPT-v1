"""Compile explicitly separated preferences into one atomic revision specification."""
import re
import food_groups

def parse(message, catalog, meal_names, match_food):
    # A supported single removal with a conversational preface is not two actions.
    if re.fullmatch(r'i am (?:fed up with|tired of|bored of) [^;]+;\s*(?:please\s+)?(?:take .+ off|leave .+ out)[^;]*',message,re.I):
        return None
    clauses=re.split(r'\s*;\s*|\s+and\s+(?=(?:please\s+)?(?:include|add|put|remove|exclude|avoid|no|keep)\b)',message,flags=re.I)
    if len(clauses)<2:return None
    spec={'excluded_food_groups':[],'locked_food_quantities':{},'preferred_food_ids':[],'meal_requirements':{}}
    summaries=[]
    for clause in clauses:
        clause=clause.strip().rstrip('.!?')
        group=food_groups.exclusion_request(clause)
        if group:
            spec['excluded_food_groups'].append(group);summaries.append('Exclude '+group);continue
        placement=re.fullmatch(r'(?:please\s+)?(?:include|add|put)\s+([1-3]|one|two|three)\s+(?:catalog\s+)?servings?\s+of\s+(.+?)\s+(?:at|for)\s+('+ '|'.join(re.escape(m) for m in meal_names) +r')',clause,re.I)
        if not placement:
            raise ValueError('I could not represent every clause. Use separate group exclusions and food/serving/meal requests; no changes have been requested.')
        amount,subject,meal=placement.groups()
        # Require complete catalog identity here; substring matching would drop
        # instructions embedded in the food phrase (e.g. "rice without salt").
        items=[{'food_id':f.food_id,'name':f.name} for f in catalog if subject.casefold() in (f.name.casefold(),str(f.food_id).casefold())]
        if len(items)!=1:raise ValueError('A food is ambiguous. Please use its full catalog name in the complete request.')
        fid=items[0]['food_id']
        if fid in spec['meal_requirements']:raise ValueError('The same food appears in multiple placements. Please clarify its daily allocation.')
        quantity=int(amount) if amount.isdigit() else {'one':1,'two':2,'three':3}[amount.lower()]
        meal=next(m for m in meal_names if m.casefold()==meal.casefold())
        spec['locked_food_quantities'][fid]=quantity;spec['meal_requirements'][fid]=meal;spec['preferred_food_ids'].append(fid)
        summaries.append(f"{quantity} daily catalog servings of {items[0]['name']} at {meal}")
    forbidden=set(food_groups.excluded_ids(spec['excluded_food_groups'],catalog))
    if forbidden.intersection(spec['locked_food_quantities']):raise ValueError('The request both excludes and includes the same food group. Please resolve that conflict.')
    return spec, '; '.join(summaries)
