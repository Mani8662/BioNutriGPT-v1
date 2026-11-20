"""Data-driven catalog group preferences; not allergen certification."""
import json
import re
from pathlib import Path

def definitions():
    return json.loads((Path(__file__).resolve().parent.parent/'food_group_preferences.json').read_text())['groups']

def excluded_ids(groups, catalog):
    known={f.food_id for f in catalog}; result=set(); definitions_by_group=definitions()
    for group in groups:
        if group not in definitions_by_group:raise ValueError('Unknown food group')
        result.update(definitions_by_group[group]['food_ids'])
    if result-known:raise ValueError('Food group index does not match active catalog')
    return sorted(result)

def exclusion_request(message):
    text=message.strip().casefold().rstrip('.!?')
    # Parse a preference operation independently of the catalog's group vocabulary.
    prefix=r"(?:please\s+)?(?:remove|exclude|avoid|no|without|i\s+(?:do\s+not|don't|dont)\s+want)\s+(?:to\s+)?(?:any\s+|all\s+)?"
    suffix=r"(?:\s+(?:in|on|from)\s+(?:my|the)\s+(?:meal\s+)?plan)?"
    for group,data in definitions().items():
        for alias in data['aliases']:
            if re.fullmatch(prefix+re.escape(alias)+suffix,text) or re.fullmatch(re.escape(alias)+r'-free(?:\s+plan)?',text):
                return group
    return None
