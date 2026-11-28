"""Six synthetic language requests using real catalog names; --run enables GPU calls."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
import chat_service as core
import chat_interpreter as parser


def main(run):
    catalog=core.ps.s3.load_food_database()
    foods=catalog[:2]
    basket=[{'food_id':f.food_id,'name':f.name,'quantity':1} for f in foods]
    a,b=foods
    scenarios=[
        (f'Could you tell me the protein in {a.name}?','nutrient',str(a.food_id)),
        (f'How much energy would one serving of {a.name} contain?','nutrient',str(a.food_id)),
        ('What is the protein total for lunch?','nutrient',None),
        (f'I am bored of {a.name}; please leave it out.','remove',str(a.food_id)),
        (f'Replace {a.name} with 2 servings of {b.name}','replace',str(a.food_id)),
        (f'Do not remove {a.name}','clarification',None),
    ]
    labels={}
    for alias,info in core.NUTRIENT_ALIASES.items():labels.setdefault(info[0],alias)
    if not run:
        print(json.dumps({'cases':scenarios,'scope':'Synthetic language smoke pilot only; no plan solving, patients or training'},indent=2))
        return
    dest=ROOT/'Saved Model/stage4_interpreter_v2_pilot_r2.jsonl'
    if dest.exists():raise RuntimeError('Pilot output exists; refusing overwrite')
    for message,intent,food_id in scenarios:
        result=parser.interpret(message,catalog,basket,{'plan_version':1},[],labels,diagnostics=True)
        if 'request' in result:result['request']=result['request'].model_dump()
        row={'message':message,'expected_intent':intent,'expected_food_id':food_id,'result':result}
        # Record field checks, not a broad chatbot success claim.
        req=result.get('request',{})
        row['intent_matches']=req.get('intent')==intent
        row['food_matches']=food_id is None or req.get('food_id')==food_id
        with dest.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');main(ap.parse_args().run)
