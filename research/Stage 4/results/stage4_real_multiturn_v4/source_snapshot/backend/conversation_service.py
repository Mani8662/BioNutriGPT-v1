"""Version-aware conversation layer over the validated Stage 4 tools.

This first release adds deterministic conversational context; it does not claim
general LLM understanding. Clinical/revision guards remain in chat_service.
"""
from __future__ import annotations

import re
from typing import Any
import chat_service as core
import plan_store
import chat_interpreter

_AMOUNT = r'(?:\d+|one|two|three|four|five)'


def _normalize_replacement(text):
    """Normalize explicit instead-of wording without inferring amounts or scope."""
    clean = text.strip().rstrip('.!?')
    m = re.fullmatch(r"(?:i\s+(?:would|['’]d)\s+rather\s+have|i['’]d\s+rather\s+have|can\s+i\s+have|could\s+i\s+have|please\s+use|use)\s+(.+?)\s+(?:instead\s+of|in\s+place\s+of)\s+(.+)", clean, re.I)
    if not m:
        return text
    target, source = m.groups()
    tail = re.search(r',\s*('+_AMOUNT+r')\s+servings?(?:\s+please)?$', source, re.I)
    if tail:
        target = tail[1]+' servings of '+target
        source = source[:tail.start()]
    return 'Replace '+source+' with '+target


def _foods(text, items):
    """Return candidates, avoiding the legacy longest-name ambiguity heuristic."""
    norm = core._normalize_text(text)
    direct = [x for x in items if text.strip().casefold() == str(x['food_id']).casefold()
              or (' ' + core._normalize_text(x['name']) + ' ') in (' ' + norm + ' ')]
    if direct:
        return direct
    # Match only meaningful food words, never arbitrary numbers in a question.
    words = set(norm.split()) - set('how much many protein calories energy fibre fiber sodium potassium fat sugar carbohydrate carbs in of the my this that its it serving servings portion portions for is has have does and please food meal plan total daily grams'.split())
    hits = [x for x in items if words.intersection(core._normalize_text(x['name']).split())]
    return hits


def _use_standard_route(question, basket):
    """Recognized requests bypass generation, even when natural mode is enabled."""
    if core._nutrient(question) is not None:
        return True
    if re.search(r'\b(instead|rather)\b|\bleave\b.+\bout\b', question, re.I) and not re.search(r'\b(replace|swap|remove)\b', question, re.I):
        return False
    if re.search(r'\b(remove|replace|swap|exclude|drop|add|include|prefer|portion|servings?|grams?|weight|weigh|fewer|maximum|rollback|undo)\b', question, re.I):
        return True
    if re.fullmatch(r'(?:please\s+)?(?:show|list|explain)\s+(?:me\s+)?(?:my|the|current)?\s*(?:current\s+)?(?:plan|meals|foods)[.!?]*', question, re.I):
        return True
    return any(question.strip().casefold() in (str(x['food_id']).casefold(), x['name'].casefold()) for x in basket)


def answer(message: str, dossier: dict[str, Any], *, session_id=None,
           use_llm_explanation=False, use_llm_interpretation=False, store_dir=None):
    active, record = core._resolve_active_dossier(dossier, session_id, store_dir=store_dir)
    stage2, stage3, catalog = core._context(active)
    if record is None:
        record = plan_store.create_session(active, store_dir=store_dir)
    sid = record['session_id']; version = record['current_version']
    context = record.get('conversation', {})
    if context.get('plan_version') != version:
        context = {'plan_version': version}
    basket = stage3['solved_food_basket']
    original = message.strip(); question = original
    interpretation_status = 'deterministic'
    interpreted_request = None
    confirmed = False
    quantity_pending = context.pop('pending_quantity', None)
    if quantity_pending:
        amount = re.fullmatch(r'\s*('+_AMOUNT+r')(?:\s+(?:catalog\s+)?servings?)?(?:\s+please)?[.!]?\s*', original, re.I)
        if amount:
            question = f"Replace {quantity_pending['source']} with {amount[1]} servings of {quantity_pending['target']}"
    action = context.pop('pending_interpretation', None)
    if action and original == 'Check this interpreted request':
        question = action['command']
        confirmed = True
        interpretation_status = 'confirmed'
        interpreted_request = question
    elif action and original == 'Cancel interpreted request':
        question = 'Show my current plan'
        confirmed = True
    pending = context.pop('pending', None)
    if pending:
        selected = [x for x in pending['choices'] if original.casefold() in (x['name'].casefold(), str(x['food_id']).casefold())]
        if len(selected) == 1:
            question = pending['template'].replace('{food}', selected[0]['name'])
            context.pop('pending', None)

    def finish(result, focus=None):
        # Reload after tool execution so proposal/allergy/version writes survive.
        latest = plan_store.get_session(sid, store_dir=store_dir)
        if latest['current_version'] != version:
            context.clear()
        context['plan_version'] = latest['current_version']
        if focus:
            context['food_id'] = str(focus['food_id'])
        turns = latest.get('conversation_turns', [])
        turns.append({'question': original, 'answer': result['answer'], 'plan_version': latest['current_version']})
        latest['conversation_turns'] = turns[-40:]
        latest['conversation'] = context
        plan_store.save_session(latest, store_dir=store_dir)
        result.update(session_id=sid, plan_version=latest['current_version'])
        result.update(interpretation_status=interpretation_status, interpreted_request=interpreted_request)
        result['plan_flagged_unsuitable'] = bool(result.get('plan_flagged_unsuitable') or latest.get('preferences', {}).get('reported_allergies'))
        return result

    def reply(text, intent='clarification', choices=None, sources=None, tools=None):
        return {'intent': intent, 'answer': text, 'sources': sources or [],
                'tool_calls_used': tools or [], 'clarification_options': choices or []}

    def clarify(choices, template, text):
        context['pending'] = {'choices': choices, 'template': template}
        return finish(reply(text, choices=[x['name'] for x in choices]))

    # Keep medical/allergy/constraint/version/screening handling authoritative.
    guarded = re.search(r'\b(allerg\w*|intoleran\w*|anaphylax\w*|medicine|medication|prescrib\w*|diagnos\w*|pregnan\w*|lactat\w*|renal|dialysis|child|pediatric|ignore|relax|disable|bypass|override|skip|rollback|revert|undo|history|versions?|risk|probability|screening)\b|\bkidney\b(?!\s*beans?\b)', question, re.I)
    if guarded:
        return finish(core.answer(question, active, session_id=sid, store_dir=store_dir))

    if re.search(r"\b(don['’]?t|do not|never|not)\b", question, re.I) and re.search(r'\b(remove|replace|swap|exclude|drop|add|include)\b|\bleave\b.+\bout\b', question, re.I):
        return finish(reply('I have not requested a change. Please state the change you want, and which foods to keep.'))

    normalized_replacement = _normalize_replacement(question)
    if normalized_replacement != question:
        if re.search(r"\b(not|never|no|don['’]?t)\b", question, re.I):
            return finish(reply('Please clarify whether you want a replacement. I have not requested a change.'))
        question = normalized_replacement

    standard_route = _use_standard_route(question, basket)
    if use_llm_interpretation and standard_route and not confirmed:
        interpretation_status = 'standard_route'
    if use_llm_interpretation and not standard_route and not confirmed and question == original:
        labels = {}
        for alias, info in core.NUTRIENT_ALIASES.items():
            labels.setdefault(info[0], alias)
        turns = [t for t in record.get('conversation_turns', []) if t.get('plan_version') == version]
        parsed = chat_interpreter.interpret(question, catalog, basket, context, turns, labels)
        interpretation_status = parsed['status']
        if parsed['status'] == 'parsed':
            req = parsed['request']; interpreted_request = parsed['command']
            try:
                chat_interpreter.validate_action_evidence(req, original)
            except ValueError:
                interpretation_status = 'semantic_rejected'
                interpreted_request = None
                return finish(reply('I could not safely match that interpretation to your request. Please state the food and the single change you want.'))
            if req.intent in ('clarification','unsupported'):
                return finish(reply('Please name the food or meal and the information or single change you want. I can check portions, nutrients, removals and exact replacements.'))
            if req.intent in ('remove','replace'):
                context['pending_interpretation'] = {'command': parsed['command']}
                return finish(reply('I understood: '+parsed['command']+'. Is that the request you want me to check?',
                    choices=['Check this interpreted request','Cancel interpreted request']))
            question = parsed['command']
        else:
            # Unresolved language must not drop into broad legacy keyword routing.
            return finish(reply('I could not interpret that wording reliably. Try naming the food and asking for its portion or nutrients, or state one replacement with a serving quantity.'))

    # Resolve follow-ups only against the same accepted plan version.
    if re.search(r'\b(it|its|that dish|this dish)\b', question, re.I):
        focus = next((x for x in basket if str(x['food_id']) == context.get('food_id')), None)
        if focus is None:
            return clarify(basket, re.sub(r'\b(it|its|that dish|this dish)\b', '{food}', question, flags=re.I), 'Which food do you mean?')
        question = re.sub(r'\b(it|its|that dish|this dish)\b', focus['name'], question, flags=re.I)

    mutation = re.search(r'\b(remove|replace|swap|exclude|drop|add|include|prefer)\b', question, re.I)
    if mutation and re.search(r"\b(don't|do not|never|not)\b", question, re.I):
        return finish(reply('I have not requested a change. Please state the change you want, and which foods to keep.'))

    swap = re.search(r'\b(?:replace|swap)\s+(.+?)\s+(?:with|for)\s+(.+)$', question.strip().rstrip('.!?'), re.I)
    if swap:
        source, target = swap.groups()
        if re.search(r'\b(grams?|kg|ml|litres?)\b', target, re.I):
            return finish(reply('Please specify whole catalog servings. I cannot treat weights or volumes as serving counts.'))
        # Compound constraints need explicit parsing before any solver call.
        if re.search(r'\b(and|but|keep|while|except|breakfast|lunch|dinner|snack)\b|[.!?;]', source+' '+target, re.I):
            return finish(reply('This replacement changes daily servings. Please clarify additional instructions or meal-specific changes before I check it.'))
        sources = _foods(source, basket)
        candidates = [{'food_id': f.food_id, 'name': f.name} for f in catalog]
        targets = _foods(target, candidates)
        if len(sources) != 1:
            return clarify(sources or basket, 'Replace {food} with ' + target, 'Which food should I replace?')
        if len(targets) != 1:
            amount = re.match(r'\s*(\d+|one|two|three|four|five)\s+', target, re.I)
            prefix = amount[1] + ' servings of ' if amount else ''
            return clarify(targets or candidates, 'Replace ' + sources[0]['name'] + ' with '+prefix+'{food}', 'Which replacement food do you mean?')
        qty_match = re.match(r'\s*(\d+|one|two|three|four|five)\s+', target, re.I)
        if not qty_match:
            context['pending_quantity'] = {'source':sources[0]['name'],'target':targets[0]['name']}
            return finish(reply('How many catalog servings of ' + targets[0]['name'] + '? You can reply “2 servings”, using your intended quantity.', choices=['1 serving','2 servings','3 servings']))
        token = qty_match[1].lower(); qty = int(token) if token.isdigit() else {'one':1,'two':2,'three':3,'four':4,'five':5}[token]
        if not 1 <= qty <= 20 or re.search(r'\b(grams?|kg|ml|litres?)\b', target, re.I):
            return finish(reply('Please specify 1–20 whole catalog servings. Weight-to-serving conversion needs an explicit serving reference.'))
        old, new = str(sources[0]['food_id']), str(targets[0]['food_id'])
        if old == new:
            return finish(reply('Those names identify the same food. Please choose a different replacement.'))
        spec = {'excluded_food_ids':[old], 'preferred_food_ids':[new], 'locked_food_quantities':{new:qty}}
        proposal, diff, conflict = core.tool_propose_revision(stage2, stage3, catalog, spec,
            reason=f'Replace {sources[0]["name"]} with {qty} catalog servings of {targets[0]["name"]}', session_id=sid, store_dir=store_dir)
        if proposal is None:
            return finish(reply('No checked replacement was found. ' + (conflict or {}).get('summary', 'The requested combination could not be verified.'), 'revision', tools=['tool_propose_revision']))
        response = reply('Review this proposal: ' + core._format_diff_summary(diff or {}) + ' Apply it only if you want the revised plan.', 'revision', tools=['tool_propose_revision'])
        response.update(proposed_meal_plan=proposal, revision_diff=diff)
        return finish(response)

    if mutation:
        return finish(core.answer(question, active, session_id=sid, store_dir=store_dir))

    nutrients = []
    remaining = question
    for alias, info in sorted(core.NUTRIENT_ALIASES.items(), key=lambda x:-len(x[0])):
        if re.search(r'\b'+re.escape(alias)+r'\b', remaining, re.I):
            if info[0] not in [x[0] for x in nutrients]: nutrients.append(info)
            remaining = re.sub(r'\b'+re.escape(alias)+r'\b', ' ', remaining, flags=re.I)
    if nutrients:
        meal = re.search(r'\b(breakfast|lunch|dinner|snack)\b', remaining, re.I)
        candidates = _foods(remaining, basket)
        if len(candidates) > 1:
            return clarify(candidates, 'How much ' + ' and '.join(x[1] for x in nutrients) + ' in {food}?', 'Which food do you mean?')
        selected = basket; scope = 'your daily basket'; focus = None
        if meal:
            schedule = core.tool_get_plan(active, record)['meal_schedule']
            slots = [s for s in schedule if meal[1].lower() in s.get('meal','').lower()]
            quantities = {}
            for slot in slots:
                for row in slot['items']: quantities[str(row['food_id'])] = quantities.get(str(row['food_id']),0) + int(row['quantity'])
            selected = [{**row, 'quantity': quantities[str(row['food_id'])]} for row in basket if str(row['food_id']) in quantities]
            scope = meal[1].lower()
        if candidates:
            focus = candidates[0]; selected = [x for x in selected if x['food_id']==focus['food_id']]
            scope = focus['name'] + (' at '+scope if meal else ' (all planned servings)')
        if re.search(r'\b(per serving|one serving|1 serving|single serving)\b', question, re.I):
            if not focus: return clarify(basket, 'How much '+ ' and '.join(x[1] for x in nutrients)+' in one serving of {food}?', 'One serving of which food?')
            selected = [{**focus,'quantity':1}]; scope = 'one catalog serving of '+focus['name']
        if not selected: return finish(reply('No matching food is scheduled for that meal.'))
        # Do not quietly turn an unresolved food-specific question into daily totals.
        if not candidates and not meal and re.search(r'\b(in|of|from)\s+(?:the\s+)?\w+', remaining, re.I) and not re.search(r'\b(plan|day|daily|basket|total|meals)\b', remaining, re.I):
            return clarify(basket, 'How much '+' and '.join(x[1] for x in nutrients)+' in {food}?', 'Which food should I calculate this for?')
        lookup = {f.food_id:f for f in catalog}; parts=[]
        for code,label,unit in nutrients:
            unknown=[x['name'] for x in selected if code in lookup[x['food_id']].unknown_nutrients]
            total=sum(x['quantity']*float(lookup[x['food_id']].nutrients_per_serving[code]) for x in selected if code not in lookup[x['food_id']].unknown_nutrients)
            parts.append(f'{label}: {total:.2f} {unit}' + (' known subtotal; missing data for '+', '.join(unknown) if unknown else ''))
        return finish(reply('For '+scope+': '+ '; '.join(parts)+'. These are catalog estimates.', 'nutrient', sources=['food_catalog:nutrients_per_serving','stage3.solved_food_basket'], tools=['scoped_nutrient_totals']), focus)

    result = core.answer(question, active, session_id=sid, store_dir=store_dir)
    matches = _foods(question,basket)
    return finish(result, matches[0] if len(matches)==1 else None)
