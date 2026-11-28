"""Persistent backend store for versioned meal plans, proposals, and preferences.

Observed patient screening inputs (Stage 1/2) remain strictly separated from
proposed and accepted meal plans (Stage 3/4). Conversation history alone is
never treated as the authoritative record of the current diet.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = Path(__file__).resolve().parent / "plan_sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str, store_dir: Path | None = None) -> Path:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))
    if not safe:
        raise ValueError("Invalid session_id.")
    base = store_dir or SESSIONS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{safe}.json"


def create_session(
    dossier: dict[str, Any],
    *,
    session_id: str | None = None,
    store_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a new versioned plan session from a completed dossier."""
    stage2 = dossier.get("stage2_counterfactual_prescription") or dossier.get("stage2_prescription")
    stage3 = dossier.get("stage3_discrete_meal_synthesis") or dossier.get("stage3_meal_plan")
    if not isinstance(stage2, dict) or not isinstance(stage3, dict):
        raise ValueError("Creating a plan session requires Stage 2 and Stage 3 outputs.")
    if not stage3.get("accepted"):
        raise ValueError("Cannot create a versioned plan session from an unaccepted Stage 3 plan.")

    sid = session_id or f"plan_{uuid.uuid4().hex[:12]}"
    allergies = sorted(set(stage3.get("preferences_applied", {}).get("reported_allergies", [])
                           + stage3.get("flagged_unsuitable_for_allergies", [])))
    record = {
        "session_id": sid,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        # Observed patient state — immutable across diet substitutions
        "observed_patient_profile": {
            "patient_metadata": copy.deepcopy(dossier.get("patient_metadata", {})),
            "stage1_diagnostic_assessment": copy.deepcopy(dossier.get("stage1_diagnostic_assessment", {})),
            "stage2_constraint_specification": copy.deepcopy(stage2),
        },
        "preferences": {
            "excluded_food_ids": list(stage3.get("revision", {}).get("excluded_food_ids", [])),
            "excluded_ingredient_codes": list(stage3.get("revision", {}).get("excluded_ingredient_codes", [])),
            "preferred_food_ids": list(stage3.get("revision", {}).get("preferred_food_ids", [])),
            "locked_food_quantities": dict(stage3.get("revision", {}).get("locked_food_quantities", {})),
            "max_active_dishes": stage3.get("revision", {}).get("max_active_dishes", stage3.get("preferences_applied", {}).get("max_active_dishes")),
            "reported_allergies": allergies,
        },
        "current_version": 1,
        "versions": [
            {
                "version": 1,
                "created_at": _now_iso(),
                "reason": "Initial Stage 3 accepted meal plan",
                "diff_from_previous": None,
                "stage3_plan": copy.deepcopy(stage3),
                "flagged_unsuitable_for_allergies": list(allergies),
            }
        ],
        "pending_proposals": {},
        "audit_log": [
            {"timestamp": _now_iso(), "event": "session_created", "version": 1}
        ],
    }
    _session_path(sid, store_dir).write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return record


def get_session(session_id: str, *, store_dir: Path | None = None) -> dict[str, Any]:
    path = _session_path(session_id, store_dir)
    if not path.is_file():
        raise KeyError(f"Plan session not found: {session_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(record: dict[str, Any], *, store_dir: Path | None = None) -> dict[str, Any]:
    record["updated_at"] = _now_iso()
    _session_path(str(record["session_id"]), store_dir).write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return record


def get_current_version_entry(record: dict[str, Any]) -> dict[str, Any]:
    cur = int(record["current_version"])
    for entry in record["versions"]:
        if int(entry["version"]) == cur:
            return entry
    raise ValueError(f"Current version {cur} missing from session {record.get('session_id')}")


def store_pending_proposal(
    session_id: str,
    proposal_id: str,
    proposal: dict[str, Any],
    diff: dict[str, Any],
    reason: str,
    *,
    store_dir: Path | None = None,
) -> dict[str, Any]:
    record = get_session(session_id, store_dir=store_dir)
    record["pending_proposals"][proposal_id] = {
        "proposal_id": proposal_id,
        "created_at": _now_iso(),
        "base_version": record["current_version"],
        "reason": reason,
        "diff": diff,
        "proposal": copy.deepcopy(proposal),
    }
    record["audit_log"].append({
        "timestamp": _now_iso(),
        "event": "proposal_created",
        "proposal_id": proposal_id,
        "base_version": record["current_version"],
        "reason": reason,
    })
    return save_session(record, store_dir=store_dir)


def commit_accepted_version(
    session_id: str,
    accepted_stage3: dict[str, Any],
    diff: dict[str, Any],
    reason: str,
    *,
    updated_preferences: dict[str, Any] | None = None,
    store_dir: Path | None = None,
) -> dict[str, Any]:
    record = get_session(session_id, store_dir=store_dir)
    new_version = max(int(v["version"]) for v in record["versions"]) + 1
    if updated_preferences:
        record["preferences"].update(updated_preferences)
    record["preferences"]["reported_allergies"] = sorted(set(
        record["preferences"].get("reported_allergies", [])
        + get_current_version_entry(record).get("flagged_unsuitable_for_allergies", [])
        + accepted_stage3.get("flagged_unsuitable_for_allergies", [])
        + accepted_stage3.get("preferences_applied", {}).get("reported_allergies", [])
    ))
    record["versions"].append({
        "version": new_version,
        "created_at": _now_iso(),
        "reason": reason,
        "diff_from_previous": diff,
        "stage3_plan": copy.deepcopy(accepted_stage3),
        "flagged_unsuitable_for_allergies": list(record["preferences"].get("reported_allergies", [])),
    })
    record["current_version"] = new_version
    record["pending_proposals"] = {}
    record["audit_log"].append({
        "timestamp": _now_iso(),
        "event": "version_accepted",
        "version": new_version,
        "reason": reason,
        "total_serving_delta": diff.get("total_serving_delta"),
    })
    return save_session(record, store_dir=store_dir)


def rollback_version(
    session_id: str,
    target_version: int,
    *,
    store_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = get_session(session_id, store_dir=store_dir)
    target = next((v for v in record["versions"] if int(v["version"]) == int(target_version)), None)
    if target is None:
        raise ValueError(f"Version {target_version} does not exist in session {session_id}.")
    previous_version = int(record["current_version"])
    record["current_version"] = int(target_version)
    record["pending_proposals"] = {}
    rev_info = target["stage3_plan"].get("revision", {})
    record["preferences"]["excluded_food_ids"] = list(rev_info.get("excluded_food_ids", []))
    initial = target["stage3_plan"].get("preferences_applied", {})
    for key, default in (("excluded_food_groups", []), ("meal_requirements", {}), ("locked_food_quantities", {}), ("preferred_food_ids", []),
                         ("excluded_ingredient_codes", []), ("max_active_dishes", None)):
        record["preferences"][key] = copy.deepcopy(rev_info.get(key, initial.get(key, default)))
    target["flagged_unsuitable_for_allergies"] = sorted(set(
        target.get("flagged_unsuitable_for_allergies", [])
        + record["preferences"].get("reported_allergies", [])
    ))
    record["audit_log"].append({
        "timestamp": _now_iso(),
        "event": "version_rollback",
        "from_version": previous_version,
        "to_version": int(target_version),
    })
    save_session(record, store_dir=store_dir)
    return record, target


def flag_allergy(
    session_id: str,
    allergen: str,
    *,
    store_dir: Path | None = None,
) -> dict[str, Any]:
    """Immediately record a user-reported allergy and flag the current plan unsuitable."""
    record = get_session(session_id, store_dir=store_dir)
    clean_allergen = allergen.strip().lower()
    reported = record["preferences"].setdefault("reported_allergies", [])
    if clean_allergen and clean_allergen not in reported:
        reported.append(clean_allergen)
    current = get_current_version_entry(record)
    flags = current.setdefault("flagged_unsuitable_for_allergies", [])
    if clean_allergen and clean_allergen not in flags:
        flags.append(clean_allergen)
    record["pending_proposals"] = {}
    record["audit_log"].append({
        "timestamp": _now_iso(),
        "event": "allergy_flagged_plan_unsuitable",
        "allergen": clean_allergen,
        "flagged_version": current["version"],
    })
    return save_session(record, store_dir=store_dir)


def build_dossier_from_session(record: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a complete dossier from the immutable patient profile + active plan version."""
    obs = record["observed_patient_profile"]
    current = get_current_version_entry(record)
    stage3 = copy.deepcopy(current["stage3_plan"])
    stage3["plan_version"] = current["version"]
    stage3["flagged_unsuitable_for_allergies"] = sorted(set(
        current.get("flagged_unsuitable_for_allergies", [])
        + record["preferences"].get("reported_allergies", [])
        + stage3.get("flagged_unsuitable_for_allergies", [])
        + stage3.get("preferences_applied", {}).get("reported_allergies", [])
    ))
    return {
        "session_id": record["session_id"],
        "patient_metadata": copy.deepcopy(obs["patient_metadata"]),
        "stage1_diagnostic_assessment": copy.deepcopy(obs["stage1_diagnostic_assessment"]),
        "stage2_counterfactual_prescription": copy.deepcopy(obs["stage2_constraint_specification"]),
        "stage3_discrete_meal_synthesis": stage3,
        "stage3_meal_plan": stage3,
    }
