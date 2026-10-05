"""Read-only operational scorecards for persisted decisions and replay journals."""

import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ACTIONS = ("BUY", "SELL", "HOLD")


class AuditedCache:
    """Marker for harness-owned exact-input caching; provider output cannot declare hits."""


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Latency measurements must be finite and nonnegative")
    return {
        "samples": len(values),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(values.max()),
    }


def action_drift(rows):
    """Descriptive earlier/later total variation; no significance or quality claim."""
    rows = sorted(rows, key=lambda r: r["decision"]["as_of"])
    n = len(rows) // 2
    if n < 10:
        return {
            "available": False,
            "reason": "Need >=20 decisions within one identity/market/position",
        }
    distributions = []
    for group in (rows[:n], rows[n:]):
        counts = Counter(r["decision"]["proposed_action"] for r in group)
        distributions.append({a: counts[a] / len(group) for a in ACTIONS})
    return {
        "available": True,
        "earlier": distributions[0],
        "later": distributions[1],
        "total_variation": sum(abs(distributions[0][a] - distributions[1][a]) for a in ACTIONS) / 2,
        "semantics": "Descriptive action-distribution shift; changing regimes can explain it",
    }


def scorecard(rows, evaluated_at=None):
    evaluated_at = int(time.time() * 1000) if evaluated_at is None else evaluated_at
    guards, members, groups = Counter(), Counter(), defaultdict(list)
    timings, replay_timings, member_timings = [], [], []
    member_by_provider = defaultdict(list)
    actions, proposals = Counter(), Counter()
    matured, correct, directional, traced, cache_hits, missing = 0, 0, 0, 0, 0, 0
    failures = 0
    for row in rows:
        decision = row.get("decision")
        if not decision:
            failures += 1
            continue
        op = decision.get("operational", {})
        guards.update(decision.get("guardrails", []))
        actions[decision["action"]] += 1
        proposals[decision.get("proposed_action", decision["action"])] += 1
        if op.get("model_call_and_validation_ms") is not None:
            timings.append(op["model_call_and_validation_ms"])
        if row.get("inference_and_replay_ms") is not None:
            replay_timings.append(row["inference_and_replay_ms"])
        missing += int(op.get("missing_intervals", 0))
        traced += bool(
            op.get("input_sha256")
            and op.get("risk_config_sha256")
            and (decision.get("model_version") or op.get("configured_model_version"))
        )
        for vote in (decision.get("consensus") or {}).get("votes", []):
            members[vote["member"] + ":" + vote["status"]] += 1
            cache_hits += bool(vote.get("cache_hit"))
            if vote.get("inference_ms") is not None and not vote.get("cache_hit"):
                member_timings.append(vote["inference_ms"])
                member_by_provider[vote["member"]].append(vote["inference_ms"])
        outcome = row.get("outcome") or {}
        if outcome.get("observed_at", evaluated_at + 1) <= evaluated_at:
            matured += 1
            action = decision.get("proposed_action", decision["action"])
            if action != "HOLD" and outcome.get("direction_label") in ACTIONS:
                directional += 1
                correct += action == outcome["direction_label"]
        key = (
            decision.get("model_version"),
            decision["symbol"],
            decision["timeframe"],
            op.get("position", "unknown"),
        )
        groups[key].append(
            {
                "decision": {
                    **decision,
                    "proposed_action": decision.get("proposed_action", decision["action"]),
                }
            }
        )
    valid = sum(actions.values())
    return {
        "contract": "operational-scorecard-v1",
        "evaluated_at": evaluated_at,
        "records": len(rows),
        "decisions": valid,
        "rejected_or_failed_records": failures,
        "final_actions": dict(actions),
        "proposed_actions": dict(proposals),
        "directional_coverage": (actions["BUY"] + actions["SELL"]) / valid if valid else 0,
        "hold_fraction": actions["HOLD"] / valid if valid else None,
        "guard_counts": dict(guards),
        "provider_status_counts": dict(members),
        "observed_missing_intervals": missing,
        "traceable_decisions": traced,
        "matured_outcomes": matured,
        "outcomes_not_attached_or_not_matured": valid - matured,
        "matured_directional_samples": directional,
        "matured_direction_accuracy": correct / directional if directional else None,
        "model_call_and_validation": quantiles(timings),
        "uncached_member_call": quantiles(member_timings),
        "uncached_call_by_provider": {
            name: quantiles(values) for name, values in sorted(member_by_provider.items())
        },
        "inference_and_replay_combined": quantiles(replay_timings),
        "reviewer_cache_hits": cache_hits,
        "action_drift_by_identity_market_position": [
            {"identity": list(key), **action_drift(group)}
            for key, group in sorted(groups.items(), key=lambda item: str(item[0]))
        ],
        "observation_coverage": None,
        "mode": "research_only",
        "real_execution_enabled": False,
        "limitations": [
            "Coverage requires an independently declared schedule; record counts cannot establish it.",
            "Latency scopes are separate; cached or combined replay timings are not GPU inference benchmarks.",
            "Direction accuracy uses the recorded direction label, not net profitability or expert judgment.",
            "Rejected inputs absent from a journal/database cannot be counted retrospectively.",
        ],
    }


def export_scorecard(inputs, output):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Scorecard exists; use a new immutable output")
    rows, sources = [], []
    for name in inputs:
        path = Path(name).resolve()
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if path.suffix in (".sqlite", ".db"):
            db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                for decision, feedback in db.execute(
                    "SELECT decision,feedback FROM decisions ORDER BY as_of,id"
                ):
                    rows.append(
                        {
                            "decision": json.loads(decision),
                            "outcome": json.loads(feedback) if feedback else None,
                        }
                    )
            finally:
                db.close()
        else:
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    result = {**scorecard(rows), "sources": sources}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def database_scorecard(path, limit=2000, scope=None):
    """Configured database only; no inference, migrations, writes or unknown path creation."""
    if not 1 <= limit <= 10000:
        raise ValueError("Choose 1–10000 persisted decisions")
    path = Path(path).resolve()
    if not path.exists():
        return {**scorecard([]), "database_available": False, "sample_limit": limit}
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        where, parameters = ("WHERE scope=?", [scope]) if scope is not None else ("", [])
        query = (
            f"SELECT decision,feedback FROM decisions {where} ORDER BY as_of DESC,id DESC LIMIT ?"
        )
        rows = [
            {"decision": json.loads(d), "outcome": json.loads(f) if f else None}
            for d, f in db.execute(query, [*parameters, limit])
        ]
    finally:
        db.close()
    return {
        **scorecard(rows),
        "database_available": True,
        "sample_limit": limit,
        "scope": scope,
        "sample_scope": "Most recent persisted decisions; not a complete observation schedule",
    }
