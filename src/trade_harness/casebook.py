"""Real historical case sampling and separately labeled input-robustness fixtures."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .hybrid_training import digest
from .research_data import load_research
from .storage import timeframe_ms


def case_tags(row):
    """Observed-context tags are diagnostic annotations, never extra model inputs."""
    market, f = row["market"], row["evidence"]["features"]
    tags = ["existing_long" if row["position"] == "long" else "flat_account"]
    if abs(f["return_20"]) <= 0.01:
        tags.append("sideways")
    if f["return_20"] < -0.03:
        tags.append("observed_drawdown")
    if f["volatility_20"] > 0.015:
        tags.append("high_volatility")
    close = np.asarray([bar[3] for bar in market["ohlc"]])
    changes = np.diff(close) / close[:-1]
    if abs(changes[-1]) > max(0.015, 3 * float(changes[:-1].std())):
        tags.append("price_spike")
    if any(
        right - left != timeframe_ms(market["timeframe"])
        for left, right in zip(market["timestamps"][:-1], market["timestamps"][1:])
    ):
        tags.append("missing_intervals")
    if row["outcome"]["next_open_return"] > 0 and row["outcome"]["net_long_return"] <= 0:
        tags.append("positive_move_erased_by_costs")
    return tags


def export_cases(datasets, output, per_case=8):
    if not 1 <= per_case <= 100:
        raise ValueError("Choose 1–100 examples per diagnostic case")
    destination, manifest_path = Path(output), Path(str(output) + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise ValueError("Casebook exists; use a new immutable output")
    buckets, sources, counts = defaultdict(list), [], Counter()
    for dataset in datasets:
        rows, manifest = load_research(dataset)
        sources.append({"path": str(dataset), "sha256": manifest["dataset_sha256"]})
        for row in rows:
            tags = case_tags(row)
            counts.update(tags)
            # Manual label review cannot inspect held-out examples to revise training criteria.
            if row["phase"] == "train":
                for tag in tags:
                    buckets[tag].append((row, tags, manifest["dataset_sha256"]))
    selected = {}
    for tag, group in sorted(buckets.items()):
        group.sort(key=lambda entry: (entry[0]["symbol"], entry[0]["as_of"], entry[0]["position"]))
        for index in np.linspace(0, len(group) - 1, min(per_case, len(group)), dtype=int):
            row, tags, source = group[index]
            key = (source, row["symbol"], row["as_of"], row["position"])
            selected[key] = {
                "id": ":".join(map(str, key)),
                "kind": "real_training_case",
                "source_dataset_sha256": source,
                "example": row,
                "review": {
                    "tags": tags,
                    "status": "pending_human_review",
                    "reviewed_action": None,
                    "notes": "",
                },
            }
    # Gaps/invalid bounds test input handling; no forecast or trading target is fabricated.
    if selected:
        sample = next(iter(selected.values()))["example"]
        market = json.loads(json.dumps(sample["market"]))
        market["timestamps"][-2] -= timeframe_ms(market["timeframe"]) // 2
        selected[("fixture", "gap")] = {
            "id": "fixture:irregular_intervals",
            "kind": "synthetic_robustness_fixture",
            "market": market,
            "expected_behavior": "Reject irregular forecast context or guard missing intervals",
            "training_eligible": False,
            "future_target": None,
        }
        market = json.loads(json.dumps(sample["market"]))
        market["ohlc"][-1][1] = market["ohlc"][-1][2] / 2
        selected[("fixture", "ohlc")] = {
            "id": "fixture:invalid_ohlc",
            "kind": "synthetic_robustness_fixture",
            "market": market,
            "expected_behavior": "Reject invalid OHLC before inference",
            "training_eligible": False,
            "future_target": None,
        }
    payload = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in selected.values()
    )
    import hashlib

    manifest = {
        "contract": "training-casebook-v1",
        "sources": sources,
        "cases": len(selected),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "observed_tag_counts_all_phases": dict(counts),
        "selected_case_counts": dict(
            Counter(t for r in selected.values() for t in r.get("review", {}).get("tags", []))
        ),
        "review_phase": "train_only",
        "human_review_completed": False,
        "case_criteria_sha256": digest(
            {
                "sideways_20": 0.01,
                "drawdown_20": -0.03,
                "high_volatility_20": 0.015,
                "spike_floor": 0.015,
                "spike_sigma": 3,
            }
        ),
        "limitations": [
            "Tag-based sampling changes class mix; it is not a trading benchmark.",
            "Cost-erased tag uses a later outcome and must remain outside inference inputs.",
            "Synthetic input fixtures are excluded from supervised market training.",
            "Reviewed actions are pending; no expert labels are claimed or automatically applied.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(payload)
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return manifest
