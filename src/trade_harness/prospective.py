"""Append-only live paper observations tied to a declaration made before future outcomes."""

import json
import time
from pathlib import Path

from .hybrid_training import digest
from .storage import timeframe_ms


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    signature = plan.pop("plan_sha256")
    if signature != digest(plan) or plan["policy"] != "research-promotion-v2":
        raise ValueError("Prospective plan digest or policy mismatch")
    plan["plan_sha256"] = signature
    return plan


def runtime_digest():
    import hashlib

    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob("*.py"))})


class ForwardJournal:
    def __init__(self, plan_path, output, model, config):
        self.plan = load_plan(plan_path)
        self.path = Path(output)
        expected = dict(self.plan["risk_config"])
        expected["allow_research"] = True
        if (model.version != self.plan["model_version"] or config.model_dump() != expected or
                runtime_digest() != self.plan["runtime_sha256"]):
            raise ValueError("Forward model or risk policy differs from locked plan")
        self.last = self.plan["plan_sha256"]
        self.last_times = {}
        if self.path.exists():
            for row in read_journal(self.path, self.plan):
                self.last = row["sha256"]
                self.last_times[row["symbol"]] = row["observed_at"]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, market, quote, observed_at, result, baselines):
        now = int(time.time() * 1000)
        first = max(self.plan["evaluation_not_before"], self.plan["last_examined_at"] + 1)
        if (market.symbol not in self.plan["symbols"] or
                (market.timeframe, market.horizon) != (self.plan["timeframe"], self.plan["horizon"])):
            raise ValueError("Forward market differs from plan")
        if market.timestamps[-1] < first or abs(now - observed_at) > 30000:
            raise ValueError("Forward observations must be fresh and after the locked research period")
        if not 0 <= observed_at - market.timestamps[-1] <= timeframe_ms(market.timeframe) * 1.5:
            raise ValueError("Forward candle is stale or future")
        if observed_at <= self.last_times.get(market.symbol, -1):
            raise ValueError("Forward observations must advance per asset")
        decision = result.get("decision")
        if decision and decision["model_version"] != self.plan["model_version"]:
            raise ValueError("Forward decision model changed")
        if decision and decision.get("consensus"):
            versions = {v["member"]: v["model_version"] for v in decision["consensus"]["votes"]}
            expected_versions = {"numerical": self.plan["primary_version"], **self.plan["reviewer_versions"]}
            if versions != expected_versions:
                raise ValueError("Forward member weights changed after declaration")
        expected_run = f"forward-{self.plan['plan_sha256'][:12]}-{market.symbol}"
        if result.get("run_id") != expected_run:
            raise ValueError("Forward observation must use the plan's funded account")
        row = {"plan_sha256": self.plan["plan_sha256"], "previous_sha256": self.last,
               "symbol": market.symbol, "as_of": market.timestamps[-1], "observed_at": observed_at,
               "recorded_at": now, "quote": quote.model_dump(), "input_sha256": digest(market.model_dump()),
               "candidate": result, "baselines": baselines}
        row["sha256"] = digest(row)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        self.last, self.last_times[market.symbol] = row["sha256"], observed_at


def read_journal(path, plan):
    previous, last_times = plan["plan_sha256"], {}
    with Path(path).open() as stream:
        yield from _journal_lines(stream, plan, previous, last_times)


def _journal_lines(stream, plan, previous, last_times):
    for line in stream:
        row = json.loads(line)
        fingerprint = row.pop("sha256")
        if (digest(row) != fingerprint or row["previous_sha256"] != previous or
                row["plan_sha256"] != plan["plan_sha256"]):
            raise ValueError("Forward journal chain or plan mismatch")
        row["sha256"] = fingerprint
        if row["observed_at"] <= last_times.get(row["symbol"], -1):
            raise ValueError("Forward observations are not chronological")
        if row["as_of"] <= max(plan["locked_at"], plan["last_examined_at"]):
            raise ValueError("Historical rows cannot become prospective evidence")
        previous, last_times[row["symbol"]] = fingerprint, row["observed_at"]
        yield row


def summarize_forward(plan_path, journal_path, paths, output):
    """Join matured market outcomes after capture; never amend the original predictions."""
    import numpy as np

    from .diagnostics import paired_block_interval, reliability
    from .learning import load_series
    from .promotion import PromotionEvidence

    plan = load_plan(plan_path)
    journals = [journal_path] if isinstance(journal_path, (str, Path)) else journal_path
    rows = (row for path in journals for row in read_journal(path, plan))
    series, _ = load_series(paths)
    scores, correct, decisions, completed = [], [], [], []
    now = int(time.time() * 1000)
    daily_marks, fills = {}, {}
    fixed, recorded_before = True, True
    observations, start, end, first_as_of = 0, None, None, None
    window_days = plan.get("validation_window_days", 28)
    window_ms = window_days * 86400000
    block_days = plan.get("block_days", 7)
    deadline = plan.get("evaluation_end")
    horizon_days = max(1, int(np.ceil(timeframe_ms(plan["timeframe"]) * plan["horizon"] / 86400000)))
    expected_matured, calibration_windows = 0, {}
    observed_signals = {symbol: set() for symbol in plan["symbols"]}
    for row in rows:
        if deadline is not None and row["observed_at"] > deadline:
            continue  # A later favorable period cannot extend the declared test.
        observations += 1
        start = row["observed_at"] if start is None else min(start, row["observed_at"])
        end = row["observed_at"] if end is None else max(end, row["observed_at"])
        first_as_of = row["as_of"] if first_as_of is None else min(first_as_of, row["as_of"])
        if set(row["baselines"]) != {"momentum", "double_cost"}:
            raise ValueError("Forward journal lacks paired baseline/stress accounts")
        expected_run = f"forward-{plan['plan_sha256'][:12]}-{row['symbol']}"
        if row["candidate"].get("run_id") != expected_run:
            raise ValueError("Forward account differs from the plan")
        candidates = {"candidate": row["candidate"], **row["baselines"]}
        for name, result in candidates.items():
            day = row["observed_at"] // 86400000
            daily_marks.setdefault((name, row["symbol"]), {})[day] = result["equity"]
            # Closing fills are captured from the ledger per tick, including protective exits.
            for event in result.get("new_fills", []):
                key = (name, row["symbol"], event["id"])
                if key not in fills:
                    fills[key] = event
        decision = row["candidate"].get("decision")
        if not decision:
            continue
        if (decision["as_of"] != row["as_of"] or row["recorded_at"] < row["observed_at"] or
                row["recorded_at"] - row["observed_at"] > 30000):
            raise ValueError("Forward decision or capture timestamp mismatch")
        observed_signals[row["symbol"]].add(row["as_of"])
        fixed = fixed and decision["model_version"] == plan["model_version"]
        if row["as_of"] + plan["horizon"] * timeframe_ms(plan["timeframe"]) > min(now, deadline or now):
            continue
        expected_matured += 1
        market = series.get(row["symbol"])
        if market is None:
            raise ValueError("Missing market outcomes for a planned asset")
        index = {t: i for i, t in enumerate(market.timestamps)}
        i = index.get(row["as_of"])
        if i is None or i + plan["horizon"] >= len(market.timestamps):
            continue
        target_index = i + plan["horizon"]
        outcome_at = market.timestamps[target_index]
        if deadline is not None and outcome_at > deadline:
            continue  # Gaps in an archive cannot move the target past the declared endpoint.
        if outcome_at > now:
            raise ValueError("Future outcome cannot be observed yet")
        recorded_before = recorded_before and row["recorded_at"] < outcome_at
        realized = market.ohlc[target_index][3] / market.ohlc[i + 1][0] - 1
        label = "BUY" if realized > 0.003 else "SELL" if realized < -0.003 else "HOLD"
        if decision["proposed_action"] != "HOLD":
            scores.append(decision["confidence"])
            correct.append(decision["proposed_action"] == label)
            period = (row["as_of"] - plan["evaluation_not_before"]) // window_ms
            calibration_windows.setdefault(period, []).append((decision["confidence"], decision["proposed_action"] == label))
        completed.append({"decision_id": decision["id"], "as_of": row["as_of"],
                          "observed_at": outcome_at, "next_open_return": realized, "direction_label": label})
        decisions.append(decision)
    if not observations:
        raise ValueError("No forward observations recorded")
    # Include each funded asset; missing marks carry cash or prior equity, never drop losing assets.
    daily, per_asset_daily = {}, {}
    for name in ("candidate", "momentum", "double_cost"):
        per_asset = []
        for symbol in plan["symbols"]:
            marks = daily_marks.get((name, symbol), {})
            previous, returns = plan["risk_config"]["initial_cash"], []
            for day in range(start // 86400000, end // 86400000 + 1):
                value = marks.get(day, previous)
                returns.append(value / previous - 1)
                previous = value
            per_asset.append(returns)
            per_asset_daily[(name, symbol)] = np.asarray(returns)
        daily[name] = np.mean(per_asset, axis=0)
    paired = paired_block_interval(daily["candidate"], daily["momentum"], block_days)
    cash = paired_block_interval(daily["candidate"], np.zeros(len(daily["candidate"])), block_days)
    calibration = reliability(scores, correct)
    # Only completed 28-day windows count; a partly observed period cannot inflate fold count.
    folds = (plan.get("minimum_completed_windows", 5) if deadline is not None and now >= deadline else
             min((end - plan["evaluation_not_before"]) // window_ms, (end - start) // window_ms))
    closing = [e for (name, _, _), e in fills.items()
               if name == "candidate" and e["payload"]["net_pnl"] is not None]
    per_asset = {}
    for symbol in plan["symbols"]:
        candidate = per_asset_daily[("candidate", symbol)]
        stressed = per_asset_daily[("double_cost", symbol)]
        comparison = paired_block_interval(candidate, per_asset_daily[("momentum", symbol)], block_days)
        cash_asset = paired_block_interval(candidate, np.zeros(len(candidate)), block_days)
        trades = sum(name == "candidate" and asset == symbol and e["payload"]["net_pnl"] is not None
                     for (name, asset, _), e in fills.items())
        per_asset[symbol] = {"net_return": float(np.prod(1 + candidate) - 1),
                             "stressed_return": float(np.prod(1 + stressed) - 1),
                             "closed_trades": trades, "paired_interval": comparison,
                             "cash_interval": cash_asset}
        per_asset[symbol]["passed"] = (trades >= 20 and per_asset[symbol]["net_return"] > 0 and
                                        per_asset[symbol]["stressed_return"] > 0 and
                                        comparison["interval"][0] > 0 and cash_asset["interval"][0] > 0)
    expected = max(1, ((deadline if deadline and now >= deadline else end) -
                      plan["evaluation_not_before"]) // timeframe_ms(plan["timeframe"]))
    coverage = min(1.0, min(len(s) for s in observed_signals.values()) / expected)
    evidence = PromotionEvidence(
        plan_sha256=plan["plan_sha256"], model_version=plan["model_version"],
        locked_at=plan["locked_at"], last_examined_at=plan["last_examined_at"],
        evaluation_start=first_as_of, evaluation_end=end,
        predictions_recorded_before_outcomes=recorded_before and bool(completed),
        model_and_policy_locked=fixed, trial_count=plan["trial_count"],
        fixed_evaluation_endpoint=deadline is not None and now >= deadline,
        fold_count=max(0, folds), closed_trades=len(closing),
        positive_folds=sum(np.prod(1 + daily["candidate"][i:i + window_days]) > 1
                           for i in range(0, len(daily["candidate"]) - window_days + 1, window_days)),
        positive_net_return=bool(np.prod(1 + daily["candidate"]) > 1),
        positive_double_cost_return=bool(np.prod(1 + daily["double_cost"]) > 1),
        paired_interval_lower=paired["interval"][0], cash_interval_lower=cash["interval"][0],
        effective_time_blocks=paired["effective_blocks"], block_days=paired["block_days"],
        horizon_dependency_days=horizon_days,
        directional_samples=calibration["samples"], minimum_decision_bin=calibration["minimum_populated_bin"],
        directional_ece=calibration["ece"],
        probability_calibrated_on_past=bool(decisions) and all(
            d["probability_calibration"] != "uncalibrated" for d in decisions),
        purged_boundaries=bool(completed) and fixed,
        all_planned_assets_observed=all(observed_signals.values()),
        per_asset_checks_passed=all(v["passed"] for v in per_asset.values()),
        forward_observation_coverage=coverage,
        matured_outcome_coverage=len(completed) / expected_matured if expected_matured else 0,
        calibration_windows=sum(len(values) >= 30 and reliability(
            [v[0] for v in values], [v[1] for v in values])["ece"] <= 0.1
                                for period, values in calibration_windows.items() if period < folds),
    )
    report = {"mode": "paper", "real_execution_enabled": False, "plan": plan,
              "observations": observations, "matured_decisions": len(completed), "per_asset": per_asset,
              "promotion": evidence.model_dump(), "decision_reliability": calibration,
              "paired_interval": paired, "cash_interval": cash, "outcomes": completed,
              "limitations": ["Hash chaining detects edits relative to the retained plan; it is not independent timestamp attestation.",
                              "Each asset has its own funded paper account; no shared portfolio execution.",
                              "Consensus scores require separate past-only calibration before promotion."]}
    Path(output).write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    return report
