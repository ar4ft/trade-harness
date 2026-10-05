"""Same-schedule portfolio replay of numerical and reviewer ablations, with row-level audit."""

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .diagnostics import decision_diagnostics, feature_diagnostics, paired_block_interval, regime
from .hybrid import HybridModel
from .hybrid_training import digest
from .learning import MODEL_FEATURES, Dataset, DecisionModel, fit, load_series, window
from .models import BaselineModel, load_model
from .operations import AuditedCache, scorecard
from .orchestrator import Orchestrator, OrchestratorConfig
from .paper import PaperEngine
from .promotion import PromotionEvidence
from .research_data import load_research
from .risk import RiskConfig
from .storage import Store, timeframe_ms
from .strategies import (
    STRATEGY_FEATURES,
    STRATEGY_VERSION,
    StrategyModel,
    strategy_features,
    strategy_signals,
)


class CachedForecaster:
    def __init__(self, records, contract):
        from .schemas import Forecast, Proposal

        self.version = contract["model_version"]
        self.predictions = {}
        for row in records:
            evidence = row["evidence"]["forecast"]
            self.predictions[(row["symbol"], row["as_of"])] = Proposal(
                action="HOLD", confidence=0, rationale="Frozen causal foundation forecast",
                model_version=self.version, forecast_interval=evidence["return_interval"],
                forecast=Forecast(horizon=evidence["horizon"], expected_return=evidence["expected_return"],
                                  projected_close=row["market"]["ohlc"][-1][3] *
                                  (1 + evidence["expected_return"]), method=evidence["source"]))

    def predict(self, market, history):
        return self.predictions[(market.symbol, market.timestamps[-1])].model_copy(deep=True)


class MemoizedReviewer(AuditedCache):
    """Cache by exact input/history/evidence AND weights; never cache outcome labels."""
    def __init__(self, model, path):
        self.model, self.name, self.version = model, model.name, model.version
        self.trained_until = getattr(model, "trained_until", None)
        self.metadata = getattr(model, "metadata", {})
        self.model_family = getattr(model, "model_family", None)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["proposal"]

    def predict_with_evidence(self, market, history, evidence):
        from .schemas import Proposal

        key = digest({"version": self.version, "market": market.model_dump(),
                      "history": history, "evidence": evidence})
        hit = key in self.cache
        if not hit:
            value = self.model.predict_with_evidence(market, history, evidence)
            self.cache[key] = value.model_dump()
            with self.path.open("a") as stream:
                stream.write(json.dumps({"key": key, "proposal": self.cache[key]}, allow_nan=False) + "\n")
        value = Proposal.model_validate(self.cache[key]).model_copy(deep=True)
        value.review_details["research_cache_hit"] = hit
        return value


class ObservedStrategyModel(DecisionModel):
    name = "observed-strategy-numerical-v1"
    feature_names = MODEL_FEATURES + STRATEGY_FEATURES

    def feature_vector(self, market):
        from .learning import model_features

        return np.concatenate([model_features(market), strategy_features(strategy_signals(market))])


class CashModel:
    name = version = "cash-baseline-v1"
    uses_history = False

    def predict(self, market, history):
        from .schemas import Forecast, Proposal

        return Proposal(action="HOLD", confidence=1, model_version=self.version,
                        rationale="Fixed cash reference; no market probability claim",
                        forecast=Forecast(horizon=market.horizon, expected_return=0,
                                          projected_close=market.ohlc[-1][3], method="cash"))


def replay_candidate(series, model, schedule, start, end, config, examples, ledger):
    summaries, daily = {}, {}
    for symbol, market in sorted(series.items()):
        store = Store(":memory:")
        engine = PaperEngine(model, store, run_id=f"research-{symbol}-{start}", symbol=symbol,
                             timeframe=market.timeframe, config=config)
        rows = []
        for i, timestamp in enumerate(market.timestamps):
            if i < 99 or not start <= timestamp < end:
                continue
            key = (symbol, timestamp)
            active = key in schedule
            view = window(market, i, 100)
            before = time.monotonic()
            result = engine.replay(view, market.ohlc[i], generate_signal=active)
            if active:
                decision = result["decision"]
                source = examples[key]
                proposal = {**decision, "action": decision["proposed_action"]}
                row = {"symbol": symbol, "as_of": timestamp, "recorded_at": int(time.time() * 1000),
                       "regime": regime(source["evidence"]["features"]),
                       "input_sha256": decision["operational"]["input_sha256"],
                       "proposal": proposal, "decision": decision, "outcome": source["outcome"],
                       "inference_and_replay_ms": (time.monotonic() - before) * 1000}
                rows.append(row)
                ledger.write(json.dumps(row, allow_nan=False) + "\n")
                ledger.flush()
        summary = engine.summary()
        curve = summary.pop("equity_curve")
        summaries[symbol] = summary
        daily[symbol] = {int(t) // 86400000: float(e) for t, e in curve}
        summaries[symbol]["decisions"] = rows
        summaries[symbol]["fills"] = [r for r in reversed(store.events(engine.run_id, 100000))
                                       if r["kind"] in ("fill", "cancel")]
        store.db.close()
    return summaries, daily


def _daily_returns(daily, start, end, initial_cash):
    days = range(start // 86400000, (end - 1) // 86400000 + 1)
    returns = []
    for _, marks in sorted(daily.items()):
        previous, values = initial_cash, []
        for day in days:
            current = marks.get(day, previous)
            values.append(current / previous - 1)
            previous = current
        returns.append(values)
    return np.mean(returns, axis=0)


def _artifact(data, start, contract, feature_config, observed_only=False):
    available = np.unique(data.timestamp[data.timestamp < start])
    cut = int(available[int(len(available) * 0.8)])
    train = data.subset((data.timestamp < cut) & (data.observed_at < cut))
    calibration = data.subset((data.timestamp >= cut) & (data.observed_at < start))
    names = contract["feature_names"]
    if observed_only:
        selected_names = MODEL_FEATURES + STRATEGY_FEATURES if observed_only == "strategies" else MODEL_FEATURES
        indices = [names.index(name) for name in selected_names]
        train.x, calibration.x = train.x[:, indices], calibration.x[:, indices]
        names = selected_names
    artifact = fit(train, calibration, "logistic", names)
    if not observed_only:
        artifact.update({"forecast_contract": contract["forecast_contract"],
                         "feature_config": feature_config, "strategy_version": STRATEGY_VERSION})
    artifact["model_version"] = digest(artifact)[:16]
    return artifact


def evaluate_research(dataset, reviewer_names, output, folds=5, max_decisions=0, reviewers=None,
                      allow_unknown_cutoff=False):
    if not 5 <= folds <= 12 or max_decisions < 0:
        raise ValueError("Use 5–12 chronological folds and a nonnegative decision budget")
    config = OrchestratorConfig(reviewers=reviewer_names)
    records, manifest = load_research(dataset)
    flat = [r for r in records if r["position"] == "flat"]
    examples = {(r["symbol"], r["as_of"]): r for r in flat}
    paths = [p["path"] for p in manifest["forecast_contract"]["datasets"]]
    series, actual_manifest = load_series(paths)
    if actual_manifest != manifest["forecast_contract"]["datasets"]:
        raise ValueError("Market inputs differ from the research dataset")
    data = Dataset(np.asarray([[r["evidence"]["features"][n] for n in manifest["feature_names"]]
                               for r in flat]),
                   np.asarray([("BUY", "SELL", "HOLD").index(r["outcome"]["direction_label"]) for r in flat]),
                   np.asarray([r["outcome"]["next_open_return"] for r in flat]),
                   np.asarray([r["as_of"] for r in flat]),
                   np.asarray([r["outcome"]["observed_at"] for r in flat]),
                   np.asarray([r["symbol"] for r in flat]), actual_manifest,
                   manifest["horizon"], manifest["timeframe"])
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan_path = Path(str(output) + ".plan.json")
    if destination.exists() or plan_path.exists():
        raise ValueError("Evaluation already exists; use a new immutable output")
    providers = reviewers or {name: load_model(name) for name in reviewer_names}
    if set(providers) != set(reviewer_names):
        raise ValueError("Reviewer membership differs from the declared plan")
    unknown = []
    for name, provider in providers.items():
        cutoff = getattr(provider, "trained_until", None)
        if cutoff is None:
            if not allow_unknown_cutoff or not max_decisions:
                raise ValueError(f"{name} lacks a known training cutoff; cannot claim chronological validation")
            unknown.append(name)
        elif cutoff >= manifest["boundaries"]["validation"]:
            raise ValueError(f"{name} overlaps validation; train a reviewer on the research train/calibration partitions")
    providers = {name: MemoizedReviewer(provider, f"artifacts/research/reviewer-cache-{provider.version}.jsonl")
                 for name, provider in providers.items()}
    times = np.unique([r["as_of"] for r in flat if r["phase"] == "validation"])
    boundaries = np.linspace(0, len(times), folds + 1, dtype=int)
    if len(times) < folds * 3:
        raise ValueError("Insufficient chronological validation timestamps")
    intervals = [(int(times[left]), int(times[right]) if right < len(times) else
                  manifest["boundaries"]["test"]) for left, right in zip(boundaries[:-1], boundaries[1:])]
    intervals.append((manifest["boundaries"]["test"], int(max(data.observed_at)) + 1))
    eligible = sorted({r["as_of"] for r in flat if r["phase"] in ("validation", "test")})
    if max_decisions:
        # Sample within each interval to keep fold coverage; the budget is per interval.
        selected = set()
        for start, end in intervals:
            group = [t for t in eligible if start <= t < end]
            selected.update(group[i] for i in np.linspace(0, len(group) - 1,
                                                          min(max_decisions, len(group)), dtype=int))
    else:
        selected = set(eligible)
    schedule = {key for key in examples if key[1] in selected}
    plan = {"dataset_sha256": manifest["dataset_sha256"], "folds": folds,
            "risk_config": manifest["risk_config"], "intervals": intervals,
            "schedule_sha256": digest(sorted(schedule)), "decision_budget_per_interval": max_decisions,
            "reviewers": {n: p.version for n, p in providers.items()},
            "policy": config.model_dump(), "selection": "No selection or automatic activation",
            "prospective": False, "declared_at": int(time.time() * 1000)}
    plan.update({"unknown_foundation_cutoff_reviewers": unknown,
                 "reviewer_chronology_certified": not unknown,
                 "benchmark_contract": "matched-policy-ablation-v2"})
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    report = {"mode": "decision_only", "real_execution_enabled": False, "plan": plan,
              "dataset_manifest": manifest, "feature_diagnostics": feature_diagnostics(records),
              "folds": [], "final_test": {}, "candidates": {},
              "limitations": ["Previously examined historical data; no fresh forward evidence.",
                              "Same sparse schedule for all ablations; not hourly model evaluation.",
                              "Separate account per asset; aligned equal-capital returns are diagnostic aggregation.",
                              "Consensus scores remain heuristic even when reviewers are calibrated.",
                              "Multiple searches require a locked fresh forward confirmation, not winner promotion."]}
    collected, daily_streams, closed, positive, versions = {}, {}, {}, {}, {}
    horizon_days = max(1, int(np.ceil(timeframe_ms(data.timeframe) * data.horizon / 86400000)))
    block_days = max(7, horizon_days)
    forecaster = CachedForecaster(flat, manifest["forecast_contract"])
    for index, (start, end) in enumerate(intervals):
        artifact = _artifact(data, start, manifest, manifest["feature_config"])
        primary = HybridModel(artifact=artifact, forecaster=forecaster)
        observed = DecisionModel(artifact=_artifact(data, start, manifest, {}, observed_only=True))
        observed_strategies = ObservedStrategyModel(artifact=_artifact(data, start, manifest, {}, observed_only="strategies"))
        members = {"observed_numerical": observed, "observed_strategies": observed_strategies, "numerical": primary,
                   "cash": CashModel(), "momentum": BaselineModel(), "strategy": StrategyModel()}
        for size in range(1, len(reviewer_names) + 1):
            for subset in itertools.combinations(reviewer_names, size):
                members["consensus_" + "_".join(subset)] = Orchestrator(
                    primary=primary, reviewers={n: providers[n] for n in subset},
                    config=OrchestratorConfig(reviewers=list(subset)))
        phase = "validation" if index < folds else "test"
        entry = {"start": start, "end": end, "models": {},
                 "fit_labels_last": artifact["fit_label_last_timestamp"],
                 "calibration_labels_last": artifact["trained_until"]}
        interval_schedule = {key for key in schedule if start <= key[1] < end
                             and examples[key]["outcome"]["observed_at"] < end}
        if not interval_schedule:
            raise ValueError("No labels mature inside a declared interval; lengthen the window")
        entry["validation_labels_last"] = max(examples[key]["outcome"]["observed_at"]
                                               for key in interval_schedule)
        for name, model in members.items():
            entry["models"][name] = {"model_version": getattr(model, "version", model.name),
                                    "feature_names": getattr(model, "feature_names", None),
                                    "trained_until": getattr(model, "trained_until", None)}
            for cost_mode in ("normal", "double_cost"):
                risk = RiskConfig.model_validate(manifest["risk_config"]).model_copy(update={"allow_research": True})
                if cost_mode == "double_cost":
                    risk = risk.model_copy(update={"fee_bps": risk.fee_bps * 2,
                                                   "slippage_bps": risk.slippage_bps * 2})
                path = destination.parent / f"{destination.stem}-fold{index + 1}-{name}-{cost_mode}.jsonl"
                with path.open("x") as stream:
                    summary, daily = replay_candidate(series, model, interval_schedule, start, end, risk, examples, stream)
                decisions = [r for s in summary.values() for r in s.pop("decisions")]
                entry["models"][name][cost_mode] = {"per_asset": summary,
                    "mean_account_return": float(np.mean([s["return"] for s in summary.values()])),
                    "ledger": str(path), "ledger_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "diagnostics": decision_diagnostics(decisions),
                    "operations": scorecard(decisions)}
                if phase == "validation":
                    closed[name] = closed.get(name, 0) + (sum(s["closed_trades"] for s in summary.values())
                                                         if cost_mode == "normal" else 0)
                    daily_streams.setdefault((name, cost_mode), []).extend(_daily_returns(daily, start, end, risk.initial_cash))
                    if cost_mode == "normal":
                        collected.setdefault(name, []).extend(decisions)
                        positive[name] = positive.get(name, 0) + int(entry["models"][name][cost_mode]["mean_account_return"] > 0)
                        versions.setdefault(name, []).append(getattr(model, "version", model.name))
            print(json.dumps({"phase": phase, "fold": index + 1, "candidate": name,
                              "normal_return": entry["models"][name]["normal"]["mean_account_return"]}), flush=True)
        if phase == "validation":
            report["folds"].append(entry)
        else:
            report["final_test"] = entry
    for name, rows in collected.items():
        diagnostics = decision_diagnostics(rows)
        reliability = diagnostics["pooled"]["proposal_directional"]
        daily = np.asarray(daily_streams[(name, "normal")])
        paired = paired_block_interval(daily, daily_streams[("momentum", "normal")], block_days)
        cash = paired_block_interval(daily, np.zeros(len(daily)), block_days)
        stress = np.asarray(daily_streams[(name, "double_cost")])
        evidence = PromotionEvidence(model_version=versions[name][-1],
            fold_count=folds, closed_trades=closed[name], positive_folds=positive[name],
            trial_count=len(collected), positive_net_return=bool(np.prod(1 + daily) > 1),
            positive_double_cost_return=bool(np.prod(1 + stress) > 1),
            paired_interval_lower=paired["interval"][0], cash_interval_lower=cash["interval"][0],
            effective_time_blocks=paired["effective_blocks"], block_days=paired["block_days"],
            horizon_dependency_days=horizon_days,
            directional_samples=reliability["samples"], minimum_decision_bin=reliability["minimum_populated_bin"],
            directional_ece=reliability["ece"], probability_calibrated_on_past=name in ("numerical", "observed_numerical", "observed_strategies"),
            purged_boundaries=all(f["fit_labels_last"] < f["calibration_labels_last"] < f["start"]
                                 and f["validation_labels_last"] < f["end"] for f in report["folds"]))
        report["candidates"][name] = {"diagnostics": diagnostics, "closed_trades": closed[name],
            "normal_compounded_diagnostic_return": float(np.prod(1 + daily) - 1),
            "paired_momentum_interval": paired, "cash_interval": cash,
            "promotion": evidence.model_dump(), "fold_model_versions": versions[name]}
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def lock_research(dataset, reviewer_names, output, trial_count=1):
    """Lock versions/risk before future observations; never pretend history is prospective."""
    records, manifest = load_research(dataset)
    from .prospective import runtime_digest

    if trial_count < 1:
        raise ValueError("Declare the complete number of prior research trials")
    primary = load_model("hybrid")
    reviewers = {name: load_model(name) for name in reviewer_names}
    if any(getattr(m, "supports_locked_forward", True) is False for m in reviewers.values()):
        raise ValueError("Forward reviewers must pin weights; mutable hosted aliases cannot certify a locked experiment")
    if primary.feature_names != manifest["feature_names"]:
        raise ValueError("Locked numerical features differ from the research dataset")
    if any(not isinstance(getattr(m, "version", None), str) for m in reviewers.values()):
        raise ValueError("Forward reviewers must expose a pinned model version")
    model = Orchestrator(primary=primary, reviewers=reviewers,
                         config=OrchestratorConfig(reviewers=reviewer_names))
    now = int(time.time() * 1000)
    horizon_days = max(1, int(np.ceil(timeframe_ms(manifest["timeframe"]) * manifest["horizon"] / 86400000)))
    plan = {"policy": "research-promotion-v2", "locked_at": now,
            "evaluation_not_before": now + 1, "last_examined_at": max(r["outcome"]["observed_at"] for r in records),
            "model_version": model.version, "primary_version": primary.version,
            "reviewer_versions": {n: m.version for n, m in reviewers.items()},
            "research_dataset_sha256": manifest["dataset_sha256"], "risk_config": manifest["risk_config"],
            "orchestrator_config": model.config.model_dump(), "symbols": manifest["symbols"],
            "timeframe": manifest["timeframe"], "horizon": manifest["horizon"],
            "baselines": ["cash", "momentum"], "block_days": max(7, horizon_days),
            "minimum_closed_trades": 100, "minimum_time_blocks": 20,
            "minimum_directional_samples": 100, "minimum_decision_bin": 30,
            "max_directional_ece": 0.1, "automatic_activation": False}
    plan.update({"runtime_sha256": runtime_digest(), "trial_count": trial_count,
                 "validation_window_days": max(28, 2 * horizon_days), "minimum_completed_windows": 5})
    plan["evaluation_end"] = plan["evaluation_not_before"] + plan["validation_window_days"] * 5 * 86400000
    path = Path(output)
    if path.exists():
        raise ValueError("Prospective plan exists; use a new immutable path")
    plan["plan_sha256"] = digest(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    return plan
