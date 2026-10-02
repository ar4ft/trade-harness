"""Resumable historical forecasts, purged ablations, and conservative paper comparisons."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import ensure_indicators
from .evaluation import CachedModel, _mean_return, calibration_metrics
from .evaluation import _simulate as simulate_all
from .feature_engineering import FeatureConfig, FitParameters, feature_names, transform_features
from .hybrid import FORECAST_FEATURES, HYBRID_FEATURES, HybridModel, forecast_features
from .learning import (
    MODEL_FEATURES,
    Dataset,
    fit,
    load_series,
    model_features,
    predict_batch,
    window,
)
from .models import BaselineModel
from .schemas import Forecast, Proposal
from .strategies import STRATEGY_FEATURES, STRATEGY_VERSION, StrategyModel, strategy_features
from .strategies import strategy_signals as signals_for
from .timesfm_model import TimesFMModel
from .validation import summarize_folds


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def input_digest(market):
    return digest({k: v for k, v in market.model_dump().items() if k != "position"})


def build_forecast_dataset(paths, cache, stride=48, context=100, forecaster=None):
    if not 1 <= stride <= 1000 or not 21 <= context <= 10000:
        raise ValueError("Forecast stride/context is outside the supported range")
    series, manifest = load_series(paths)
    if not series:
        raise ValueError("No training market files")
    provider = forecaster or TimesFMModel()
    if forecaster is None:
        # Construct version and context together, without mutating process-wide environment.
        from .timesfm_model import REVISION

        provider.batch_size = 16
        provider.context_length = context
        provider.version = f"timesfm3-{REVISION}-ctx{context}-interval-policy-v1"
    contract = {
        "schema_version": 1, "datasets": manifest, "stride": stride,
        "context_length": context, "model_version": provider.version,
        "strategy_version": STRATEGY_VERSION,
    }
    path = Path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        with path.open() as stream:
            if json.loads(next(stream)) != contract:
                raise ValueError("Forecast cache contract changed; use a new cache file")
            for line in stream:
                record = json.loads(line)
                key = (record["symbol"], record["as_of"])
                if key in existing:
                    raise ValueError("Duplicate forecast cache record")
                existing[key] = record
    else:
        path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    rows = []
    for symbol, market in sorted(series.items()):
        indices = list(range(max(20, context - 1), len(market.ohlc) - market.horizon, stride))
        missing = [i for i in indices if (symbol, market.timestamps[i]) not in existing]
        for offset in range(0, len(missing), 16):
            chunk = missing[offset : offset + 16]
            views = [ensure_indicators(window(market, i, context)) for i in chunk]
            predictions = (provider.predict_many(views) if hasattr(provider, "predict_many")
                           else [provider.predict(view, []) for view in views])
            if len(predictions) != len(views):
                raise ValueError("Forecaster batch count mismatch")
            with path.open("a") as stream:
                for view, prediction in zip(views, predictions):
                    record = {"symbol": symbol, "as_of": view.timestamps[-1],
                              "input_sha256": input_digest(view), "proposal": prediction.model_dump()}
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    existing[(symbol, view.timestamps[-1])] = record
            print(json.dumps({"symbol": symbol, "generated": min(offset + 16, len(missing)),
                              "required": len(missing)}), flush=True)
        count = 0
        for i in range(max(20, context - 1), len(market.ohlc) - market.horizon, stride):
            view = ensure_indicators(window(market, i, context))
            key = (symbol, market.timestamps[i])
            fingerprint = input_digest(view)
            record = existing.get(key)
            if record is None:
                forecast = provider.predict(view, [])
                record = {"symbol": symbol, "as_of": key[1], "input_sha256": fingerprint,
                          "proposal": forecast.model_dump()}
                with path.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            if record["input_sha256"] != fingerprint:
                raise ValueError("Cached forecast input differs from the historical prefix")
            forecast = Proposal.model_validate(record["proposal"])
            if (forecast.model_version != provider.version or
                    forecast.forecast.horizon != market.horizon or
                    forecast.forecast_interval is None or
                    not np.isclose(forecast.forecast.projected_close,
                                   view.ohlc[-1][3] * (1 + forecast.forecast.expected_return))):
                raise ValueError("Cached forecast contract mismatch")
            expected = forecast.forecast.expected_return
            if not forecast.forecast_interval[0] <= expected <= forecast.forecast_interval[1]:
                raise ValueError("Cached forecast interval excludes its point")
            strategies = signals_for(view)
            x = np.concatenate([model_features(view), strategy_features(strategies),
                                forecast_features(view, forecast)])
            realized = market.ohlc[i + market.horizon][3] / market.ohlc[i + 1][0] - 1
            label = 0 if realized > 0.003 else 1 if realized < -0.003 else 2
            rows.append((key[1], symbol, x, realized, label,
                         market.timestamps[i + market.horizon], forecast))
            count += 1
            if count % 100 == 0:
                print(json.dumps({"symbol": symbol, "forecast_samples": count}), flush=True)
    rows.sort(key=lambda r: (r[0], r[1]))
    if len(rows) < 300:
        raise ValueError("Need at least 300 forecast examples for this benchmark")
    first = next(iter(series.values()))
    data = Dataset(
        np.asarray([r[2] for r in rows]), np.asarray([r[4] for r in rows]),
        np.asarray([r[3] for r in rows]), np.asarray([r[0] for r in rows], dtype=np.int64),
        np.asarray([r[5] for r in rows], dtype=np.int64), np.asarray([r[1] for r in rows]),
        manifest, first.horizon, first.timeframe,
    )
    proposals = {(r[1], r[0]): r[6] for r in rows}
    contract["cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data, series, contract, proposals


class ScheduledModel:
    uses_history = False

    def __init__(self, model, allowed, forecasts=None):
        self.model, self.allowed, self.forecasts = model, allowed, forecasts
        self.name = model.name + "-scheduled"
        self.version = getattr(model, "version", "fixed-policy-v1")

    def predict(self, market, history):
        key = (market.symbol, market.timestamps[-1])
        if key in self.allowed:
            return self.forecasts[key] if self.forecasts is not None else self.model.predict(market, [])
        return Proposal(
            action="HOLD", confidence=0, rationale="Outside the predeclared decision schedule.",
            forecast=Forecast(horizon=market.horizon, expected_return=0,
                              projected_close=market.ohlc[-1][3], method="schedule"),
        )


def _simulate(series, model, start, end, config):
    schedule = getattr(model, "allowed", set(getattr(model, "predictions", {})))
    return simulate_all(series, model, start, end, config, decision_times=schedule)


def evaluate_hybrid(paths, cache="artifacts/hybrid/forecasts.jsonl", stride=48, context=100,
                    output="reports/hybrid-comparison.json",
                    model_output="src/trade_harness/assets/hybrid_model.json", folds=3,
                    forecaster=None, experiment_trials=None):
    if not 3 <= folds <= 6:
        raise ValueError("Use three to six walk-forward folds")
    from .risk import RiskConfig

    data, series, contract, forecasts = build_forecast_dataset(
        paths, cache, stride, context, forecaster,
    )
    market_end = len(MODEL_FEATURES)
    strategy_end = market_end + len(STRATEGY_FEATURES)
    definitions = {
        "market_only": (slice(0, market_end), MODEL_FEATURES),
        "market_boosted_reference": (slice(0, market_end), MODEL_FEATURES),
        "strategy_only": (slice(market_end, strategy_end), STRATEGY_FEATURES),
        "market_strategy": (slice(0, strategy_end), MODEL_FEATURES + STRATEGY_FEATURES),
        "forecast_only": (slice(strategy_end, None), FORECAST_FEATURES),
        "combined": (slice(None), HYBRID_FEATURES),
    }
    plans = {}
    if experiment_trials is not None:
        if not 2 <= len(experiment_trials) <= 16:
            raise ValueError("Use two to sixteen predeclared trials")
        for trial in experiment_trials:
            if set(trial) != {"name", "features", "parameters"}:
                raise ValueError("Trial fields must be name, features, parameters")
            name = trial["name"]
            if not isinstance(name, str) or not name.isidentifier() or name in plans:
                raise ValueError("Trial names must be unique identifiers")
            plans[name] = {"features": FeatureConfig.model_validate(trial["features"]).model_dump(),
                           "parameters": FitParameters.model_validate(trial["parameters"]).model_dump()}
        definitions = {name: (slice(None), feature_names(plan["features"]))
                       for name, plan in plans.items()}
    candidates = {}
    for name, (columns, names) in definitions.items():
        view = data.subset(np.ones(len(data.y), dtype=bool))
        view.x = transform_features(data.x, plans[name]["features"]) if name in plans else data.x[:, columns]
        candidates[name] = (view, names)
    unique = np.unique(data.timestamp)
    development_end = int(len(unique) * 0.8)
    test_start = int(unique[development_end])
    boundaries = np.linspace(int(development_end * 0.4), development_end, folds + 1, dtype=int)
    config = RiskConfig(allow_research=True)
    stressed = config.model_copy(update={"fee_bps": config.fee_bps * 2,
                                        "slippage_bps": config.slippage_bps * 2})
    schedule = set(forecasts)
    momentum = ScheduledModel(BaselineModel(), schedule)
    strategy_rule = ScheduledModel(StrategyModel(), schedule)
    forecast_rule = ScheduledModel(TimesFMModel(), schedule, forecasts)
    report = {
        "mode": "decision_only", "real_execution_enabled": False,
        "forecast_contract": contract, "strategy_version": STRATEGY_VERSION,
        "samples": len(data.y), "assets": sorted(series), "test_start": test_start,
        "risk_config": config.model_dump(), "folds": [], "candidates": {}, "final_test": {},
        "selection_rule": "Lowest mean validation log loss across five predeclared logistic ablations; final test excluded. Combined artifact shipped independently and remains research_only.",
        "experiment": {"learner": "logistic_and_ridge",
                       "reference": "Current boosted architecture retrained on the identical sampled cohort", "candidate_features":
                       {name: names for name, (_, names) in candidates.items()}},
        "limitations": [
            "Historical research only; TimesFM 3.0 use restrictions apply to forecast outputs.",
            "Pretraining overlap is unknown; this is not untouched prospective evidence.",
            "Previously examined market history; no profitable edge claim can follow this experiment alone.",
            "Sparse decisions every stride candles; every intervening bar is replayed for exits.",
            "Comparisons use identical sampled decision times; results do not measure hourly operation.",
            "One account per asset, fixed simulated costs, no exchange orders.",
            "Only three folds; bootstrap intervals are descriptive.",
            "Decision history and account state are not fitted features in this first combined model.",
        ],
    }
    if plans:
        report["trial_plan"] = plans
        report["selection_rule"] = "Lowest mean purged validation log loss; final test compares only the locked winner and untuned base. No automatic activation or consensus-agreement objective."
        report["limitations"].append("Multiple parameter trials increase selection bias; fresh confirmation is required.")
        plan_path = Path(str(output) + ".plan.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan = {"trials": plans, "test_start": test_start, "forecast_contract": contract,
                "selection_rule": report["selection_rule"]}
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Experiment plan changed; use a new output path")
        plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    for number, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        start, end = int(unique[left]), int(unique[right])
        calibration_start = int(unique[int(left * 0.8)])
        train_mask = (data.timestamp < calibration_start) & (data.observed_at < calibration_start)
        cal_mask = (data.timestamp >= calibration_start) & (data.observed_at < start)
        val_mask = (data.timestamp >= start) & (data.observed_at < end)
        prior = np.bincount(data.y[train_mask], minlength=3) / train_mask.sum()
        validation = data.subset(val_mask)
        baseline = calibration_metrics(np.tile(prior, (len(validation.y), 1)), validation.y)
        record = {
            "fold": number, "start": start, "end": end, "calibration_first": calibration_start,
            "fit_labels_last": int(data.observed_at[train_mask].max()),
            "calibration_labels_last": int(data.observed_at[cal_mask].max()),
            "validation_labels_last": int(data.observed_at[val_mask].max()),
            "training_samples": int(train_mask.sum()), "calibration_samples": int(cal_mask.sum()),
            "validation_samples": int(val_mask.sum()), "train_prior_baseline": baseline,
            "prior_metrics_per_asset": {
                s: calibration_metrics(np.tile(prior, (int((validation.symbol == s).sum()), 1)),
                                       validation.y[validation.symbol == s]) for s in series},
            "momentum_baseline": _simulate(series, momentum, start, end, config), "models": {},
            "strategy_rule": _simulate(series, strategy_rule, start, end, config),
            "forecast_rule": _simulate(series, forecast_rule, start, end, config),
        }
        for name, (candidate, names) in candidates.items():
            train, calibration, val = (candidate.subset(m) for m in (train_mask, cal_mask, val_mask))
            artifact = fit(train, calibration,
                           "boosted" if name == "market_boosted_reference" else "logistic", names,
                           parameters=plans[name]["parameters"] if name in plans else None)
            p, expected = predict_batch(artifact, val.x)
            model = CachedModel(artifact, val)
            normal = _simulate(series, model, start, end, config)
            stress = _simulate(series, model, start, end, stressed)
            record["models"][name] = {
                "metrics": calibration_metrics(p, val.y),
                "return_mae": float(np.mean(abs(expected - val.returns))),
                "zero_return_mae": float(np.mean(abs(val.returns))),
                "per_asset": normal, "cost_stress_per_asset": stress,
                "metrics_per_asset": {s: calibration_metrics(p[val.symbol == s], val.y[val.symbol == s])
                                      for s in series},
            }
            print(json.dumps({"fold": number, "candidate": name,
                              "net_return": _mean_return(normal)}), flush=True)
        report["folds"].append(record)
    for name in candidates:
        evidence = summarize_folds(report["folds"], name, data.timeframe, data.horizon, test_start)
        report["candidates"][name] = {
            "mean_log_loss": float(np.mean([f["models"][name]["metrics"]["log_loss"]
                                           for f in report["folds"]])),
            "trading_validation": evidence.model_dump(),
            "validation_per_asset": {s: summarize_folds(report["folds"], name, data.timeframe,
                                                       data.horizon, test_start, s).model_dump()
                                     for s in series},
        }
    report["selected_candidate"] = min((n for n in candidates if n != "market_boosted_reference"),
                                       key=lambda n: report["candidates"][n]["mean_log_loss"])
    if plans:
        scores = {name: np.asarray([f["models"][name]["metrics"]["log_loss"]
                                   for f in report["folds"]]) for name in plans}
        winner = report["selected_candidate"]
        # Paired fold differences show when a tiny winning score is within validation variation.
        report["parameter_stability"] = {
            "phase": "validation_only", "trial_count": len(plans),
            "winner": winner, "comparisons": {
                name: {"mean_log_loss_gap": float(np.mean(values - scores[winner])),
                       "paired_fold_gap_std": float(np.std(values - scores[winner])),
                       "recipe": plans[name]["features"]["recipe"],
                       "parameters": plans[name]["parameters"]}
                for name, values in scores.items()},
            "interpretation": "Inspect neighboring settings and fold variation; no significance or edge claim. Fresh locked confirmation accounts for the complete search.",
        }
    calibration_start = int(unique[int(development_end * 0.8)])
    train_mask = (data.timestamp < calibration_start) & (data.observed_at < calibration_start)
    cal_mask = (data.timestamp >= calibration_start) & (data.observed_at < test_start)
    test_mask = data.timestamp >= test_start
    final_artifacts = {}
    final_names = set(candidates)
    if plans:
        final_names = {report["selected_candidate"]}
        final_names.update(name for name, plan in plans.items()
                           if plan["features"]["recipe"] == "base" and
                           plan["parameters"] == FitParameters().model_dump())
    for name, (candidate, names) in candidates.items():
        if name not in final_names:
            continue
        training, calibration, test = (candidate.subset(m) for m in (train_mask, cal_mask, test_mask))
        artifact = fit(training, calibration,
                       "boosted" if name == "market_boosted_reference" else "logistic", names,
                           parameters=plans[name]["parameters"] if name in plans else None)
        artifact.update({"forecast_contract": contract, "strategy_version": STRATEGY_VERSION,
                         "validation_status": "research_only"})
        if name in plans:
            artifact["feature_config"] = plans[name]["features"]
        artifact["model_version"] = digest(artifact)[:16]
        for key in ("trading_validation", "validation_per_asset"):
            artifact[key] = report["candidates"][name][key]
        artifact["trading_validation"]["model_version"] = artifact["model_version"]
        for evidence in artifact["validation_per_asset"].values():
            evidence["model_version"] = artifact["model_version"]
        p, expected = predict_batch(artifact, test.x)
        final_artifacts[name] = artifact
        report["final_test"][name] = {
            "metrics": calibration_metrics(p, test.y),
            "return_mae": float(np.mean(abs(expected - test.returns))),
            "per_asset": _simulate(series, CachedModel(artifact, test), test_start,
                                   int(unique[-1] + 1), config),
        }
    report["final_test"]["momentum_rule"] = _simulate(series, momentum, test_start,
                                                    int(unique[-1] + 1), config)
    report["final_test"]["strategy_rule"] = _simulate(series, strategy_rule, test_start,
                                                    int(unique[-1] + 1), config)
    report["final_test"]["forecast_rule"] = _simulate(series, forecast_rule, test_start,
                                                    int(unique[-1] + 1), config)
    output_candidate = report["selected_candidate"] if plans else "combined"
    report["output_candidate"] = output_candidate
    combined = final_artifacts[output_candidate]
    target = Path(model_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(combined, allow_nan=False, separators=(",", ":")))
    report["combined_model_version"] = combined["model_version"]
    report["validation_status"] = "research_only"
    # Exercise the exact portable combined predictor after artifact generation.
    model = HybridModel(artifact=combined, forecaster=forecaster)
    for market in series.values():
        result = model.predict(window(market, len(market.ohlc) - 1, context), [])
        assert result.forecast_evidence is not None
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, allow_nan=False, indent=2))
    print(json.dumps({"selected": report["selected_candidate"], "report": str(dest),
                      "model": str(target), "validation_status": "research_only"}), flush=True)
    return report
