"""Purged chronological model comparison; final test cannot influence selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from .learning import ACTIONS, DecisionModel, fit, load_dataset, load_series, predict_batch, window
from .models import BaselineModel
from .paper import PaperEngine
from .risk import RiskConfig
from .schemas import Forecast, Proposal
from .storage import Store


class CachedModel:
    uses_history = False
    """Precomputed causal-feature predictions accelerate replay, without accessing labels."""

    name = "replay-cached-market-model"

    def __init__(self, artifact, data):
        self.version = artifact.get("model_version", "evaluation")
        self.trained_until = artifact["trained_until"]
        probability, expected = predict_batch(artifact, data.x)
        self.predictions = {
            (symbol, int(t)): (p, e)
            for symbol, t, p, e in zip(data.symbol, data.timestamp, probability, expected)
        }
        self.kind = artifact["kind"]

    def predict(self, market, history):
        if market.timestamps[-1] <= self.trained_until:
            raise ValueError("Replay prediction overlaps fit/calibration period")
        found = self.predictions.get((market.symbol, market.timestamps[-1]))
        if found is None:
            return Proposal(
                action="HOLD",
                confidence=0,
                rationale="No labeled feature window",
                forecast=Forecast(
                    horizon=market.horizon,
                    expected_return=0,
                    projected_close=market.ohlc[-1][3],
                    method="window_boundary",
                ),
            )
        probability, expected = found
        return Proposal(
            action=ACTIONS[int(probability.argmax())],
            confidence=float(probability.max()),
            probabilities={a: float(p) for a, p in zip(ACTIONS, probability)},
            probability_calibration="chronological temperature calibration",
            validation_status="research_only",
            rationale=f"{self.kind} causal feature replay",
            forecast=Forecast(
                horizon=market.horizon,
                expected_return=float(expected),
                projected_close=market.ohlc[-1][3] * (1 + expected),
                method="return_regression",
            ),
        )


def calibration_metrics(probability, y):
    truth = np.eye(3)[y]
    selected = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    accuracy = selected == y
    ece = 0.0
    bins = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        count = int(mask.sum())
        if count:
            actual = float(accuracy[mask].mean())
            predicted = float(confidence[mask].mean())
            ece += count / len(y) * abs(actual - predicted)
            bins.append(
                {
                    "lower": float(low),
                    "upper": float(high),
                    "samples": count,
                    "mean_confidence": predicted,
                    "observed_accuracy": actual,
                }
            )
    return {
        "samples": len(y),
        "accuracy": float(accuracy.mean()),
        "log_loss": float(log_loss(y, probability, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((probability - truth) ** 2, axis=1))),
        "expected_calibration_error": float(ece),
        "confidence_bins": bins,
    }


def _simulate(series, model, start, end, config):
    reports = {}
    for symbol, market in series.items():
        store = Store(":memory:")
        engine = PaperEngine(
            model,
            store,
            run_id=f"eval-{symbol}",
            symbol=symbol,
            timeframe=market.timeframe,
            config=config,
        )
        indices = [i for i, t in enumerate(market.timestamps) if start <= t < end and i >= 20]
        for i in indices:
            engine.replay(window(market, i), market.ohlc[i])
        summary = engine.summary()
        curve = summary.pop("equity_curve")
        days = {int(t) // 86400000: value for t, value in curve}
        values = np.array(list(days.values()))
        daily = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])
        summary["annualized_daily_sharpe"] = (
            float(np.mean(daily) / np.std(daily) * np.sqrt(365))
            if len(daily) > 1 and np.std(daily) > 1e-12
            else None
        )
        summary["trade_events"] = [
            r for r in reversed(store.events(engine.run_id, 100000)) if r["kind"] == "fill"
        ]
        if indices:
            first, last = indices[0], indices[-1]
            entry = market.ohlc[first][0] * (1 + config.slippage_bps / 10000)
            gross = market.ohlc[last][3] / entry - 1
            passive = (1 + gross) * (1 - config.fee_bps / 10000) / (1 + config.fee_bps / 10000) - 1
            summary["cash_baseline_return"] = 0.0
            summary["buy_hold_full_capital_return"] = float(passive)
            summary["buy_hold_capped_exposure_return"] = float(
                passive * config.max_position_fraction
            )
        reports[symbol] = summary
        store.db.close()
    return reports


def _mean_return(reports):
    return float(np.mean([r["return"] for r in reports.values()]))


def evaluate(
    paths,
    output="reports/walk-forward.json",
    model_output="src/trade_harness/assets/decision_model.json",
    folds=3,
):
    if folds < 2 or folds > 6:
        raise ValueError("Use two to six chronological folds")
    config = RiskConfig(allow_research=True)
    data = load_dataset(paths, config.roundtrip_cost)
    series, _ = load_series(paths)
    unique = np.unique(data.timestamp)
    development_end = int(len(unique) * 0.8)
    test_start = int(unique[development_end])
    boundaries = np.linspace(int(development_end * 0.4), development_end, folds + 1, dtype=int)
    report = {
        "dataset": data.datasets,
        "samples": len(data.y),
        "assets": sorted(series),
        "objective": "Long/cash, hourly bars, 3-candle return forecast, net outcome after costs",
        "risk_config": config.model_dump(),
        "test_start": test_start,
        "selection_rule": "Lowest mean walk-forward log loss; deployment requires positive mean net return, at least 20 closed trades, positive returns in >=2 folds, and Brier improvement over the train-prior baseline.",
        "folds": [],
        "candidates": {},
        "final_test": {},
        "limitations": [
            "Independent funded account per asset; not a shared portfolio",
            "One final market period; not proof of future profitability",
            "Fixed fee/slippage model",
            "OHLC stops are conservative when both stop and target touch",
            "Pending final orders are not filled after the test boundary",
        ],
    }
    performances = {kind: [] for kind in ("logistic", "boosted")}
    for number, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        start = int(unique[left])
        end = int(unique[right]) if right < len(unique) else int(unique[-1] + 1)
        calibration_start = int(unique[int(left * 0.8)])
        training = data.subset(
            (data.timestamp < calibration_start) & (data.observed_at < calibration_start)
        )
        calibration = data.subset(
            (data.timestamp >= calibration_start) & (data.observed_at < start)
        )
        validation = data.subset((data.timestamp >= start) & (data.observed_at < end))
        prior = np.bincount(training.y, minlength=3) / len(training.y)
        baseline = calibration_metrics(np.tile(prior, (len(validation.y), 1)), validation.y)
        record = {
            "fold": number,
            "start": start,
            "end": end,
            "training_samples": len(training.y),
            "calibration_samples": len(calibration.y),
            "validation_samples": len(validation.y),
            "fit_labels_last": int(training.observed_at.max()),
            "calibration_first": calibration_start,
            "train_prior_baseline": baseline,
            "models": {},
        }
        for kind in performances:
            artifact = fit(training, calibration, kind)
            p, expected = predict_batch(artifact, validation.x)
            metrics = calibration_metrics(p, validation.y)
            metrics["return_mae"] = float(np.mean(abs(expected - validation.returns)))
            metrics["zero_return_mae"] = float(np.mean(abs(validation.returns)))
            simulation = _simulate(series, CachedModel(artifact, validation), start, end, config)
            entry = {
                "metrics": metrics,
                "per_asset": simulation,
                "mean_account_return": _mean_return(simulation),
                "closed_trades": sum(r["closed_trades"] for r in simulation.values()),
            }
            record["models"][kind] = entry
            performances[kind].append((entry, baseline))
            print(
                json.dumps(
                    {
                        "fold": number,
                        "kind": kind,
                        "log_loss": metrics["log_loss"],
                        "net_return": entry["mean_account_return"],
                        "closed_trades": entry["closed_trades"],
                    }
                ),
                flush=True,
            )
        # Momentum follows the same position sizing, costs, stops, and confidence policy.
        record["momentum_baseline"] = _simulate(series, BaselineModel(), start, end, config)
        report["folds"].append(record)
    for kind, results in performances.items():
        logloss = float(np.mean([entry["metrics"]["log_loss"] for entry, _ in results]))
        net = float(np.mean([entry["mean_account_return"] for entry, _ in results]))
        closed = sum(entry["closed_trades"] for entry, _ in results)
        positive = sum(entry["mean_account_return"] > 0 for entry, _ in results)
        brier_improvement = float(
            np.mean([baseline["brier"] - entry["metrics"]["brier"] for entry, baseline in results])
        )
        gates = {
            "positive_net_return": net > 0,
            "at_least_20_closed_trades": closed >= 20,
            "two_positive_folds": positive >= 2,
            "brier_better_than_prior": brier_improvement > 0,
        }
        report["candidates"][kind] = {
            "mean_log_loss": logloss,
            "mean_net_return": net,
            "closed_trades": closed,
            "positive_folds": positive,
            "brier_improvement": brier_improvement,
            "deployment_gates": gates,
            "validation_status": "validated" if all(gates.values()) else "research_only",
        }
    selected = min(performances, key=lambda kind: report["candidates"][kind]["mean_log_loss"])
    report["selected_model"] = selected
    calibration_start = int(unique[int(development_end * 0.8)])
    training = data.subset(
        (data.timestamp < calibration_start) & (data.observed_at < calibration_start)
    )
    calibration = data.subset(
        (data.timestamp >= calibration_start) & (data.observed_at < test_start)
    )
    artifact = fit(training, calibration, selected)
    artifact["validation_status"] = report["candidates"][selected]["validation_status"]
    artifact["validation_summary"] = report["candidates"][selected]
    artifact["model_version"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True).encode()
    ).hexdigest()[:16]
    test = data.subset(data.timestamp >= test_start)
    # Selection/deployment criteria are locked above; the final test never chooses model or thresholds.
    p, expected = predict_batch(artifact, test.x)
    prior = np.bincount(training.y, minlength=3) / len(training.y)
    final = {
        "metrics": calibration_metrics(p, test.y),
        "prior_baseline": calibration_metrics(np.tile(prior, (len(test.y), 1)), test.y),
        "return_mae": float(np.mean(abs(expected - test.returns))),
        "zero_return_mae": float(np.mean(abs(test.returns))),
        "research_per_asset": _simulate(
            series, CachedModel(artifact, test), test_start, int(unique[-1] + 1), config
        ),
        "momentum_per_asset": _simulate(
            series, BaselineModel(), test_start, int(unique[-1] + 1), config
        ),
    }
    final["mean_research_account_return"] = _mean_return(final["research_per_asset"])
    final["mean_momentum_account_return"] = _mean_return(final["momentum_per_asset"])
    report["final_test"] = final
    target = Path(model_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, allow_nan=False, separators=(",", ":")))
    # Exercise the same portable predictor used in CLI/API.
    model = DecisionModel(str(target))
    for market in series.values():
        model.predict(window(market, len(market.ohlc) - 1), [])
    report["model_version"] = artifact["model_version"]
    report["validation_status"] = artifact["validation_status"]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, allow_nan=False, indent=2))
    print(
        json.dumps(
            {
                "selected": selected,
                "validation_status": artifact["validation_status"],
                "final_test_accuracy": final["metrics"]["accuracy"],
                "final_test_net_return": final["mean_research_account_return"],
                "report": str(path),
                "model": str(target),
            }
        ),
        flush=True,
    )
    return report
