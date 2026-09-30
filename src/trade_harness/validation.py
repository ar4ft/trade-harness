"""Evidence gates for decision research; passing them never enables order execution."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TradingValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    policy_version: Literal["walk-forward-edge-v1"] = "walk-forward-edge-v1"
    method: Literal["purged_walk_forward_paper_replay"] = "purged_walk_forward_paper_replay"
    symbol: str | None = None
    timeframe: str | None = None
    horizon: int | None = Field(default=None, ge=1)
    model_version: str | None = None
    fold_count: int = Field(default=0, ge=0)
    closed_trades: int = Field(default=0, ge=0)
    positive_folds: int = Field(default=0, ge=0)
    mean_net_return: float | None = None
    mean_momentum_return: float | None = None
    mean_cost_stressed_return: float | None = None
    mean_return_interval: list[float] | None = None
    brier_improvement: float | None = None
    purged_boundaries: bool = False
    evaluated_until: int | None = Field(default=None, ge=0)
    final_test_start: int | None = Field(default=None, ge=0)
    gates: dict[str, bool] = Field(default_factory=dict)
    failed_gates: list[str] = Field(default_factory=list)
    positive_edge: bool = False
    status: Literal["validated", "research_only", "unknown"] = "unknown"

    @model_validator(mode="after")
    def recompute_gates(self):
        interval = self.mean_return_interval
        if interval is not None and (len(interval) != 2 or interval[0] > interval[1]):
            raise ValueError("Mean return interval must be ordered [lower, upper]")
        if self.positive_folds > self.fold_count:
            raise ValueError("Positive folds exceed evaluated folds")
        # Supplied status/boolean claims cannot override measured evidence.
        self.gates = {
            "at_least_three_folds": self.fold_count >= 3,
            "positive_net_return": self.mean_net_return is not None and self.mean_net_return > 0,
            "at_least_20_closed_trades": self.closed_trades >= 20,
            "two_positive_folds": self.positive_folds >= 2,
            "brier_better_than_prior": self.brier_improvement is not None
            and self.brier_improvement > 0,
            "beats_momentum_after_costs": (
                self.mean_net_return is not None
                and self.mean_momentum_return is not None
                and self.mean_net_return > self.mean_momentum_return
            ),
            "positive_with_double_costs": (
                self.mean_cost_stressed_return is not None and self.mean_cost_stressed_return > 0
            ),
            "positive_interval_lower_bound": interval is not None and interval[0] > 0,
            "purged_chronological_boundaries": self.purged_boundaries,
            "final_test_excluded": (
                self.evaluated_until is not None
                and self.final_test_start is not None
                and self.evaluated_until < self.final_test_start
            ),
        }
        self.failed_gates = [key for key, passed in self.gates.items() if not passed]
        self.positive_edge = not self.failed_gates
        self.status = (
            "unknown"
            if self.mean_net_return is None
            else "validated"
            if self.positive_edge
            else "research_only"
        )
        return self

    def applies_to(self, market, model_version):
        return (
            self.symbol == market.symbol
            and self.timeframe == market.timeframe
            and self.horizon == market.horizon
            and self.model_version == model_version
            and self.evaluated_until is not None
            and self.evaluated_until < market.timestamps[-1]
        )


def summarize_folds(records, kind, timeframe, horizon, test_start, symbol=None):
    """Use validation folds only; never read held-out test performance or headline claims."""
    returns, momentum, stress, improvements = [], [], [], []
    closed = 0
    purged = bool(records)
    previous_end = None
    for fold in records:
        entry = fold["models"][kind]
        purged = purged and (
            fold["fit_labels_last"] < fold["calibration_first"] < fold["start"]
            and fold.get("calibration_labels_last", fold["start"]) < fold["start"]
            and fold.get("validation_labels_last", fold["end"]) < fold["end"]
            and fold["end"] <= test_start
            and (previous_end is None or previous_end <= fold["start"])
        )
        previous_end = fold["end"]
        symbols = [symbol] if symbol else sorted(entry["per_asset"])
        returns.append(float(np.mean([entry["per_asset"][s]["return"] for s in symbols])))
        momentum.append(float(np.mean([fold["momentum_baseline"][s]["return"] for s in symbols])))
        if "cost_stress_per_asset" in entry:
            stress.append(
                float(np.mean([entry["cost_stress_per_asset"][s]["return"] for s in symbols]))
            )
        closed += sum(entry["per_asset"][s]["closed_trades"] for s in symbols)
        baseline = fold["train_prior_baseline"]
        metrics = entry["metrics"]
        if symbol:
            baseline = fold.get("prior_metrics_per_asset", {}).get(symbol)
            metrics = entry.get("metrics_per_asset", {}).get(symbol)
        if baseline is not None and metrics is not None:
            improvements.append(baseline["brier"] - metrics["brier"])
    interval = None
    if len(returns) >= 3:
        # Descriptive block bootstrap over non-overlapping folds, preserving each fold as a block.
        bootstrap = (
            np.random.default_rng(42).choice(returns, size=(10000, len(returns))).mean(axis=1)
        )
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return TradingValidation(
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        fold_count=len(returns),
        closed_trades=closed,
        positive_folds=sum(value > 0 for value in returns),
        mean_net_return=float(np.mean(returns)) if returns else None,
        mean_momentum_return=float(np.mean(momentum)) if momentum else None,
        mean_cost_stressed_return=float(np.mean(stress))
        if len(stress) == len(returns) and stress
        else None,
        mean_return_interval=interval,
        brier_improvement=float(np.mean(improvements))
        if len(improvements) == len(returns) and improvements
        else None,
        purged_boundaries=purged,
        evaluated_until=max(f["end"] - 1 for f in records) if records else None,
        final_test_start=test_start,
    )


def validation_report(model, symbol=None):
    artifact = getattr(model, "artifact", {})
    symbols = [symbol] if symbol is not None else artifact.get("symbols", [])
    validator = getattr(model, "validation_for", None)
    per_asset = {}
    for asset in symbols:
        evidence = (
            TradingValidation.model_validate(validator(asset))
            if callable(validator)
            else TradingValidation(symbol=asset)
        )
        per_asset[asset] = evidence.model_dump()
    global_evidence = TradingValidation.model_validate(artifact.get("trading_validation", {}))
    return {
        "mode": "decision_only",
        "real_execution_enabled": False,
        "backend": model.name,
        "model_version": getattr(model, "version", None),
        "validation": global_evidence.model_dump(),
        "per_asset": per_asset,
        "interpretation": "Evidence for a fixed simulated decision policy; passing does not guarantee future profitability or enable orders.",
    }
