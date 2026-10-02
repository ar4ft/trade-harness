"""Decision-band reliability, causal regime summaries, and training-only signal checks."""

import numpy as np
from scipy.stats import spearmanr

from .evaluation import calibration_metrics


def reliability(scores, correct, bins=5):
    scores, correct = np.asarray(scores), np.asarray(correct)
    table, ece = [], 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        n = int(mask.sum())
        if n:
            mean, rate = float(scores[mask].mean()), float(correct[mask].mean())
            ece += n / max(len(scores), 1) * abs(mean - rate)
            table.append({"lower": float(lower), "upper": float(upper), "samples": n,
                          "mean_score": mean, "observed_rate": rate})
    return {"samples": len(scores), "ece": float(ece) if len(scores) else None,
            "minimum_populated_bin": min((b["samples"] for b in table), default=0), "bins": table}


def probability_diagnostics(probabilities, truth):
    probabilities, truth = np.asarray(probabilities), np.asarray(truth, dtype=int)
    if not len(truth):
        return {"samples": 0, "directional": reliability([], []), "by_class": {}}
    chosen = probabilities.argmax(axis=1)
    directional = chosen != 2
    return {**calibration_metrics(probabilities, truth),
            "directional": reliability(probabilities.max(axis=1)[directional],
                                       (chosen == truth)[directional]),
            "by_class": {action: reliability(probabilities[:, i], truth == i)
                         for i, action in enumerate(("BUY", "SELL", "HOLD"))}}


def regime(features):
    trend = ("uptrend" if features["return_20"] > 0.01 else
             "downtrend" if features["return_20"] < -0.01 else "range")
    vol = "high_volatility" if features["volatility_20"] > 0.015 else "normal_volatility"
    return f"{trend}/{vol}"


def _correlation(x, y, rank=False):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(spearmanr(x, y).statistic if rank else np.corrcoef(x, y)[0, 1])


def feature_diagnostics(rows):
    # Paired position examples are one market observation, never two independent samples.
    rows = [r for r in rows if r["position"] == "flat" and r["phase"] == "train"]
    names = sorted(rows[0]["evidence"]["features"]) if rows else []
    result = {}
    groups = {"pooled": rows}
    for asset in sorted({r["symbol"] for r in rows}):
        groups[asset] = [r for r in rows if r["symbol"] == asset]
    for key in sorted({regime(r["evidence"]["features"]) for r in rows}):
        groups[key] = [r for r in rows if regime(r["evidence"]["features"]) == key]
    for key, group in groups.items():
        returns = np.asarray([r["outcome"]["next_open_return"] for r in group])
        shuffled = np.random.default_rng(42).permutation(returns)
        result[key] = {"samples": len(group), "features": {}}
        for name in names:
            signal = np.asarray([r["evidence"]["features"][name] for r in group])
            result[key]["features"][name] = {
                "pearson_ic": _correlation(signal, returns),
                "rank_ic": _correlation(signal, returns, True),
                "shuffled_ic": _correlation(signal, shuffled),
            }
    return {"phase": "train_only", "groups": result,
            "limitations": ["Correlation is not causation or net trading edge.",
                            "Shuffle is an alignment diagnostic, not a dependence-aware significance test.",
                            "Regime cutoffs are fixed beforehand; no test-period feature selection."]}


def decision_diagnostics(rows):
    """Rows contain actual proposals/actions, including abstentions and guarded decisions."""
    groups = {"pooled": rows}
    for symbol in sorted({r["symbol"] for r in rows}):
        groups[symbol] = [r for r in rows if r["symbol"] == symbol]
    for key in sorted({r["regime"] for r in rows}):
        groups[key] = [r for r in rows if r["regime"] == key]
    output = {}
    for key, group in groups.items():
        selected = [r for r in group if r["proposal"]["action"] != "HOLD"]
        probability_rows = [r for r in group if r["proposal"].get("probabilities")]
        scores = [r["proposal"]["confidence"] for r in selected]
        correct = [r["proposal"]["action"] == r["outcome"]["direction_label"] for r in selected]
        counts = {a: sum(r["decision"]["action"] == a for r in group) for a in ("BUY", "SELL", "HOLD")}
        output[key] = {"samples": len(group), "final_actions": counts,
                       "decision_coverage": (counts["BUY"] + counts["SELL"]) / len(group) if group else 0,
                       "proposal_directional": reliability(scores, correct),
                       "forecast_mae": float(np.mean([
                           abs(r["proposal"]["forecast"]["expected_return"] -
                               r["outcome"]["next_open_return"]) for r in group])) if group else None,
                       "decision_interval_coverage": float(np.mean([
                           r["proposal"]["forecast_interval"][0] <= r["outcome"]["next_open_return"] <=
                           r["proposal"]["forecast_interval"][1]
                           for r in group if r["proposal"].get("forecast_interval")]))
                       if any(r["proposal"].get("forecast_interval") for r in group) else None}
        if probability_rows:
            output[key]["probability_quality"] = probability_diagnostics(
                np.asarray([[r["proposal"]["probabilities"][a] for a in ("BUY", "SELL", "HOLD")]
                            for r in probability_rows]),
                np.asarray([("BUY", "SELL", "HOLD").index(r["outcome"]["direction_label"])
                            for r in probability_rows]))
    return output


def paired_block_interval(candidate, baseline, block_days=7, samples=2000):
    """Paired contiguous daily blocks; assets share the same sampled time indices."""
    candidate, baseline = np.asarray(candidate, float), np.asarray(baseline, float)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or block_days < 1:
        raise ValueError("Bootstrap requires aligned daily return arrays and a positive block length")
    n = len(candidate)
    if not n:
        return {"interval": None, "effective_blocks": 0, "block_days": block_days}
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("Bootstrap returns must be finite")
    length = min(block_days, n)
    rng = np.random.default_rng(42)
    count = int(np.ceil(n / length))
    starts = rng.integers(0, n - length + 1, size=(samples, count))
    indices = (starts[..., None] + np.arange(length)).reshape(samples, -1)[:, :n]
    means = (candidate - baseline)[indices].mean(axis=1)
    return {"interval": np.quantile(means, [0.025, 0.975]).tolist(),
            "effective_blocks": n // length, "block_days": block_days,
            "samples": samples, "seed": 42, "daily_samples": n,
            "method": "paired_moving_time_block_bootstrap"}
