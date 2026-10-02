"""Timestamped, purged shared-evidence examples with explicit execution-aware targets."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .feature_engineering import feature_names, transform_features
from .hybrid_training import build_forecast_dataset, digest
from .learning import window
from .local_language import REVIEW_CONTRACT, review_prompt
from .orchestrator import shared_evidence
from .risk import RiskConfig
from .schemas import ForecastEvidence, MarketInput
from .storage import timeframe_ms
from .strategies import strategy_signals

LABEL_CONTRACT = "next-open-horizon-close-cost-aware-long-cash-v1"
PHASES = ("train", "calibration", "validation", "test")


def target_action(realized, position, config):
    fee, slip = config.fee_bps / 10000, config.slippage_bps / 10000
    net = (1 + realized) * (1 - fee) * (1 - slip) / ((1 + fee) * (1 + slip)) - 1
    if position == "flat":
        return ("BUY" if net > config.min_net_edge else "HOLD"), net
    return ("SELL" if realized < -config.roundtrip_cost - config.min_net_edge else "HOLD"), net


def snapshot_evidence(view, forecast, vector, names):
    forecast = forecast.model_copy(deep=True)
    forecast.forecast_evidence = ForecastEvidence(
        source=forecast.forecast.method, model_version=forecast.model_version,
        as_of=view.timestamps[-1], horizon=view.horizon,
        expected_return=forecast.forecast.expected_return, return_interval=forecast.forecast_interval,
    )
    forecast.strategy_signals = strategy_signals(view)
    forecast.feature_evidence = {n: float(v) for n, v in zip(names, vector)}
    return shared_evidence(view, forecast)


def build_examples(data, series, forecasts, feature_config=None, risk_config=None):
    """Global 50/10/20/20 split; crossing labels purged; paired flat/long snapshots."""
    risk = risk_config or RiskConfig()
    unique = np.unique(data.timestamp)
    boundaries = [int(unique[int(len(unique) * fraction)]) for fraction in (0.5, 0.6, 0.8)]
    names = feature_names(feature_config or {})
    vectors = transform_features(data.x, feature_config or {})
    indices = {symbol: {t: i for i, t in enumerate(m.timestamps)} for symbol, m in series.items()}
    previous = {}
    rows, purged = [], 0
    for k, (timestamp, symbol) in enumerate(zip(data.timestamp, data.symbol)):
        as_of = int(timestamp)
        phase_index = sum(as_of >= cut for cut in boundaries)
        if phase_index < 3 and data.observed_at[k] >= boundaries[phase_index]:
            purged += 1
            continue
        view = window(series[symbol], indices[symbol][as_of], 100)
        history = []
        prior = previous.get(symbol)
        if prior and prior["feedback"]["observed_at"] < as_of:
            history = [prior]
        for position in ("flat", "long"):
            market = view.model_copy(update={"position": position})
            evidence = snapshot_evidence(market, forecasts[(symbol, as_of)], vectors[k], names)
            label, net = target_action(float(data.returns[k]), position, risk)
            rows.append({
                "schema_version": 1, "phase": PHASES[phase_index], "symbol": symbol,
                "as_of": as_of, "position": position, "market": market.model_dump(),
                "history": history, "evidence": evidence, "evidence_sha256": digest(evidence),
                "prompt": review_prompt(market, history, evidence), "label": label,
                "outcome": {"observed_at": int(data.observed_at[k]),
                            "next_open_return": float(data.returns[k]), "net_long_return": net,
                            "direction_label": ("BUY" if data.returns[k] > 0.003 else
                                                "SELL" if data.returns[k] < -0.003 else "HOLD")},
            })
        # Historical fixed-strategy feedback is knowable, not a future reviewer answer.
        signal = strategy_signals(view)[0]
        previous[symbol] = {"as_of": as_of, "symbol": symbol, "timeframe": view.timeframe,
                            "action": signal.action,
                            "feedback": {"observed_at": int(data.observed_at[k]),
                                         "realized_return": float(data.returns[k])}}
    return rows, {"boundaries": dict(zip(PHASES[1:], boundaries)),
                  "purged_snapshots": purged, "feature_names": names}


def export_research(paths, cache, output, stride=48, context=100, feature_config=None,
                    forecaster=None):
    if context != 100:
        raise ValueError("Reviewer training currently uses the runtime 100-candle context")
    data, series, forecast_contract, forecasts = build_forecast_dataset(
        paths, cache, stride, context, forecaster,
    )
    config = RiskConfig(max_holding_candles=data.horizon, cooldown_candles=data.horizon)
    rows, split = build_examples(data, series, forecasts, feature_config, config)
    destination = Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise ValueError("Research dataset exists; choose a new immutable output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    manifest = {
        "schema_version": 1, "prompt_contract": REVIEW_CONTRACT, "label_contract": LABEL_CONTRACT,
        "source": "historical_automatic_labels", "prospective": False,
        "dataset_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "forecast_contract": forecast_contract, "risk_config": config.model_dump(),
        "feature_config": feature_config or {}, "timeframe": data.timeframe,
        "horizon": data.horizon, "symbols": sorted(series), **split,
        "partitions": {phase: {"examples": sum(r["phase"] == phase for r in rows),
                               "labels": dict(Counter(r["label"] for r in rows if r["phase"] == phase))}
                       for phase in PHASES},
        "limitations": [
            "Future-return labels are automatic supervision, not expert judgments.",
            "Labels estimate a horizon trade, not path-dependent stop/target profitability.",
            "Flat and long views of one timestamp always share a partition.",
            "Previous feedback follows a fixed trend hypothesis, not reviewer self-play.",
            "Historical evidence; unknown foundation pretraining overlap.",
        ],
    }
    destination.write_text(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_research(path):
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Research dataset digest mismatch")
    if manifest["prompt_contract"] != REVIEW_CONTRACT or manifest["label_contract"] != LABEL_CONTRACT:
        raise ValueError("Unsupported research dataset contract")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        market = MarketInput.model_validate(row["market"])
        key = (row["symbol"], row["as_of"], row["position"])
        if key in seen:
            raise ValueError("Duplicate research example")
        seen.add(key)
        phase = sum(row["as_of"] >= manifest["boundaries"][p] for p in PHASES[1:])
        if row["phase"] != PHASES[phase] or row["outcome"]["observed_at"] <= row["as_of"]:
            raise ValueError("Research partition or outcome timing mismatch")
        if phase < 3 and row["outcome"]["observed_at"] >= manifest["boundaries"][PHASES[phase + 1]]:
            raise ValueError("Research label crosses partition boundary")
        if row["evidence_sha256"] != digest(row["evidence"]) or row["prompt"] != review_prompt(
                market, row["history"], row["evidence"]):
            raise ValueError("Training prompt differs from the runtime renderer")
        if (market.symbol, market.timestamps[-1], market.position) != key:
            raise ValueError("Research market identity mismatch")
        if sorted(row["evidence"]["features"]) != sorted(manifest["feature_names"]):
            raise ValueError("Research feature names differ from the manifest")
        if row["outcome"]["observed_at"] != row["as_of"] + timeframe_ms(market.timeframe) * market.horizon:
            raise ValueError("Research outcome differs from the declared horizon")
        expected, net = target_action(row["outcome"]["next_open_return"], market.position,
                                     RiskConfig.model_validate(manifest["risk_config"]))
        if row["label"] != expected or not np.isclose(row["outcome"]["net_long_return"], net):
            raise ValueError("Research target differs from the cost-aware label contract")
    if any(not any(r["phase"] == phase for r in rows) for phase in PHASES):
        raise ValueError("All four chronological research partitions must contain examples")
    return rows, manifest
