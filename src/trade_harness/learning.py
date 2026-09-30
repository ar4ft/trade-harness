"""Portable, calibrated numerical decision models and causal multi-asset datasets."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import ensure_indicators
from .features import FEATURE_NAMES, INDICATOR_FEATURE_NAMES, features
from .schemas import Forecast, MarketInput, Proposal

ACTIONS = ["BUY", "SELL", "HOLD"]
MODEL_FEATURES = (
    FEATURE_NAMES
    + INDICATOR_FEATURE_NAMES
    + [
        "return_3",
        "return_10",
        "volatility_5",
        "body",
        "upper_wick",
        "lower_wick",
        "volume_delta",
        "rsi_delta_3",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]
)


def model_features(market: MarketInput) -> np.ndarray:
    market = ensure_indicators(market)
    close = np.array([r[3] for r in market.ohlc])
    recent = np.diff(close[-6:]) / close[-6:-1]
    opening, high, low, last = market.ohlc[-1]
    rsi = next(i.values for i in market.indicators if i.name == "rsi_14")
    now = datetime.fromtimestamp(market.timestamps[-1] / 1000, tz=timezone.utc)
    hour = (now.hour + now.minute / 60) / 24 * 2 * np.pi
    weekday = now.weekday() / 7 * 2 * np.pi
    extra = [
        last / close[-4] - 1,
        last / close[-11] - 1,
        recent.std(),
        (last - opening) / last,
        (high - max(opening, last)) / last,
        (min(opening, last) - low) / last,
        market.volume[-1] / max(market.volume[-2], 1e-12) - 1,
        ((rsi[-1] if rsi[-1] is not None else 50) - (rsi[-4] if rsi[-4] is not None else 50)) / 100,
        np.sin(hour),
        np.cos(hour),
        np.sin(weekday),
        np.cos(weekday),
    ]
    return np.concatenate([features(market, True), extra])


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    returns: np.ndarray
    timestamp: np.ndarray
    observed_at: np.ndarray
    symbol: np.ndarray
    datasets: list[dict]
    horizon: int
    timeframe: str

    def subset(self, mask):
        return Dataset(
            *(
                getattr(self, k)[mask]
                for k in ("x", "y", "returns", "timestamp", "observed_at", "symbol")
            ),
            self.datasets,
            self.horizon,
            self.timeframe,
        )


def load_series(paths: list[str]):
    """Validate each bounded source file, then join trusted histories internally."""
    grouped = {}
    manifest = []
    contract = None
    for name in sorted(paths):
        path = Path(name)
        market = MarketInput.model_validate_json(path.read_text())
        current = (market.timeframe, market.horizon)
        if contract is not None and current != contract:
            raise ValueError("All training files must share timeframe and horizon")
        contract = current
        grouped.setdefault(market.symbol, []).append(market)
        manifest.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "symbol": market.symbol,
                "candles": len(market.ohlc),
            }
        )
    joined = {}
    from .data import add_indicators
    from .storage import timeframe_ms

    for symbol, segments in grouped.items():
        segments.sort(key=lambda m: m.timestamps[0])
        timestamps = [t for m in segments for t in m.timestamps]
        if any(b - a != timeframe_ms(contract[0]) for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("Source histories overlap or contain missing intervals")
        # This internal history exceeds the HTTP input cap; every constituent file was validated.
        combined = MarketInput.model_construct(
            symbol=symbol,
            timeframe=contract[0],
            horizon=contract[1],
            timestamps=timestamps,
            ohlc=[r for m in segments for r in m.ohlc],
            volume=[v for m in segments for v in m.volume],
            indicators=[],
            position="flat",
        )
        joined[symbol] = add_indicators(combined)
    return joined, manifest


def load_dataset(paths: list[str], roundtrip_cost=0.003) -> Dataset:
    series, manifest = load_series(paths)
    rows = []
    contract = None
    for market in series.values():
        contract = (market.timeframe, market.horizon)
        for i in range(20, len(market.ohlc) - market.horizon):
            view = window(market, i)
            realized = market.ohlc[i + market.horizon][3] / market.ohlc[i + 1][0] - 1
            label = 0 if realized > roundtrip_cost else 1 if realized < -roundtrip_cost else 2
            rows.append(
                (
                    market.timestamps[i],
                    model_features(view),
                    label,
                    realized,
                    market.timestamps[i + market.horizon],
                    market.symbol,
                )
            )
    if not rows:
        raise ValueError("No labeled candles")
    rows.sort(key=lambda r: (r[0], r[5]))
    return Dataset(
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]),
        np.array([r[3] for r in rows]),
        np.array([r[0] for r in rows], dtype=np.int64),
        np.array([r[4] for r in rows], dtype=np.int64),
        np.array([r[5] for r in rows]),
        manifest,
        contract[1],
        contract[0],
    )


def window(market, index, length=100):
    start = max(0, index - length + 1)
    return market.model_copy(
        update={
            "ohlc": market.ohlc[start : index + 1],
            "volume": market.volume[start : index + 1],
            "timestamps": market.timestamps[start : index + 1],
            "indicators": [
                item.model_copy(update={"values": item.values[start : index + 1]})
                for item in market.indicators
            ],
        }
    )


def _trees(model):
    return [
        [
            [
                {
                    key: (
                        int(node[key])
                        if key in ("feature_idx", "left", "right", "is_leaf", "missing_go_to_left")
                        else float(node[key])
                    )
                    for key in (
                        "value",
                        "feature_idx",
                        "num_threshold",
                        "left",
                        "right",
                        "is_leaf",
                        "missing_go_to_left",
                    )
                }
                for node in tree.nodes
            ]
            for tree in stage
        ]
        for stage in model._predictors
    ]


def _tree_value(nodes, x):
    index = 0
    while not nodes[index]["is_leaf"]:
        node = nodes[index]
        value = x[node["feature_idx"]]
        go_left = node["missing_go_to_left"] if np.isnan(value) else value <= node["num_threshold"]
        index = node["left"] if go_left else node["right"]
    return nodes[index]["value"]


def raw_logits(artifact, x):
    classifier = artifact["classifier"]
    if classifier["kind"] == "logistic":
        scaled = (x - np.array(classifier["mean"])) / np.array(classifier["scale"])
        return np.array(classifier["coef"]) @ scaled + np.array(classifier["intercept"])
    result = np.array(classifier["baseline"], dtype=float)
    for stage in classifier["trees"]:
        result += np.array([_tree_value(tree, x) for tree in stage])
    return result


def expected_return(artifact, x):
    reg = artifact["regressor"]
    if reg["kind"] == "ridge":
        scaled = (x - np.array(reg["mean"])) / np.array(reg["scale"])
        return float(np.dot(scaled, reg["coef"]) + reg["intercept"])
    return float(reg["baseline"] + sum(_tree_value(stage[0], x) for stage in reg["trees"]))


def fit(train: Dataset, calibration: Dataset, kind="boosted") -> dict:
    if set(train.y.tolist()) != {0, 1, 2} or len(calibration.y) < 30:
        raise ValueError("Need all three training classes and >=30 calibration examples")
    with threadpool_limits(limits=4):
        if kind == "logistic":
            scaler = StandardScaler().fit(train.x)
            classifier = LogisticRegression(C=0.5, max_iter=500, random_state=42).fit(
                scaler.transform(train.x), train.y
            )
            reg = Ridge(alpha=10).fit(scaler.transform(train.x), train.returns)
            logits = classifier.decision_function(scaler.transform(calibration.x))
            forecast = reg.predict(scaler.transform(calibration.x))
            state = {
                "classifier": {
                    "kind": "logistic",
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": classifier.coef_.tolist(),
                    "intercept": classifier.intercept_.tolist(),
                },
                "regressor": {
                    "kind": "ridge",
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": reg.coef_.tolist(),
                    "intercept": float(reg.intercept_),
                },
            }
        elif kind == "boosted":
            kwargs = {
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "max_depth": 4,
                "min_samples_leaf": 60,
                "l2_regularization": 2,
                "learning_rate": 0.06,
                "early_stopping": False,
                "random_state": 42,
            }
            classifier = HistGradientBoostingClassifier(**kwargs).fit(train.x, train.y)
            reg = HistGradientBoostingRegressor(**kwargs).fit(train.x, train.returns)
            logits = classifier._raw_predict(calibration.x)
            forecast = reg.predict(calibration.x)
            state = {
                "classifier": {
                    "kind": "boosted",
                    "baseline": classifier._baseline_prediction.reshape(-1).tolist(),
                    "trees": _trees(classifier),
                },
                "regressor": {
                    "kind": "boosted",
                    "baseline": float(reg._baseline_prediction[0, 0]),
                    "trees": _trees(reg),
                },
            }
        else:
            raise ValueError("Unknown model family")
    temperatures = np.geomspace(0.5, 4, 51)
    temperature = float(
        min(
            temperatures,
            key=lambda t: log_loss(calibration.y, softmax(logits / t, axis=1), labels=[0, 1, 2]),
        )
    )
    residual = calibration.returns - forecast
    state.update(
        {
            "version": 1,
            "kind": kind,
            "features": MODEL_FEATURES,
            "actions": ACTIONS,
            "symbols": sorted(set(train.symbol.tolist())),
            "horizon": train.horizon,
            "timeframe": train.timeframe,
            "trained_until": int(max(train.observed_at.max(), calibration.observed_at.max())),
            "fit_last_timestamp": int(train.timestamp.max()),
            "calibration_first_timestamp": int(calibration.timestamp.min()),
            "fit_label_last_timestamp": int(train.observed_at.max()),
            "temperature": temperature,
            "residual_quantiles": np.quantile(residual, [0.1, 0.9]).tolist(),
            "validation_status": "research_only",
            "datasets": train.datasets,
            "calibration_metrics": {
                "raw_log_loss": float(
                    log_loss(calibration.y, softmax(logits, axis=1), labels=[0, 1, 2])
                ),
                "temperature_log_loss": float(
                    log_loss(calibration.y, softmax(logits / temperature, axis=1), labels=[0, 1, 2])
                ),
            },
            "target": "next-open to horizon-close market return; direction threshold equals roundtrip modeled costs",
        }
    )
    # Verify exported trees/coefficients reproduce sklearn exactly; no executable pickle artifacts.
    for i in np.linspace(0, len(calibration.x) - 1, min(12, len(calibration.x)), dtype=int):
        if not np.allclose(raw_logits(state, calibration.x[i]), logits[i], atol=1e-9):
            raise ValueError("Portable classifier serialization mismatch")
        if not np.isclose(expected_return(state, calibration.x[i]), forecast[i], atol=1e-9):
            raise ValueError("Portable return serialization mismatch")
    return state


class DecisionModel:
    uses_history = False
    name = "calibrated-market-decisions-v1"

    def __init__(self, path=None, artifact=None):
        self.artifact = artifact or json.loads(Path(path).read_text())
        if self.artifact["features"] != MODEL_FEATURES or self.artifact["actions"] != ACTIONS:
            raise ValueError("Unsupported decision model feature/action contract")
        self.trained_until = self.artifact["trained_until"]
        self.version = self.artifact.get("model_version", "development")

    def predict(self, market, history):
        a = self.artifact
        if market.symbol not in a["symbols"] or (market.timeframe, market.horizon) != (
            a["timeframe"],
            a["horizon"],
        ):
            raise ValueError("Input differs from trained asset/timeframe/horizon contract")
        if market.timestamps[-1] <= self.trained_until:
            raise ValueError("Input timestamp overlaps model fit or probability calibration")
        x = model_features(market)
        probability = softmax(raw_logits(a, x) / a["temperature"])
        action = ACTIONS[int(probability.argmax())]
        expected = float(np.clip(expected_return(a, x), -0.95, 1))
        interval = [float(expected + q) for q in a["residual_quantiles"]]
        return Proposal(
            action=action,
            confidence=float(max(probability)),
            rationale=f"{a['kind']} model: temperature-calibrated directional scores; gross expected return {expected:.3%}.",
            forecast=Forecast(
                horizon=market.horizon,
                expected_return=expected,
                projected_close=market.ohlc[-1][3] * (1 + expected),
                method="trained return regression",
            ),
            probabilities={key: float(p) for key, p in zip(ACTIONS, probability)},
            probability_calibration="temperature fitted on chronological calibration data",
            validation_status=a["validation_status"],
            model_version=self.version,
            forecast_interval=interval,
        )


def _tree_batch(nodes, x):
    values = np.array([n["value"] for n in nodes])
    leaf = np.array([n["is_leaf"] for n in nodes], dtype=bool)
    feature = np.array([n["feature_idx"] for n in nodes])
    threshold = np.array([n["num_threshold"] for n in nodes])
    left = np.array([n["left"] for n in nodes])
    right = np.array([n["right"] for n in nodes])
    missing = np.array([n["missing_go_to_left"] for n in nodes], dtype=bool)
    indices = np.zeros(len(x), dtype=int)
    active = ~leaf[indices]
    while active.any():
        positions = np.flatnonzero(active)
        old = indices[positions]
        observed = x[positions, feature[old]]
        go_left = np.where(np.isnan(observed), missing[old], observed <= threshold[old])
        indices[positions] = np.where(go_left, left[old], right[old])
        active = ~leaf[indices]
    return values[indices]


def predict_batch(artifact, x):
    classifier = artifact["classifier"]
    if classifier["kind"] == "logistic":
        scaled = (x - np.array(classifier["mean"])) / np.array(classifier["scale"])
        logits = scaled @ np.array(classifier["coef"]).T + np.array(classifier["intercept"])
    else:
        logits = np.tile(np.array(classifier["baseline"]), (len(x), 1))
        for stage in classifier["trees"]:
            logits += np.column_stack([_tree_batch(tree, x) for tree in stage])
    reg = artifact["regressor"]
    if reg["kind"] == "ridge":
        scaled = (x - np.array(reg["mean"])) / np.array(reg["scale"])
        expected = scaled @ np.array(reg["coef"]) + reg["intercept"]
    else:
        expected = np.full(len(x), reg["baseline"], dtype=float)
        for stage in reg["trees"]:
            expected += _tree_batch(stage[0], x)
    return softmax(logits / artifact["temperature"], axis=1), np.clip(expected, -0.95, 1)
