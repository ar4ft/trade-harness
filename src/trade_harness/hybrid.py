"""A trained decision layer over observed market data, strategies, and TimesFM evidence."""

import numpy as np

from .data import ensure_indicators
from .learning import MODEL_FEATURES, DecisionModel, model_features
from .schemas import ForecastEvidence
from .strategies import STRATEGY_FEATURES, STRATEGY_VERSION, strategy_features, strategy_signals
from .timesfm_model import TimesFMModel

FORECAST_FEATURES = [
    "forecast_return", "forecast_lower", "forecast_upper", "forecast_width", "forecast_vs_momentum",
]
HYBRID_FEATURES = MODEL_FEATURES + STRATEGY_FEATURES + FORECAST_FEATURES


def forecast_features(market, forecast):
    lower, upper = forecast.forecast_interval
    expected = forecast.forecast.expected_return
    momentum = market.ohlc[-1][3] / market.ohlc[-6][3] - 1
    return np.asarray([expected, lower, upper, upper - lower, expected - momentum], dtype=float)


class HybridModel(DecisionModel):
    name = "timesfm-strategy-decisions-v1"
    feature_names = HYBRID_FEATURES

    def __init__(self, path=None, artifact=None, forecaster=None):
        super().__init__(path, artifact)
        self.forecaster = forecaster or TimesFMModel(
            context_length=self.artifact["forecast_contract"]["context_length"]
        )
        if self.artifact["forecast_contract"]["model_version"] != self.forecaster.version:
            raise ValueError("Hybrid forecasting checkpoint/context differs from training")
        if self.artifact.get("strategy_version") != STRATEGY_VERSION:
            raise ValueError("Hybrid strategy policy differs from training")

    def feature_vector(self, market):
        # Parent predict is intentionally bypassed below: one forecast per decision, no shared state.
        raise RuntimeError("Use the hybrid prediction path")

    def predict(self, market, history):
        a = self.artifact
        if market.symbol not in a["symbols"] or (market.timeframe, market.horizon) != (
            a["timeframe"], a["horizon"],
        ):
            raise ValueError("Input differs from trained asset/timeframe/horizon contract")
        if market.timestamps[-1] <= self.trained_until:
            raise ValueError("Input timestamp overlaps fit or calibration")
        market = ensure_indicators(market)
        forecast = self.forecaster.predict(market, [])
        signals = strategy_signals(market)
        x = np.concatenate([
            model_features(market), strategy_features(signals), forecast_features(market, forecast),
        ])
        # Reuse the portable decision predictor with a per-call feature vector, safe for concurrent API calls.
        class PreparedDecision(DecisionModel):
            feature_names = HYBRID_FEATURES

            def feature_vector(self, market):
                return x

        decision = PreparedDecision(artifact=self.artifact).predict(market, [])
        decision.validation_status = "research_only"  # TimesFM 3.0 remains a research pipeline.
        decision.strategy_signals = signals
        decision.forecast_evidence = ForecastEvidence(
            source=forecast.forecast.method, model_version=forecast.model_version,
            as_of=market.timestamps[-1], horizon=market.horizon,
            expected_return=forecast.forecast.expected_return,
            return_interval=forecast.forecast_interval,
        )
        decision.rationale += (
            " Decision uses observed market features, fixed strategy hypotheses, and TimesFM "
            "forecast evidence. Forecast quantiles are uncalibrated; strategy scores are heuristic."
        )
        return decision
