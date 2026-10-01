"""Optional TimesFM 3.0 research forecasts; weights are never shipped in releases."""

import os
import threading

import numpy as np

from .schemas import Forecast, Proposal
from .storage import timeframe_ms

CHECKPOINT = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


class TimesFMModel:
    name = "timesfm-3.0-research-v1"
    uses_history = False

    def __init__(self, context_length=None):
        self.context_length = int(
            context_length if context_length is not None
            else os.environ.get("TRADING_TIMESFM_CONTEXT", "512")
        )
        if not 21 <= self.context_length <= 10000:
            raise ValueError("TRADING_TIMESFM_CONTEXT must be between 21 and 10000")
        self.batch_size = 1
        self.device = os.environ.get("TRADING_TIMESFM_DEVICE", "cpu")
        self.version = f"timesfm3-{REVISION}-ctx{self.context_length}-interval-policy-v1"
        self._forecaster = None
        self._lock = threading.Lock()

    def _load(self):
        if self._forecaster is None:
            try:
                from timesfm3 import ModelConfig, TimesFM3Forecaster
            except ImportError as error:
                raise RuntimeError("Install trade-harness[timesfm] to use TimesFM 3.0") from error
            self._forecaster = TimesFM3Forecaster(
                ModelConfig(
                    checkpoint_path=CHECKPOINT,
                    revision=REVISION,
                    device=self.device,
                    per_core_batch_size=self.batch_size,
                    local_files_only=os.environ.get("TRADING_TIMESFM_OFFLINE") == "1",
                )
            )
        return self._forecaster

    def _inputs(self, market):
        interval = timeframe_ms(market.timeframe)
        if any(b - a != interval for a, b in zip(market.timestamps, market.timestamps[1:])):
            raise ValueError("TimesFM requires regularly spaced closed candles")
        # Independent OHLC target channels; no future indicator values are supplied.
        prices = np.asarray(market.ohlc[-self.context_length :], dtype=np.float32).T
        n = prices.shape[1]
        covariates = [np.asarray(market.volume[-n:], dtype=np.float32)]
        for indicator in sorted(market.indicators, key=lambda i: i.name):
            values = indicator.values[-n:]
            # Drop channels with leading warm-up gaps rather than backfill from the future.
            if any(v is None for v in values):
                continue
            covariates.append(np.asarray(values, dtype=np.float32))
        past = np.stack(covariates)
        if not np.isfinite(prices).all() or not np.isfinite(past).all():
            raise ValueError("TimesFM inputs exceed float32 range")
        return prices, past

    def predict(self, market, history):
        prices, past = self._inputs(market)
        with self._lock:
            output = self._load().predict(
                prices, horizon=market.horizon, past_only_covariates=past,
                return_quantiles=True, use_znorm=True, sort_quantiles=True,
            )
        return self._proposal(market, output)

    def predict_many(self, markets):
        if not markets:
            return []
        if len({m.horizon for m in markets}) != 1:
            raise ValueError("A forecast batch must share its horizon")
        inputs = [self._inputs(m) for m in markets]
        # Indicator warm-up gaps can change covariate channel counts. Batch compatible shapes.
        groups = {}
        for index, (prices, past) in enumerate(inputs):
            groups.setdefault((prices.shape, past.shape), []).append(index)
        outputs = [None] * len(markets)
        with self._lock:
            for indices in groups.values():
                batch = list(self._load().predict_batch(
                    contexts=[inputs[i][0] for i in indices], horizon=markets[0].horizon,
                    past_only_covariates=[inputs[i][1] for i in indices],
                    return_quantiles=True, use_znorm=True, sort_quantiles=True,
                ))
                if len(batch) != len(indices):
                    raise ValueError("TimesFM batch output count mismatch")
                for i, output in zip(indices, batch):
                    outputs[i] = output
        return [self._proposal(m, o) for m, o in zip(markets, outputs)]

    def _proposal(self, market, output):
        point = np.asarray(output.forecast, dtype=float)
        quantiles = np.asarray(output.quantiles, dtype=float)
        if point.shape != (4, market.horizon) or quantiles.shape != (4, market.horizon, 9):
            raise ValueError("Unexpected TimesFM multivariate forecast shape")
        if (
            not np.isfinite(point).all()
            or not np.isfinite(quantiles).all()
            or np.any(point <= 0)
            or np.any(quantiles <= 0)
            or np.any(np.diff(quantiles, axis=-1) < 0)
        ):
            raise ValueError("Invalid TimesFM price forecasts or quantile ordering")
        close = market.ohlc[-1][3]
        expected = float(point[3, -1] / close - 1)
        lower, upper = (float(quantiles[3, -1, i] / close - 1) for i in (0, 8))
        if not lower <= expected <= upper:
            raise ValueError("TimesFM point forecast is outside its quantile interval")
        threshold = 0.003
        action = "BUY" if lower > threshold else "SELL" if upper < -threshold else "HOLD"
        return Proposal(
            action=action,
            confidence=0.8 if action != "HOLD" else 0.65,
            rationale=(
                f"Research forecast: median close return {expected:.2%}; "
                f"10th–90th quantile return range [{lower:.2%}, {upper:.2%}]. "
                "Direction requires the entire range to exceed ±0.30%. "
                "Confidence is a policy heuristic; quantiles are not market-calibrated. "
                "Previous decisions are not inputs to this forecasting backend."
            ),
            forecast=Forecast(
                horizon=market.horizon,
                expected_return=expected,
                projected_close=close * (1 + expected),
                method=self.name,
            ),
            forecast_interval=[lower, upper],
            model_version=self.version,
            validation_status="research_only",
        )
