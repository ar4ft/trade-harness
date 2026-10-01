import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from trade_harness.api import app, runtime
from trade_harness.data import add_indicators
from trade_harness.harness import Harness
from trade_harness.hybrid import HYBRID_FEATURES, HybridModel
from trade_harness.hybrid_training import build_forecast_dataset
from trade_harness.learning import Dataset, fit, predict_batch
from trade_harness.schemas import Forecast, MarketInput, Proposal
from trade_harness.storage import Store
from trade_harness.strategies import STRATEGY_VERSION, StrategyModel, strategy_signals
from trade_harness.timesfm_model import REVISION, TimesFMModel


@pytest.fixture
def market():
    return add_indicators(MarketInput(
        symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[1700000000000 + i * 3600000 for i in range(100)],
        ohlc=[[100, 100.2, 99.8, 100]] * 100, volume=[1000] * 100,
    ))


class FakeForecast:
    uses_history = False
    name = "timesfm-test"

    def __init__(self, context=100):
        self.context_length = context
        self.version = f"timesfm3-{REVISION}-ctx{context}-interval-policy-v1"
        self.calls = []

    def predict(self, market, history):
        self.calls.append(market)
        assert history == []
        expected = market.ohlc[-1][3] / market.ohlc[-6][3] - 1
        return Proposal(
            action="HOLD", confidence=0.65, model_version=self.version,
            forecast=Forecast(horizon=market.horizon, expected_return=expected,
                              projected_close=market.ohlc[-1][3] * (1 + expected), method=self.name),
            forecast_interval=[expected - 0.01, expected + 0.01], rationale="Test forecaster",
        )


def artifact(market):
    rng = np.random.default_rng(42)
    n = 180
    data = Dataset(rng.normal(size=(n, len(HYBRID_FEATURES))), np.arange(n) % 3,
                   rng.normal(0, 0.01, n), np.arange(n), np.arange(n) + 3,
                   np.asarray([market.symbol] * n), [], 3, "1h")
    result = fit(data.subset(np.arange(n) < 120), data.subset(np.arange(n) >= 120),
                 "logistic", HYBRID_FEATURES)
    result.update({"strategy_version": STRATEGY_VERSION,
                   "forecast_contract": {"context_length": 100,
                       "model_version": FakeForecast().version}})
    return result


def test_strategies_abstain_in_flat_market(market):
    result = StrategyModel().predict(market, [])
    assert result.action == "HOLD"
    assert result.strategy_signals[-1].strength == 1
    assert all(s.action == "HOLD" for s in result.strategy_signals)


def test_breakout_excludes_current_high(market):
    market.ohlc[-1] = [100, 104, 100, 103]
    signals = strategy_signals(market)
    breakout = next(s for s in signals if s.name == "breakout")
    assert breakout.action == "BUY"
    assert breakout.invalidation_price < 103


def test_conflicting_strategy_hypotheses_hold(market):
    prices = [[100 + i / 10, 100.2 + i / 10, 99.8 + i / 10, 100 + i / 10]
              for i in range(100)]
    trending = add_indicators(market.model_copy(update={"ohlc": prices}))
    signals = strategy_signals(trending)
    assert signals[0].action == "BUY"
    assert signals[2].action == "SELL"
    assert StrategyModel().predict(trending, []).action == "HOLD"


def test_hybrid_feature_serialization_and_evidence(market):
    a = artifact(market)
    provider = FakeForecast()
    model = HybridModel(artifact=a, forecaster=provider)
    result = model.predict(market, [])
    assert len(provider.calls) == 1
    assert len(result.strategy_signals) == 4
    assert result.forecast_evidence.as_of == market.timestamps[-1]
    assert result.forecast_evidence.uncertainty_calibrated is False
    assert result.validation_status == "research_only"
    assert result.model_version == model.version
    assert sum(result.probabilities.values()) == pytest.approx(1)
    from trade_harness.hybrid import forecast_features
    from trade_harness.learning import model_features
    from trade_harness.strategies import strategy_features

    forecast = provider.predict(market, [])
    x = np.concatenate([model_features(market), strategy_features(strategy_signals(market)),
                        forecast_features(market, forecast)])
    probability, expected = predict_batch(a, np.array([x]))
    assert list(result.probabilities.values()) == pytest.approx(probability[0])
    assert result.forecast.expected_return == pytest.approx(expected[0])


def test_hybrid_rejects_training_overlap_and_provider_contract(market):
    a = artifact(market)
    a["trained_until"] = market.timestamps[-1]
    provider = FakeForecast()
    with pytest.raises(ValueError, match="overlaps"):
        HybridModel(artifact=a, forecaster=provider).predict(market, [])
    assert not provider.calls
    with pytest.raises(ValueError, match="checkpoint/context"):
        HybridModel(artifact=a, forecaster=FakeForecast(512))


def test_hybrid_unavailable_forecast_fails_closed(market, tmp_path):
    provider = FakeForecast()
    provider.predict = Mock(side_effect=RuntimeError("unavailable"))
    model = HybridModel(artifact=artifact(market), forecaster=provider)
    result = Harness(model, Store(str(tmp_path / "state.sqlite"))).decide(market)
    assert result.action == "HOLD"
    assert result.real_execution_enabled is False
    assert "model_failure" in result.guardrails


def test_hybrid_api_exposes_separate_forecast_and_strategy_evidence(market, tmp_path):
    harness = Harness(HybridModel(artifact=artifact(market), forecaster=FakeForecast()),
                      Store(str(tmp_path / "api.sqlite")))
    app.dependency_overrides.clear()
    from unittest.mock import patch

    import trade_harness.api as api

    with patch.object(api, "runtime", return_value=harness), TestClient(app) as client:
        response = client.post("/decisions", json=market.model_dump())
    runtime.cache_clear()
    assert response.status_code == 200
    assert response.json()["forecast_evidence"]["source"] == "timesfm-test"
    assert len(response.json()["strategy_signals"]) == 4


def test_batched_forecasts_group_warmup_shapes_and_keep_order(market):
    model = TimesFMModel()
    model._forecaster = Mock()
    for indicator in market.indicators:
        indicator.values = [indicator.values[-1]] * len(market.ohlc)
    first = market.model_copy(deep=True)
    first.indicators[0].values[0] = None
    model._forecaster.predict_batch.side_effect = lambda contexts, **kwargs: [
        SimpleNamespace(forecast=np.full((4, 3), 100), quantiles=np.full((4, 3, 9), 100))
        for _ in contexts]
    predictions = model.predict_many([first, market])
    assert len(predictions) == 2
    assert model._forecaster.predict_batch.call_count == 2
    assert all(p.action == "HOLD" for p in predictions)


def history_file(tmp_path):
    rng = np.random.default_rng(8)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, 1000)))
    market = MarketInput(symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[1700000000000 + i * 3600000 for i in range(len(close))],
        ohlc=[[float(p), float(p * 1.01), float(p * 0.99), float(p)] for p in close],
        volume=[1000] * len(close))
    path = tmp_path / "history.json"
    path.write_text(market.model_dump_json())
    return path


def test_forecast_cache_resume_and_matured_labels(tmp_path):
    path = history_file(tmp_path)
    provider = FakeForecast(21)
    cache = tmp_path / "forecasts.jsonl"
    data, _, contract, _ = build_forecast_dataset([str(path)], cache, 3, 21, provider)
    assert len(data.y) >= 300
    assert np.all(data.observed_at > data.timestamp)
    assert all(len(m.ohlc) == 21 for m in provider.calls)
    provider.calls.clear()
    again, _, _, _ = build_forecast_dataset([str(path)], cache, 3, 21, provider)
    assert not provider.calls
    assert np.array_equal(data.x, again.x)
    assert contract["cache_sha256"]
    lines = cache.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["input_sha256"] = "tampered"
    lines[1] = json.dumps(entry)
    cache.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="historical prefix"):
        build_forecast_dataset([str(path)], cache, 3, 21, provider)


def test_future_candles_do_not_change_past_forecast_features(tmp_path):
    path = history_file(tmp_path)
    first, _, _, _ = build_forecast_dataset([str(path)], tmp_path / "first.jsonl", 3, 21,
                                          FakeForecast(21))
    history = json.loads(path.read_text())
    history["ohlc"][-1] = [500, 510, 490, 500]
    path.write_text(json.dumps(history))
    second, _, _, _ = build_forecast_dataset([str(path)], tmp_path / "second.jsonl", 3, 21,
                                           FakeForecast(21))
    assert np.array_equal(first.x[:-10], second.x[:-10])


@pytest.mark.parametrize("fee_bps,slippage_bps", [(10, 5), (20, 10)])
def test_sparse_replay_preserves_fills_costs_and_risk(market, fee_bps, slippage_bps):
    from trade_harness.evaluation import _simulate
    from trade_harness.hybrid_training import ScheduledModel
    from trade_harness.risk import RiskConfig

    class Buy:
        name = "fixed-buy-test"
        version = "test"
        uses_history = False

        def predict(self, market, history):
            return Proposal(action="BUY", confidence=0.9, validation_status="research_only",
                rationale="Controlled positive forecast for execution parity",
                forecast=Forecast(horizon=3, expected_return=0.02,
                                  projected_close=102, method="test"))

    allowed = {(market.symbol, market.timestamps[i]) for i in (25, 50)}
    config = RiskConfig(allow_research=True, fee_bps=fee_bps, slippage_bps=slippage_bps)
    args = ({market.symbol: market}, ScheduledModel(Buy(), allowed),
            market.timestamps[20], market.timestamps[80], config)
    full = _simulate(*args)[market.symbol]
    sparse = _simulate(*args, decision_times=allowed)[market.symbol]
    assert full["closed_trades"] == 2
    for key in ("return", "equity", "closed_trades", "fees", "turnover", "max_drawdown",
                "halted", "open_units", "pending_order", "annualized_daily_sharpe"):
        assert sparse[key] == pytest.approx(full[key]) if full[key] is not None else sparse[key] is None
