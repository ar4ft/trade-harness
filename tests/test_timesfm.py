from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from trade_harness.harness import Harness
from trade_harness.models import load_model
from trade_harness.schemas import Indicator, MarketInput
from trade_harness.storage import Store
from trade_harness.timesfm_model import REVISION, TimesFMModel


@pytest.fixture
def market():
    return MarketInput(
        symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[1700000000000 + i * 3600000 for i in range(30)],
        ohlc=[[100, 102, 98, 100]] * 30, volume=[1000] * 30,
        indicators=[Indicator(name="rsi", values=[55] * 30),
                    Indicator(name="warmup", values=[None] + [1] * 29)],
    )


def backend(lower=101, median=102, upper=103):
    model = TimesFMModel()
    quantiles = np.broadcast_to(np.linspace(lower, upper, 9), (4, 3, 9)).copy()
    model._forecaster = Mock()
    model._forecaster.predict.return_value = SimpleNamespace(
        forecast=np.full((4, 3), median), quantiles=quantiles,
    )
    return model


@pytest.mark.parametrize("lower,median,upper,action", [
    (101, 102, 103, "BUY"), (97, 98, 99, "SELL"), (99, 102, 105, "HOLD"),
])
def test_quantile_policy_and_forecast(market, lower, median, upper, action):
    model = backend(lower, median, upper)
    result = model.predict(market, [])
    assert result.action == action
    assert result.forecast.expected_return == pytest.approx(median / 100 - 1)
    assert result.forecast.projected_close == pytest.approx(median)
    assert result.forecast_interval == pytest.approx([lower / 100 - 1, upper / 100 - 1])
    assert result.validation_status == "research_only"
    assert result.probabilities is None
    assert result.probability_calibration == "uncalibrated"
    assert REVISION in result.model_version


def test_only_closed_aligned_past_covariates(market, monkeypatch):
    monkeypatch.setenv("TRADING_TIMESFM_CONTEXT", "21")
    model = backend()
    model.predict(market, [{"action": "BUY", "future": "ignored"}])
    args, kwargs = model._forecaster.predict.call_args
    assert args[0].shape == (4, 21)
    assert kwargs["past_only_covariates"].shape == (3, 21)
    assert "past_future_covariates" not in kwargs
    model = backend()
    model.context_length = 30
    model.predict(market, [])
    assert model._forecaster.predict.call_args.kwargs["past_only_covariates"].shape == (2, 30)


@pytest.mark.parametrize("corruption", ["shape", "nan", "negative", "unordered", "outside"])
def test_invalid_outputs_fail_closed(market, tmp_path, corruption):
    model = backend()
    output = model._forecaster.predict.return_value
    if corruption == "shape":
        output.forecast = np.ones((3, 3))
    elif corruption == "nan":
        output.quantiles[3, -1, 0] = np.nan
    elif corruption == "negative":
        output.forecast[0, 0] = -1
    elif corruption == "unordered":
        output.quantiles[3, -1, 0] = 999
    else:
        output.forecast[3, -1] = 200
    result = Harness(model, Store(str(tmp_path / "decisions.sqlite"))).decide(market)
    assert result.action == "HOLD"
    assert "model_failure" in result.guardrails
    assert result.real_execution_enabled is False


def test_gap_rejected_before_inference(market):
    model = backend()
    market.timestamps[-1] += 3600000
    with pytest.raises(ValueError, match="regularly spaced"):
        model.predict(market, [])
    model._forecaster.predict.assert_not_called()


def test_factory_is_lazy_and_optional(monkeypatch):
    monkeypatch.setenv("TRADING_BACKEND", "timesfm")
    model = load_model()
    assert model._forecaster is None
    monkeypatch.setenv("TRADING_TIMESFM_CONTEXT", "10")
    with pytest.raises(ValueError, match="CONTEXT"):
        load_model()


def test_no_walk_forward_certification(market, tmp_path):
    result = Harness(backend(), Store(str(tmp_path / "decisions.sqlite"))).decide(market)
    assert result.validation_status == "research_only"
    assert result.trading_validation.positive_edge is False
    assert result.trading_validation.status == "unknown"


def test_loader_pins_checkpoint_and_offline_mode(monkeypatch):
    import sys

    factory = Mock()
    monkeypatch.setitem(sys.modules, "timesfm3", SimpleNamespace(
        ModelConfig=lambda **kwargs: kwargs, TimesFM3Forecaster=factory,
    ))
    monkeypatch.setenv("TRADING_TIMESFM_OFFLINE", "1")
    model = TimesFMModel()
    assert model._load() is model._load()
    factory.assert_called_once()
    config = factory.call_args.args[0]
    assert config["revision"] == REVISION
    assert config["local_files_only"] is True
    assert config["device"] == "cpu"
