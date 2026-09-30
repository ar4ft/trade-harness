import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from trade_harness.api import app, runtime
from trade_harness.backtest import backtest
from trade_harness.features import prefix
from trade_harness.harness import Harness
from trade_harness.models import BaselineModel, LanguageModel, TrainedModel
from trade_harness.schemas import Feedback, Forecast, MarketInput, Proposal
from trade_harness.storage import Store, timeframe_ms
from trade_harness.training import export_finetuning, train


@pytest.fixture
def market():
    return MarketInput.model_validate_json(Path("examples/market.json").read_text())


class BuyModel:
    name = "test"

    def predict(self, market, history):
        return Proposal(
            action="BUY",
            confidence=0.8,
            rationale="Test signal",
            forecast=Forecast(
                horizon=market.horizon,
                expected_return=0.01,
                projected_close=market.ohlc[-1][3] * 1.01,
                method="test",
            ),
        )


def test_invalid_data(market):
    data = market.model_dump()
    data["ohlc"][0][1] = 1
    with pytest.raises(ValidationError):
        MarketInput.model_validate(data)
    data = market.model_dump()
    data["volume"][0] = float("nan")
    with pytest.raises(ValidationError):
        MarketInput.model_validate(data)
    data = market.model_dump()
    data["timestamps"][1] = data["timestamps"][0]
    with pytest.raises(ValidationError):
        MarketInput.model_validate(data)


def test_position_guard(market):
    result = Harness(BuyModel(), Store(":memory:")).decide(
        market.model_copy(update={"position": "long"})
    )
    assert result.action == "HOLD"
    assert "already_long" in result.guardrails


def test_provider_failure_and_inconsistent_forecast(market):
    class BadModel:
        name = "bad"

        def predict(self, market, history):
            raise RuntimeError("secret provider message")

    result = Harness(BadModel(), Store(":memory:")).decide(market)
    assert result.action == "HOLD"
    assert "secret" not in result.model_dump_json()

    class WrongForecast(BuyModel):
        def predict(self, market, history):
            return (
                super()
                .predict(market, history)
                .model_copy(
                    update={
                        "forecast": Forecast(
                            horizon=99, expected_return=0.01, projected_close=1, method="bad"
                        )
                    }
                )
            )

    result = Harness(WrongForecast(), Store(":memory:")).decide(market)
    assert "model_failure" in result.guardrails


def test_history_never_exposes_future_outcomes(market):
    store = Store(":memory:")
    old = prefix(market, 30)
    decision = Harness(BuyModel(), store).decide(old)
    future = market.timestamps[40]
    store.feedback(
        Feedback(
            decision_id=decision.id, realized_return=0.02, reviewed_action="BUY", observed_at=future
        )
    )
    before = store.history(prefix(market, 35))
    assert len(before) == 1 and "feedback" not in before[0]
    assert "feedback" in store.history(prefix(market, 45))[0]
    assert store.history(prefix(market, 25)) == []
    with pytest.raises(ValueError):
        store.feedback(
            Feedback(decision_id=decision.id, realized_return=0, observed_at=old.timestamps[-1])
        )


def test_next_open_fills_and_costs(market):
    sample = prefix(market, 22)
    sample.ohlc[-1] = [200, 202, 198, 200]
    result = backtest(sample, BuyModel(), fee_bps=10, slippage_bps=5)
    assert len(result["trades"]) == 1
    assert result["trades"][0]["price"] == pytest.approx(200 * 1.0005)
    assert result["total_return"] < 0
    assert result["max_drawdown"] > 0


def test_backtest_inputs_are_prefix_only(market):
    class Spy(BuyModel):
        def __init__(self):
            self.last = -1

        def predict(self, view, history):
            assert view.timestamps[-1] > self.last
            self.last = view.timestamps[-1]
            index = market.timestamps.index(self.last)
            assert view.ohlc[-1] == market.ohlc[index]
            for entry in history:
                assert entry["as_of"] < self.last
                if "feedback" in entry:
                    assert entry["feedback"]["observed_at"] < self.last
            return super().predict(view, history)

    backtest(market, Spy())


def test_training_portability_and_leakage_guard(market, tmp_path):
    path = tmp_path / "model.json"
    metrics = train(market, str(path))
    assert metrics["train_samples"] == 169
    model = TrainedModel(str(path))
    with pytest.raises(ValueError, match="overlaps"):
        model.predict(prefix(market, 100), [])
    model.predict(market, [])
    assert backtest(market, model)["equity"][0]["timestamp"] > model.trained_until
    altered = market.model_copy(deep=True)
    for row in altered.ohlc[192:]:
        row[:] = [v * 2 for v in row]
    second = tmp_path / "second.json"
    train(altered, str(second))
    a, b = json.loads(path.read_text()), json.loads(second.read_text())
    assert a["coef"] == b["coef"]
    assert a["mean"] == b["mean"]


def test_finetuning_export_excludes_future_labels(market, tmp_path):
    store = Store(":memory:")
    decision = Harness(BuyModel(), store).decide(market)
    store.feedback(
        Feedback(
            decision_id=decision.id,
            realized_return=0.123,
            reviewed_action="BUY",
            observed_at=decision.as_of + 3 * timeframe_ms("1h"),
        )
    )
    path = tmp_path / "train.jsonl"
    assert export_finetuning(store, str(path))["examples"] == 1
    item = json.loads(path.read_text())
    user = json.loads(item["messages"][1]["content"])
    assert "realized_return" not in user
    assert len(item["messages"]) == 3


def test_api_contract_and_feedback(market):
    harness = Harness(BaselineModel(), Store(":memory:"))
    app.dependency_overrides.clear()
    # API runtime is a cached factory, so replace its bound module reference.
    from unittest.mock import patch

    with patch("trade_harness.api.runtime", return_value=harness):
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        response = client.post("/decisions", json=market.model_dump())
        assert response.status_code == 200
        decision = response.json()
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert client.post("/decisions", json={}).status_code == 422
        assert (
            client.post(
                "/feedback", json={"decision_id": "missing", "realized_return": 0, "observed_at": 0}
            ).status_code
            == 404
        )
    runtime.cache_clear()


def test_llm_parses_provider_response(market, monkeypatch):
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("LLM_MODEL", "local-trading-model")
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": BuyModel().predict(market, []).model_dump_json()}}]
    }
    with patch("trade_harness.models.httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value = response
        assert LanguageModel().predict(market, []).action == "BUY"
