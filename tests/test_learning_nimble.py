from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trade_harness.data import add_indicators
from trade_harness.harness import Harness
from trade_harness.learning import (
    DecisionModel,
    fit,
    load_dataset,
    model_features,
    predict_batch,
    raw_logits,
)
from trade_harness.nimble import NimbleModel
from trade_harness.schemas import Forecast, MarketInput, Proposal
from trade_harness.storage import Store


@pytest.mark.parametrize("kind", ["logistic", "boosted"])
def test_portable_prediction_and_purged_fit(kind):
    data = load_dataset(["examples/market.json"])
    boundary = data.timestamp[150]
    training = data.subset(data.observed_at < boundary)
    calibration = data.subset((data.timestamp >= boundary) & (data.timestamp < data.timestamp[190]))
    artifact = fit(training, calibration, kind)
    assert artifact["fit_label_last_timestamp"] < artifact["calibration_first_timestamp"]
    p, expected = predict_batch(artifact, data.x[-5:])
    for i in range(5):
        logits = raw_logits(artifact, data.x[-5 + i]) / artifact["temperature"]
        shifted = np.exp(logits - max(logits))
        assert np.allclose(p[i], shifted / sum(shifted), atol=1e-10)
    assert np.isfinite(expected).all()
    original = MarketInput.model_validate_json(Path("examples/market.json").read_text())
    with pytest.raises(ValueError, match="overlaps"):
        DecisionModel(artifact=artifact).predict(
            original.model_copy(update={"timestamps": [0] * len(original.timestamps)}), []
        )
    latest = original.model_copy(update={"symbol": artifact["symbols"][0]})
    result = DecisionModel(artifact=artifact).predict(latest, [])
    assert set(result.probabilities) == {"BUY", "SELL", "HOLD"}


def test_features_are_causal_and_cost_labels():
    source = add_indicators(
        MarketInput.model_validate_json(Path("examples/market.json").read_text())
    )
    from trade_harness.learning import window

    before = model_features(window(source, 100))
    changed = source.model_copy(deep=True)
    for row in changed.ohlc[101:]:
        row[:] = [p * 3 for p in row]
    assert np.array_equal(before, model_features(window(changed, 100)))
    data = load_dataset(["examples/market.json"])
    assert np.all(data.y[data.returns > 0.003] == 0)
    assert np.all(data.y[data.returns < -0.003] == 1)
    assert np.all(data.observed_at > data.timestamp)


def test_default_model_is_new_calibrated_model(monkeypatch):
    monkeypatch.delenv("TRADING_BACKEND", raising=False)
    from trade_harness.models import load_model

    model = load_model()
    assert isinstance(model, DecisionModel)
    assert model.artifact["validation_status"] == "research_only"
    assert (
        model.artifact["fit_label_last_timestamp"] < model.artifact["calibration_first_timestamp"]
    )


def test_nimble_contract_and_failure_fallback(monkeypatch):
    market = MarketInput.model_validate_json(
        Path("src/trade_harness/assets/latest.json").read_text()
    )

    class ForecastModel:
        trained_until = 0
        version = "test"

        def predict(self, market, history):
            return Proposal(
                action="HOLD",
                confidence=0.8,
                rationale="Numerical forecast",
                forecast=Forecast(
                    horizon=market.horizon,
                    expected_return=0.002,
                    projected_close=market.ohlc[-1][3] * 1.002,
                    method="test",
                ),
            )

    model = NimbleModel(ForecastModel())
    response = MagicMock()
    response.json.return_value = {
        "answers": {
            "direction": {
                "choice": "HOLD",
                "probabilities": {"BUY": 0.1, "SELL": 0.1, "HOLD": 0.8},
            },
            "evidence_sufficient": {"noul": 0.1},
        }
    }
    with patch("trade_harness.nimble.httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value = response
        result = model.predict(market, [])
        assert result.action == "HOLD" and result.validation_status == "research_only"
        args = client.return_value.__enter__.return_value.post.call_args
        assert args.kwargs["json"]["questions"]["direction"]["type"] == "choice"
        response.raise_for_status.side_effect = RuntimeError("secret provider message")
        decision = Harness(model, Store(":memory:")).decide(market)
        assert decision.action == "HOLD"
        assert "model_failure" in decision.guardrails
        assert "secret provider message" not in decision.rationale
