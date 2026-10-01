import copy
import json
from unittest.mock import Mock

import httpx
import pytest

from trade_harness.data import add_indicators
from trade_harness.ollaya import OllayaModel
from trade_harness.orchestrator import Orchestrator
from trade_harness.schemas import Forecast, ForecastEvidence, MarketInput, Proposal

DIGEST = "a" * 64


@pytest.fixture
def market():
    return add_indicators(MarketInput(symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[1700000000000 + i * 3600000 for i in range(30)],
        ohlc=[[100, 101, 99, 100]] * 30, volume=[1000] * 30))


def evidence(market):
    return {"symbol": market.symbol, "timeframe": market.timeframe,
            "forecast": ForecastEvidence(source="timesfm-test", model_version="tf-v1",
                as_of=market.timestamps[-1], horizon=3, expected_return=0.02,
                return_interval=[0.01, 0.03]).model_dump(),
            "features": {"return_1": 0.001}, "strategies": []}


def response():
    return {"model": "laya:en", "routing": None, "done_reason": "decide", "state_truncated": False,
            "answers": {"direction": {"type": "choice", "choice": "BUY", "confidence": 0.7,
                "probabilities": {"BUY": 0.8, "SELL": 0.1, "HOLD": 0.1}},
                "evidence_sufficient": {"type": "noul", "noul": 0.9}}}


def transport(monkeypatch, body=None, digest=None):
    requests = []
    original = httpx.Client
    monkeypatch.setenv("OLLAYA_MODEL", "laya:en")
    monkeypatch.delenv("OLLAYA_MODEL_DIGEST", raising=False)
    monkeypatch.delenv("OLLAYA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAYA_TIMEOUT_SECONDS", raising=False)

    def handler(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "laya:en", "digest": digest or DIGEST}]})
        assert request.url.path == "/api/decide"
        return httpx.Response(200, json=body or response())

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    return requests


def test_shared_forecast_and_independent_typed_review(market, monkeypatch):
    calls = transport(monkeypatch)
    source = Mock()
    model = OllayaModel(source)
    result = model.predict_with_evidence(market, [], evidence(market))
    assert result.action == "BUY" and result.confidence == 0.8  # not provider confidence=0.7
    assert result.forecast.method == "timesfm-test"
    source.predict.assert_not_called()
    state = json.loads(calls[-1].content)["state"]
    assert state["shared_evidence"] == evidence(market)
    assert "numerical_direction_probabilities" not in state
    assert "primary_action" not in state
    assert result.review_details["manifest_digest"] == DIGEST
    assert result.review_details["evidence_sufficient"] == 0.9
    assert result.probability_calibration == "uncalibrated"


def test_documented_rounding_is_normalized(market, monkeypatch):
    body = response()
    body["answers"]["direction"]["probabilities"] = {"BUY": 0.3334, "SELL": 0.3333, "HOLD": 0.3334}
    transport(monkeypatch, body)
    result = OllayaModel().predict_with_evidence(market, [], evidence(market))
    assert sum(result.probabilities.values()) == pytest.approx(1)


@pytest.mark.parametrize("change", [
    {"state_truncated": True}, {"state_truncated": None}, {"done_reason": "load"},
    {"model": "laya:multilingual"}, {"routing": {"model": "laya:en"}},
])
def test_incomplete_or_routed_reviews_are_rejected(market, monkeypatch, change):
    body = response()
    body.update(change)
    transport(monkeypatch, body)
    with pytest.raises(ValueError):
        OllayaModel().predict_with_evidence(market, [], evidence(market))


@pytest.mark.parametrize("probabilities,choice", [
    ({"BUY": 0.9, "SELL": 0.9, "HOLD": 0.1}, "BUY"),
    ({"BUY": 0.8, "SELL": 0.1, "HOLD": 0.1}, "SELL"),
    ({"BUY": 0.8, "SELL": 0.2}, "BUY"),
])
def test_invalid_distributions_rejected(market, monkeypatch, probabilities, choice):
    body = response()
    body["answers"]["direction"].update(probabilities=probabilities, choice=choice)
    transport(monkeypatch, body)
    with pytest.raises(ValueError):
        OllayaModel().predict_with_evidence(market, [], evidence(market))


def test_manifest_drift_and_pin_rejected(market, monkeypatch):
    transport(monkeypatch)
    model = OllayaModel()
    monkeypatch.setattr(model, "_model_digest", lambda: "b" * 64)
    with pytest.raises(ValueError, match="changed"):
        model.predict_with_evidence(market, [], evidence(market))
    monkeypatch.setenv("OLLAYA_MODEL_DIGEST", "b" * 64)
    with pytest.raises(ValueError, match="configured digest"):
        OllayaModel()


def test_consensus_audit_carries_provider_provenance(market, monkeypatch):
    transport(monkeypatch)
    primary = Mock(name="primary")
    primary.name, primary.version = "numerical", "v1"
    primary.artifact = {}
    primary.predict.return_value = Proposal(action="BUY", confidence=0.8, rationale="test",
        forecast=Forecast(horizon=3, expected_return=0.02, projected_close=102, method="test"),
        forecast_evidence=ForecastEvidence.model_validate(evidence(market)["forecast"]))
    model = Orchestrator(primary=primary, reviewers={"ollaya": OllayaModel()}, config={"reviewers": ["ollaya"]})
    result = model.predict(market, [])
    assert result.consensus.accepted
    assert result.consensus.votes[-1].review_details["manifest_digest"] == DIGEST
    assert result.validation_status == "research_only"


def test_duplicate_nimble_family_cannot_add_vote():
    primary = Mock()
    ollaya = Mock(model_family="nimble")
    with pytest.raises(ValueError, match="separate votes"):
        Orchestrator(primary=primary, reviewers={"nimble": Mock(), "ollaya": ollaya},
                     config={"reviewers": ["nimble", "ollaya"]})


def test_stale_shared_forecast_rejected(market, monkeypatch):
    transport(monkeypatch)
    state = copy.deepcopy(evidence(market))
    state["forecast"]["as_of"] -= 1
    with pytest.raises(ValueError, match="contract mismatch"):
        OllayaModel().predict_with_evidence(market, [], state)


@pytest.mark.parametrize("value", ["0", "121", "nan", "inf"])
def test_inference_timeout_is_bounded(monkeypatch, value):
    monkeypatch.setenv("OLLAYA_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="timeout"):
        OllayaModel()
