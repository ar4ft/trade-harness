import json
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from trade_harness.data import add_indicators
from trade_harness.feature_engineering import (
    BASE_FEATURES,
    FeatureConfig,
    FitParameters,
    feature_names,
    transform_features,
)
from trade_harness.harness import Harness
from trade_harness.models import LanguageModel
from trade_harness.orchestrator import Orchestrator, OrchestratorConfig
from trade_harness.schemas import Forecast, ForecastEvidence, MarketInput, Proposal
from trade_harness.storage import Store


@pytest.fixture
def market():
    return add_indicators(MarketInput(symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[1700000000000 + i * 3600000 for i in range(30)],
        ohlc=[[100, 101, 99, 100]] * 30, volume=[1000] * 30))


def proposal(action="BUY", confidence=0.8):
    return Proposal(action=action, confidence=confidence, rationale="Controlled test vote",
        probabilities={a: confidence if a == action else (1 - confidence) / 2
                       for a in ("BUY", "SELL", "HOLD")},
        forecast=Forecast(horizon=3, expected_return=0.02, projected_close=102, method="test"))


class Primary:
    name = "test-numerical"
    version = "primary-v1"

    def __init__(self, action="BUY"):
        self.action = action

    def predict(self, market, history):
        value = proposal(self.action)
        value.forecast_evidence = ForecastEvidence(source="timesfm-test", model_version="tf-v1",
            as_of=market.timestamps[-1], horizon=3, expected_return=0.02,
            return_interval=[0.01, 0.03])
        value.feature_evidence = {"return_1": 0.001}
        return value


def reviewer(action="BUY", confidence=0.8):
    value = Mock()
    value.name = "review-test"
    value.version = "review-v1"
    value.predict_with_evidence.return_value = proposal(action, confidence)
    return value


def orchestrator(review, primary_action="BUY", **config):
    return Orchestrator(primary=Primary(primary_action), reviewers={"llm": review},
                        config={"reviewers": ["llm"], **config})


def test_confirmation_and_separate_vote_fractions(market):
    peer = reviewer()
    result = orchestrator(peer).predict(market, [])
    assert result.action == "BUY"
    assert result.probabilities is None
    assert result.consensus.accepted is True
    assert result.consensus.agreement_fraction == 1
    assert result.consensus.is_calibrated_probability is False
    assert result.validation_status == "research_only"
    assert len({v.evidence_sha256 for v in result.consensus.votes}) == 1
    evidence = peer.predict_with_evidence.call_args.args[2]
    assert evidence["forecast"]["source"] == "timesfm-test"
    assert evidence["features"] == {"return_1": 0.001}
    assert "primary_action" not in evidence
    assert "numerical_direction_probabilities" not in evidence


@pytest.mark.parametrize("action", ["SELL", "HOLD"])
def test_disagreement_abstains(market, action):
    result = orchestrator(reviewer(action)).predict(market, [])
    assert result.action == "HOLD"
    assert result.confidence == 0
    assert result.consensus.agreement_fraction == 0.5
    assert "insufficient_agreement" in result.consensus.reason_codes


def test_missing_peer_cannot_reduce_quorum(market):
    result = orchestrator(None).predict(market, [])
    assert result.action == "HOLD"
    assert result.consensus.configured_members == 2
    assert result.consensus.valid_members == 1
    assert result.consensus.votes[-1].status == "unavailable"


def test_provider_errors_do_not_leak(market):
    peer = reviewer()
    peer.predict_with_evidence.side_effect = RuntimeError("credential-sensitive-provider-error")
    result = orchestrator(peer).predict(market, [])
    assert "credential-sensitive" not in result.model_dump_json()
    assert result.consensus.votes[-1].status == "unavailable"


def test_malformed_forecast_is_invalid_vote(market):
    peer = reviewer()
    peer.predict_with_evidence.return_value.forecast.projected_close = 999
    result = orchestrator(peer).predict(market, [])
    assert result.action == "HOLD"
    assert result.consensus.votes[-1].status == "invalid"


def test_low_supporting_score_cannot_create_direction(market):
    result = orchestrator(reviewer(confidence=0.4)).predict(market, [])
    assert result.action == "HOLD"
    assert "supporting_score_below_policy_threshold" in result.consensus.reason_codes


def test_language_majority_cannot_override_numerical_direction(market):
    model = Orchestrator(primary=Primary("HOLD"),
        reviewers={"llm": reviewer("BUY"), "nimble": reviewer("BUY")},
        config={"reviewers": ["llm", "nimble"], "agreement_fraction": 0.66})
    result = model.predict(market, [])
    assert result.action == "HOLD"
    assert result.consensus.accepted is False


def test_history_only_exposes_past_matured_matching_records(market):
    peer = reviewer()
    now = market.timestamps[-1]
    past = {"as_of": now - 1000, "action": "BUY", "symbol": market.symbol,
            "timeframe": market.timeframe,
            "feedback": {"observed_at": now + 1000, "realized_return": 0.5}}
    future = {**past, "as_of": now + 1}
    wrong_asset = {**past, "symbol": "ETHUSDT"}
    orchestrator(peer).predict(market, [past, future, wrong_asset])
    history = peer.predict_with_evidence.call_args.args[1]
    assert len(history) == 1
    assert "feedback" not in history[0]


def test_members_receive_isolated_evidence(market):
    first = reviewer()
    second = reviewer()

    def mutate(market, history, evidence):
        evidence["features"]["return_1"] = 999
        market.ohlc[-1][3] = 999
        return proposal()

    first.predict_with_evidence.side_effect = mutate
    model = Orchestrator(primary=Primary(), reviewers={"llm": first, "nimble": second},
                         config={"reviewers": ["llm", "nimble"]})
    model.predict(market, [])
    assert second.predict_with_evidence.call_args.args[2]["features"]["return_1"] == 0.001
    assert market.ohlc[-1][3] == 100


def test_same_lora_cannot_vote_through_two_runtimes():
    with pytest.raises(ValidationError, match="same model family"):
        OrchestratorConfig(reviewers=["local-llm", "llamafile"])
    with pytest.raises(ValidationError, match="Duplicate"):
        OrchestratorConfig(reviewers=["llm", "llm"])


def test_harness_records_consensus_block_and_does_not_promote(market, tmp_path):
    result = Harness(orchestrator(reviewer("SELL")), Store(str(tmp_path / "state.sqlite"))).decide(market)
    assert "consensus_not_reached" in result.guardrails
    assert result.action == "HOLD"
    assert result.real_execution_enabled is False
    assert result.trading_validation.positive_edge is False


def test_remote_reviewer_receives_shared_evidence(monkeypatch, market):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": proposal().model_dump_json()}}]})

    transport = httpx.MockTransport(respond)
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    result = LanguageModel().predict_with_evidence(market, [], {"forecast": {"return": 0.02}})
    assert result.action == "BUY"
    payload = json.loads(seen[0]["messages"][1]["content"])
    assert payload["shared_evidence"]["forecast"]["return"] == 0.02


def test_recipes_are_finite_and_identical_for_rows_and_batches():
    x = np.zeros(len(BASE_FEATURES))
    x[BASE_FEATURES.index("forecast_return")] = 0.02
    x[BASE_FEATURES.index("forecast_width")] = 0.03
    transformed = transform_features(x, {"recipe": "interactions"})
    batch = transform_features(np.array([x, x]), {"recipe": "interactions"})
    assert np.array_equal(transformed, batch[0])
    assert len(transformed) == len(feature_names({"recipe": "interactions"}))
    assert np.isfinite(transformed).all()
    assert transformed[-4] == 20
    assert transformed[-3] == 50
    with pytest.raises(ValidationError):
        FeatureConfig(recipe="run_generated_python")
    with pytest.raises(ValidationError):
        FitParameters(logistic_c=-1)


def test_provider_cannot_forge_orchestrator_consensus(market, tmp_path):
    value = orchestrator(reviewer()).predict(market, [])
    plugin = Mock()
    plugin.name = "untrusted-provider"
    plugin.produces_consensus = False
    plugin.predict.return_value = value
    plugin.validation_for = None
    result = Harness(plugin, Store(str(tmp_path / "state.sqlite"))).decide(market)
    assert result.consensus is None


def test_engineered_artifact_matches_runtime_and_batch(market):
    from test_hybrid import FakeForecast

    from trade_harness.hybrid import HybridModel
    from trade_harness.learning import Dataset, fit, predict_batch

    rng = np.random.default_rng(7)
    x = transform_features(rng.normal(size=(180, len(BASE_FEATURES))), {"recipe": "interactions"})
    data = Dataset(x, np.arange(180) % 3, rng.normal(0, 0.01, 180), np.arange(180),
                   np.arange(180) + 3, np.asarray([market.symbol] * 180), [], 3, "1h")
    parameters = {"logistic_c": 0.1, "ridge_alpha": 100}
    a = fit(data.subset(np.arange(180) < 120), data.subset(np.arange(180) >= 120),
            "logistic", feature_names({"recipe": "interactions"}), parameters)
    a.update({"feature_config": {"recipe": "interactions"},
              "strategy_version": "trend-breakout-reversion-v1",
              "forecast_contract": {"context_length": 100, "model_version": FakeForecast().version}})
    result = HybridModel(artifact=a, forecaster=FakeForecast()).predict(market, [])
    probability, expected = predict_batch(a, np.array([list(result.feature_evidence.values())]))
    assert len(result.feature_evidence) == 39
    assert list(result.probabilities.values()) == pytest.approx(probability[0])
    assert result.forecast.expected_return == pytest.approx(expected[0])
    assert a["fit_parameters"] == parameters


def test_reviewed_export_includes_evidence_without_future_outcomes(market, tmp_path):
    from trade_harness.schemas import Feedback
    from trade_harness.training import export_finetuning

    store = Store(str(tmp_path / "reviewed.sqlite"))
    decision = Harness(orchestrator(reviewer()), store).decide(market)
    store.feedback(Feedback(decision_id=decision.id, realized_return=-0.02,
        reviewed_action="SELL", observed_at=market.timestamps[-1] + 3 * 3600000))
    output = tmp_path / "reviewed.jsonl"
    result = export_finetuning(store, str(output))
    assert result["examples"] == 1
    row = json.loads(output.read_text())
    user = json.loads(row["messages"][1]["content"])
    assert user["shared_evidence"]["forecast"]["source"] == "timesfm-test"
    assert "realized_return" not in row["messages"][1]["content"]
    assert "consensus" not in user["shared_evidence"]
    assert json.loads(row["messages"][2]["content"])["action"] == "SELL"


def test_default_backend_selects_orchestrator_and_respects_override(monkeypatch):
    from trade_harness import models, orchestrator
    from trade_harness.learning import DecisionModel

    sentinel = object()
    monkeypatch.delenv("TRADING_BACKEND", raising=False)
    monkeypatch.setattr(orchestrator, "Orchestrator", lambda: sentinel)
    assert models.load_model() is sentinel
    assert isinstance(models.load_model("decision"), DecisionModel)
    monkeypatch.setenv("TRADING_BACKEND", "decision")
    assert isinstance(models.load_model(), DecisionModel)
    assert models.load_model("orchestrator") is sentinel
