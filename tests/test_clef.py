import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from trade_harness.clef import ClefModel, export_clef, review_state
from trade_harness.clef_contract import CONTRACT, MAX_REQUEST_BYTES, QUESTIONS, RELEASES
from trade_harness.clef_server import create_app
from trade_harness.data import add_indicators
from trade_harness.harness import Harness
from trade_harness.models import load_model
from trade_harness.orchestrator import Orchestrator, OrchestratorConfig
from trade_harness.schemas import Forecast, ForecastEvidence, MarketInput, Proposal
from trade_harness.storage import Store

HTTP_CLIENT = httpx.Client


@pytest.fixture
def market():
    return add_indicators(
        MarketInput(
            symbol="BTCUSDT",
            timeframe="1h",
            horizon=3,
            timestamps=[1700000000000 + i * 3600000 for i in range(30)],
            ohlc=[[100, 101, 99, 100]] * 30,
            volume=[1000] * 30,
        )
    )


def evidence(market):
    return {
        "as_of": market.timestamps[-1],
        "symbol": market.symbol,
        "timeframe": market.timeframe,
        "horizon": market.horizon,
        "position": market.position,
        "forecast": ForecastEvidence(
            source="timesfm-test",
            model_version="tf-v1",
            as_of=market.timestamps[-1],
            horizon=3,
            expected_return=0.02,
            return_interval=[0.01, 0.03],
        ).model_dump(),
        "features": {"return_1": 0.001},
        "strategies": [],
    }


def response(variant="clef"):
    return {
        "model": variant,
        "answers": {
            "direction": {
                "type": "choice",
                "choice": "BUY",
                "confidence": 0.7,
                "probabilities": {"BUY": 0.8, "SELL": 0.1, "HOLD": 0.1},
            },
            "evidence_sufficient": {"type": "noul", "noul": 0.9},
            "risk_level": {
                "type": "score",
                "score": 1,
                "confidence": 0.4,
                "probabilities": {"0": 0.3, "1": 0.4, "2": 0.3, "3": 0.0},
            },
        },
        "usage": {"input_tokens": 1000, "output_tokens": 0},
    }


def manifest(variant="clef"):
    return {
        "model": variant,
        "repository": RELEASES[variant][0],
        "revision": RELEASES[variant][1],
        "contract": CONTRACT,
        "input_truncation": "reject",
        "max_length": 16384,
        "server_sha256": "a" * 64,
    }


def transport(monkeypatch, body=None, local=False, variant="clef", status=200):
    calls = []
    original = HTTP_CLIENT
    for key in ("CLEF_BASE_URL", "CLEF_REVISION", "CLEF_API_KEY", "CLEF_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "test-token")
    if local:
        monkeypatch.setenv("CLEF_BASE_URL", "http://127.0.0.1:11436")
    answer = copy.deepcopy(body if body is not None else response(variant))
    if local:
        answer.update(provenance=manifest(variant), state_truncated=False)

    def handler(request):
        calls.append(request)
        if request.url.path == "/metadata":
            return httpx.Response(200, json=manifest(variant))
        if local:
            assert request.url.path == "/v1/systemone"
        else:
            assert request.url.path.endswith("/ai/run/@cf/cloudflare/" + variant)
            assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            status, json=answer if local else {"success": True, "errors": [], "result": answer}
        )

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    return calls


@pytest.mark.parametrize("variant", ["clef", "clef-flash"])
def test_native_workers_review_preserves_independent_forecast(market, monkeypatch, variant):
    calls = transport(monkeypatch, variant=variant)
    source = Mock()
    model = ClefModel(variant, source)
    value = model.predict_with_evidence(market, [], evidence(market))
    assert value.action == "BUY" and value.confidence == 0.8
    assert value.forecast.method == "timesfm-test" and value.forecast.projected_close == 102
    assert (
        value.probability_calibration == "uncalibrated"
        and value.validation_status == "research_only"
    )
    assert value.review_details["risk_level"]["score"] == 1
    assert value.review_details["evidence_sufficient"] == 0.9
    assert not model.supports_locked_forward and model.trained_until is None
    source.predict.assert_not_called()
    request = json.loads(calls[-1].content)
    assert request["questions"] == QUESTIONS
    assert request["state"]["shared_evidence"] == evidence(market)
    assert (
        "primary_action" not in request["state"]
        and "numerical_direction_probabilities" not in request["state"]
    )
    assert request["state"]["reference_costs"]["semantics"].startswith("Research reference")


def test_local_pin_drift_and_response_provenance(market, monkeypatch):
    transport(monkeypatch, local=True)
    model = load_model("clef")
    value = model.predict_with_evidence(market, [], evidence(market))
    assert model.supports_locked_forward and value.review_details["weights_pinned"]
    assert value.review_details["manifest"]["revision"] == RELEASES["clef"][1]
    monkeypatch.setattr(model, "_manifest", lambda: {**manifest(), "revision": "b" * 40})
    with pytest.raises(ValueError, match="changed"):
        model.predict_with_evidence(market, [], evidence(market))


def test_local_unpinned_snapshot_rejected(monkeypatch):
    transport(monkeypatch, local=True)
    monkeypatch.setenv("CLEF_REVISION", "b" * 40)
    with pytest.raises(ValueError, match="pinned release"):
        ClefModel()


@pytest.mark.parametrize(
    "change",
    [
        "wrong_model",
        "missing_question",
        "wrong_choice",
        "extra_option",
        "negative",
        "nan",
        "bad_total",
        "bad_noul",
        "bad_score",
        "truncated",
    ],
)
def test_invalid_decisions_rejected(market, monkeypatch, change):
    body = response()
    direction = body["answers"]["direction"]
    if change == "wrong_model":
        body["model"] = "clef-flash"
    elif change == "missing_question":
        del body["answers"]["risk_level"]
    elif change == "wrong_choice":
        direction["choice"] = "SELL"
    elif change == "extra_option":
        direction["probabilities"]["SHORT"] = 0
    elif change == "negative":
        direction["probabilities"]["BUY"] = -0.8
    elif change == "nan":
        direction["probabilities"]["BUY"] = "nan"
    elif change == "bad_total":
        direction["probabilities"]["BUY"] = 0.9
    elif change == "bad_noul":
        body["answers"]["evidence_sufficient"]["noul"] = 1.1
    elif change == "bad_score":
        body["answers"]["risk_level"]["score"] = 3
    else:
        body["state_truncated"] = True
    transport(monkeypatch, body)
    with pytest.raises(ValueError):
        ClefModel().predict_with_evidence(market, [], evidence(market))


def test_four_decimal_rounding_and_ordinal_order(market, monkeypatch):
    body = response()
    body["answers"]["direction"]["probabilities"] = {"BUY": 0.3334, "SELL": 0.3333, "HOLD": 0.3334}
    body["answers"]["risk_level"]["probabilities"] = {"3": 0, "2": 0.3, "1": 0.4, "0": 0.3}
    transport(monkeypatch, body)
    value = ClefModel().predict_with_evidence(market, [], evidence(market))
    assert sum(value.probabilities.values()) == pytest.approx(1)
    assert value.review_details["risk_level"]["score"] == 1


def test_stale_evidence_and_oversized_state_not_sent(market, monkeypatch):
    calls = transport(monkeypatch)
    model = ClefModel()
    stale = evidence(market)
    stale["as_of"] -= 1
    with pytest.raises(ValueError, match="contract mismatch"):
        model.predict_with_evidence(market, [], stale)
    oversized = evidence(market)
    oversized["rules"] = "x" * MAX_REQUEST_BYTES
    with pytest.raises(ValueError, match="byte budget"):
        model.predict_with_evidence(market, [], oversized)
    assert not calls


def test_past_only_feedback_and_context_alignment(market):
    now = market.timestamps[-1]
    history = [
        {
            "symbol": market.symbol,
            "timeframe": "1h",
            "as_of": now - 3600000,
            "action": "BUY",
            "feedback": {"observed_at": now + 1, "realized_return": 0.9},
        },
        {"symbol": market.symbol, "timeframe": "1h", "as_of": now, "action": "SELL"},
    ]
    state = review_state(market, history, evidence(market))
    assert len(state["past_decisions"]) == 1
    assert "feedback" not in state["past_decisions"][0]
    assert len(state["market"]["timestamps"]) == len(state["market"]["ohlc"])


def test_provider_failure_stays_hold_and_hides_secrets(market, monkeypatch, tmp_path):
    transport(monkeypatch, status=401)
    result = Harness(ClefModel(), Store(str(tmp_path / "state.sqlite"))).decide(market)
    assert result.action == "HOLD" and "model_failure" in result.guardrails
    assert "test-token" not in result.model_dump_json()
    assert not result.real_execution_enabled


def test_native_review_can_confirm_or_abstain_under_existing_policy(market, monkeypatch):
    transport(monkeypatch)
    primary = Mock()
    primary.name, primary.version, primary.artifact = "numerical", "v1", {}
    primary.predict.return_value = Proposal(
        action="BUY",
        confidence=0.8,
        rationale="Controlled numerical proposal",
        forecast=Forecast(horizon=3, expected_return=0.02, projected_close=102, method="test"),
        forecast_evidence=ForecastEvidence.model_validate(evidence(market)["forecast"]),
    )
    model = Orchestrator(
        primary=primary, reviewers={"clef": ClefModel()}, config={"reviewers": ["clef"]}
    )
    result = model.predict(market, [])
    assert result.consensus.accepted and result.action == "BUY"
    assert result.consensus.votes[-1].review_details["provider"] == "workers-ai"
    hold = response()
    hold["answers"]["direction"].update(
        choice="HOLD", probabilities={"BUY": 0.1, "SELL": 0.1, "HOLD": 0.8}
    )
    transport(monkeypatch, hold)
    result = model.predict(market, [])
    assert result.action == "HOLD" and not result.consensus.accepted
    assert "insufficient_agreement" in result.consensus.reason_codes
    assert result.consensus.votes[-1].status == "ok"


def test_variants_cannot_inflate_consensus():
    with pytest.raises(ValueError, match="separate candidates"):
        OrchestratorConfig(reviewers=["clef", "clef-flash"])


@pytest.mark.parametrize("value", ["0", "601", "nan", "inf"])
def test_timeout_is_bounded(monkeypatch, value):
    monkeypatch.setenv("CLEF_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="timeout"):
        ClefModel()


def test_declared_short_context_retains_shared_evidence_and_changes_identity(market, monkeypatch):
    transport(monkeypatch)
    full = ClefModel()
    monkeypatch.setenv("CLEF_CONTEXT_CANDLES", "21")
    compact = ClefModel()
    assert compact.version != full.version
    calls = transport(monkeypatch)
    compact.predict_with_evidence(market, [], evidence(market))
    state = json.loads(calls[-1].content)["state"]
    assert state["input_scope"]["raw_context_candles"] == 21
    assert len(state["market"]["ohlc"]) == len(state["market"]["indicators"][0]["values"]) == 21
    assert state["shared_evidence"] == evidence(market)


def test_separate_server_rejects_token_overflow_before_inference():
    processor = SimpleNamespace(tokenizer=Mock())
    scorer, encoder = (
        Mock(return_value=response()),
        Mock(return_value=SimpleNamespace(input_ids=list(range(11)))),
    )
    meta = {**manifest(), "max_length": 10}
    app = create_app(Mock(), processor, meta, encoder, scorer, api_key="local-test")
    with TestClient(app) as client:
        assert client.get("/metadata").status_code == 401
        assert (
            client.get("/metadata", headers={"Authorization": "Bearer local-test"}).json() == meta
        )
        request = {"model": "clef", "state": {}, "questions": QUESTIONS}
        assert (
            client.post(
                "/v1/systemone", json=request, headers={"Authorization": "Bearer local-test"}
            ).status_code
            == 422
        )
        scorer.assert_not_called()
        encoder.return_value = SimpleNamespace(input_ids=list(range(10)))
        result = client.post(
            "/v1/systemone", json=request, headers={"Authorization": "Bearer local-test"}
        )
        assert result.status_code == 200
        assert result.json()["provenance"] == meta and result.json()["state_truncated"] is False


def test_native_export_preserves_phase_targets_and_prompt_outcome_separation(tmp_path):
    # Use the same synthetic causal export fixture as the complete-policy tests.
    from test_research import dataset as research_fixture

    dataset_path = research_fixture.__wrapped__(tmp_path)
    output = tmp_path / "clef.jsonl"
    result = export_clef(dataset_path, output)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    source = [json.loads(line) for line in Path(dataset_path).read_text().splitlines()]
    assert result["targeted_questions"] == ["direction"]
    assert len(records) == len(source)
    for exported, original in zip(records, source):
        assert exported["phase"] == original["phase"]
        assert exported["targets"]["direction"] == original["label"]
        assert "outcome" not in exported["request"]["state"]
        assert "label" not in exported["request"]["state"]
        assert exported["request"]["state"] == review_state(
            MarketInput.model_validate(original["market"]),
            original["history"],
            original["evidence"],
        )
    with pytest.raises(ValueError, match="immutable"):
        export_clef(dataset_path, output)


def test_mutable_hosted_weights_cannot_lock_forward_or_certify_old_history(tmp_path, monkeypatch):
    from test_research import dataset as research_fixture

    from trade_harness.research_data import load_research
    from trade_harness.research_evaluation import evaluate_research, lock_research

    dataset_path = research_fixture.__wrapped__(tmp_path)
    _, source = load_research(dataset_path)
    calls = transport(monkeypatch)
    hosted = ClefModel()
    primary = SimpleNamespace(feature_names=source["feature_names"])
    monkeypatch.setattr(
        "trade_harness.research_evaluation.load_model",
        lambda name: primary if name == "hybrid" else hosted,
    )
    with pytest.raises(ValueError, match="mutable hosted aliases"):
        lock_research(dataset_path, ["clef"], tmp_path / "plan.json")
    with pytest.raises(ValueError, match="known training cutoff"):
        evaluate_research(
            dataset_path, ["clef"], tmp_path / "evaluation.json", reviewers={"clef": hosted}
        )
    assert not calls
    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "evaluation.json").exists()
