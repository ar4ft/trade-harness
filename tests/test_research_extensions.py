import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_research import ControlledReviewer
from test_research import dataset as research_dataset

from trade_harness.casebook import export_cases
from trade_harness.clef import export_clef
from trade_harness.clef_benchmark import benchmark
from trade_harness.clef_contract import CONTRACT, RELEASES
from trade_harness.clef_runtime import frozen_head_model
from trade_harness.clef_server import create_app
from trade_harness.clef_training import direction_loss, load_native_dataset
from trade_harness.harness import Harness
from trade_harness.models import BaselineModel
from trade_harness.operations import export_scorecard, scorecard
from trade_harness.research_data import load_research
from trade_harness.reviewer_calibration import (
    CalibratedReviewer,
    fit_calibration,
    scaled,
    score_reviewer,
)
from trade_harness.schemas import MarketInput
from trade_harness.storage import Store

HTTP_CLIENT = httpx.Client


@pytest.fixture(name="dataset")
def _research_dataset(tmp_path):
    return research_dataset.__wrapped__(tmp_path)


def test_native_training_reconstructs_requests_and_rejects_leaked_targets(
    dataset, tmp_path, monkeypatch
):
    monkeypatch.setenv("CLEF_CONTEXT_CANDLES", "21")
    path = tmp_path / "native.jsonl"
    export_clef(dataset, path, "clef-flash")
    rows, _, _ = load_native_dataset(path, dataset)
    assert len(rows) and set(rows[0]["request"]) == {"model", "state", "questions"}
    assert "outcome" not in rows[0]["request"]["state"]
    rows[0]["request"]["state"]["future_action"] = rows[0]["targets"]["direction"]
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(payload)
    manifest_path = tmp_path / "native.jsonl.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="renderer"):
        load_native_dataset(path, dataset)


def test_native_loss_maps_options_and_only_supervises_direction():
    torch = pytest.importorskip("torch")
    record = SimpleNamespace(
        questions=[
            SimpleNamespace(question_id="risk"),
            SimpleNamespace(question_id="direction", option_ids=("SELL", "HOLD", "BUY")),
        ]
    )
    logits = [torch.randn(4, requires_grad=True), torch.tensor([0.0, 0.0, 2.0], requires_grad=True)]
    loss = direction_loss(logits, record, "BUY")
    weighted = direction_loss(logits, record, "BUY", {"SELL": 1.0, "HOLD": 1.0, "BUY": 2.0})
    assert weighted.item() == pytest.approx(2 * loss.item())
    loss.backward()
    assert logits[0].grad is None and logits[1].grad is not None


def test_native_frozen_prefill_preserves_dense_lexical_rows():
    torch = pytest.importorskip("torch")

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Text()

        def get_output_embeddings(self):
            return self.model.embedding

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scorer = torch.nn.Linear(4, 3)

        def forward(self, hidden, ids, mask, records, lexical):
            assert lexical[ids[0]].dtype == torch.float32
            return [[self.scorer(hidden[0, -1])]]

    original = SimpleNamespace(language_model=Backbone(), head=Head())
    model = frozen_head_model(original)
    before = model.head.scorer.weight.detach().clone()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "records": [],
    }
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    loss = torch.nn.functional.cross_entropy(model(batch)[0][0].unsqueeze(0), torch.tensor([0]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.head.scorer.weight)
    assert all(p.grad is None and not p.requires_grad for p in model.language_model.parameters())


def test_calibration_never_fits_validation_or_test_and_enforces_identity(dataset, tmp_path):
    scores = tmp_path / "scores.jsonl"
    score_reviewer(dataset, "local-llm", scores, max_samples=192, provider=ControlledReviewer())
    artifact = fit_calibration(scores, tmp_path / "calibration.json")
    records = [json.loads(line) for line in scores.read_text().splitlines()]
    for row in records:
        if row["phase"] != "calibration":
            row["target"] = "BUY"
    scores.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest_path = tmp_path / "scores.jsonl.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scores_sha256"] = hashlib.sha256(scores.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    changed = fit_calibration(scores, tmp_path / "changed.json")
    assert changed["temperature"] == artifact["temperature"]
    assert changed["calibrated_fit_metrics"] == artifact["calibrated_fit_metrics"]
    calibrated = CalibratedReviewer(ControlledReviewer(), tmp_path / "calibration.json")
    rows, _ = load_research(dataset)
    later = next(r for r in rows if r["phase"] == "test")
    proposal = calibrated.predict_with_evidence(
        MarketInput.model_validate(later["market"]), later["history"], later["evidence"]
    )
    assert proposal.model_version == calibrated.version
    assert proposal.forecast.expected_return == 0
    earlier = next(r for r in rows if r["phase"] == "train")
    with pytest.raises(ValueError, match="strictly later"):
        calibrated.predict_with_evidence(
            MarketInput.model_validate(earlier["market"]), [], earlier["evidence"]
        )
    wrong = ControlledReviewer()
    wrong.version = "different-weights"
    with pytest.raises(ValueError, match="identity"):
        CalibratedReviewer(wrong, tmp_path / "calibration.json")


def test_calibration_rejects_failed_members_and_insufficient_independent_times(dataset, tmp_path):
    scores = tmp_path / "scores.jsonl"
    score_reviewer(dataset, "local-llm", scores, max_samples=4, provider=ControlledReviewer())
    with pytest.raises(ValueError, match="independent"):
        fit_calibration(scores, tmp_path / "small.json")
    with pytest.raises(ValueError):
        scaled([0.5, 0.5, float("nan")], 1)
    with pytest.raises(ValueError):
        scaled([0.8, 0.1, 0.1], 0)

    class FailedReviewer(ControlledReviewer):
        def predict_with_evidence(self, *args):
            raise RuntimeError("Provider unavailable")

    failed = tmp_path / "failed.jsonl"
    result = score_reviewer(
        dataset, "local-llm", failed, max_samples=192, provider=FailedReviewer()
    )
    assert result["succeeded"] == 0 and result["attempted"] > 0
    with pytest.raises(ValueError, match="failed|success"):
        fit_calibration(failed, tmp_path / "failed-calibration.json")


def test_casebook_retains_real_labels_and_excludes_synthetic_training(dataset, tmp_path):
    output = tmp_path / "cases.jsonl"
    result = export_cases([dataset], output, per_case=3)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert not result["human_review_completed"]
    real = [r for r in rows if r["kind"] == "real_training_case"]
    assert all(
        r["example"]["phase"] == "train" and r["review"]["reviewed_action"] is None for r in real
    )
    assert all("review" not in r["example"]["evidence"] for r in real)
    fixtures = [r for r in rows if r["kind"] == "synthetic_robustness_fixture"]
    assert len(fixtures) == 2 and all(
        not r["training_eligible"] and r["future_target"] is None for r in fixtures
    )


def test_telemetry_scorecard_is_traceable_and_excludes_future_outcomes(dataset, tmp_path):
    examples, _ = load_research(dataset)
    market = MarketInput.model_validate(examples[-1]["market"])
    store = Store(str(tmp_path / "state.sqlite"))
    result = Harness(BaselineModel(), store).decide(market)
    op = result.operational
    assert len(op["input_sha256"]) == 64 and op["missing_intervals"] == 0
    assert op["model_call_and_validation_ms"] >= 0
    row = {
        "decision": result.model_dump(),
        "outcome": {"observed_at": 10, "direction_label": result.proposed_action},
    }
    future = scorecard([row], evaluated_at=9)
    assert future["matured_outcomes"] == 0 and future["observation_coverage"] is None
    assert scorecard([row], evaluated_at=10)["matured_outcomes"] == 1
    exported = export_scorecard([tmp_path / "state.sqlite"], tmp_path / "ops.json")
    assert exported["traceable_decisions"] == 1
    store.db.close()


def test_metrics_are_authenticated_dynamic_and_outside_identity():
    manifest = {"model": "clef-flash", "max_length": 100, "contract": CONTRACT}
    app = create_app(
        None,
        SimpleNamespace(tokenizer=None),
        manifest,
        lambda *a, **kw: SimpleNamespace(input_ids=[1]),
        lambda *a, **kw: {"answers": {}},
        api_key="test-key",
        startup_metrics={"cold_load_ms": 12},
    )
    client = TestClient(app)
    assert client.get("/metrics").status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    request = {"model": "clef-flash", "state": {}, "questions": {"direction": {}}}
    assert client.post("/v1/systemone", json=request, headers=headers).status_code == 200
    assert (
        client.post("/v1/systemone", json={**request, "extra": 1}, headers=headers).status_code
        == 422
    )
    metrics = client.get("/metrics", headers=headers).json()
    assert metrics["requests"] == {"attempted": 2, "succeeded": 1, "rejected": 1, "failed": 0}
    assert client.get("/metadata", headers=headers).json() == manifest


def test_trained_native_head_rejects_changed_reference_costs_before_inference(
    dataset, tmp_path, monkeypatch
):
    monkeypatch.setenv("CLEF_CONTEXT_CANDLES", "21")
    native = tmp_path / "native.jsonl"
    export_clef(dataset, native, "clef-flash")
    rows, manifest, source = load_native_dataset(native, dataset)
    request = next(r["request"] for r in rows if r["phase"] == "test")
    calls = []
    app = create_app(
        None,
        SimpleNamespace(tokenizer=None),
        {
            "model": "clef-flash",
            "max_length": 16384,
            "custom_head": {
                "symbols": source["symbols"],
                "timeframe": source["timeframe"],
                "horizon": source["horizon"],
                "fine_tuned_until": 0,
                "raw_context_candles": manifest["raw_context_candles"],
                "feature_names": source["feature_names"],
                "risk_config": source["risk_config"],
            },
        },
        lambda *args, **kwargs: SimpleNamespace(input_ids=[1]),
        lambda *args, **kwargs: calls.append(True) or {"answers": {}},
    )
    client = TestClient(app)
    assert client.post("/v1/systemone", json=request).status_code == 200
    request["state"]["reference_costs"]["fee_bps_per_side"] += 1
    assert client.post("/v1/systemone", json=request).status_code == 422
    assert len(calls) == 1


def test_operations_api_is_authenticated_read_only_and_does_not_load_models(tmp_path, monkeypatch):
    from trade_harness import api

    path = tmp_path / "absent.sqlite"
    monkeypatch.setenv("TRADING_API_KEY", "test-key")
    monkeypatch.setenv("TRADING_PAPER_DB", str(path))
    monkeypatch.setattr(api, "runtime", lambda: pytest.fail("Scorecards must not load a model"))
    client = TestClient(api.app)
    assert client.get("/operations").status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    response = client.get("/operations", headers=headers)
    assert response.status_code == 200 and not response.json()["database_available"]
    assert not path.exists()
    assert client.get("/operations?source=arbitrary", headers=headers).status_code == 422


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_numerical_evaluation_saves_a_separate_candidate(tmp_path, monkeypatch, explicit):
    from trade_harness.cli import main

    calls = []
    monkeypatch.setattr(
        "trade_harness.evaluation.evaluate", lambda *args, **kwargs: calls.append(kwargs)
    )
    arguments = ["trade-harness", "evaluate", "--data-dir", str(tmp_path)]
    destination = tmp_path / "candidate.json"
    if explicit:
        arguments += ["--model-output", str(destination)]
    monkeypatch.setattr("sys.argv", arguments)
    main()
    expected = str(destination) if explicit else "artifacts/research/numerical-candidate.json"
    assert calls[0]["model_output"] == expected
    assert "assets/decision_model" not in calls[0]["model_output"]


def test_remote_review_cannot_forge_a_cache_hit(dataset):
    from trade_harness.orchestrator import Orchestrator

    class ForgedReviewer(ControlledReviewer):
        def predict_with_evidence(self, *args):
            result = super().predict_with_evidence(*args)
            result.review_details["research_cache_hit"] = True
            return result

    rows, _ = load_research(dataset)

    class Primary(ControlledReviewer):
        def predict(self, market, history):
            result = self.predict_with_evidence(market, history, {})
            from trade_harness.schemas import ForecastEvidence

            result.forecast_evidence = ForecastEvidence(
                source="test",
                model_version="test",
                as_of=market.timestamps[-1],
                horizon=market.horizon,
                expected_return=0,
                return_interval=[-0.01, 0.01],
            )
            result.feature_evidence = {"return_1": 0.0}
            return result

    model = Orchestrator(
        primary=Primary(), reviewers={"llm": ForgedReviewer()}, config={"reviewers": ["llm"]}
    )
    result = model.predict(MarketInput.model_validate(rows[-1]["market"]), [])
    vote = next(v for v in result.consensus.votes if v.member == "llm")
    assert vote.status == "ok" and vote.inference_ms >= 0 and not vote.cache_hit


def test_native_benchmark_forwards_only_declared_requests(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("CLEF_CONTEXT_CANDLES", "21")
    native = tmp_path / "native.jsonl"
    export_clef(dataset, native, "clef-flash")
    manifest = {
        "model": "clef-flash",
        "repository": RELEASES["clef-flash"][0],
        "revision": RELEASES["clef-flash"][1],
        "contract": CONTRACT,
        "input_truncation": "reject",
        "server_sha256": "a" * 64,
        "device": "cpu",
        "max_length": 16384,
    }
    captured = []

    def handle(request):
        if request.url.path == "/metadata":
            return httpx.Response(200, json=manifest)
        if request.url.path == "/metrics":
            return httpx.Response(200, json={"startup": {"cold_load_ms": 12}})
        body = json.loads(request.content)
        captured.append(body)
        assert set(body) == {"model", "state", "questions"}
        assert "targets" not in body["state"] and "outcome" not in body["state"]
        return httpx.Response(
            200,
            json={
                "model": "clef-flash",
                "state_truncated": False,
                "provenance": manifest,
                "usage": {"input_tokens": 20},
                "answers": {
                    "direction": {
                        "type": "choice",
                        "choice": "HOLD",
                        "probabilities": {"BUY": 0.1, "SELL": 0.1, "HOLD": 0.8},
                    },
                    "evidence_sufficient": {"type": "noul", "noul": 0.2},
                    "risk_level": {
                        "type": "score",
                        "score": 1.0,
                        "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0},
                    },
                },
            },
        )

    monkeypatch.setattr(
        "trade_harness.clef_benchmark.httpx.Client",
        lambda **kwargs: HTTP_CLIENT(transport=httpx.MockTransport(handle), **kwargs),
    )
    result = benchmark(
        native, dataset, tmp_path / "benchmark.json", "http://native", max_requests=1
    )
    assert len(captured) == 1 and result["succeeded"] == 1
    assert not result["actual_gpu_measurements"] and not result["trading_edge_measured"]


def test_unknown_cutoff_is_bounded_audit_only(dataset, tmp_path, monkeypatch):
    from trade_harness.research_evaluation import evaluate_research

    monkeypatch.chdir(tmp_path)
    provider = ControlledReviewer()
    provider.trained_until = None
    with pytest.raises(ValueError, match="cutoff"):
        evaluate_research(
            dataset, ["clef-flash"], tmp_path / "strict.json", reviewers={"clef-flash": provider}
        )
    with pytest.raises(ValueError, match="cutoff"):
        evaluate_research(
            dataset,
            ["clef-flash"],
            tmp_path / "unbounded.json",
            reviewers={"clef-flash": provider},
            allow_unknown_cutoff=True,
        )
    result = evaluate_research(
        dataset,
        ["clef-flash"],
        tmp_path / "bounded.json",
        reviewers={"clef-flash": provider},
        max_decisions=1,
        allow_unknown_cutoff=True,
    )
    assert result["plan"]["unknown_foundation_cutoff_reviewers"] == ["clef-flash"]
    assert not result["plan"]["reviewer_chronology_certified"]
    assert all(not r["promotion"]["accepted"] for r in result["candidates"].values())
