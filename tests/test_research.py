import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_hybrid import FakeForecast
from test_trading_validation import passing_evidence

from trade_harness.diagnostics import (
    feature_diagnostics,
    paired_block_interval,
    probability_diagnostics,
)
from trade_harness.feature_engineering import BASE_FEATURES, feature_names, transform_features
from trade_harness.hybrid_training import digest
from trade_harness.local_language import REVIEW_CONTRACT, LocalLanguageModel, review_prompt
from trade_harness.promotion import PromotionEvidence
from trade_harness.prospective import (
    ForwardJournal,
    read_journal,
    runtime_digest,
    summarize_forward,
)
from trade_harness.research_data import export_research, load_research, target_action
from trade_harness.research_evaluation import evaluate_research
from trade_harness.risk import Quote, RiskConfig
from trade_harness.schemas import Forecast, MarketInput, Proposal
from trade_harness.validation import TradingValidation


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(14)
    n = 1000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    market = MarketInput(symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[1700000000000 + i * 3600000 for i in range(n)],
        ohlc=[[float(p), float(p * 1.01), float(p * 0.99), float(p)] for p in close],
        volume=[1000] * n)
    source = tmp_path / "market.json"
    source.write_text(market.model_dump_json())
    output = tmp_path / "research.jsonl"
    export_research([str(source)], str(tmp_path / "forecasts.jsonl"), str(output),
                    stride=2, forecaster=FakeForecast(100))
    return output


def test_position_targets_use_costs_and_allowed_actions():
    config = RiskConfig()
    assert target_action(0.003, "flat", config)[0] == "HOLD"
    assert target_action(0.01, "flat", config)[0] == "BUY"
    assert target_action(-0.01, "flat", config)[0] == "HOLD"
    assert target_action(-0.01, "long", config)[0] == "SELL"
    assert target_action(0.01, "long", config)[0] == "HOLD"


def test_dataset_prompt_parity_purge_and_future_feedback(dataset):
    rows, manifest = load_research(dataset)
    assert manifest["prompt_contract"] == REVIEW_CONTRACT
    for phase in ("train", "calibration", "validation"):
        group = [r for r in rows if r["phase"] == phase]
        next_phase = {"train": "calibration", "calibration": "validation", "validation": "test"}[phase]
        assert max(r["outcome"]["observed_at"] for r in group) < manifest["boundaries"][next_phase]
    for row in rows:
        assert "next_open_return" not in row["prompt"]
        assert row["prompt"] == review_prompt(MarketInput.model_validate(row["market"]),
                                               row["history"], row["evidence"])
    row = rows[0]
    view = MarketInput.model_validate(row["market"])
    history = [{"symbol": view.symbol, "timeframe": view.timeframe, "as_of": row["as_of"] - 1,
                "action": "HOLD", "feedback": {"observed_at": row["as_of"] + 1,
                                               "realized_return": 0.9}}]
    first = review_prompt(view, history, row["evidence"])
    history[0]["feedback"]["realized_return"] = -0.5
    assert first == review_prompt(view, history, row["evidence"])


def test_dataset_rejects_tampering_and_duplicate_examples(dataset):
    rows, manifest = load_research(dataset)
    dataset.write_text(dataset.read_text() + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="digest"):
        load_research(dataset)
    manifest["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    Path(str(dataset) + ".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        load_research(dataset)


def test_train_only_signal_diagnostics_ignore_future_outcomes(dataset):
    rows, _ = load_research(dataset)
    before = feature_diagnostics(rows)
    for row in rows:
        if row["phase"] != "train":
            row["outcome"]["next_open_return"] = 100
    assert before == feature_diagnostics(rows)


def test_quant_features_are_causal_bounded_and_batch_equivalent():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, len(BASE_FEATURES)))
    result = transform_features(x, {"recipe": "quant"})
    assert result.shape == (50, len(feature_names({"recipe": "quant"})))
    assert np.isfinite(result).all()
    assert np.array_equal(result[0], transform_features(x[0], {"recipe": "quant"}))
    x[1:] *= 100
    assert np.array_equal(result[0], transform_features(x, {"recipe": "quant"})[0])


def test_decision_calibration_reports_class_and_action_bands():
    p = np.tile([0.8, 0.1, 0.1], (100, 1))
    report = probability_diagnostics(p, np.tile([0, 1], 50))
    assert report["directional"]["ece"] == pytest.approx(0.3)
    assert report["directional"]["samples"] == 100
    assert set(report["by_class"]) == {"BUY", "SELL", "HOLD"}
    assert probability_diagnostics(np.tile([0.1, 0.1, 0.8], (100, 1)), np.full(100, 2))["directional"]["samples"] == 0


def test_paired_block_bootstrap_preserves_pairing_and_warns_small_cohorts():
    stream = np.random.default_rng(1).normal(0, 0.01, 140)
    result = paired_block_interval(stream, stream)
    assert result["interval"] == [0, 0]
    assert result["effective_blocks"] == 20
    assert paired_block_interval(stream + 0.01, stream)["interval"][0] == pytest.approx(0.01)
    assert paired_block_interval(stream[:3], stream[:3])["effective_blocks"] == 1


def test_legacy_or_mismatched_promotion_cannot_certify_a_model():
    values = passing_evidence().model_dump()
    assert TradingValidation.model_validate(values).positive_edge
    values["promotion"]["model_version"] = "different-model"
    assert not TradingValidation.model_validate(values).positive_edge
    values.pop("promotion")
    values.update(status="validated", positive_edge=True)
    assert not TradingValidation.model_validate(values).positive_edge
    evidence = passing_evidence().promotion.model_dump()
    evidence.update(evaluation_start=5, accepted=True, failed_gates=[])
    assert not PromotionEvidence.model_validate(evidence).accepted
    evidence = passing_evidence().promotion.model_dump()
    evidence.update(horizon_dependency_days=21, block_days=7)
    assert "20_time_blocks" in PromotionEvidence.model_validate(evidence).failed_gates
    evidence = passing_evidence().promotion.model_dump()
    evidence.update(fixed_evaluation_endpoint=False, accepted=True, failed_gates=[])
    assert "predeclared_evaluation_endpoint" in PromotionEvidence.model_validate(evidence).failed_gates


class ControlledReviewer:
    name = "controlled-reviewer"
    version = "controlled-reviewer-v1"
    trained_until = 0
    metadata = {"feature_contract": REVIEW_CONTRACT}

    def predict_with_evidence(self, market, history, evidence):
        # Valid typed action on exactly the evidence supplied at this time.
        return Proposal(action="HOLD", confidence=0.8, model_version=self.version,
            rationale="Controlled test abstention", probabilities={"BUY": 0.1, "SELL": 0.1, "HOLD": 0.8},
            forecast=Forecast(horizon=market.horizon, expected_return=0,
                              projected_close=market.ohlc[-1][3], method="test"))


def test_full_replay_reports_abstention_costs_and_unavailable_edge(dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = str(tmp_path / "evaluation.json")
    report = evaluate_research(str(dataset), ["local-llm"], output, folds=5,
                               max_decisions=1, reviewers={"local-llm": ControlledReviewer()})
    assert len(report["folds"]) == 5 and report["final_test"]
    assert set(report["candidates"]) == {"observed_numerical", "observed_strategies", "numerical", "cash", "momentum", "strategy", "consensus_local-llm"}
    for candidate in report["candidates"].values():
        assert not candidate["promotion"]["accepted"]
        assert "fresh_locked_forward_period" in candidate["promotion"]["failed_gates"]
    ledger = Path(report["folds"][0]["models"]["consensus_local-llm"]["normal"]["ledger"])
    row = json.loads(ledger.read_text().splitlines()[0])
    assert row["decision"]["consensus"]["configured_members"] == 2
    assert row["decision"]["real_execution_enabled"] is False
    with pytest.raises(ValueError, match="already exists"):
        evaluate_research(str(dataset), ["local-llm"], output, reviewers={"local-llm": ControlledReviewer()})


def test_reviewer_requires_trained_feature_contract(dataset):
    rows, manifest = load_research(dataset)
    row = next(r for r in rows if r["phase"] == "validation")
    model = LocalLanguageModel.__new__(LocalLanguageModel)
    model.metadata = {"symbols": manifest["symbols"], "timeframe": "1h", "horizon": 3,
                      "feature_contract": REVIEW_CONTRACT, "feature_names": manifest["feature_names"],
                      "class_returns": {"BUY": 0.01, "SELL": -0.01, "HOLD": 0}}
    model.trained_until = 0
    prompts = []

    def score(prompt):
        prompts.append(prompt)
        return np.asarray([0.1, 0.1, 0.8])

    model.probabilities = score
    market = MarketInput.model_validate(row["market"])
    model.predict_with_evidence(market, row["history"], row["evidence"])
    assert prompts[0] == row["prompt"]
    with pytest.raises(ValueError, match="requires shared"):
        model.predict(market, [])
    row["evidence"]["features"]["future_price"] = 100
    with pytest.raises(ValueError, match="feature contract"):
        model.predict_with_evidence(market, [], row["evidence"])


def test_forward_journal_rejects_history_changes_and_detects_edits(tmp_path, monkeypatch):
    now = 1800000000000
    monkeypatch.setattr("trade_harness.prospective.time.time", lambda: now / 1000)
    model = ControlledReviewer()
    config = RiskConfig(allow_research=True)
    plan = {"policy": "research-promotion-v2", "model_version": model.version,
            "locked_at": now - 10000, "last_examined_at": now - 20000,
            "evaluation_not_before": now - 9999, "symbols": ["BTCUSDT"],
            "timeframe": "1h", "horizon": 3, "risk_config": RiskConfig().model_dump(),
            "runtime_sha256": runtime_digest()}
    plan["plan_sha256"] = digest(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    market = MarketInput(symbol="BTCUSDT", timeframe="1h", horizon=3,
                        timestamps=[now - 1000 - i * 3600000 for i in range(21)][::-1],
                        ohlc=[[100, 101, 99, 100]] * 21, volume=[1000] * 21)
    path = tmp_path / "journal.jsonl"
    journal = ForwardJournal(plan_path, path, model, config)
    journal.append(market, Quote(bid=100, ask=100, observed_at=now), now,
                   {"decision": None, "run_id": f"forward-{plan['plan_sha256'][:12]}-BTCUSDT"}, {})
    assert len(list(read_journal(path, plan))) == 1
    with pytest.raises(ValueError, match="advance"):
        journal.append(market, Quote(bid=100, ask=100, observed_at=now), now, {}, {})
    row = json.loads(path.read_text())
    row["as_of"] -= 100
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="chain"):
        list(read_journal(path, plan))


def test_paired_live_capture_and_matured_outcome_join(tmp_path, monkeypatch):
    from test_risk_paper import FixedModel

    from trade_harness.data import add_indicators
    from trade_harness.live import run_live

    now = 1800000000000
    monkeypatch.setattr("trade_harness.prospective.time.time", lambda: now / 1000)
    model = FixedModel()
    market = add_indicators(MarketInput(symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[now - 1000 - i * 3600000 for i in range(25)][::-1],
        ohlc=[[100, 100.2, 99.8, 100]] * 25, volume=[1000] * 25))
    plan = {"policy": "research-promotion-v2", "model_version": model.version,
            "locked_at": now - 10000, "last_examined_at": now - 20000,
            "evaluation_not_before": now - 9999, "symbols": ["BTCUSDT"],
            "timeframe": "1h", "horizon": 3, "risk_config": RiskConfig().model_dump(),
            "runtime_sha256": runtime_digest(), "trial_count": 12,
            "evaluation_end": now + 4 * 3600000}
    plan["plan_sha256"] = digest(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    class Feed:
        def fetch(self, *args):
            return market, Quote(bid=100, ask=100, observed_at=now), now

    journal = tmp_path / "observations.jsonl"
    run_live(model, db=str(tmp_path / "paper.sqlite"), steps=1, feed=Feed(),
             config=RiskConfig(allow_research=True), prospective_plan=str(plan_path),
             audit_output=str(journal))
    captured = list(read_journal(journal, plan))
    assert set(captured[0]["baselines"]) == {"momentum", "double_cost"}
    assert captured[0]["candidate"]["new_fills"]
    later = market.model_copy(update={
        "timestamps": market.timestamps + [market.timestamps[-1] + 3600000 * i for i in range(1, 5)],
        "ohlc": market.ohlc + [[101, 102, 100, 101]] * 4,
        "volume": market.volume + [1000] * 4, "indicators": []})
    source = tmp_path / "outcomes.json"
    source.write_text(later.model_dump_json())
    monkeypatch.setattr("trade_harness.prospective.time.time", lambda: (now + 5 * 3600000) / 1000)
    report = summarize_forward(plan_path, journal, [str(source)], tmp_path / "forward.json")
    assert report["matured_decisions"] == 1
    # Next open is 101; horizon close is 101. A close-to-close target would incorrectly say +1%.
    assert report["outcomes"][0]["next_open_return"] == 0
    assert report["promotion"]["predictions_recorded_before_outcomes"]
    assert not report["promotion"]["accepted"]
    assert "100_closed_trades" in report["promotion"]["failed_gates"]
    assert report["promotion"]["fixed_evaluation_endpoint"]
    monkeypatch.setattr("trade_harness.prospective.time.time", lambda: (now + 3.5 * 3600000) / 1000)
    early = summarize_forward(plan_path, journal, [str(source)], tmp_path / "early.json")
    assert not early["promotion"]["fixed_evaluation_endpoint"]
    assert "predeclared_evaluation_endpoint" in early["promotion"]["failed_gates"]
    # A sparse outcome archive cannot extend the declared target into later observations.
    later.timestamps[-2:] = [now + 10 * 3600000, now + 11 * 3600000]
    source.write_text(later.model_dump_json())
    with pytest.raises(ValueError, match="missing intervals"):
        summarize_forward(plan_path, journal, [str(source)], tmp_path / "bounded.json")


def test_cached_candidate_likelihood_matches_independent_teacher_forcing():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    model = LocalLanguageModel.__new__(LocalLanguageModel)
    model.model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2)).eval()
    encoded = torch.tensor([[1, 2, 3, 4, 5]])

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return encoded

    model.tokenizer = Tokenizer()
    model.candidates = [[6], [7, 8], [9, 10, 11]]
    scores = []
    with torch.inference_mode():
        for tokens in model.candidates:
            full = torch.cat([encoded, torch.tensor([tokens])], dim=1)
            logits = model.model(input_ids=full).logits[0].log_softmax(dim=-1)
            scores.append(sum(logits[encoded.shape[1] - 1 + i, token].item()
                              for i, token in enumerate(tokens)))
    expected = np.exp(np.asarray(scores) - max(scores))
    expected /= expected.sum()
    assert np.allclose(model.probabilities("same prompt"), expected, atol=1e-6)
    model.candidates.reverse()
    assert np.allclose(model.probabilities("same prompt"), expected[::-1], atol=1e-6)
