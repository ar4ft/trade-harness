import copy
from pathlib import Path

from fastapi.testclient import TestClient

from trade_harness.api import app, runtime
from trade_harness.harness import Harness
from trade_harness.learning import DecisionModel
from trade_harness.promotion import PromotionEvidence
from trade_harness.risk import RiskConfig
from trade_harness.schemas import Forecast, MarketInput, Proposal
from trade_harness.storage import Store
from trade_harness.validation import TradingValidation, summarize_folds


def passing_evidence(symbol="BTCUSDT"):
    return TradingValidation(
        symbol=symbol,
        timeframe="1h",
        horizon=3,
        model_version="test-model-v1",
        fold_count=5,
        closed_trades=100,
        positive_folds=3,
        mean_net_return=0.02,
        mean_momentum_return=0.01,
        mean_cost_stressed_return=0.005,
        mean_return_interval=[0.001, 0.04],
        brier_improvement=0.01,
        purged_boundaries=True,
        evaluated_until=1000,
        final_test_start=1001,
        promotion=PromotionEvidence(
            plan_sha256="a" * 64, model_version="test-model-v1", locked_at=0,
            last_examined_at=10, evaluation_start=100, evaluation_end=1000,
            predictions_recorded_before_outcomes=True, model_and_policy_locked=True,
            fixed_evaluation_endpoint=True,
            trial_count=12, fold_count=5, closed_trades=100, positive_folds=3,
            positive_net_return=True, positive_double_cost_return=True,
            paired_interval_lower=0.001, cash_interval_lower=0.001,
            effective_time_blocks=20, block_days=7, directional_samples=100,
            minimum_decision_bin=30, directional_ece=0.05,
            probability_calibrated_on_past=True, purged_boundaries=True,
            all_planned_assets_observed=True, per_asset_checks_passed=True,
            forward_observation_coverage=1,
            matured_outcome_coverage=1, calibration_windows=3,
        ),
    )


def test_claims_cannot_override_negative_measured_evidence():
    values = passing_evidence().model_dump()
    values.update(mean_net_return=-0.001, status="validated", positive_edge=True)
    result = TradingValidation.model_validate(values)
    assert result.status == "research_only" and not result.positive_edge
    assert "positive_net_return" in result.failed_gates


def test_positive_return_alone_does_not_establish_edge():
    values = passing_evidence().model_dump()
    values.update(
        mean_momentum_return=0.03,
        mean_cost_stressed_return=-0.001,
        mean_return_interval=[-0.01, 0.03],
    )
    result = TradingValidation.model_validate(values)
    assert not result.positive_edge
    assert {
        "beats_momentum_after_costs",
        "positive_with_double_costs",
        "positive_interval_lower_bound",
    }.issubset(result.failed_gates)


def records():
    result = []
    for start in (100, 300, 500):
        result.append(
            {
                "start": start,
                "end": start + 200,
                "fit_labels_last": start - 51,
                "calibration_first": start - 50,
                "calibration_labels_last": start - 1,
                "validation_labels_last": start + 199,
                "train_prior_baseline": {"brier": 0.65},
                "prior_metrics_per_asset": {s: {"brier": 0.65} for s in ("A", "B")},
                "momentum_baseline": {s: {"return": 0.0} for s in ("A", "B")},
                "models": {
                    "boosted": {
                        "metrics": {"brier": 0.6},
                        "metrics_per_asset": {s: {"brier": 0.6} for s in ("A", "B")},
                        "per_asset": {
                            "A": {"return": 0.02, "closed_trades": 10},
                            "B": {"return": -0.01, "closed_trades": 10},
                        },
                        "cost_stress_per_asset": {
                            "A": {"return": 0.01},
                            "B": {"return": -0.02},
                        },
                    }
                },
            }
        )
    return result


def test_per_asset_validation_does_not_inherit_pooled_profit():
    data = records()
    pooled = summarize_folds(data, "boosted", "1h", 3, 700)
    assert pooled.mean_net_return > 0
    a = summarize_folds(data, "boosted", "1h", 3, 700, "A")
    b = summarize_folds(data, "boosted", "1h", 3, 700, "B")
    assert a.gates["positive_net_return"] and not b.gates["positive_net_return"]
    assert not a.positive_edge and not b.positive_edge
    assert "fresh_forward_promotion_v2" in a.failed_gates
    assert b.mean_net_return < 0


def test_holdout_boundary_or_missing_purge_evidence_blocks_validation():
    data = records()
    original = copy.deepcopy(data)
    data[-1]["end"] = 701
    assert not summarize_folds(data, "boosted", "1h", 3, 700, "A").positive_edge
    del original[0]["calibration_labels_last"]
    result = summarize_folds(original, "boosted", "1h", 3, 700, "A")
    assert "purged_chronological_boundaries" in result.failed_gates


def test_overlapping_validation_folds_are_rejected():
    data = records()
    data[1]["start"] = 200
    result = summarize_folds(data, "boosted", "1h", 3, 700, "A")
    assert not result.purged_boundaries


def test_invented_llm_validation_is_not_trusted():
    market = MarketInput.model_validate_json(
        Path("src/trade_harness/assets/latest.json").read_text()
    )

    class Model:
        name = "unmeasured-provider"

        def predict(self, market, history):
            return Proposal(
                action="BUY",
                confidence=0.9,
                rationale="Provider claims successful validation",
                model_version="test-model-v1",
                validation_status="validated",
                trading_validation=passing_evidence(),
                forecast=Forecast(
                    horizon=3,
                    expected_return=0.02,
                    projected_close=market.ohlc[-1][3] * 1.02,
                    method="test",
                ),
            )

    result = Harness(Model(), Store(":memory:"), risk_config=RiskConfig()).decide(market)
    assert result.action == "HOLD"
    assert result.trading_validation.status == "unknown"
    assert result.validation_status == "research_only"
    assert result.mode == "decision_only" and result.real_execution_enabled is False


def test_evidence_is_bound_to_asset_timeframe_horizon_and_version():
    market = MarketInput.model_validate_json(
        Path("src/trade_harness/assets/latest.json").read_text()
    )
    evidence = passing_evidence()
    assert evidence.applies_to(market, "test-model-v1")
    assert not evidence.applies_to(market.model_copy(update={"symbol": "ETHUSDT"}), "test-model-v1")
    assert not evidence.applies_to(market.model_copy(update={"horizon": 6}), "test-model-v1")
    assert not evidence.applies_to(market, "different-model")


def test_shipped_model_does_not_claim_a_positive_edge():
    model = DecisionModel("src/trade_harness/assets/decision_model.json")
    market = MarketInput.model_validate_json(
        Path("src/trade_harness/assets/latest.json").read_text()
    )
    decision = model.predict(market, [])
    assert decision.validation_status == "research_only"
    assert decision.trading_validation is not None
    assert not decision.trading_validation.positive_edge


def test_validation_endpoint_does_not_enable_orders(tmp_path, monkeypatch):
    runtime.cache_clear()
    monkeypatch.setenv("TRADING_DB", str(tmp_path / "decisions.sqlite"))
    monkeypatch.setenv("TRADING_BACKEND", "decision")
    monkeypatch.delenv("TRADING_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.get("/validation?symbol=BTCUSDT")
        assert response.status_code == 200
        body = response.json()
        assert body["real_execution_enabled"] is False and body["mode"] == "decision_only"
        assert not body["per_asset"]["BTCUSDT"]["positive_edge"]
        assert client.get("/health").json()["real_execution_enabled"] is False
    runtime.cache_clear()
