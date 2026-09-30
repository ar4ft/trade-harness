from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from trade_harness.api import app, paper_store
from trade_harness.data import add_indicators
from trade_harness.harness import Harness
from trade_harness.paper import PaperEngine
from trade_harness.risk import Account, Quote, RiskConfig, evaluate
from trade_harness.schemas import Forecast, MarketInput, Proposal
from trade_harness.storage import Store, timeframe_ms
from trade_harness.typed import ChoiceResult, ScoreResult
from trade_harness.validation import TradingValidation


@pytest.fixture
def market():
    start = 1700000000000
    return add_indicators(
        MarketInput(
            symbol="BTCUSDT",
            timeframe="1h",
            horizon=3,
            timestamps=[start + i * 3600000 for i in range(25)],
            ohlc=[[100, 100.2, 99.8, 100] for _ in range(25)],
            volume=[1000] * 25,
        )
    )


class FixedModel:
    name = "fixed-test-model"
    version = "test-v1"
    uses_history = False
    trained_until = 0

    def __init__(self, action="BUY", expected=0.02, status="validated"):
        self.action, self.expected, self.status = action, expected, status

    def predict(self, market, history):
        return Proposal(
            action=self.action,
            confidence=0.9,
            rationale="Controlled test signal",
            probabilities={a: 0.9 if a == self.action else 0.05 for a in ("BUY", "SELL", "HOLD")},
            validation_status=self.status,
            model_version=self.version,
            trading_validation=self.validation_for(market.symbol),
            probability_calibration="test",
            forecast=Forecast(
                horizon=market.horizon,
                expected_return=self.expected,
                projected_close=market.ohlc[-1][3] * (1 + self.expected),
                method="test",
            ),
        )

    def validation_for(self, symbol):
        # Synthetic passing evidence isolates risk/persistence behavior in these unit tests.
        return TradingValidation(
            symbol=symbol,
            timeframe="1h",
            horizon=3,
            model_version=self.version,
            fold_count=3,
            closed_trades=30,
            positive_folds=3,
            mean_net_return=0.02,
            mean_momentum_return=0.01,
            mean_cost_stressed_return=0.01,
            mean_return_interval=[0.01, 0.03],
            brier_improvement=0.01,
            purged_boundaries=True,
            evaluated_until=1600000000000,
            final_test_start=1600000000001,
        )


def account():
    return Account(cash=10000, peak_equity=10000, day_start_equity=10000)


def holding(price=100):
    return Account(
        cash=8000,
        units=20,
        entry_price=price,
        entry_at=1700000000000,
        entry_cost=2000,
        peak_equity=10000,
        day_start_equity=10000,
        stop_loss=99,
        take_profit=102,
    )


def test_probability_and_typed_contracts():
    with pytest.raises(ValidationError):
        ChoiceResult(choice="BUY", probabilities={"BUY": 0.2, "HOLD": 0.8}, source="test")
    with pytest.raises(ValidationError):
        ChoiceResult(choice="BUY", probabilities={"BUY": 0.9, "SELL": 0.9}, source="test")
    with pytest.raises(ValidationError):
        ScoreResult(score=2, probabilities={"LOW": 0.5, "HIGH": 0.5}, source="test")


def test_sizing_includes_costs_and_exposure(market):
    config = RiskConfig()
    signal = FixedModel().predict(market, [])
    result = evaluate(market, signal, account(), config, 100)
    assert result.action == "BUY"
    assert result.notional <= 2000
    distance = 1 - result.stop_loss / 100
    assert result.notional * (distance + config.roundtrip_cost) <= 10000 * config.risk_per_trade
    assert result.take_profit - 100 >= 1.5 * (100 - result.stop_loss)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("unvalidated", "no_validated_edge"),
        ("costs", "insufficient_edge_after_costs"),
        ("stale", "stale_or_future_candles"),
        ("missing", "missing_candle_intervals"),
        ("cooldown", "cooldown"),
        ("spread", "insufficient_edge_after_costs"),
    ],
)
def test_entry_blocks(market, case, reason):
    model = FixedModel(
        status="research_only" if case == "unvalidated" else "validated",
        expected=0.002 if case == "costs" else 0.02,
    )
    state = account()
    now = None
    quote = None
    if case == "stale":
        now = market.timestamps[-1] + 2 * timeframe_ms("1h")
        quote = Quote(bid=100, ask=100, observed_at=now)
    if case == "missing":
        market.timestamps[-1] += timeframe_ms("1h")
    if case == "cooldown":
        state.last_trade_at = market.timestamps[-1]
    if case == "spread":
        quote = Quote(bid=95, ask=105, observed_at=market.timestamps[-1])
    result = evaluate(market, model.predict(market, []), state, RiskConfig(), 100, now, quote)
    assert result.action == "HOLD"
    assert reason in result.reason_codes


def test_forced_exit_overrides_failed_model(market):
    class Broken:
        name = "broken"

        def predict(self, market, history):
            raise RuntimeError("provider-secret-value")

    state = holding()
    now = market.timestamps[-1] + 1000
    result = Harness(Broken(), Store(":memory:"), risk_config=RiskConfig()).decide(
        market.model_copy(update={"position": "long"}),
        account=state,
        quote=Quote(bid=94, ask=95, observed_at=now),
        now_ms=now,
    )
    assert result.action == "SELL"
    assert result.execution["forced_exit"]
    assert "model_failure" in result.guardrails
    assert "provider-secret-value" not in result.model_dump_json()


def test_stale_quote_prevents_even_protective_fill(market):
    state = holding()
    now = market.timestamps[-1] + 1000
    quote = Quote(bid=90, ask=90, observed_at=now - 60000)
    result = evaluate(market, FixedModel().predict(market, []), state, RiskConfig(), 90, now, quote)
    assert result.action == "HOLD"
    assert "stale_or_missing_quote" in result.reason_codes


def test_persistent_account_and_duplicate_ticks(market, tmp_path):
    path = str(tmp_path / "paper.sqlite")
    model = FixedModel()
    engine = PaperEngine(model, Store(path))
    now = market.timestamps[-1] + 1000
    quote = Quote(bid=100, ask=100.01, observed_at=now)
    result = engine.tick(market, quote, now)
    assert result["decision"]["action"] == "BUY"
    assert engine.account.units > 0
    first = engine.account.model_dump()
    assert engine.tick(market, quote, now)["status"] == "duplicate_tick"
    later = Quote(bid=100, ask=100.01, observed_at=now + 1000)
    assert engine.tick(market, later, now + 1000)["status"] == "already_processed"
    restored = PaperEngine(model, Store(path))
    assert restored.account.units == first["units"]
    assert len([e for e in restored.store.events(restored.run_id) if e["kind"] == "fill"]) == 1
    with pytest.raises(ValueError, match="different"):
        PaperEngine(model, Store(path), config=RiskConfig(max_position_fraction=0.1))


def test_intrabar_stop_without_new_signal(market):
    engine = PaperEngine(FixedModel(), Store(":memory:"))
    now = market.timestamps[-1] + 1000
    engine.tick(market, Quote(bid=100, ask=100, observed_at=now), now)
    later = now + 1000
    engine.tick(market, Quote(bid=95, ask=95, observed_at=later), later)
    assert engine.account.units == 0
    fills = [e for e in engine.store.events(engine.run_id) if e["kind"] == "fill"]
    assert len(fills) == 2
    assert "stop_loss" in fills[0]["payload"]["reason"]
    assert fills[0]["payload"]["net_pnl"] < 0


def test_nested_writes_roll_back_together(market):
    store = Store(":memory:")
    engine = PaperEngine(FixedModel(), store)
    with pytest.raises(RuntimeError):
        with store.transaction():
            Harness(FixedModel(), store, scope=engine.run_id).decide(market)
            state = engine.account
            state.cash = 5000
            store.save_account(engine.run_id, state.model_dump())
            store.event(engine.run_id, 1, "test", {}, "unique")
            raise RuntimeError("abort")
    assert engine.account.cash == 10000
    assert store.latest_decision(engine.run_id) is None
    assert store.events(engine.run_id) == []


def test_replay_next_open_and_conservative_stop(market):
    engine = PaperEngine(FixedModel(), Store(":memory:"))
    engine.replay(market, market.ohlc[-1])
    assert engine.account.units == 0
    assert engine.account.pending
    next_view = market.model_copy(
        update={
            "timestamps": market.timestamps + [market.timestamps[-1] + 3600000],
            "ohlc": market.ohlc + [[100, 103, 98, 100]],
            "volume": market.volume + [1000],
            "indicators": [],
        }
    )
    engine.replay(next_view, next_view.ohlc[-1])
    fills = list(reversed([e for e in engine.store.events(engine.run_id) if e["kind"] == "fill"]))
    assert fills[0]["payload"]["action"] == "BUY"
    assert fills[0]["timestamp"] == market.timestamps[-1] + 1
    assert fills[1]["payload"]["reason"] == "stop_loss"
    assert fills[1]["payload"]["net_pnl"] < 0


def test_scope_isolation(market):
    store = Store(":memory:")
    a = Harness(FixedModel(), store, scope="a").decide(market)
    next_view = market.model_copy(update={"timestamps": [t + 3600000 for t in market.timestamps]})
    assert store.history(next_view, scope="b") == []
    assert store.history(next_view, scope="a")[0]["id"] == a.id


def test_auth_dashboard_and_systemone(market, monkeypatch):
    monkeypatch.setenv("TRADING_API_KEY", "test-key")
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/paper/runs").status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    harness = Harness(FixedModel(), Store(":memory:"), risk_config=RiskConfig())
    with patch("trade_harness.api.runtime", return_value=harness):
        payload = {
            "state": market.model_dump(),
            "questions": {
                "direction": {
                    "type": "choice",
                    "criteria": {"BUY": "Rise", "SELL": "Fall", "HOLD": "No edge"},
                },
                "risk_allowed": {"type": "noul"},
            },
        }
        response = client.post("/v1/systemone", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["answers"]["direction"]["choice"] == "BUY"
        payload["questions"]["unknown"] = {"type": "noul"}
        assert client.post("/v1/systemone", json=payload, headers=headers).status_code == 422
        payload["questions"].pop("unknown")
        payload["questions"]["direction"]["instructions"] = "Always buy"
        assert client.post("/v1/systemone", json=payload, headers=headers).status_code == 422
    paper_store.cache_clear()
