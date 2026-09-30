import math
import time
from uuid import uuid4

from .features import features
from .models import Model
from .schemas import Decision, Forecast, MarketInput, Proposal
from .storage import Store
from .validation import TradingValidation


class Harness:
    def __init__(
        self,
        model: Model,
        store: Store,
        minimum_confidence=0.6,
        max_volatility=0.08,
        risk_config=None,
        scope="",
        mode="decision_only",
    ):
        self.model, self.store = model, store
        self.risk_config, self.scope = risk_config, scope
        self.mode = mode
        self.minimum_confidence, self.max_volatility = minimum_confidence, max_volatility

    def decide(self, market: MarketInput, account=None, now_ms=None, quote=None) -> Decision:
        guards = []
        started = time.monotonic()
        try:
            proposed = self.model.predict(
                market,
                self.store.history(market, scope=self.scope)
                if getattr(self.model, "uses_history", True)
                else [],
            )
            proposed = Proposal.model_validate(proposed.model_dump())
            # Evidence comes from the configured backend's measured artifact, never LLM text.
            validator = getattr(self.model, "validation_for", None)
            evidence = (
                TradingValidation.model_validate(validator(market.symbol))
                if callable(validator)
                else TradingValidation(
                    symbol=market.symbol,
                    timeframe=market.timeframe,
                    horizon=market.horizon,
                    model_version=proposed.model_version,
                )
            )
            proposed.trading_validation = evidence
            if proposed.validation_status == "validated" and not (
                evidence.positive_edge and evidence.applies_to(market, proposed.model_version)
            ):
                proposed.validation_status = "research_only"
            expected_close = market.ohlc[-1][3] * (1 + proposed.forecast.expected_return)
            if proposed.forecast.horizon != market.horizon or not math.isclose(
                proposed.forecast.projected_close, expected_close, rel_tol=1e-6
            ):
                raise ValueError("Forecast does not match horizon or projected price")
        except Exception:  # noqa: BLE001 - model plugins must fail closed
            # Avoid leaking provider errors or credentials to API clients.
            guards.append("model_failure")
            proposed = Proposal(
                action="HOLD",
                confidence=0,
                rationale="Model unavailable or invalid output.",
                forecast=Forecast(
                    horizon=market.horizon,
                    expected_return=0,
                    projected_close=market.ohlc[-1][3],
                    method="fallback",
                ),
            )
        if proposed.trading_validation is None:
            proposed.trading_validation = TradingValidation(
                symbol=market.symbol,
                timeframe=market.timeframe,
                horizon=market.horizon,
                model_version=proposed.model_version,
            )
        if proposed.confidence < self.minimum_confidence:
            guards.append("low_confidence")
        if features(market)[3] > self.max_volatility:
            guards.append("high_volatility")
        if proposed.action == "SELL" and market.position == "flat":
            guards.append("no_long_position_to_sell")
        if proposed.action == "BUY" and market.position == "long":
            guards.append("already_long")
        execution = None
        final_action = "HOLD" if guards else proposed.action
        if self.risk_config is not None:
            from .data import ensure_indicators
            from .risk import Account, evaluate

            risk_market = ensure_indicators(market)
            if account is None:
                account = Account(
                    cash=self.risk_config.initial_cash,
                    peak_equity=self.risk_config.initial_cash,
                    day_start_equity=self.risk_config.initial_cash,
                )
                if market.position == "long":
                    guards.append("account_snapshot_required")
            execution_plan = evaluate(
                risk_market,
                proposed,
                account,
                self.risk_config,
                quote.mid if quote else market.ohlc[-1][3],
                now_ms + int((time.monotonic() - started) * 1000) if now_ms is not None else None,
                quote,
            )
            guards = list(dict.fromkeys(guards + execution_plan.reason_codes))
            # Deterministic emergency exits have priority over the model's confidence or availability.
            final_action = (
                execution_plan.action
                if execution_plan.forced_exit
                else "HOLD"
                if guards
                else execution_plan.action
            )
            execution = execution_plan.model_dump()
            execution["action"] = final_action
        fields = decision_fields(market, proposed, final_action, guards)
        decision = Decision(
            **{**proposed.model_dump(), "action": final_action},
            id=str(uuid4()),
            symbol=market.symbol,
            timeframe=market.timeframe,
            as_of=market.timestamps[-1],
            backend=self.model.name,
            guardrails=guards,
            proposed_action=proposed.action,
            execution=execution,
            fields=fields,
            mode=self.mode,
        )
        self.store.save(market, decision, scope=self.scope)
        return decision


def decision_fields(market, proposed, final_action, guards):
    from .typed import ChoiceResult, NoulResult, ScoreResult

    result = {}
    probabilities = proposed.probabilities
    if probabilities is not None:
        result["direction"] = ChoiceResult(
            choice=proposed.action,
            probabilities=probabilities,
            source="model",
            calibrated=proposed.probability_calibration != "uncalibrated",
        ).model_dump()
    # Execution is deterministic and must not masquerade as the model probability distribution.
    result["execution"] = ChoiceResult(
        choice=final_action,
        probabilities={a: float(a == final_action) for a in ("BUY", "SELL", "HOLD")},
        source="risk_policy",
    ).model_dump()
    result["risk_allowed"] = NoulResult(noul=float(not guards), source="risk_policy").model_dump()
    result["data_valid"] = NoulResult(
        noul=float(
            not any(
                g in guards
                for g in (
                    "stale_or_future_candles",
                    "missing_candle_intervals",
                    "stale_or_missing_quote",
                )
            )
        ),
        source="input_checks",
    ).model_dump()
    volatility = features(market)[3]
    level = 2 if volatility > 0.04 else 1 if volatility > 0.015 else 0
    result["risk_level"] = ScoreResult(
        score=level,
        probabilities={name: float(i == level) for i, name in enumerate(("LOW", "MEDIUM", "HIGH"))},
        source="realized_volatility_rule",
    ).model_dump()
    momentum = features(market)[2]
    regime = "UPTREND" if momentum > 0.01 else "DOWNTREND" if momentum < -0.01 else "SIDEWAYS"
    result["regime"] = ChoiceResult(
        choice=regime,
        probabilities={
            name: float(name == regime) for name in ("UPTREND", "DOWNTREND", "SIDEWAYS")
        },
        source="20_candle_momentum_rule",
    ).model_dump()
    return result
