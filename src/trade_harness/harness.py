import math
from uuid import uuid4

from .features import features
from .models import Model
from .schemas import Decision, Forecast, MarketInput, Proposal
from .storage import Store


class Harness:
    def __init__(self, model: Model, store: Store, minimum_confidence=0.6, max_volatility=0.08):
        self.model, self.store = model, store
        self.minimum_confidence, self.max_volatility = minimum_confidence, max_volatility

    def decide(self, market: MarketInput) -> Decision:
        guards = []
        try:
            proposed = self.model.predict(market, self.store.history(market))
            proposed = Proposal.model_validate(proposed.model_dump())
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
        if proposed.confidence < self.minimum_confidence:
            guards.append("low_confidence")
        if features(market)[3] > self.max_volatility:
            guards.append("high_volatility")
        if proposed.action == "SELL" and market.position == "flat":
            guards.append("no_long_position_to_sell")
        if proposed.action == "BUY" and market.position == "long":
            guards.append("already_long")
        decision = Decision(
            **{**proposed.model_dump(), "action": "HOLD" if guards else proposed.action},
            id=str(uuid4()),
            symbol=market.symbol,
            timeframe=market.timeframe,
            as_of=market.timestamps[-1],
            backend=self.model.name,
            guardrails=guards,
        )
        self.store.save(market, decision)
        return decision
