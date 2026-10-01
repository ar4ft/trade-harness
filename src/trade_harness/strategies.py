"""Versioned strategy hypotheses computed exclusively from observed candles."""

import numpy as np

from .data import ensure_indicators
from .schemas import Forecast, Proposal, StrategySignal

STRATEGY_VERSION = "trend-breakout-reversion-v1"
STRATEGY_FEATURES = [
    "trend_direction", "trend_strength", "breakout_direction", "breakout_strength",
    "reversion_direction", "reversion_strength", "strategy_agreement", "no_trade_strength",
]


def strategy_signals(market):
    market = ensure_indicators(market)
    prices = np.asarray(market.ohlc, dtype=float)
    close = prices[:, 3]
    last = close[-1]
    indicators = {i.name: i.values[-1] for i in market.indicators}
    atr = float(indicators["atr_14"])
    rsi = float(indicators["rsi_14"])
    volatility = float(np.std(np.diff(close[-21:]) / close[-21:-1]))
    momentum = last / close[-21] - 1
    trend = 1 if momentum > 0.01 else -1 if momentum < -0.01 else 0
    past_high, past_low = prices[-21:-1, 1].max(), prices[-21:-1, 2].min()
    breakout = 1 if last > past_high else -1 if last < past_low else 0
    deviation = (last - close[-20:].mean()) / max(float(close[-20:].std()), last * 1e-8)
    reversion = 1 if deviation < -1.5 and rsi < 40 else -1 if deviation > 1.5 and rsi > 60 else 0
    rows = [
        ("trend", trend, min(abs(momentum) / 0.03, 1), momentum * market.horizon / 20,
         "20-candle momentum beyond ±1%; exit hypothesis at two ATR."),
        ("breakout", breakout, min(abs(last / (past_high if breakout >= 0 else past_low) - 1) / 0.01, 1),
         breakout * atr / last, "Close exceeds prior 20-candle high/low, excluding current candle."),
        ("mean_reversion", reversion, min(abs(deviation) / 3, 1),
         (close[-20:].mean() / last - 1) if reversion else 0,
         "Deviation beyond ±1.5 standard deviations with confirming RSI."),
    ]
    signals = []
    for name, direction, strength, expected, reason in rows:
        signals.append(StrategySignal(
            name=name, action="BUY" if direction > 0 else "SELL" if direction < 0 else "HOLD",
            strength=float(strength if direction else 0),
            expected_return=float(np.clip(expected if direction else 0, -0.95, 1)),
            invalidation_price=float(max(last - direction * 2 * atr, last * 0.01)),
            holding_horizon=market.horizon, reason=reason,
        ))
    directions = {s.action for s in signals if s.action != "HOLD"}
    abstain = volatility > 0.04 or len(directions) != 1
    signals.append(StrategySignal(
        name="no_trade", action="HOLD", strength=float(abstain), expected_return=0,
        invalidation_price=float(last), holding_horizon=market.horizon,
        reason="Abstain when directional hypotheses conflict, none activate, or volatility exceeds 4%.",
    ))
    return signals


def strategy_features(signals):
    directions = [1 if s.action == "BUY" else -1 if s.action == "SELL" else 0 for s in signals[:3]]
    return np.asarray([
        *[v for d, s in zip(directions, signals[:3]) for v in (d, s.strength)],
        sum(directions) / 3, signals[-1].strength,
    ], dtype=float)


class StrategyModel:
    name = STRATEGY_VERSION
    version = STRATEGY_VERSION
    uses_history = False

    def predict(self, market, history):
        signals = strategy_signals(market)
        active = [s for s in signals[:3] if s.action != "HOLD"]
        action = active[0].action if active and signals[-1].strength == 0 else "HOLD"
        expected = float(np.mean([s.expected_return for s in active])) if action != "HOLD" else 0
        return Proposal(
            action=action, confidence=0.65, validation_status="research_only",
            model_version=self.version, strategy_signals=signals,
            rationale="Fixed strategy hypotheses; conflicting signals abstain. Scores are heuristic.",
            forecast=Forecast(horizon=market.horizon, expected_return=expected,
                              projected_close=market.ohlc[-1][3] * (1 + expected), method=self.name),
        )
