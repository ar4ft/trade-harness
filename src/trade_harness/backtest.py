from .features import prefix
from .harness import Harness
from .models import Model
from .schemas import Feedback, MarketInput
from .storage import Store


def backtest(
    market: MarketInput,
    model: Model,
    fee_bps=10.0,
    slippage_bps=5.0,
    initial_cash=10000.0,
    window=100,
):
    """Signals at candle close; fills at the NEXT open, long/flat only."""
    if fee_bps < 0 or slippage_bps < 0 or fee_bps + slippage_bps >= 10000:
        raise ValueError("Invalid trading costs")
    if initial_cash <= 0 or window < 21:
        raise ValueError("Cash must be positive and window >= 21")
    harness = Harness(model, Store(":memory:"))
    cash, units, peak, drawdown = initial_cash, 0.0, initial_cash, 0.0
    trades, equity, pending = [], [], []
    start = 20
    if hasattr(model, "trained_until"):
        start = max(
            start,
            next(
                (i for i, t in enumerate(market.timestamps) if t > model.trained_until),
                len(market.ohlc),
            ),
        )
    if start >= len(market.ohlc) - 1:
        raise ValueError("No out-of-sample candles available for backtest")
    for i in range(start, len(market.ohlc) - 1):
        for target, decision_id, entry_close in list(pending):
            if target <= i:
                harness.store.feedback(
                    Feedback(
                        decision_id=decision_id,
                        realized_return=market.ohlc[target][3] / entry_close - 1,
                        observed_at=market.timestamps[i],
                    )
                )
                pending.remove((target, decision_id, entry_close))
        view = prefix(market, i + 1)
        view = view.model_copy(
            update={
                "ohlc": view.ohlc[-window:],
                "volume": view.volume[-window:],
                "timestamps": view.timestamps[-window:],
                "position": "long" if units else "flat",
                "indicators": [
                    a.model_copy(update={"values": a.values[-window:]}) for a in view.indicators
                ],
            }
        )
        decision = harness.decide(view)
        pending.append((i + market.horizon, decision.id, market.ohlc[i][3]))
        next_open = market.ohlc[i + 1][0]
        fee, slip = fee_bps / 10000, slippage_bps / 10000
        if decision.action == "BUY" and not units:
            fill = next_open * (1 + slip)
            units = cash / (fill * (1 + fee))
            cash = 0.0
            trades.append({"action": "BUY", "timestamp": market.timestamps[i + 1], "price": fill})
        elif decision.action == "SELL" and units:
            fill = next_open * (1 - slip)
            cash = units * fill * (1 - fee)
            units = 0.0
            trades.append({"action": "SELL", "timestamp": market.timestamps[i + 1], "price": fill})
        value = cash + units * market.ohlc[i + 1][3]
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
        equity.append({"timestamp": market.timestamps[i + 1], "equity": value})
    return {
        "total_return": equity[-1]["equity"] / initial_cash - 1,
        "max_drawdown": drawdown,
        "trades": trades,
        "equity": equity,
        "open_units": units,
        "valuation": "mark-to-market; final position not liquidated",
    }
