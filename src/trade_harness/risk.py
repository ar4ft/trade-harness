"""Deterministic portfolio controls; no model may override them."""

from typing import Literal

from pydantic import Field, model_validator

from .schemas import MarketInput, Proposal, StrictModel
from .storage import timeframe_ms


class RiskConfig(StrictModel):
    initial_cash: float = Field(default=10000, gt=0)
    fee_bps: float = Field(default=10, ge=0, le=100)
    slippage_bps: float = Field(default=5, ge=0, le=100)
    max_position_fraction: float = Field(default=0.20, gt=0, le=1)
    risk_per_trade: float = Field(default=0.005, gt=0, le=0.05)
    stop_atr_multiple: float = Field(default=2, gt=0, le=10)
    min_stop_fraction: float = Field(default=0.01, gt=0, le=0.2)
    max_stop_fraction: float = Field(default=0.08, gt=0, le=0.5)
    reward_risk_ratio: float = Field(default=2, ge=1.5, le=10)
    daily_loss_limit: float = Field(default=0.02, gt=0, le=0.2)
    max_drawdown: float = Field(default=0.10, gt=0, le=0.5)
    max_consecutive_losses: int = Field(default=3, ge=1, le=20)
    cooldown_candles: int = Field(default=3, ge=0, le=100)
    max_holding_candles: int = Field(default=3, ge=1, le=1000)
    min_confidence: float = Field(default=0.55, ge=0.33, le=0.99)
    min_net_edge: float = Field(default=0.001, ge=0, le=0.1)
    max_volatility: float = Field(default=0.08, gt=0, le=1)
    max_candle_age_intervals: float = Field(default=1.5, gt=0, le=3)
    max_quote_age_ms: int = Field(default=30000, ge=1000, le=300000)
    max_entry_gap_fraction: float = Field(default=0.01, gt=0, le=0.1)
    allow_research: bool = False

    @model_validator(mode="after")
    def stop_bounds(self):
        if self.min_stop_fraction > self.max_stop_fraction:
            raise ValueError("Minimum stop distance exceeds maximum")
        return self

    @property
    def roundtrip_cost(self):
        return 2 * (self.fee_bps + self.slippage_bps) / 10000


class Account(StrictModel):
    cash: float = Field(ge=0)
    units: float = Field(default=0, ge=0)
    entry_price: float | None = Field(default=None, gt=0)
    entry_at: int | None = None
    entry_cost: float = Field(default=0, ge=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    peak_equity: float = Field(gt=0)
    day_start_equity: float = Field(gt=0)
    day: str = ""
    last_trade_at: int | None = None
    consecutive_losses: int = Field(default=0, ge=0)
    halted: bool = False
    last_signal_at: int | None = None
    last_tick_at: int | None = None
    pending: dict | None = None

    def equity(self, price):
        return self.cash + self.units * price


class Quote(StrictModel):
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    observed_at: int = Field(ge=0)

    @model_validator(mode="after")
    def bounds(self):
        if self.bid > self.ask:
            raise ValueError("Crossed quote")
        return self

    @property
    def mid(self):
        return (self.bid + self.ask) / 2


class ExecutionPlan(StrictModel):
    action: Literal["BUY", "SELL", "HOLD"]
    notional: float = Field(default=0, ge=0)
    units: float = Field(default=0, ge=0)
    stop_loss: float | None = None
    take_profit: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    net_expected_return: float
    forced_exit: bool = False


def evaluate(
    market: MarketInput,
    proposal: Proposal,
    account: Account,
    config: RiskConfig,
    price: float,
    now_ms: int | None = None,
    quote: Quote | None = None,
) -> ExecutionPlan:
    """Decide execution separately from model direction and probability."""
    reasons = []
    equity = account.equity(price)
    interval = timeframe_ms(market.timeframe)
    spread = (quote.ask - quote.bid) / quote.mid if quote else 0.0
    net = proposal.forecast.expected_return - config.roundtrip_cost - spread
    if now_ms is not None:
        age = now_ms - market.timestamps[-1]
        if age < 0 or age > interval * config.max_candle_age_intervals:
            reasons.append("stale_or_future_candles")
        if quote is None or not 0 <= now_ms - quote.observed_at <= config.max_quote_age_ms:
            reasons.append("stale_or_missing_quote")
    if any(b - a != interval for a, b in zip(market.timestamps, market.timestamps[1:])):
        reasons.append("missing_candle_intervals")
    daily_loss = 1 - equity / account.day_start_equity
    drawdown = 1 - equity / account.peak_equity
    emergency = []
    if account.halted:
        emergency.append("account_halted")
    if daily_loss >= config.daily_loss_limit:
        emergency.append("daily_loss_limit")
    if drawdown >= config.max_drawdown:
        emergency.append("max_drawdown")
    if account.consecutive_losses >= config.max_consecutive_losses:
        emergency.append("consecutive_loss_limit")
    if (
        account.units
        and account.entry_at is not None
        and (now_ms if now_ms is not None else market.timestamps[-1] + 1) - account.entry_at
        >= config.max_holding_candles * interval
    ):
        emergency.append("maximum_holding_period")
    if account.units and account.stop_loss is not None and price <= account.stop_loss:
        emergency.append("stop_loss")
    if account.units and account.take_profit is not None and price >= account.take_profit:
        emergency.append("take_profit")
    if emergency:
        action = "SELL" if account.units else "HOLD"
        # Stale quotes must never be used even for an emergency fill; retain exit intent for retry.
        if "stale_or_missing_quote" in reasons:
            action = "HOLD"
        return ExecutionPlan(
            action=action,
            units=account.units if action == "SELL" else 0,
            notional=account.units * price if action == "SELL" else 0,
            reason_codes=reasons + emergency,
            net_expected_return=net,
            forced_exit=action == "SELL",
        )
    action = proposal.action
    if action == "SELL" and not account.units:
        reasons.append("no_long_position_to_sell")
    if action == "BUY" and account.units:
        reasons.append("already_long")
    if proposal.confidence < config.min_confidence:
        reasons.append("low_confidence")
    if proposal.validation_status != "validated" and not config.allow_research and action == "BUY":
        reasons.append("no_validated_edge")
    if action == "BUY" and net < config.min_net_edge:
        reasons.append("insufficient_edge_after_costs")
    if (
        action == "BUY"
        and account.last_trade_at is not None
        and market.timestamps[-1] - account.last_trade_at < config.cooldown_candles * interval
    ):
        reasons.append("cooldown")
    from .features import features

    if features(market)[3] > config.max_volatility:
        reasons.append("high_volatility")
    if action == "HOLD" or reasons:
        return ExecutionPlan(action="HOLD", reason_codes=reasons, net_expected_return=net)
    if action == "SELL":
        return ExecutionPlan(
            action="SELL",
            units=account.units,
            notional=account.units * price,
            net_expected_return=net,
        )
    atr = next((i.values[-1] for i in market.indicators if i.name == "atr_14"), None)
    distance = max(
        config.min_stop_fraction, (atr or price * 0.01) * config.stop_atr_multiple / price
    )
    distance = min(distance, config.max_stop_fraction)
    budget = min(
        equity * config.max_position_fraction,
        equity * config.risk_per_trade / (distance + config.roundtrip_cost),
        account.cash / (1 + config.fee_bps / 10000),
    )
    if budget <= 0:
        return ExecutionPlan(action="HOLD", reason_codes=["no_cash"], net_expected_return=net)
    return ExecutionPlan(
        action="BUY",
        notional=budget,
        units=budget / price,
        stop_loss=price * (1 - distance),
        take_profit=price * (1 + distance * config.reward_risk_ratio),
        net_expected_return=net,
    )
