"""Atomic, restartable long/cash paper execution with no exchange order API."""

from datetime import datetime, timezone

from .data import ensure_indicators
from .harness import Harness
from .risk import Account, Quote, RiskConfig
from .schemas import Feedback, Forecast, Proposal
from .storage import Store, timeframe_ms


class PaperEngine:
    def __init__(
        self,
        model,
        store: Store,
        run_id="paper-BTCUSDT-1h",
        symbol="BTCUSDT",
        timeframe="1h",
        config=None,
    ):
        self.model, self.store, self.run_id = model, store, run_id
        self.symbol, self.timeframe = symbol, timeframe
        self.config = config or RiskConfig()
        self.harness = Harness(
            model,
            store,
            minimum_confidence=self.config.min_confidence,
            max_volatility=self.config.max_volatility,
            risk_config=self.config,
            scope=run_id,
            mode="paper",
        )
        version = getattr(model, "version", model.name)
        with store.transaction():
            previous = store.paper_account(run_id)
            if previous is None:
                account = Account(
                    cash=self.config.initial_cash,
                    peak_equity=self.config.initial_cash,
                    day_start_equity=self.config.initial_cash,
                )
                store.create_account(
                    run_id,
                    symbol,
                    timeframe,
                    version,
                    self.config.model_dump(),
                    account.model_dump(),
                )
            elif (
                previous["symbol"],
                previous["timeframe"],
                previous["model_version"],
                previous["config"],
            ) != (symbol, timeframe, version, self.config.model_dump()):
                raise ValueError(
                    "Run exists with a different market, model, or risk configuration; use a new run ID"
                )

    @property
    def account(self):
        return Account.model_validate(self.store.paper_account(self.run_id)["state"])

    def _event(self, timestamp, kind, payload, key):
        self.store.event(self.run_id, timestamp, kind, payload, key)

    def _mark(self, account, price, timestamp):
        value = account.equity(price)
        day = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date().isoformat()
        if account.day != day:
            account.day = day
            account.day_start_equity = value
        account.peak_equity = max(account.peak_equity, value)
        if (
            1 - value / account.peak_equity >= self.config.max_drawdown
            or 1 - value / account.day_start_equity >= self.config.daily_loss_limit
            or account.consecutive_losses >= self.config.max_consecutive_losses
        ):
            account.halted = True
        return value

    def _fill(self, account, order, price, timestamp):
        action = order["action"]
        fee = self.config.fee_bps / 10000
        slip = self.config.slippage_bps / 10000
        if action == "BUY":
            if account.units or account.halted:
                return
            if abs(price / order["reference_price"] - 1) > self.config.max_entry_gap_fraction:
                self._event(
                    timestamp,
                    "cancel",
                    {"reason": "entry_price_gap", "decision_id": order["decision_id"]},
                    "cancel:" + order["decision_id"],
                )
                return
            fill = price * (1 + slip)
            notional = min(
                order["notional"],
                account.cash / (1 + fee),
                account.equity(price) * self.config.max_position_fraction,
            )
            quantity = notional / fill
            if quantity <= 0:
                return
            account.cash = max(0, account.cash - notional * (1 + fee))
            account.units = quantity
            account.entry_price = fill
            account.entry_at = timestamp
            account.entry_cost = notional * (1 + fee)
            distance = order["stop_fraction"]
            account.stop_loss = fill * (1 - distance)
            account.take_profit = fill * (1 + distance * self.config.reward_risk_ratio)
            pnl = None
        elif action == "SELL":
            if not account.units:
                return
            fill = price * (1 - slip)
            quantity = account.units
            notional = quantity * fill
            pnl = notional * (1 - fee) - account.entry_cost
            account.cash += notional * (1 - fee)
            account.consecutive_losses = account.consecutive_losses + 1 if pnl < 0 else 0
            account.units = 0
            account.entry_price = account.entry_at = account.stop_loss = account.take_profit = None
            account.entry_cost = 0
        else:
            raise ValueError("Unsupported paper order")
        account.last_trade_at = timestamp
        self._event(
            timestamp,
            "fill",
            {
                "action": action,
                "price": fill,
                "quantity": quantity,
                "notional": notional,
                "fee": notional * fee,
                "net_pnl": pnl,
                "reason": order["reason"],
                "decision_id": order["decision_id"],
            },
            "fill:" + order["decision_id"],
        )
        self._mark(account, price, timestamp)

    def _order(self, decision, price):
        plan = decision.execution
        if decision.action not in ("BUY", "SELL") or not plan:
            return None
        return {
            "action": decision.action,
            "notional": plan["notional"],
            "decision_id": decision.id,
            "reference_price": price,
            "stop_fraction": 1 - plan["stop_loss"] / price if plan["stop_loss"] else 0,
            "reason": ",".join(plan["reason_codes"]) or "model_signal",
            "signal_at": decision.as_of,
        }

    def _outcomes(self, market):
        # Only mature outcomes present in the chronological feed; never wall-clock-hindsight labels.
        index = {t: i for i, t in enumerate(market.timestamps)}
        for old in self.store.pending_outcomes(
            self.run_id, market.timestamps[0], market.timestamps[-1]
        ):
            if old["as_of"] not in index:
                continue
            start = index[old["as_of"]]
            end = start + old["horizon"]
            if end < len(market.timestamps) and market.timestamps[end] < market.timestamps[-1]:
                self.store.feedback(
                    Feedback(
                        decision_id=old["id"],
                        realized_return=market.ohlc[end][3] / market.ohlc[start][3] - 1,
                        observed_at=market.timestamps[end],
                    )
                )

    def tick(self, market, quote: Quote, now_ms: int):
        """Live tick: executable quotes, one signal per completed candle, stops on each tick."""
        if (market.symbol, market.timeframe) != (self.symbol, self.timeframe):
            raise ValueError("Feed differs from paper account market")
        with self.store.transaction():
            account = self.account
            if account.last_tick_at is not None and now_ms <= account.last_tick_at:
                return {"status": "duplicate_tick", "run_id": self.run_id}
            if market.timestamps[-1] > now_ms:
                return {"status": "rejected", "reason": "future_candles"}
            if not 0 <= now_ms - quote.observed_at <= self.config.max_quote_age_ms:
                return {"status": "rejected", "reason": "stale_or_future_quote"}
            self._mark(account, quote.mid, now_ms)
            view = ensure_indicators(
                market.model_copy(update={"position": "long" if account.units else "flat"})
            )
            new_signal = (
                account.last_signal_at is None or market.timestamps[-1] > account.last_signal_at
            )
            if new_signal:
                self._outcomes(view)
                decision = self.harness.decide(view, account=account, quote=quote, now_ms=now_ms)
                account.last_signal_at = view.timestamps[-1]
                order = self._order(decision, quote.mid)
            else:
                # Protective exits do not wait for a new candle or a successful model request.
                from .risk import evaluate

                dummy = Proposal(
                    action="HOLD",
                    confidence=0,
                    rationale="Protective tick",
                    forecast=Forecast(
                        horizon=view.horizon,
                        expected_return=0,
                        projected_close=view.ohlc[-1][3],
                        method="risk_only",
                    ),
                )
                plan = evaluate(view, dummy, account, self.config, quote.mid, now_ms, quote)
                decision = None
                order = (
                    {
                        "action": "SELL",
                        "decision_id": f"protective-{self.run_id}-{now_ms}",
                        "reason": ",".join(plan.reason_codes),
                        "reference_price": quote.mid,
                    }
                    if plan.forced_exit
                    else None
                )
            if order:
                self._fill(
                    account, order, quote.ask if order["action"] == "BUY" else quote.bid, now_ms
                )
            account.last_tick_at = now_ms
            value = self._mark(account, quote.mid, now_ms)
            self._event(
                now_ms,
                "mark",
                {
                    "equity": value,
                    "cash": account.cash,
                    "units": account.units,
                    "halted": account.halted,
                    "quote": quote.model_dump(),
                },
                f"mark:{now_ms}",
            )
            self.store.save_account(self.run_id, account.model_dump())
            return {
                "status": "processed" if new_signal else "already_processed",
                "run_id": self.run_id,
                "equity": value,
                "account": account.model_dump(),
                "decision": decision.model_dump() if decision else None,
            }

    def replay(self, view, bar):
        """Replay one completed bar; previous signal fills at this bar's open."""
        with self.store.transaction():
            account = self.account
            timestamp = view.timestamps[-1]
            if account.last_signal_at is not None and timestamp <= account.last_signal_at:
                return {"status": "already_processed"}
            opening, high, low, close = bar
            open_at = timestamp - timeframe_ms(view.timeframe) + 1
            # Reset daily risk reference BEFORE this bar's P&L, not after a loss already happened.
            self._mark(account, opening, open_at)
            if account.pending:
                order = account.pending
                account.pending = None
                if order["signal_at"] + timeframe_ms(view.timeframe) == timestamp:
                    self._fill(account, order, opening, open_at)
                else:
                    self._event(
                        open_at,
                        "cancel",
                        {"reason": "missing_next_bar"},
                        "cancel:" + order["decision_id"],
                    )
            # Conservative OHLC assumption: when stop and take-profit both touch, stop wins.
            protective = None
            if account.units and account.stop_loss is not None and low <= account.stop_loss:
                protective = ("stop_loss", min(opening, account.stop_loss))
            elif account.units and account.take_profit is not None and high >= account.take_profit:
                protective = ("take_profit", max(opening, account.take_profit))
            if protective:
                reason, price = protective
                self._fill(
                    account,
                    {
                        "action": "SELL",
                        "decision_id": f"{self.run_id}-{timestamp}-{reason}",
                        "reason": reason,
                    },
                    price,
                    timestamp,
                )
            self._mark(account, close, timestamp)
            self._outcomes(view)
            view = ensure_indicators(
                view.model_copy(update={"position": "long" if account.units else "flat"})
            )
            decision = self.harness.decide(view, account=account)
            account.pending = self._order(decision, close)
            account.last_signal_at = timestamp
            value = self._mark(account, close, timestamp)
            self._event(
                timestamp,
                "mark",
                {
                    "equity": value,
                    "cash": account.cash,
                    "units": account.units,
                    "halted": account.halted,
                },
                f"mark:{timestamp}",
            )
            self.store.save_account(self.run_id, account.model_dump())
            return {
                "status": "processed",
                "equity": value,
                "decision": decision.model_dump(),
                "account": account.model_dump(),
            }

    def summary(self):
        rows = self.store.events(self.run_id, limit=100000)
        marks = list(reversed([r for r in rows if r["kind"] == "mark"]))
        fills = list(reversed([r for r in rows if r["kind"] == "fill"]))
        equity = [r["payload"]["equity"] for r in marks]
        closing = [r["payload"]["net_pnl"] for r in fills if r["payload"]["net_pnl"] is not None]
        value = equity[-1] if equity else self.config.initial_cash
        peak = self.config.initial_cash
        drawdown = 0
        for e in equity:
            peak = max(peak, e)
            drawdown = max(drawdown, 1 - e / peak)
        turnover = sum(r["payload"]["notional"] for r in fills) / self.config.initial_cash
        gains = sum(p for p in closing if p > 0)
        losses = -sum(p for p in closing if p < 0)
        return {
            "run_id": self.run_id,
            "return": value / self.config.initial_cash - 1,
            "equity": value,
            "max_drawdown": drawdown,
            "fills": len(fills),
            "closed_trades": len(closing),
            "win_rate": sum(p > 0 for p in closing) / len(closing) if closing else None,
            "profit_factor": gains / losses if losses else None,
            "turnover": turnover,
            "fees": sum(r["payload"]["fee"] for r in fills),
            "halted": self.account.halted,
            "open_units": self.account.units,
            "pending_order": self.account.pending is not None,
            "equity_curve": [[r["timestamp"], r["payload"]["equity"]] for r in marks],
        }
