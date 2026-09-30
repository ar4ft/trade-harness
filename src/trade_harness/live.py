"""Public market-data polling; simulation only, with no private exchange endpoints."""

import json
import time
from pathlib import Path

import httpx

from .data import add_indicators
from .paper import PaperEngine
from .risk import Quote, RiskConfig
from .schemas import MarketInput
from .storage import Store


class BinanceFeed:
    def __init__(self, base_url="https://data-api.binance.vision", timeout=15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(self, symbol="BTCUSDT", timeframe="1h", horizon=3):
        with httpx.Client(timeout=self.timeout) as client:
            server = client.get(self.base_url + "/api/v3/time")
            server.raise_for_status()
            server_now = int(server.json()["serverTime"])
            response = client.get(
                self.base_url + "/api/v3/klines",
                params={"symbol": symbol, "interval": timeframe, "limit": 150},
            )
            response.raise_for_status()
            rows = [row for row in response.json() if int(row[6]) < server_now]
            response = client.get(
                self.base_url + "/api/v3/ticker/bookTicker", params={"symbol": symbol}
            )
            response.raise_for_status()
            book = response.json()
        now = int(time.time() * 1000)
        # Reject material local clock skew, rather than interpreting future candles as closed.
        if abs(now - server_now) > 30000:
            raise ValueError(
                "Local clock differs from exchange server time by more than 30 seconds"
            )
        market = add_indicators(
            MarketInput(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                timestamps=[int(r[6]) for r in rows],
                ohlc=[[float(v) for v in r[1:5]] for r in rows],
                volume=[float(r[5]) for r in rows],
            )
        )
        quote = Quote(bid=float(book["bidPrice"]), ask=float(book["askPrice"]), observed_at=now)
        return market, quote, now


def run_live(
    model,
    db="paper.sqlite",
    run_id="paper-BTCUSDT-1h",
    symbol="BTCUSDT",
    timeframe="1h",
    horizon=3,
    steps=1,
    poll_seconds=5,
    config=None,
    feed=None,
):
    if steps < 0 or poll_seconds < 1:
        raise ValueError("Steps must be nonnegative and polling interval at least one second")
    store = Store(db)
    engine = PaperEngine(model, store, run_id, symbol, timeframe, config or RiskConfig())
    feed = feed or BinanceFeed()
    count = 0
    try:
        while steps == 0 or count < steps:
            try:
                market, quote, now = feed.fetch(symbol, timeframe, horizon)
                result = engine.tick(market, quote, now)
                print(json.dumps(result), flush=True)
            except (httpx.HTTPError, ValueError) as error:
                # HTTP providers can include sensitive URL query values; publish stable error types only.
                now = int(time.time() * 1000)
                store.event(
                    run_id,
                    now,
                    "feed_error",
                    {"error_type": type(error).__name__},
                    f"feed_error:{now}",
                )
                print(
                    json.dumps({"status": "feed_error", "error_type": type(error).__name__}),
                    flush=True,
                )
            count += 1
            if steps == 0 or count < steps:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print(json.dumps({"status": "stopped", "run_id": run_id}), flush=True)
    return engine.summary()


def load_risk_config(path=None, allow_research=False):
    config = RiskConfig.model_validate_json(Path(path).read_text()) if path else RiskConfig()
    return config.model_copy(update={"allow_research": True}) if allow_research else config
