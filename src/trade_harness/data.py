"""Download checksum-verified public spot OHLCV archives without API credentials."""

import csv
import hashlib
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from .schemas import Indicator, MarketInput
from .storage import timeframe_ms

ARCHIVE_ROOT = "https://data.binance.vision/data/spot/monthly/klines"


def month_range(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start + "-01")
    last = date.fromisoformat(end + "-01")
    if first > last:
        raise ValueError("Start month must precede end month")
    result = []
    while first <= last:
        result.append(first.strftime("%Y-%m"))
        first = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    return result


def normalize_timestamp(value: str) -> int:
    raw = int(value)
    # Binance spot archives switched from milliseconds to microseconds in Jan 2025.
    return raw // 1000 if raw >= 100_000_000_000_000 else raw


def read_archive(content: bytes) -> list[tuple[int, list[float], float]]:
    rows = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one candle CSV")
        with archive.open(names[0]) as raw:
            for row in csv.reader(io.TextIOWrapper(raw)):
                if not row or row[0] in ("open_time", "Open time"):
                    continue
                # row[6] is the closed candle end time (inclusive), not the open time.
                rows.append(
                    (normalize_timestamp(row[6]), [float(v) for v in row[1:5]], float(row[5]))
                )
    return rows


def add_indicators(market: MarketInput) -> MarketInput:
    close = np.array([row[3] for row in market.ohlc])
    high = np.array([row[1] for row in market.ohlc])
    low = np.array([row[2] for row in market.ohlc])
    sma = [None if i < 19 else float(close[i - 19 : i + 1].mean()) for i in range(len(close))]
    ema_fast, ema_slow, rsi, atr = [], [], [], []
    fast = slow = float(close[0])
    gains = losses = average_range = 0.0
    for i, value in enumerate(close):
        fast = fast + 2 / 13 * (value - fast)
        slow = slow + 2 / 27 * (value - slow)
        ema_fast.append(float(fast))
        ema_slow.append(float(slow))
        delta = value - close[i - 1] if i else 0.0
        true_range = (
            max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            if i
            else high[i] - low[i]
        )
        if i < 14:
            gains += max(delta, 0) / 14
            losses += max(-delta, 0) / 14
            average_range += true_range / 14
            rsi.append(None)
            atr.append(None)
        else:
            gains = (gains * 13 + max(delta, 0)) / 14
            losses = (losses * 13 + max(-delta, 0)) / 14
            average_range = (average_range * 13 + true_range) / 14
            rsi.append(
                float(100 - 100 / (1 + gains / losses)) if losses else 100.0 if gains else 50.0
            )
            atr.append(float(average_range))
    indicators = [
        Indicator(name="sma_20", values=sma),
        Indicator(name="rsi_14", values=rsi),
        Indicator(name="macd", values=[f - s for f, s in zip(ema_fast, ema_slow)]),
        Indicator(name="atr_14", values=atr),
    ]
    return market.model_copy(update={"indicators": indicators})


def download(
    symbol="BTCUSDT",
    timeframe="1h",
    start="2025-09",
    end="2026-08",
    output="data/BTCUSDT-1h.json",
    horizon=3,
):
    if not re.fullmatch(r"[A-Z0-9]{3,20}", symbol):
        raise ValueError("Invalid exchange symbol")
    if timeframe not in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"):
        raise ValueError("Unsupported Binance archive timeframe")
    months = month_range(start, end)
    if len(months) > 24:
        raise ValueError("Download at most 24 months per request")
    estimated = len(months) * 31 * 86400000 // timeframe_ms(timeframe)
    if estimated > 24000:
        raise ValueError("Requested dataset is too large; use fewer months or a larger timeframe")

    def fetch(month):
        filename = f"{symbol}-{timeframe}-{month}.zip"
        url = f"{ARCHIVE_ROOT}/{symbol}/{timeframe}/{filename}"
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            checksum = client.get(url + ".CHECKSUM")
            checksum.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        expected = checksum.text.split()[0]
        if digest != expected:
            raise ValueError(f"Archive checksum mismatch for {filename}")
        return read_archive(response.content), {
            "url": url,
            "sha256": digest,
            "bytes": len(response.content),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(fetch, months))
    rows = sorted(row for group, _ in downloaded for row in group)
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = [row for row in rows if row[0] < now]
    if any(b[0] - a[0] != timeframe_ms(timeframe) for a, b in zip(rows, rows[1:])):
        raise ValueError("Archive has duplicate candles or missing intervals")
    market = add_indicators(
        MarketInput(
            symbol=symbol,
            timeframe=timeframe,
            timestamps=[r[0] for r in rows],
            ohlc=[r[1] for r in rows],
            volume=[r[2] for r in rows],
            horizon=horizon,
        )
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(market.model_dump_json())
    metadata = {
        "source": "Binance public spot monthly klines",
        "symbol": symbol,
        "timeframe": timeframe,
        "start_month": start,
        "end_month": end,
        "candles": len(rows),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "archives": [m for _, m in downloaded],
        "timestamp_units": "Unix milliseconds; closed candle end",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def ensure_indicators(market: MarketInput) -> MarketInput:
    required = {"sma_20", "rsi_14", "macd", "atr_14"}
    existing = {item.name: item for item in market.indicators}
    missing = {
        name for name in required if name not in existing or existing[name].values[-1] is None
    }
    if not missing:
        return market
    for item in add_indicators(market).indicators:
        if item.name in missing:
            existing[item.name] = item
    return market.model_copy(update={"indicators": list(existing.values())})
