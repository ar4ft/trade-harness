"""Reproduce the shipped models and report; language training is optional."""

import argparse
import subprocess
import sys
from pathlib import Path

from trade_harness.data import download
from trade_harness.features import prefix
from trade_harness.schemas import MarketInput
from trade_harness.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-language", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    source = Path("data/BTCUSDT-1h.json")
    if not source.exists():
        download(output=str(source))
    market = MarketInput.model_validate_json(source.read_text())
    development = prefix(market, int(len(market.ohlc) * 0.85))
    target = Path("src/trade_harness/assets/btcusdt_1h_forecast.json")
    train(development, str(target), include_indicators=True)
    latest = market.model_copy(
        update={
            "ohlc": market.ohlc[-100:],
            "volume": market.volume[-100:],
            "timestamps": market.timestamps[-100:],
            "indicators": [
                i.model_copy(update={"values": i.values[-100:]}) for i in market.indicators
            ],
        }
    )
    target.with_name("latest.json").write_text(latest.model_dump_json())
    if args.with_language:
        subprocess.run(
            [sys.executable, "scripts/train_market_language.py", "--steps", str(args.steps)],
            check=True,
        )
    subprocess.run([sys.executable, "scripts/evaluate_market.py"], check=True)


if __name__ == "__main__":
    main()
