"""Produce a reproducible report on the untouched final 15% of real candles."""

import argparse
import json
from pathlib import Path

import numpy as np

from trade_harness.backtest import backtest
from trade_harness.features import prefix
from trade_harness.models import TrainedModel
from trade_harness.schemas import MarketInput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/BTCUSDT-1h.json")
    parser.add_argument("--model", default="src/trade_harness/assets/btcusdt_1h_forecast.json")
    parser.add_argument("--output", default="reports/real-data-evaluation.json")
    args = parser.parse_args()
    market = MarketInput.model_validate_json(Path(args.input).read_text())
    model = TrainedModel(args.model)
    start = int(len(market.ohlc) * 0.85)
    errors = []
    zeros = []
    correct = []
    for i in range(start, len(market.ohlc) - market.horizon):
        expected = model.predict(prefix(market, i + 1), []).forecast.expected_return
        observed = market.ohlc[i + market.horizon][3] / market.ohlc[i][3] - 1
        errors.append(abs(expected - observed))
        zeros.append(abs(observed))
        correct.append(np.sign(expected) == np.sign(observed))
    simulation = backtest(market, model, start_at=market.timestamps[start])
    report = {
        "dataset": {
            "path": args.input,
            "candles": len(market.ohlc),
            "symbol": market.symbol,
            "timeframe": market.timeframe,
            "horizon": market.horizon,
            "test_start": market.timestamps[start],
            "test_end": market.timestamps[-1],
        },
        "forecast": {
            "test_samples": len(errors),
            "mae": float(np.mean(errors)),
            "zero_return_mae": float(np.mean(zeros)),
            "direction_accuracy": float(np.mean(correct)),
        },
        "backtest": {key: value for key, value in simulation.items() if key != "equity"},
        "costs": {"fee_bps": 10, "slippage_bps": 5},
        "limitations": [
            "Single asset and timeframe",
            "One chronological test period",
            "No demonstrated profitable edge",
            "No live exchange execution",
            "Pretrained LLM may know historical market events",
        ],
    }
    language_path = Path("src/trade_harness/assets/trading_lora/trading_metadata.json")
    if language_path.exists():
        report["language_model"] = json.loads(language_path.read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
