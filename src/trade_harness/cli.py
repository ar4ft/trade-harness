import argparse
import json
import os
from pathlib import Path

from .backtest import backtest
from .harness import Harness
from .models import load_model
from .schemas import MarketInput
from .storage import Store
from .training import export_finetuning, train

DEFAULT_MARKET = Path(__file__).parent / "assets/latest.json"


def main():
    parser = argparse.ArgumentParser(description="Trained trading decision CLI")
    parser.add_argument(
        "command",
        choices=["decide", "backtest", "train", "fetch", "train-language", "export-finetuning"],
    )
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--db", default=os.environ.get("TRADING_DB", "decisions.sqlite"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2025-09", help="Archive start month YYYY-MM")
    parser.add_argument("--end", default="2026-08", help="Archive end month YYYY-MM")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--with-indicators", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=96)
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--position", choices=["flat", "long"])
    parser.add_argument("--backend", choices=["baseline", "trained", "local-llm", "llm"])
    args = parser.parse_args()
    if args.backend:
        os.environ["TRADING_BACKEND"] = args.backend
    if args.command == "fetch":
        from .data import download

        result = download(
            args.symbol,
            args.timeframe,
            args.start,
            args.end,
            args.output or f"data/{args.symbol}-{args.timeframe}.json",
            args.horizon,
        )
    elif args.command == "train-language":
        from .language_training import main as train_language

        train_language(
            [
                "--input",
                args.input or "data/BTCUSDT-1h.json",
                "--output",
                args.output or "artifacts/trading-lora",
                "--steps",
                str(args.steps),
                "--stride",
                str(args.stride),
                "--eval-samples",
                str(args.eval_samples),
                "--base-model",
                args.base_model,
            ]
        )
        return
    elif args.command == "export-finetuning":
        result = export_finetuning(Store(args.db), args.output or "artifacts/trading-chat.jsonl")
    else:
        input_path = args.input or (
            str(DEFAULT_MARKET) if args.command == "decide" else "data/BTCUSDT-1h.json"
        )
        market = MarketInput.model_validate_json(Path(input_path).read_text())
        if args.position:
            market = market.model_copy(update={"position": args.position})
        if args.command == "train":
            result = train(market, args.output or "artifacts/forecast.json", args.with_indicators)
        elif args.command == "backtest":
            result = backtest(market, load_model())
        else:
            result = Harness(load_model(), Store(args.db)).decide(market).model_dump()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
