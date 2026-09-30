import argparse
import json
import os

from .backtest import backtest
from .harness import Harness
from .models import load_model
from .schemas import MarketInput
from .storage import Store
from .training import export_finetuning, train


def main():
    parser = argparse.ArgumentParser(description="Trading research harness")
    parser.add_argument("command", choices=["decide", "backtest", "train", "export-finetuning"])
    parser.add_argument("--input", default="examples/market.json")
    parser.add_argument("--output", default="artifacts/forecast.json")
    parser.add_argument("--db", default=os.environ.get("TRADING_DB", "decisions.sqlite"))
    args = parser.parse_args()
    if args.command == "export-finetuning":
        result = export_finetuning(Store(args.db), args.output)
    else:
        with open(args.input) as file:
            market = MarketInput.model_validate(json.load(file))
        if args.command == "train":
            result = train(market, args.output)
        elif args.command == "backtest":
            result = backtest(market, load_model())
        else:
            result = Harness(load_model(), Store(args.db)).decide(market).model_dump()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
