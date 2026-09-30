from pathlib import Path

from trade_harness.evaluation import evaluate

if __name__ == "__main__":
    paths = [
        str(p)
        for p in Path("data/markets").glob("*.json")
        if not p.name.endswith(".provenance.json")
    ]
    evaluate(paths)
