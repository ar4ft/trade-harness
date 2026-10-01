"""Run the CLI-equivalent, resumable frozen-forecast research experiment."""

from pathlib import Path

from trade_harness.hybrid_training import evaluate_hybrid

if __name__ == "__main__":
    paths = [str(p) for p in Path("data/markets").glob("*.json")
             if not p.name.endswith(".provenance.json")]
    evaluate_hybrid(paths)
