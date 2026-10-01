"""Small held-after-training agreement audit; not a trading-edge validation."""

import argparse
import json
from pathlib import Path

import numpy as np

from trade_harness.feature_engineering import transform_features
from trade_harness.hybrid_training import build_forecast_dataset
from trade_harness.learning import ACTIONS, DecisionModel, window
from trade_harness.models import load_model
from trade_harness.orchestrator import Orchestrator
from trade_harness.schemas import ForecastEvidence
from trade_harness.strategies import strategy_signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output", default="reports/consensus-comparison.json")
    args = parser.parse_args()
    if not 1 <= args.samples <= 200:
        parser.error("Use 1–200 operational audit samples")
    paths = [str(p) for p in Path("data/markets").glob("*.json")
             if not p.name.endswith(".provenance.json")]
    data, series, contract, forecasts = build_forecast_dataset(
        paths, "artifacts/hybrid/forecasts.jsonl", 48, 100,
    )
    artifact = json.loads(Path("src/trade_harness/assets/hybrid_model.json").read_text())
    vectors = {(s, int(t)): x for s, t, x in zip(data.symbol, data.timestamp, data.x)}

    class CachedEvidence(DecisionModel):
        name = "cached-hybrid-evidence-audit"
        feature_names = artifact["features"]

        def feature_vector(self, market):
            return transform_features(vectors[(market.symbol, market.timestamps[-1])],
                                      artifact.get("feature_config", {}))

        def predict(self, market, history):
            value = super().predict(market, history)
            forecast = forecasts[(market.symbol, market.timestamps[-1])]
            value.forecast_evidence = ForecastEvidence(source=forecast.forecast.method,
                model_version=forecast.model_version, as_of=market.timestamps[-1],
                horizon=market.horizon, expected_return=forecast.forecast.expected_return,
                return_interval=forecast.forecast_interval)
            value.strategy_signals = strategy_signals(market)
            value.feature_evidence = dict(zip(self.feature_names, self.feature_vector(market)))
            return value

    reviewer = load_model("local-llm")
    primary = CachedEvidence(artifact=artifact)
    model = Orchestrator(primary=primary, reviewers={"local-llm": reviewer},
                         config={"reviewers": ["local-llm"]})
    eligible = np.flatnonzero((data.symbol == "BTCUSDT") &
                             (data.timestamp > max(primary.trained_until, reviewer.trained_until)))
    selected = eligible[-args.samples:]
    rows = []
    market = series["BTCUSDT"]
    index = {t: i for i, t in enumerate(market.timestamps)}
    for row in selected:
        as_of = int(data.timestamp[row])
        result = model.predict(window(market, index[as_of], 100), [])
        rows.append({"as_of": as_of, "realized_label": ACTIONS[int(data.y[row])],
                     "realized_next_open_return": float(data.returns[row]),
                     "action": result.action, "consensus": result.consensus.model_dump()})
        if len(rows) % 4 == 0:
            print(json.dumps({"audit_samples": len(rows), "total": len(selected)}), flush=True)
    if not rows:
        raise ValueError("No audit samples after both models' fit cutoffs")
    report = {
        "mode": "decision_only", "real_execution_enabled": False, "samples": len(rows),
        "ensemble_version": model.version, "forecast_contract": contract,
        "numeric_trained_until": primary.trained_until,
        "reviewer_trained_until": reviewer.trained_until,
        "agreement_rate": sum(r["consensus"]["accepted"] for r in rows) / len(rows),
        "numeric_direction_accuracy": sum(r["consensus"]["votes"][0]["action"] == r["realized_label"]
                                          for r in rows) / len(rows),
        "reviewer_direction_accuracy": sum(r["consensus"]["votes"][1]["action"] == r["realized_label"]
                                           for r in rows) / len(rows),
        "consensus_direction_accuracy": sum(r["action"] == r["realized_label"] for r in rows) / len(rows),
        "hold_rate": sum(r["action"] == "HOLD" for r in rows) / len(rows),
        "records": rows,
        "limitations": [
            "Only a small BTC operational agreement audit; not walk-forward trading validation.",
            "No fees/slippage replay in this audit; automatic direction labels are not expert judgments.",
            "Both known training cutoffs precede samples; TimesFM pretraining overlap remains unknown.",
            "The local LoRA was not trained on the extended shared-evidence prompt.",
            "This already examined historical period is not prospective confirmation.",
        ],
    }
    Path(args.output).write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ("records", "forecast_contract", "limitations")}), flush=True)


if __name__ == "__main__":
    main()
