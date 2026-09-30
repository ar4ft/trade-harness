import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, INDICATOR_FEATURE_NAMES, features, prefix
from .models import SYSTEM_PROMPT, TrainedModel
from .schemas import MarketInput, Proposal


def train(market: MarketInput, output: str, include_indicators=False):
    n, horizon = len(market.ohlc), market.horizon
    split = int(n * 0.8)
    # Purge labels that would mature in the chronological validation partition.
    train_indices = list(range(20, split - horizon))
    test_indices = list(range(split, n - horizon))
    if len(train_indices) < 30 or len(test_indices) < 10:
        raise ValueError("Need more candles for purged chronological train/validation split")

    def xy(indices):
        return (
            np.array([features(prefix(market, i + 1), include_indicators) for i in indices]),
            np.array([market.ohlc[i + horizon][3] / market.ohlc[i][3] - 1 for i in indices]),
        )

    x, y = xy(train_indices)
    xt, yt = xy(test_indices)
    scaler = StandardScaler().fit(x)
    estimator = Ridge(alpha=1.0).fit(scaler.transform(x), y)
    prediction = estimator.predict(scaler.transform(xt))
    metrics = {
        "validation_mae": float(np.abs(prediction - yt).mean()),
        "zero_return_mae": float(np.abs(yt).mean()),
        "direction_accuracy": float((np.sign(prediction) == np.sign(yt)).mean()),
        "train_samples": len(y),
        "validation_samples": len(yt),
    }
    artifact = {
        "version": 1,
        "features": FEATURE_NAMES + (INDICATOR_FEATURE_NAMES if include_indicators else []),
        "horizon": horizon,
        "symbol": market.symbol,
        "timeframe": market.timeframe,
        "trained_until": market.timestamps[split - 1],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_.tolist(),
        "intercept": float(estimator.intercept_),
        "metrics": metrics,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))
    # Verify portable artifact loads without pickle/joblib execution.
    TrainedModel(str(path))
    return metrics


def export_finetuning(store, output: str):
    count = 0
    with open(output, "w") as file:
        for market, decision, feedback in store.reviewed_examples():
            answer = {k: decision[k] for k in ("action", "confidence", "rationale", "forecast")}
            answer["action"] = feedback["reviewed_action"]
            # Realized returns remain in feedback; never insert future outcomes into model input.
            user = {
                "market": market,
                "prior_decisions": [],
                "output_schema": Proposal.model_json_schema(),
            }
            file.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(user)},
                            {"role": "assistant", "content": json.dumps(answer)},
                        ]
                    }
                )
                + "\n"
            )
            count += 1
    return {"examples": count, "path": output}
