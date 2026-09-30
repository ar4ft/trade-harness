"""Small locally trained causal language model for trading direction decisions."""

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, features, prefix
from .schemas import Forecast, MarketInput, Proposal

ACTIONS = ["BUY", "SELL", "HOLD"]
COMPACT_SYSTEM = (
    "Use the historical market features to choose a trading direction. Reply BUY, SELL, or HOLD."
)


def compact_prompt(market: MarketInput, history: list[dict]) -> str:
    numbers = features(market)
    fields = [f"{key}={value:.4f}" for key, value in zip(FEATURE_NAMES, numbers)]
    close = market.ohlc[-1][3]
    indicators = {item.name: item.values[-1] for item in market.indicators}
    for name in ("sma_20", "rsi_14", "macd", "atr_14"):
        value = indicators.get(name)
        if value is not None:
            normalized = value / 100 if name == "rsi_14" else value / close
            if name == "sma_20":
                normalized -= 1
            fields.append(f"{name}={normalized:.4f}")
    # Unknown indicators remain available to the remote LLM backend; this tiny model has a fixed contract.
    prior_action = history[-1]["action"] if history else "NONE"
    outcomes = [entry["feedback"]["realized_return"] for entry in history if entry.get("feedback")]
    prior_return = f"{outcomes[-1]:.4f}" if outcomes else "NONE"
    return (
        f"{market.symbol} {market.timeframe} horizon={market.horizon}\n"
        + " ".join(fields)
        + f"\nprevious_action={prior_action} previous_observed_return={prior_return}\nDirection:"
    )


def direction(realized_return: float, threshold=0.003) -> str:
    return (
        "BUY" if realized_return > threshold else "SELL" if realized_return < -threshold else "HOLD"
    )


def training_records(market: MarketInput, stride=8):
    """70% train / 15% validation / 15% test, with purged boundary labels."""
    n, h = len(market.ohlc), market.horizon
    train_end, validation_end = int(n * 0.7), int(n * 0.85)
    partitions = {
        "train": range(21, train_end - h, stride),
        "validation": range(train_end, validation_end - h, stride),
        "test": range(validation_end, n - h, stride),
    }
    records = {}
    for name, indices in partitions.items():
        group = []
        for i in indices:
            view = prefix(market, i + 1)
            realized = market.ohlc[i + h][3] / market.ohlc[i][3] - 1
            # Simulated prior momentum decision/outcome, both knowable strictly before current as_of.
            old = i - h - 1
            history = []
            if old >= 20:
                old_market = prefix(market, old + 1)
                prior = direction(features(old_market)[1] * h / 5)
                outcome = market.ohlc[old + h][3] / market.ohlc[old][3] - 1
                history = [{"action": prior, "feedback": {"realized_return": outcome}}]
            group.append(
                {
                    "prompt": compact_prompt(view, history),
                    "label": direction(realized),
                    "realized_return": realized,
                    "as_of": market.timestamps[i],
                    "label_observed_at": market.timestamps[i + h],
                }
            )
        records[name] = group
    return records


class LocalLanguageModel:
    name = "smollm2-trading-lora-v1"

    def __init__(self, path: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = Path(path)
        self.metadata = json.loads((self.path / "trading_metadata.json").read_text())
        self.trained_until = self.metadata["trained_until"]
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
        base = AutoModelForCausalLM.from_pretrained(
            self.metadata["base_model"], revision=self.metadata["base_revision"]
        )
        self.model = PeftModel.from_pretrained(base, str(self.path)).eval()
        self.candidates = [
            self.tokenizer.encode(action, add_special_tokens=False) for action in ACTIONS
        ]

    def probabilities(self, prompt: str) -> np.ndarray:
        import torch

        messages = [
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        # Candidate likelihoods support multi-token labels without free-form JSON failures.
        scores = []
        with torch.inference_mode():
            for tokens in self.candidates:
                suffix = torch.tensor([tokens], dtype=torch.long)
                inputs = torch.cat([encoded, suffix], dim=1)
                logits = self.model(input_ids=inputs).logits[0]
                start = encoded.shape[1] - 1
                logp = logits[start : start + len(tokens)].log_softmax(dim=-1)
                score = sum(logp[j, token].item() for j, token in enumerate(tokens))
                scores.append(score)
        shifted = np.asarray(scores) - max(scores)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    def predict(self, market: MarketInput, history: list[dict]) -> Proposal:
        m = self.metadata
        if (market.symbol, market.timeframe, market.horizon) != (
            m["symbol"],
            m["timeframe"],
            m["horizon"],
        ):
            raise ValueError("Symbol, timeframe, or horizon differs from local LLM training")
        if market.timestamps[-1] <= self.trained_until:
            raise ValueError("Forecast timestamp overlaps local LLM training")
        from .data import ensure_indicators

        market = ensure_indicators(market)
        probabilities = self.probabilities(compact_prompt(market, history))
        best = int(probabilities.argmax())
        # Return projection is a probability-weighted mean of training-only class returns.
        expected = float(sum(p * m["class_returns"][a] for p, a in zip(probabilities, ACTIONS)))
        return Proposal(
            action=ACTIONS[best],
            confidence=float(probabilities[best]),
            probabilities={a: float(p) for a, p in zip(ACTIONS, probabilities)},
            validation_status="research_only",
            rationale="Local trained language-model direction scores: "
            + ", ".join(f"{a}={p:.3f}" for a, p in zip(ACTIONS, probabilities)),
            forecast=Forecast(
                horizon=market.horizon,
                expected_return=expected,
                projected_close=market.ohlc[-1][3] * (1 + expected),
                method="language-model scores with training-class return means",
            ),
        )
