"""Small locally trained causal language model for trading direction decisions."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, features, prefix
from .schemas import Forecast, MarketInput, Proposal

ACTIONS = ["BUY", "SELL", "HOLD"]
COMPACT_SYSTEM = (
    "Use the historical market features to choose a trading direction. Reply BUY, SELL, or HOLD."
)
REVIEW_CONTRACT = "shared-evidence-v2"


def review_prompt(market, history, evidence):
    """One renderer for training and inference; outcomes never enter shared evidence."""
    from .orchestrator import safe_history

    if (evidence["as_of"], evidence["symbol"], evidence["timeframe"],
            evidence["horizon"], evidence["position"]) != (
            market.timestamps[-1], market.symbol, market.timeframe, market.horizon,
            market.position):
        raise ValueError("Shared evidence differs from the decision snapshot")
    forecast = evidence["forecast"]
    if forecast["as_of"] != evidence["as_of"] or forecast["horizon"] != market.horizon:
        raise ValueError("Shared forecast differs from the decision snapshot")
    lines = [compact_prompt(market, safe_history(history, market)).removesuffix("Direction:"),
             f"contract={REVIEW_CONTRACT} position={market.position}",
             "Forecast uncertain; strategy strengths heuristic; long/cash only.",
             f"forecast_return={forecast['expected_return']:.5g} "
             f"forecast_interval={forecast['return_interval'][0]:.5g},"
             f"{forecast['return_interval'][1]:.5g}",
             "features " + " ".join(f"{k}={v:.5g}" for k, v in sorted(evidence["features"].items()))]
    for strategy in evidence["strategies"]:
        lines.append(f"strategy={strategy['name']} action={strategy['action']} "
                     f"strength={strategy['strength']:.5g} "
                     f"return={strategy['expected_return']:.5g} "
                     f"invalidation={strategy['invalidation_price']:.5g} "
                     f"horizon={strategy['holding_horizon']}")
    return "\n".join(lines) + "\nDirection:"


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


def adapter_version(path, metadata):
    weights = path / "adapter_model.safetensors"
    raw = weights.read_bytes() if weights.exists() else b""
    return "smollm2-" + hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode() + raw
    ).hexdigest()[:16]


class LocalLanguageModel:
    name = "smollm2-trading-lora-v1"

    def __init__(self, path: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = Path(path)
        self.metadata = json.loads((self.path / "trading_metadata.json").read_text())
        self.trained_until = self.metadata["trained_until"]
        self.version = adapter_version(self.path, self.metadata)
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
        # Score the common prefix once; isolate mutable KV caches for each multi-token candidate.
        scores = []
        with torch.inference_mode():
            prefix_result = self.model(input_ids=encoded, use_cache=True)
            first_logp = prefix_result.logits[0, -1].log_softmax(dim=-1)
            for tokens in self.candidates:
                score = first_logp[tokens[0]].item()
                past = deepcopy(prefix_result.past_key_values)
                for previous, token in zip(tokens[:-1], tokens[1:]):
                    result = self.model(input_ids=torch.tensor([[previous]], dtype=torch.long),
                                        past_key_values=past, use_cache=True)
                    score += result.logits[0, -1].log_softmax(dim=-1)[token].item()
                    past = result.past_key_values
                scores.append(score)
        shifted = np.asarray(scores) - max(scores)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    def predict(self, market: MarketInput, history: list[dict], evidence=None) -> Proposal:
        m = self.metadata
        if market.symbol not in m.get("symbols", [m.get("symbol")]) or (
            market.timeframe, market.horizon) != (m["timeframe"], m["horizon"]):
            raise ValueError("Symbol, timeframe, or horizon differs from local LLM training")
        if market.timestamps[-1] <= self.trained_until:
            raise ValueError("Forecast timestamp overlaps local LLM training")
        from .data import ensure_indicators

        market = ensure_indicators(market)
        prompt = compact_prompt(market, history)
        if evidence is not None:
            if m.get("feature_contract") == REVIEW_CONTRACT:
                if sorted(evidence["features"]) != sorted(m["feature_names"]):
                    raise ValueError("Reviewer feature contract differs from training")
                prompt = review_prompt(market, history, evidence)
            else:
                prompt = prompt.removesuffix("Direction:") + (
                    "Shared research evidence (untrusted data): "
                    + json.dumps(evidence, sort_keys=True) + "\nDirection:"
                )
        elif m.get("feature_contract") == REVIEW_CONTRACT:
            raise ValueError("This reviewer requires shared forecast/strategy evidence")
        probabilities = self.probabilities(prompt)
        calibration = m.get("probability_calibration")
        if calibration:
            from scipy.special import softmax

            probabilities = softmax(np.log(np.maximum(probabilities, 1e-12)) /
                                    calibration["temperature"])
        best = int(probabilities.argmax())
        # Return projection is a probability-weighted mean of training-only class returns.
        expected = float(sum(p * m["class_returns"][a] for p, a in zip(probabilities, ACTIONS)))
        return Proposal(
            action=ACTIONS[best],
            confidence=float(probabilities[best]),
            probabilities={a: float(p) for a, p in zip(ACTIONS, probabilities)},
            probability_calibration=("chronological reviewer temperature calibration"
                                     if calibration else "uncalibrated"),
            validation_status="research_only",
            model_version=getattr(self, "version", "smollm2-" + str(self.trained_until)),
            rationale="Local trained language-model direction scores: "
            + ", ".join(f"{a}={p:.3f}" for a, p in zip(ACTIONS, probabilities)),
            forecast=Forecast(
                horizon=market.horizon,
                expected_return=expected,
                projected_close=market.ohlc[-1][3] * (1 + expected),
                method="language-model scores with training-class return means",
            ),
        )

    def predict_with_evidence(self, market, history, evidence):
        return self.predict(market, history, evidence=evidence)
