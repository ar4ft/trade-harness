"""Score the trained multi-token trading labels through a local llama.cpp server."""

import json
import math
import os
from pathlib import Path

import httpx
import numpy as np

from .local_language import ACTIONS, COMPACT_SYSTEM, LocalLanguageModel, adapter_version


def chat_prompt(prompt):
    # Exact template bundled with this adapter; do not use a generic JSON-chat prompt.
    return (
        f"<|im_start|>system\n{COMPACT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )


def token_log_probability(response, token):
    records = response.get("completion_probabilities", response.get("probs", []))
    if len(records) != 1:
        raise ValueError("Server must return one token's likelihoods")
    record = records[0]
    for item in record.get("top_logprobs", record.get("top_probs", record.get("probs", []))):
        if item.get("id") == token:
            if "logprob" in item and math.isfinite(item["logprob"]):
                return float(item["logprob"])
            probability = item.get("prob", item.get("p"))
            if probability is not None and 0 < probability <= 1:
                return math.log(probability)
    # Never invent a probability for a label missing from the returned top tokens.
    raise ValueError("Trading candidate token missing from server likelihoods")


class LlamafileModel(LocalLanguageModel):
    name = "smollm2-trading-lora-llamafile-v1"

    def __init__(self, path=None):
        self.path = Path(path or Path(__file__).parent / "assets/trading_lora")
        self.metadata = json.loads((self.path / "trading_metadata.json").read_text())
        self.trained_until = self.metadata["trained_until"]
        self.version = adapter_version(self.path, self.metadata)
        self.base_url = os.environ.get("LLAMAFILE_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
        self.key = os.environ.get("LLAMAFILE_API_KEY", "")

    def probabilities(self, prompt):
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        scores = []
        with httpx.Client(timeout=60, headers=headers) as client:

            def tokenize(text, special=False):
                response = client.post(
                    self.base_url + "/tokenize",
                    json={"content": text, "add_special": False, "parse_special": special},
                )
                response.raise_for_status()
                return response.json()["tokens"]

            prefix = tokenize(chat_prompt(prompt), special=True)
            cache = {}
            for action in ACTIONS:
                tokens = tokenize(action)
                score = 0.0
                for index, token in enumerate(tokens):
                    context = tuple(prefix + tokens[:index])
                    if context not in cache:
                        response = client.post(
                            self.base_url + "/completion",
                            json={
                                "prompt": list(context),
                                "n_predict": 1,
                                "n_probs": 512,
                                "temperature": 1,
                                "top_k": 0,
                                "top_p": 1,
                                "min_p": 0,
                                "repeat_penalty": 1,
                                "cache_prompt": True,
                                "seed": 42,
                            },
                        )
                        response.raise_for_status()
                        result = response.json()
                        if result.get("truncated"):
                            raise ValueError("Server truncated the trading prompt")
                        cache[context] = result
                    score += token_log_probability(cache[context], token)
                scores.append(score)
        probabilities = np.exp(np.asarray(scores) - max(scores))
        return probabilities / probabilities.sum()
