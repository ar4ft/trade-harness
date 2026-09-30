import math

import httpx
import numpy as np
import pytest

from trade_harness.harness import Harness
from trade_harness.llamafile_model import LlamafileModel, token_log_probability
from trade_harness.schemas import MarketInput
from trade_harness.storage import Store


def test_multitoken_likelihoods_are_teacher_forced(monkeypatch):
    original = httpx.Client
    contexts = []

    def respond(request):
        import json

        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(
                200,
                json={
                    "tokens": {"BUY": [1, 2], "SELL": [3, 4], "HOLD": [5, 6]}.get(
                        body["content"], [90, 91]
                    )
                },
            )
        context = body["prompt"]
        contexts.append(context)
        values = {1: 0.6, 3: 0.3, 5: 0.1} if len(context) == 2 else {2: 0.2, 4: 0.9, 6: 0.5}
        return httpx.Response(
            200,
            json={
                "completion_probabilities": [
                    {"top_logprobs": [{"id": k, "logprob": math.log(v)} for k, v in values.items()]}
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs)
    )
    result = LlamafileModel().probabilities("market features")
    expected = np.array([0.6 * 0.2, 0.3 * 0.9, 0.1 * 0.5])
    np.testing.assert_allclose(result, expected / expected.sum())
    assert contexts == [[90, 91], [90, 91, 1], [90, 91, 3], [90, 91, 5]]


def test_missing_likelihood_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        token_log_probability({"completion_probabilities": [{"top_logprobs": []}]}, 2)
    assert token_log_probability(
        {"completion_probabilities": [{"probs": [{"id": 2, "prob": 0.2}]}]}, 2
    ) == pytest.approx(math.log(0.2))


def test_llamafile_unavailable_fails_to_hold(monkeypatch):
    model = LlamafileModel()
    monkeypatch.setattr(
        model,
        "probabilities",
        lambda prompt: (_ for _ in ()).throw(ValueError("server unavailable")),
    )
    from pathlib import Path

    market = MarketInput.model_validate_json(
        Path("src/trade_harness/assets/latest.json").read_text()
    )
    result = Harness(model, Store(":memory:")).decide(market)
    assert result.action == "HOLD" and "model_failure" in result.guardrails
