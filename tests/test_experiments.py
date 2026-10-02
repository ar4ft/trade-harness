import json
from pathlib import Path

import numpy as np
import pytest
from test_hybrid import FakeForecast

from trade_harness.experiments import default_trials, tune_hybrid
from trade_harness.hybrid import HybridModel
from trade_harness.learning import window
from trade_harness.schemas import MarketInput


def test_trial_registry_is_bounded_and_distinct():
    trials = default_trials()
    assert len(trials) == 12
    assert len({t["name"] for t in trials}) == 12
    assert {t["features"]["recipe"] for t in trials} == {"base", "interactions", "quant"}


def test_tuning_selection_excludes_changed_final_period(tmp_path):
    n = 1800
    rng = np.random.default_rng(14)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    market = MarketInput(symbol="BTCUSDT", timeframe="1h", horizon=3,
        timestamps=[1700000000000 + i * 3600000 for i in range(n)],
        ohlc=[[float(p), float(p * 1.01), float(p * 0.99), float(p)] for p in close],
        volume=[1000] * n)
    source = tmp_path / "source.json"
    source.write_text(market.model_dump_json())
    trials = [default_trials()[2], default_trials()[7]]

    def run(name):
        return tune_hybrid([str(source)], cache=str(tmp_path / f"{name}.jsonl"), stride=3,
                           context=21, output=str(tmp_path / f"{name}.json"),
                           model_output=str(tmp_path / f"{name}-model.json"),
                           trials=trials, forecaster=FakeForecast(21))

    first = run("first")
    first_index = market.timestamps.index(first["test_start"])
    changed = market.model_copy(deep=True)
    for i in range(first_index, n):
        scale = 1.1 + (i - first_index) / (n - first_index)
        changed.ohlc[i] = [v * scale for v in changed.ohlc[i]]
    source.write_text(changed.model_dump_json())
    second = run("second")
    assert first["selected_candidate"] == second["selected_candidate"]
    for name in first["candidates"]:
        assert first["candidates"][name]["mean_log_loss"] == pytest.approx(
            second["candidates"][name]["mean_log_loss"])
    assert first["output_candidate"] == first["selected_candidate"]
    assert first["validation_status"] == "research_only"
    assert Path(str(tmp_path / "first.json") + ".plan.json").exists()
    a = json.loads((tmp_path / "first-model.json").read_text())
    assert a["fit_parameters"]
    assert a["feature_config"]
    result = HybridModel(artifact=a, forecaster=FakeForecast(21)).predict(window(market, n - 1, 21), [])
    assert result.validation_status == "research_only"
    # Changing a predeclared plan at an existing output path is rejected.
    with pytest.raises(ValueError, match="plan changed"):
        tune_hybrid([str(source)], cache=str(tmp_path / "second.jsonl"), stride=3, context=21,
                    output=str(tmp_path / "second.json"), trials=trials[::-1] + [default_trials()[0]],
                    forecaster=FakeForecast(21))
