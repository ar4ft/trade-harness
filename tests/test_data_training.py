import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest

from trade_harness.data import add_indicators, download, normalize_timestamp, read_archive
from trade_harness.features import FEATURE_NAMES, INDICATOR_FEATURE_NAMES, features, prefix
from trade_harness.learning import DecisionModel
from trade_harness.local_language import compact_prompt, training_records
from trade_harness.models import TrainedModel, load_model
from trade_harness.schemas import MarketInput
from trade_harness.training import train


@pytest.fixture
def real_market():
    return MarketInput.model_validate_json(Path("data/BTCUSDT-1h.json").read_text())


def archive(rows):
    raw = io.StringIO()
    csv.writer(raw).writerows(rows)
    file = io.BytesIO()
    with zipfile.ZipFile(file, "w") as z:
        z.writestr("candles.csv", raw.getvalue())
    return file.getvalue()


def test_microseconds_and_csv_contract():
    assert normalize_timestamp("1700000000000") == 1700000000000
    assert normalize_timestamp("1700000000000000") == 1700000000000
    data = archive([["1700000000000000", "100", "110", "90", "105", "20", "1700003599999000"]])
    assert read_archive(data) == [(1700003599999, [100.0, 110.0, 90.0, 105.0], 20.0)]


def test_download_rejects_wrong_checksum(monkeypatch, tmp_path):
    def get(self, url):
        return httpx.Response(
            200,
            content=b"invalid archive" if url.endswith(".zip") else b"0" * 64,
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    with pytest.raises(ValueError, match="checksum mismatch"):
        download(start="2025-09", end="2025-09", output=str(tmp_path / "data.json"))
    assert not (tmp_path / "data.json").exists()


def test_indicators_do_not_use_future_candles(real_market):
    shorter = add_indicators(prefix(real_market, 100))
    longer = add_indicators(prefix(real_market, 200))
    for a, b in zip(shorter.indicators, longer.indicators):
        assert a.values == b.values[:100]
    assert len(features(shorter, True)) == len(FEATURE_NAMES + INDICATOR_FEATURE_NAMES)


def test_language_training_splits_and_future_exclusion(real_market):
    market = prefix(real_market, 240)
    before = training_records(market, stride=8)
    boundary = market.timestamps[int(len(market.ohlc) * 0.7)]
    assert all(row["label_observed_at"] < boundary for row in before["train"])
    assert all(row["as_of"] >= boundary for row in before["validation"])
    mutated = market.model_copy(deep=True)
    for row in mutated.ohlc[204:]:
        row[:] = [v * 2 for v in row]
    after = training_records(mutated, stride=8)
    assert before["train"] == after["train"]
    assert before["validation"] == after["validation"]
    assert compact_prompt(prefix(market, 100), []) == compact_prompt(prefix(mutated, 100), [])


def test_indicator_model_and_shipped_numerical(real_market, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_BACKEND", "decision")
    monkeypatch.delenv("TRADING_MODEL_PATH", raising=False)
    shipped = load_model()
    assert isinstance(shipped, DecisionModel)
    decision = shipped.predict(real_market, [])
    assert np.isfinite(decision.forecast.expected_return)
    target = tmp_path / "model.json"
    train(prefix(real_market, 240), str(target), include_indicators=True)
    assert json.loads(target.read_text())["features"] == FEATURE_NAMES + INDICATOR_FEATURE_NAMES
    TrainedModel(str(target)).predict(prefix(real_market, 250), [])
    with pytest.raises(ValueError, match="requires"):
        features(real_market.model_copy(update={"indicators": []}), True)
    provenance = json.loads(Path("data/BTCUSDT-1h.provenance.json").read_text())
    assert (
        hashlib.sha256(Path("data/BTCUSDT-1h.json").read_bytes()).hexdigest()
        == provenance["data_sha256"]
    )
