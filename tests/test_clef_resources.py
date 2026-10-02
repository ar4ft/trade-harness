import json
import sys
from types import SimpleNamespace

import pytest

from trade_harness import clef_resources as resources
from trade_harness.clef_contract import RELEASES
from trade_harness.clef_server import main


def test_full_nf4_still_needs_original_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=5 * resources.GIB)
    )
    monkeypatch.setattr(resources, "available_ram", lambda: 16 * resources.GIB)
    report = resources.inspect_resources("clef", "nf4", "cpu", tmp_path)
    assert report["snapshot_bytes"] == 54989894057
    assert report["revision"] == RELEASES["clef"][1]
    assert not report["ready_for_fresh_deployment"]
    assert not report["inference_tested"]


def test_sufficient_full_cpu_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(
        resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * resources.GIB)
    )
    monkeypatch.setattr(resources, "available_ram", lambda: 64 * resources.GIB)
    assert resources.inspect_resources("clef", "nf4", "cpu", tmp_path)["ready_for_fresh_deployment"]
    assert not resources.inspect_resources("clef", "none", "cpu", tmp_path)[
        "ready_for_fresh_deployment"
    ]


def test_custom_revision_needs_size_check(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "available_ram", lambda: 100 * resources.GIB)
    report = resources.inspect_resources("clef", "nf4", "cpu", tmp_path, "a" * 40)
    assert report["snapshot_bytes"] is None
    assert not report["ready_for_fresh_deployment"]


@pytest.mark.parametrize("cuda_available,ready", [(True, True), (False, False)])
def test_full_gpu_budget(monkeypatch, tmp_path, cuda_available, ready):
    monkeypatch.setattr(
        resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * resources.GIB)
    )
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        mem_get_info=lambda device: (40 * resources.GIB, 48 * resources.GIB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, device=lambda name: name))
    report = resources.inspect_resources("clef", "nf4", "cuda:0", tmp_path)
    assert report["ready_for_fresh_deployment"] == ready
    assert bool(report["gpu_inspection_error"]) != cuda_available


def test_unsupported_device(tmp_path):
    with pytest.raises(ValueError, match="supports"):
        resources.inspect_resources("clef", "nf4", "mps", tmp_path)


def test_check_returns_json_without_loading_weights(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clef_server",
            "--model",
            "clef",
            "--device",
            "cpu",
            "--quantization",
            "nf4",
            "--check",
            "--cache-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(resources, "available_ram", lambda: 0)
    assert main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "clef"
    assert not report["inference_tested"]
