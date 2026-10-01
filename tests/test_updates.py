import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from trade_harness import updater
from trade_harness.releases import (
    BUNDLE_NAME,
    MANIFEST_NAME,
    ReleaseManifest,
    artifact_url,
    release_identity,
)
from trade_harness.updater import UpdateError, Updater, environment_python, update_lock


def manifest(version="0.5.0", wheel=b"verified wheel"):
    name = f"trade_harness-{version}-py3-none-any.whl"
    return {
        "schema_version": 1,
        "repository": "ar4ft/trade-harness",
        "channel": "stable",
        "version": version,
        "tag": "v" + version,
        "commit": "a" * 40,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {
                "name": name,
                "url": artifact_url(version, name),
                "sha256": hashlib.sha256(wheel).hexdigest(),
                "size": len(wheel),
                "kind": "python-wheel",
                "notarized": False,
            }
        ],
    }


@pytest.fixture
def feed(monkeypatch):
    class Feed:
        def __init__(self):
            self.manifest = manifest()
            self.wheel = b"verified wheel"
            self.signed = json.dumps(self.manifest).encode()
            self.unsigned = False
            self.calls = []
            self.installs = []

        def response(self, request):
            self.calls.append(str(request.url))
            if request.url.path.endswith("/latest"):
                names = [MANIFEST_NAME, BUNDLE_NAME, self.manifest["artifacts"][0]["name"]]
                return httpx.Response(
                    200,
                    json={
                        "tag_name": self.manifest["tag"],
                        "draft": False,
                        "prerelease": False,
                        "assets": [] if self.unsigned else [{"name": name} for name in names],
                    },
                )
            if request.url.path.endswith(MANIFEST_NAME):
                return httpx.Response(200, content=json.dumps(self.manifest).encode())
            if request.url.path.endswith(BUNDLE_NAME):
                return httpx.Response(200, content=b"signature fixture")
            return httpx.Response(200, content=self.wheel)

        def verify(self, path, bundle, version):
            if path.read_bytes() != self.signed:
                raise UpdateError("Signature invalid")
            assert bundle.read_bytes() == b"signature fixture"
            assert version == self.manifest["version"]

        def install(self, wheel, directory, version):
            self.installs.append((wheel.read_bytes(), directory, version))
            python = environment_python(directory)
            python.parent.mkdir(parents=True)
            python.write_bytes(b"installed interpreter fixture")

        def resign(self):
            self.signed = json.dumps(self.manifest).encode()

    result = Feed()
    monkeypatch.setattr(updater, "verify_bundle", result.verify)
    monkeypatch.setattr(updater, "install_wheel", result.install)
    with httpx.Client(transport=httpx.MockTransport(result.response)) as client:
        result.client = client
        yield result


def runtime(tmp_path, feed, current="0.4.0"):
    return Updater(tmp_path, current_version=current, client=feed.client)


def test_check_verifies_manifest_without_installing(tmp_path, feed):
    result = runtime(tmp_path, feed).update()
    assert result["status"] == "available"
    assert result["identity"] == release_identity("0.5.0")
    assert not feed.installs
    assert not (tmp_path / "state.json").exists()


def test_successful_install_and_cached_automatic_launch(tmp_path, feed):
    manager = runtime(tmp_path, feed)
    assert manager.update(apply=True)["status"] == "installed"
    assert manager.active_python().is_file()
    count = len(feed.calls)
    assert manager.update(apply=True, automatic=True)["status"] == "recently_checked"
    assert len(feed.calls) == count
    assert manager.update()["status"] == "up_to_date"


def test_second_install_retains_previous_environment(tmp_path, feed):
    manager = runtime(tmp_path, feed)
    manager.update(apply=True)
    previous = manager.active_python()
    feed.manifest = manifest("0.6.0")
    feed.resign()
    result = manager.update(apply=True)
    assert result["previous"]["version"] == "0.5.0"
    assert previous.is_file() and previous != manager.active_python()


def test_install_failure_keeps_previous_state(tmp_path, feed, monkeypatch):
    manager = runtime(tmp_path, feed)
    manager.update(apply=True)
    before = (tmp_path / "state.json").read_bytes()
    feed.manifest = manifest("0.6.0")
    feed.resign()

    def failed(wheel, directory, version):
        directory.mkdir()
        raise UpdateError("startup check failed")

    monkeypatch.setattr(updater, "install_wheel", failed)
    with pytest.raises(UpdateError, match="startup"):
        manager.update(apply=True)
    assert (tmp_path / "state.json").read_bytes() == before
    assert len(list((tmp_path / "versions").iterdir())) == 1


def test_forged_manifest_cannot_download_or_install_wheel(tmp_path, feed):
    feed.manifest["version"] = "99.0.0"
    feed.manifest["tag"] = "v99.0.0"
    with pytest.raises(UpdateError, match="Signature invalid"):
        runtime(tmp_path, feed).update(apply=True)
    assert not feed.installs
    assert not any(call.endswith(".whl") for call in feed.calls)


@pytest.mark.parametrize("change", ["hash", "oversize", "undersize"])
def test_tampered_wheel_rejected(tmp_path, feed, change):
    feed.wheel = {
        "hash": b"maliciousbytes",
        "oversize": b"verified wheel EXTRA",
        "undersize": b"short",
    }[change]
    with pytest.raises(UpdateError):
        runtime(tmp_path, feed).update(apply=True)
    assert not feed.installs
    assert not (tmp_path / "state.json").exists()


def test_unsigned_release_is_not_an_update(tmp_path, feed):
    feed.unsigned = True
    with pytest.raises(UpdateError, match="signed update manifest"):
        runtime(tmp_path, feed).update(apply=True)


def test_downgrade_rejected(tmp_path, feed):
    with pytest.raises(UpdateError, match="downgrade"):
        runtime(tmp_path, feed, current="0.6.0").update(apply=True)


def test_installed_version_manifest_cannot_change(tmp_path, feed):
    manager = runtime(tmp_path, feed)
    manager.update(apply=True)
    feed.manifest["commit"] = "b" * 40
    feed.resign()
    with pytest.raises(UpdateError, match="manifest changed"):
        manager.update(apply=True)


def test_future_manifest_rejected(tmp_path, feed):
    feed.manifest["published_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    feed.resign()
    with pytest.raises(UpdateError, match="tag/time"):
        runtime(tmp_path, feed).update()


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", "attacker/repo"),
        ("tag", "v0.6.0"),
        ("version", "0.5.0rc1"),
    ],
)
def test_manifest_contract_rejects_untrusted_metadata(field, value):
    data = manifest()
    data[field] = value
    with pytest.raises(ValidationError):
        ReleaseManifest.model_validate(data)


@pytest.mark.parametrize(
    "name,url",
    [
        ("../package.whl", "https://github.com/attacker/file.whl"),
        ("trade_harness-0.5.0-py3-none-any.whl", "http://github.com/ar4ft/trade-harness/file.whl"),
        ("trade_harness-0.5.0-py3-none-any.whl", "https://evil.test/package.whl"),
    ],
)
def test_artifact_path_and_url_cannot_escape_release(name, url):
    data = manifest()
    data["artifacts"][0].update(name=name, url=url)
    with pytest.raises(ValidationError):
        ReleaseManifest.model_validate(data)


def test_identity_and_issuer_are_pinned(monkeypatch, tmp_path):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(UpdateError, match="identity"):
        updater.verify_bundle(tmp_path / "manifest", tmp_path / "bundle", "0.5.0")
    command = commands[0]
    assert command[command.index("--cert-identity") + 1] == (
        "https://github.com/ar4ft/trade-harness/.github/workflows/release.yml@refs/tags/v0.5.0"
    )
    assert command[3:5] == ["verify", "github"]
    assert command[command.index("--trigger") + 1] == "workflow_dispatch"
    assert command[command.index("--repository") + 1] == "ar4ft/trade-harness"
    assert command[command.index("--ref") + 1] == "refs/tags/v0.5.0"


def test_download_redirect_cannot_send_request_to_untrusted_host(tmp_path):
    calls = []

    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://localhost/admin"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(UpdateError, match="host"):
            updater.fetch_file(client, "https://github.com/start", tmp_path / "file", 1024)
    assert calls == ["https://github.com/start"]


def test_concurrent_updates_are_locked(tmp_path):
    with update_lock(tmp_path):
        with pytest.raises(UpdateError, match="Another updater"):
            with update_lock(tmp_path):
                pytest.fail("Second lock acquired")


def test_corrupt_state_never_launches_a_saved_path(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "active": {
                    "version": "0.5.0",
                    "directory": "../../evil",
                    "manifest_sha256": "a" * 64,
                },
                "previous": None,
                "last_checked": 0,
            }
        )
    )
    with pytest.raises(UpdateError, match="corrupt"):
        Updater(tmp_path, current_version="0.4.0").active_python()


def test_macos_artifact_requires_notarization():
    data = manifest()
    name = "trade-harness-0.5.0-macos-universal.pkg"
    item = {
        "name": name,
        "url": artifact_url("0.5.0", name),
        "sha256": "a" * 64,
        "size": 100,
        "kind": "macos-universal",
        "notarized": False,
    }
    data["artifacts"].append(item)
    with pytest.raises(ValidationError, match="notarized"):
        ReleaseManifest.model_validate(data)
    item["notarized"] = True
    assert ReleaseManifest.model_validate(data).artifacts[-1].notarized


def test_macos_payload_contains_bound_wheel(tmp_path):
    import importlib.util
    import sys

    source = Path("scripts/build_macos_payload.py")
    spec = importlib.util.spec_from_file_location("mac_payload", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wheel = tmp_path / "trade_harness-0.5.0-py3-none-any.whl"
    wheel.write_bytes(b"trusted wheel fixture")
    from unittest.mock import patch

    root = tmp_path / "payload"
    with patch.object(
        sys, "argv", [str(source), "--version", "0.5.0", "--wheel", str(wheel), "--root", str(root)]
    ):
        module.main()
    bootstrap = (root / "usr/local/share/trade-harness/bootstrap.py").read_text()
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() in bootstrap
    assert "__RELEASE_" not in bootstrap
    compile(bootstrap, "bootstrap.py", "exec")
    assert (root / "usr/local/bin/trade-harness-managed").stat().st_mode & 0o111


@pytest.mark.parametrize(
    "event,sign,notarize,enabled,publish,macos",
    [
        ("push", False, False, "false", False, False),
        ("push", True, True, "true", False, False),
        ("workflow_dispatch", False, True, "true", False, False),
        ("workflow_dispatch", True, False, "false", True, False),
        ("workflow_dispatch", True, True, "false", True, True),
        ("workflow_dispatch", True, False, "true", True, True),
    ],
)
def test_signing_and_notarization_require_explicit_manual_run(
    event, sign, notarize, enabled, publish, macos
):
    import yaml

    workflow = yaml.load(Path(".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["workflow_dispatch"]["inputs"]["sign_release"]["default"] == "false"
    context = {
        "github": SimpleNamespace(event_name=event),
        "inputs": SimpleNamespace(sign_release=sign, notarize_macos=notarize),
        "vars": SimpleNamespace(MACOS_NOTARIZATION_ENABLED=enabled),
        "needs": SimpleNamespace(
            wheel=SimpleNamespace(result="success"), macos=SimpleNamespace(result="success")
        ),
    }
    for job, expected in (("publish", publish), ("macos", macos)):
        expression = workflow["jobs"][job]["if"].removeprefix("${{").removesuffix("}}")
        expression = (
            expression.replace("&&", " and ").replace("||", " or ").replace("== true", "== True")
        )
        result = eval(expression, {"__builtins__": {}, "always": lambda: True}, context)
        assert result is expected


def test_unknown_update_dependency_set_is_rejected_before_install(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_UPDATE_EXTRAS", "untrusted-extra")
    with pytest.raises(UpdateError, match="dependency set"):
        updater.install_wheel(tmp_path / "package.whl", tmp_path / "environment", "0.5.0")
    assert not (tmp_path / "environment").exists()


@pytest.mark.parametrize(
    "state", [{"last_checked": 0}, {"active": None, "previous": None, "last_checked": float("inf")}]
)
def test_incomplete_state_is_rejected(tmp_path, state):
    (tmp_path / "state.json").write_text(json.dumps(state))
    with pytest.raises(UpdateError, match="corrupt"):
        Updater(tmp_path, current_version="0.4.0").state()


@pytest.mark.parametrize("backend", ["timesfm", "hybrid"])
def test_forecast_updates_select_optional_dependencies(tmp_path, monkeypatch, backend):
    monkeypatch.delenv("TRADING_UPDATE_EXTRAS", raising=False)
    monkeypatch.setenv("TRADING_BACKEND", backend)
    create = []
    commands = []
    monkeypatch.setattr(updater.venv.EnvBuilder, "create", lambda self, path: create.append(path))

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(updater.subprocess, "run", run)
    wheel = tmp_path / "package.whl"
    updater.install_wheel(wheel, tmp_path / "environment", "0.4.2")
    assert create == [tmp_path / "environment"]
    assert commands[0][-1] == str(wheel) + "[timesfm]"
