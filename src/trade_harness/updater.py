"""Verified startup updates into separate environments; never mutate a running account."""

import argparse
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import venv
from contextlib import contextmanager
from pathlib import Path

import httpx
from packaging.version import Version

from .releases import (
    BUNDLE_NAME,
    MANIFEST_NAME,
    REPOSITORY,
    VERSION_PATTERN,
    ReleaseManifest,
    artifact_url,
    release_identity,
    sha256,
)


class UpdateError(RuntimeError):
    pass


def default_home():
    if sys.platform == "win32":
        parent = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        parent = Path.home() / "Library/Application Support"
    else:
        parent = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return parent / "trade-harness"


def environment_python(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def update_lock(home):
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = home / "update.lock"
    if path.is_symlink():
        raise UpdateError("Update lock cannot be a symlink")
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                stream.write(b"0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise UpdateError("Another updater is running") from error
        yield


def fetch_file(client, url, target, limit):
    """Bound downloads, including redirects; only GitHub's HTTPS artifact hosts."""
    for _ in range(6):
        parsed = httpx.URL(url)
        if (
            parsed.scheme != "https"
            or parsed.port not in (None, 443)
            or not (
                parsed.host in ("github.com", "api.github.com")
                or parsed.host.endswith(".githubusercontent.com")
            )
        ):
            raise UpdateError("Unexpected update download host")
        with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise UpdateError("Invalid update redirect")
                url = str(response.url.join(location))
                continue
            response.raise_for_status()
            total = 0
            with target.open("xb") as output:
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > limit:
                        raise UpdateError("Update download exceeds its size limit")
                    output.write(chunk)
            return
    raise UpdateError("Too many update redirects")


def verify_bundle(manifest, bundle, version):
    # The identity is fixed by this installed bootstrap, never supplied by the manifest.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sigstore",
            "verify",
            "github",
            "--bundle",
            str(bundle),
            "--cert-identity",
            release_identity(version),
            "--trigger",
            "workflow_dispatch",
            "--repository",
            REPOSITORY,
            "--ref",
            "refs/tags/v" + version,
            str(manifest),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise UpdateError("Release signature, GitHub identity, or transparency proof is invalid")


def install_wheel(wheel, directory, version):
    extras = os.environ.get("TRADING_UPDATE_EXTRAS", "").strip()
    if not extras and os.environ.get("TRADING_BACKEND") == "local-llm":
        extras = "llm-train"
    if not extras and os.environ.get("TRADING_BACKEND") in ("timesfm", "hybrid"):
        extras = "timesfm"
    if not extras and os.environ.get("TRADING_BACKEND") == "orchestrator":
        extras = "orchestrator"
    if extras not in ("", "llm-train", "timesfm", "orchestrator"):
        raise UpdateError("Only llm-train, timesfm, and orchestrator dependency sets are supported for updates")
    requirement = str(wheel) + (f"[{extras}]" if extras else "")
    venv.EnvBuilder(with_pip=True).create(directory)
    python = environment_python(directory)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("PIP_") or name in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(name)
    env["PIP_CONFIG_FILE"] = os.devnull
    result = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "install",
            "--index-url",
            "https://pypi.org/simple",
            "--disable-pip-version-check",
            requirement,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode:
        raise UpdateError("Staged installation failed; current version retained")
    env["TRADING_BACKEND"] = "decision"
    env.pop("TRADING_DECISION_MODEL", None)
    smoke = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.metadata; from trade_harness.models import load_model; "
            "from trade_harness.api import app; "
            f"assert importlib.metadata.version('trade-harness') == {version!r}; load_model()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if smoke.returncode:
        raise UpdateError("Staged version failed its startup check; current version retained")


class Updater:
    def __init__(self, home=None, current_version=None, client=None):
        self.home = Path(home or os.environ.get("TRADING_UPDATE_HOME") or default_home())
        self.current_version = current_version or importlib.metadata.version("trade-harness")
        self.client = client

    def state(self):
        path = self.home / "state.json"
        if not path.exists():
            return {"active": None, "previous": None, "last_checked": 0}
        if path.is_symlink() or path.stat().st_size > 65536:
            raise UpdateError("Invalid managed update state")
        try:
            result = json.loads(path.read_text())
            if (
                not isinstance(result, dict)
                or not {"active", "previous", "last_checked"}.issubset(result)
                or not isinstance(result.get("last_checked"), (int, float))
            ):
                raise ValueError("Malformed state")
            if (
                not math.isfinite(result["last_checked"])
                or not 0 <= result["last_checked"] <= time.time() + 300
            ):
                raise ValueError("Invalid check timestamp")
            for key in ("active", "previous"):
                item = result.get(key)
                if item is None:
                    continue
                if not re.fullmatch(VERSION_PATTERN, item["version"]):
                    raise ValueError("Invalid installed version")
                if not re.fullmatch(r"[0-9.]+-[a-f0-9]{12}-[a-f0-9]{8}", item["directory"]):
                    raise ValueError("Invalid installation directory")
                if not re.fullmatch(r"[a-f0-9]{64}", item["manifest_sha256"]):
                    raise ValueError("Invalid installed manifest hash")
            return result
        except (ValueError, KeyError, TypeError) as error:
            raise UpdateError(
                "Managed state is corrupt; no unverified version will be launched"
            ) from error

    def active_python(self):
        active = self.state()["active"]
        if active is None:
            return Path(sys.executable)
        directory = self.home / "versions" / active["directory"]
        if (
            (self.home / "versions").is_symlink()
            or directory.is_symlink()
            or not environment_python(directory).is_file()
        ):
            raise UpdateError("Active managed installation is missing or invalid")
        return environment_python(directory)

    def update(self, apply=False, automatic=False):
        with update_lock(self.home):
            state = self.state()
            if automatic and time.time() - state["last_checked"] < 86400:
                return {"status": "recently_checked"}
            try:
                if self.client is not None:
                    return self._update(self.client, state, apply)
                with httpx.Client(
                    timeout=60, headers={"Accept": "application/vnd.github+json"}
                ) as client:
                    return self._update(client, state, apply)
            except (httpx.HTTPError, subprocess.TimeoutExpired, OSError, ValueError) as error:
                raise UpdateError(
                    f"Update failed ({type(error).__name__}); current version retained"
                ) from error

    def _update(self, client, state, apply):
        response = client.get(f"https://api.github.com/repos/{REPOSITORY}/releases/latest")
        response.raise_for_status()
        release = response.json()
        tag = release.get("tag_name", "")
        if (
            release.get("draft")
            or release.get("prerelease")
            or not re.fullmatch("v" + VERSION_PATTERN[1:], tag)
        ):
            raise UpdateError("Latest release is not a supported stable tag")
        version = tag[1:]
        asset_names = {a["name"] for a in release.get("assets", [])}
        if not {MANIFEST_NAME, BUNDLE_NAME}.issubset(asset_names):
            raise UpdateError("Release has no signed update manifest")
        with tempfile.TemporaryDirectory(prefix="download-", dir=self.home) as work:
            work = Path(work)
            manifest_path, bundle = work / MANIFEST_NAME, work / BUNDLE_NAME
            fetch_file(client, artifact_url(version, MANIFEST_NAME), manifest_path, 256 * 1024)
            fetch_file(client, artifact_url(version, BUNDLE_NAME), bundle, 2 * 1024**2)
            verify_bundle(manifest_path, bundle, version)
            manifest = ReleaseManifest.model_validate_json(manifest_path.read_bytes())
            if manifest.version != version or manifest.published_at.timestamp() > time.time() + 300:
                raise UpdateError("Signed manifest does not match the release tag/time")
            digest = sha256(manifest_path)
            active = state["active"]
            current = max(
                Version(self.current_version),
                Version(active["version"]) if active else Version("0"),
            )
            candidate = Version(version)
            if candidate < current:
                raise UpdateError("Release would downgrade the installed version")
            if active and active["version"] == version and active["manifest_sha256"] != digest:
                raise UpdateError("Published manifest changed for an installed version")
            if candidate == current:
                state["last_checked"] = time.time()
                atomic_json(self.home / "state.json", state)
                return {"status": "up_to_date", "version": version}
            artifact = next(a for a in manifest.artifacts if a.kind == "python-wheel")
            if artifact.name not in asset_names:
                raise UpdateError("Signed wheel is missing from the release")
            if not apply:
                return {
                    "status": "available",
                    "version": version,
                    "identity": release_identity(version),
                }
            wheel = work / artifact.name
            fetch_file(client, artifact.url, wheel, artifact.size)
            if wheel.stat().st_size != artifact.size or sha256(wheel) != artifact.sha256:
                raise UpdateError("Downloaded wheel does not match the signed size/hash")
            versions = self.home / "versions"
            if versions.is_symlink():
                raise UpdateError("Managed versions directory cannot be a symlink")
            versions.mkdir(exist_ok=True, mode=0o700)
            name = f"{version}-{digest[:12]}-{uuid.uuid4().hex[:8]}"
            directory = versions / name
            try:
                # Unique final path avoids relocating virtualenv scripts after installation.
                install_wheel(wheel, directory, version)
                next_state = {
                    "active": {"version": version, "directory": name, "manifest_sha256": digest},
                    "previous": active,
                    "last_checked": time.time(),
                }
                atomic_json(self.home / "state.json", next_state)
            except (UpdateError, OSError, subprocess.TimeoutExpired):
                shutil.rmtree(directory, ignore_errors=True)
                raise
            return {"status": "installed", "version": version, "previous": active}


def managed_main():
    parser = argparse.ArgumentParser(description="Launch with signed startup updates")
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    updater = Updater()
    if not args.no_update and os.environ.get("TRADING_AUTO_UPDATE", "1") != "0":
        try:
            result = updater.update(apply=True, automatic=True)
            print(json.dumps({"update": result}), file=sys.stderr)
        except UpdateError as error:
            print(str(error), file=sys.stderr)
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    python = updater.active_python()
    # -I removes inherited import paths; the child uses its own installed package/dependencies.
    os.execv(
        str(python), [str(python), "-I", "-m", "trade_harness.cli", *(arguments or ["decide"])]
    )
