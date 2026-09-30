"""Template installed by the signed macOS package. Uses only the standard library."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

VERSION = "__RELEASE_VERSION__"
WHEEL_NAME = "__RELEASE_WHEEL__"
WHEEL_SHA256 = "__RELEASE_SHA256__"


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Install Python 3.11 or newer before starting Trade Harness")
    if importlib.util.find_spec("venv") is None:
        raise SystemExit("Python's venv module is required")
    wheel = Path(__file__).resolve().parent / WHEEL_NAME
    with wheel.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WHEEL_SHA256:
            raise SystemExit("Installed bootstrap wheel hash is invalid")
    root = Path(
        os.environ.get(
            "TRADING_BOOTSTRAP_HOME",
            str(Path.home() / "Library/Application Support/trade-harness/bootstrap"),
        )
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    import fcntl

    with (root / "bootstrap.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = root / VERSION
        python = directory / "bin/python"
        marker = directory / "bootstrap.json"
        if marker.exists():
            if json.loads(marker.read_text()) != {"sha256": WHEEL_SHA256} or not python.is_file():
                raise SystemExit("Bootstrap installation differs; reinstall the signed package")
        else:
            env = os.environ.copy()
            for name in list(env):
                if name.startswith("PIP_") or name in ("PYTHONPATH", "PYTHONHOME"):
                    env.pop(name)
            env["PIP_CONFIG_FILE"] = os.devnull
            try:
                venv.EnvBuilder(with_pip=True).create(directory)
                subprocess.run(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "--isolated",
                        "install",
                        "--index-url",
                        "https://pypi.org/simple",
                        "--disable-pip-version-check",
                        str(wheel),
                    ],
                    env=env,
                    check=True,
                    timeout=600,
                )
                marker.write_text(json.dumps({"sha256": WHEEL_SHA256}))
            except (subprocess.SubprocessError, OSError):
                shutil.rmtree(directory, ignore_errors=True)
                raise SystemExit(
                    "Bootstrap installation failed; retry when dependencies are available"
                ) from None
    os.execv(
        str(python),
        [
            str(python),
            "-I",
            "-c",
            "from trade_harness.updater import managed_main; managed_main()",
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
