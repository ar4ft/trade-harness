"""Stage a pure-Python installer payload; signing/notarization runs only on macOS."""

import argparse
import shutil
from pathlib import Path

from trade_harness.releases import release_identity, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--root", default="artifacts/macos-payload")
    args = parser.parse_args()
    release_identity(args.version)
    wheel = Path(args.wheel)
    if wheel.name != f"trade_harness-{args.version}-py3-none-any.whl":
        raise SystemExit("Wheel version/name differs from the requested installer")
    root = Path(args.root)
    if root.exists():
        raise SystemExit("Payload directory already exists")
    share = root / "usr/local/share/trade-harness"
    binary = root / "usr/local/bin"
    share.mkdir(parents=True)
    binary.mkdir(parents=True)
    shutil.copyfile(wheel, share / wheel.name)
    template = Path(__file__).with_name("macos_bootstrap.py").read_text()
    template = (
        template.replace("__RELEASE_VERSION__", args.version)
        .replace("__RELEASE_WHEEL__", wheel.name)
        .replace("__RELEASE_SHA256__", sha256(wheel))
    )
    (share / "bootstrap.py").write_text(template)
    launcher = binary / "trade-harness-managed"
    launcher.write_text(
        '#!/bin/sh\nexec python3 -I /usr/local/share/trade-harness/bootstrap.py "$@"\n'
    )
    launcher.chmod(0o755)
    license_dir = share / "licenses"
    license_dir.mkdir()
    project = Path(__file__).resolve().parents[1]
    shutil.copyfile(project / "LICENSE", license_dir / "trade-harness-LICENSE.txt")
    for source in (project / "docs/licenses").glob("*.txt"):
        shutil.copyfile(source, license_dir / source.name)


if __name__ == "__main__":
    main()
