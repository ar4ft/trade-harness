"""Generate the exact manifest signed by the tag-only release workflow."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from trade_harness.releases import ReleaseArtifact, ReleaseManifest, artifact_url, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--directory", default="release")
    args = parser.parse_args()
    directory = Path(args.directory)
    artifacts = []
    for path in sorted(directory.iterdir()):
        if path.suffix not in (".whl", ".pkg"):
            continue
        kind = "python-wheel" if path.suffix == ".whl" else "macos-universal"
        artifacts.append(
            ReleaseArtifact(
                name=path.name,
                url=artifact_url(args.version, path.name),
                sha256=sha256(path),
                size=path.stat().st_size,
                kind=kind,
                notarized=path.suffix == ".pkg",
            )
        )
    manifest = ReleaseManifest(
        version=args.version,
        tag="v" + args.version,
        commit=args.commit,
        published_at=datetime.now(timezone.utc),
        artifacts=artifacts,
    )
    (directory / "release-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
