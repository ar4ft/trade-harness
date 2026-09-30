"""Signed release contract shared by publishing and the managed updater."""

import hashlib
import re
from datetime import datetime
from pathlib import Path

from pydantic import Field, model_validator

from .schemas import StrictModel

REPOSITORY = "ar4ft/trade-harness"
WORKFLOW = ".github/workflows/release.yml"
ISSUER = "https://token.actions.githubusercontent.com"
VERSION_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+$"
MANIFEST_NAME = "release-manifest.json"
BUNDLE_NAME = "release-manifest.sigstore.json"


def release_identity(version):
    if not re.fullmatch(VERSION_PATTERN, version):
        raise ValueError("Only stable numeric release versions are supported")
    return f"https://github.com/{REPOSITORY}/{WORKFLOW}@refs/tags/v{version}"


def artifact_url(version, name):
    return f"https://github.com/{REPOSITORY}/releases/download/v{version}/{name}"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ReleaseArtifact(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,160}$")
    url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(gt=0, le=1024**3)
    kind: str = Field(pattern=r"^(python-wheel|macos-universal)$")
    notarized: bool = False


class ReleaseManifest(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    repository: str = REPOSITORY
    channel: str = "stable"
    version: str = Field(pattern=VERSION_PATTERN)
    tag: str
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    published_at: datetime
    artifacts: list[ReleaseArtifact] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def contract(self):
        if self.repository != REPOSITORY or self.channel != "stable":
            raise ValueError("Release repository/channel differs from the trusted project")
        if self.tag != "v" + self.version or self.published_at.tzinfo is None:
            raise ValueError("Release tag/time is invalid")
        if len({a.kind for a in self.artifacts}) != len(self.artifacts):
            raise ValueError("Duplicate artifact kind")
        if len({a.name for a in self.artifacts}) != len(self.artifacts):
            raise ValueError("Duplicate artifact filename")
        for artifact in self.artifacts:
            if artifact.url != artifact_url(self.version, artifact.name):
                raise ValueError("Artifact URL is outside this release")
            if artifact.kind == "python-wheel":
                if artifact.name != f"trade_harness-{self.version}-py3-none-any.whl":
                    raise ValueError("Unexpected wheel name")
                if artifact.notarized or artifact.size > 50 * 1024**2:
                    raise ValueError("Invalid portable wheel metadata")
            elif (
                artifact.name != f"trade-harness-{self.version}-{artifact.kind}.pkg"
                or not artifact.notarized
            ):
                raise ValueError("macOS installers must be notarized and use the release name")
        if not any(a.kind == "python-wheel" for a in self.artifacts):
            raise ValueError("Portable wheel is required")
        return self
