# Signed releases, macOS notarization, and automatic updates

Stable releases use a signed manifest covering the exact artifact sizes and SHA256 hashes. GitHub Actions obtains a short-lived Sigstore certificate using its GitHub OIDC identity, signs the manifest, and records it in Sigstore's transparency log. No long-lived project signing key is needed.

The updater trusts only `ar4ft/trade-harness`, the manually dispatched `.github/workflows/release.yml` workflow, and the exact `refs/tags/vVERSION` identity issued by `https://token.actions.githubusercontent.com`. Trust is embedded in the installed bootstrap, not taken from downloaded metadata. Certificate claims must also identify the `workflow_dispatch` event, repository, and tag; the GitHub verifier always requires GitHub’s OIDC issuer. Tag, version, repository, URLs, size limits, signature, certificate identity, and transparency proof are checked before installation. A checksum alone is not treated as a signature.

## User commands

Install a trusted initial package using Python 3.11+, then:

```bash
# Verify the latest signed manifest and report availability:
trade-harness update --check

# Download, verify, stage, and activate a newer managed version:
trade-harness update --apply

# Automatically check/install at launch, at most once per day:
trade-harness-managed -- decide
trade-harness-managed -- paper --steps 0
trade-harness-managed -- serve --port 8000

# Start the currently installed managed version without checking:
trade-harness-managed --no-update -- decide
```

`TRADING_AUTO_UPDATE=0` also disables startup checks. Direct `trade-harness` and `uvicorn` commands do not automatically update; automatic updates belong to the managed launcher. `--check` verifies metadata without downloading/installing the wheel. The application does not grant a remote release permission to mutate a running paper account.

Updates install into a separate per-user virtual environment, run a startup smoke check, then atomically change the active-version pointer. The old environment is retained. Failed verification, interrupted downloads, installer errors, and startup-check failures keep the existing version active. A process already running keeps its interpreter and code until restarted. The launcher can run the existing version when the update service is unavailable. Corrupt local state fails closed.

The updater rejects versions lower than the installed bootstrap/active version and rejects changed signed manifests for an already installed version. It accepts stable numeric versions only. Native installer updates are not run with administrator privileges by the updater; the macOS package installs the same Python-based managed launcher, which performs later updates per user.

Managed environments live under `~/.local/share/trade-harness` on Linux, `~/Library/Application Support/trade-harness` on macOS, and `%LOCALAPPDATA%/trade-harness` on Windows. Override with `TRADING_UPDATE_HOME` when needed. The signed artifact is the project wheel; runtime dependencies are installed from PyPI over HTTPS in an isolated environment, rather than bundled or covered by the project signature. Initial installation and the local Python interpreter must be trusted. For Python LoRA inference, set `TRADING_UPDATE_EXTRAS=llm-train` before upgrades to include its optional dependencies; `TRADING_BACKEND=local-llm` also selects that dependency set. These can include large PyTorch downloads. External llamafile inference needs only the core dependencies.

Packaged numerical weights, the LoRA adapter/tokenizer, dashboard, and harness code update together. External inference servers and standalone llamafile executables are not automatically replaced. Model version changes may require a new paper run ID; persisted accounts are never silently migrated to a different model. SQLite files remain at their existing configured locations.

## Maintainer release workflow

1. Update the package/API version, release notes, and tests; merge to `main`.
2. Create a fresh immutable tag matching the package version, such as `v0.4.0`.
3. Pushes to `main` and version tags automatically test/build an **unsigned development wheel**, stored as an Actions artifact. They never sign, notarize, or publish a stable release.
4. Manually dispatch **Signed release**, select the unpublished version tag, and enable `sign_release`. This run optionally builds/notarizes macOS, signs the manifest, verifies its own signature, uploads all assets into a draft, then publishes it.

```bash
# Unsigned development build; main or a version tag is allowed:
gh workflow run release.yml --ref main -f sign_release=false

# Explicitly sign/publish the tagged stable release:
gh workflow run release.yml --ref v0.4.0 -f sign_release=true

# Also sign/notarize macOS once the Apple secrets are configured:
gh workflow run release.yml --ref v0.4.0 -f sign_release=true -f notarize_macos=true
```

`sign_release` defaults to false. Enabling notarization alone cannot trigger signing. Development artifacts have no signed manifest and are rejected by the automatic updater. Install a development wheel explicitly in a separate development virtual environment, using `pip install path/to/wheel.whl`.

The release workflow is pinned to action commit SHAs. Only the publishing job receives `contents: write` and `id-token: write`. It does not sign automatic pushes, pull requests, or a branch dispatch. Manual dispatch must use an unpublished version tag. Existing releases are never overwritten; fix an unsuccessful published version with a new version. Protect release tags from modification/deletion in GitHub repository rulesets.

Assets are `trade_harness-VERSION-py3-none-any.whl`, `release-manifest.json`, and `release-manifest.sigstore.json`. When enabled, a signed/notarized `trade-harness-VERSION-macos-universal.pkg` is included in the same signed manifest. The universal package contains the Python wheel and a launcher, not a standalone bundled Python interpreter.

Manual verification after downloading the manifest and Sigstore bundle:

```bash
python -m sigstore verify github \
  --bundle release-manifest.sigstore.json \
  --cert-identity 'https://github.com/ar4ft/trade-harness/.github/workflows/release.yml@refs/tags/v0.4.0' \
  --trigger workflow_dispatch --repository ar4ft/trade-harness \
  --ref refs/tags/v0.4.0 release-manifest.json
```

Use the actual release tag in the expected identity, then check the downloaded wheel against the authenticated manifest. For an initial install, verify the release with an independently installed Sigstore client before installing the wheel. Release signing is an authenticity check; it does not certify trading profitability.

## Enable Apple signing and notarization

Apple notarization is prepared but cannot run until the repository owner has Apple Developer Program access and supplies credentials. No Apple credentials are present in this development environment. Its GitHub integration cannot administer Actions secrets (HTTP 403), so configure them directly in **Settings → Secrets and variables → Actions**. Never commit or paste private keys into issues/chat.

| Secret | Required value |
| --- | --- |
| `MACOS_INSTALLER_P12_BASE64` | Base64 export containing a **Developer ID Installer** certificate and its private key |
| `MACOS_CERT_PASSWORD` | Password protecting that P12 export |
| `MACOS_INSTALLER_IDENTITY` | Full certificate identity, such as `Developer ID Installer: Your Organization (TEAMID)` |
| `APPLE_API_KEY_P8_BASE64` | Base64 App Store Connect API private key with notarization access |
| `APPLE_API_KEY_ID` | API key ID |
| `APPLE_API_ISSUER_ID` | API issuer UUID |

The current installer contains scripts and a wheel, so it uses **Developer ID Installer** signing. It does not contain a native `.app` needing a Developer ID Application identity. Python 3.11+ must already be installed on the Mac. First launch creates a user-owned bootstrap environment and downloads runtime dependencies; subsequent launches use the signed-update mechanism.

Set the repository **variable** `MACOS_NOTARIZATION_ENABLED=true` after configuring all six secrets. Future **manually signed** tag releases will then include notarized macOS packages. Alternatively, manually dispatch an unpublished version tag with both `sign_release` and `notarize_macos` enabled. If enabled and any credential, signature, notarization, staple, or Gatekeeper check fails, the entire release stays unpublished; there is no unsigned macOS fallback.

Apple signing keys are imported into a temporary keychain on an ephemeral GitHub macOS runner. Temporary certificate/key files are removed when the signing step exits. `notarytool` must return `Accepted`, then `stapler` and `spctl` must validate the final installer before upload. Manifest hashes are generated **after** stapling, since stapling changes the artifact.

macOS packaging/signing/notarization remains unverified until the first credentialed macOS run. A portable signed wheel release does not imply an Apple-notarized installer exists.
