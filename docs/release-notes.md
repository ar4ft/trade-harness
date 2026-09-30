Signed releases and verified automatic updates for Trade Harness.

- Automatic unsigned development builds, with signing/publication enabled only by explicit manual workflow dispatch.
- Keyless Sigstore release manifests tied to this repository's tagged GitHub release workflow.
- Managed startup updates with signature/hash checks, downgrade protection, isolated staging, atomic activation, and retention of the previous environment.
- CLI update/check commands and a managed launcher for decisions, paper trading, and the HTTP/browser monitor.
- Prepared Developer ID Installer signing, Apple notarization, ticket stapling, and Gatekeeper validation for a universal macOS Python launcher package.

The macOS installer is included only when Apple credentials and notarization are enabled. Otherwise this release contains the signed portable wheel. Python 3.11+ is required. External llamafile runtimes are distributed separately.

Trading models remain research-only; default policy blocks unvalidated entries. No live exchange order execution is enabled by installation or updating.
