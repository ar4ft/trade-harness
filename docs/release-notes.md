# 0.9.1 (development)

Adds full 27B Clef NF4/BF16 self-host launch instructions alongside Flash, plus a no-download hardware planning check for both variants. The pinned full snapshot is 55 GB; full-model inference remains untested because this workspace lacks disk, RAM and a GPU. Flash has verified native and orchestrator inference. Trading promotion still requires independent forward evidence; signing remains manual-only. See [Clef deployment](clef.md).

# 0.9.0 (development)

Adds Clef and Clef-Flash native typed reviewers through Cloudflare Workers AI or a pinned self-hosted GPU service. Adds native request/target exports using causal position/cost labels. Verifies typed outputs, ordinal rounding, context budgets and snapshot/service identity; hosted aliases cannot certify locked forward experiments. Real Flash NF4 native and complete-orchestrator CPU smoke tests succeeded; full 27B and hosted inference remain unmeasured. No trading edge is established and the default reviewer remains unchanged. Manual-only signing is preserved. See [Clef setup and evaluation](clef.md).

# 0.8.0 (development)

Adds a shared-evidence reviewer training contract with position-aware cost labels and separate chronological calibration. Adds full numerical/reviewer consensus ablations through matched paper accounts, row-level audit ledgers, decision-band and regime diagnostics, causal quant features, and twelve bounded parameter trials. Legacy historical gates now require matching `research-promotion-v2` evidence before promotion. Locked forward declarations and hash-chained paper journals preserve model/code/risk identities and attach outcomes only after capture. No artifact activation or exchange execution is automatic. Release signing remains restricted to manual workflow dispatch. See the [research workflow](research-workflow.md).

# 0.7.0 (development)

Adds Ollaya native typed decision review as an optional layer before consensus or a standalone research backend. Records model manifest provenance, validates probability rounding and rejects truncation/routing changes. TimesFM, numerical confirmation, risk and manual-only release signing remain owned by the harness. No trading edge or trading fine-tuning is claimed. See [Ollaya setup](ollaya.md).

# 0.6.1 (development)

Makes shared-evidence orchestration the default CLI/API backend, with the hybrid numerical anchor and local LoRA reviewer. Managed updates install its dependency group when no backend override is configured. Explicit backend choices remain supported. Research-only validation, decision-only operation, and manual-only signing remain in place.

# 0.6.0 (development)

Adds a shared-evidence orchestrator connecting the hybrid numerical model to local, OpenAI-compatible, Nimble, or llamafile language review. Fixed confirmation policies abstain on disagreement, low supporting scores, or member failure, and expose vote audits in CLI/API/dashboard. Adds bounded causal feature recipes and chronological regularization experiments with a separately shipped research artifact. Reviewed fine-tuning exports preserve shared snapshot evidence without future returns in prompts. No language weights were retrained, no profitable edge was established, and signing remains manual-only. See [orchestration and experiments](orchestration.md).

# 0.5.0 (development)

Adds versioned causal strategy hypotheses and an optional hybrid decision backend: observed market features plus TimesFM 3.0 forecasts feed a trained numerical decision layer. Historical forecast caching and purged ablation comparisons measure added evidence against matched baselines, with doubled-cost paper replay. Forecast and strategy evidence are separately exposed in CLI/API decisions. The default backend is retained; hybrid remains research-only; signing remains manual-only. See [hybrid setup and evaluation](hybrid-decisions.md).

# 0.4.2 (development)

Adds an optional, pinned TimesFM 3.0 research backend with multivariate OHLC forecasting, historical volume/indicator covariates, and quantile-based direction proposals. CLI/API integration and managed update dependency selection are supported. Weights download separately; no trading edge is certified. Manual-only signing and decision-only operation remain in place. See [TimesFM setup and use limits](timesfm.md).

Decision-only trading validation for Trade Harness.

- Per-asset walk-forward evidence attached to recommendations, with mode and real_execution_enabled=false.
- Purged temporal checks, momentum comparison, doubled-fee/slippage replay, and descriptive fold-bootstrap intervals.
- A standalone validate command and GET /validation endpoint.
- Independent artifact-based evidence; provider-generated validation claims cannot promote a model.

The current models and all assets remain research-only: no positive edge has been established. Passing checks does not enable real orders. Simulated replay is retained for measuring decision quality.

Development builds remain unsigned. Signing/publication continues to require explicit manual workflow dispatch; Apple notarization awaits its configured credentials.
