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
