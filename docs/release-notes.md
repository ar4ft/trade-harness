# 0.4.2 (development)

Adds an optional, pinned TimesFM 3.0 research backend with multivariate OHLC forecasting, historical volume/indicator covariates, and quantile-based direction proposals. CLI/API integration and managed update dependency selection are supported. Weights download separately; no trading edge is certified. Manual-only signing and decision-only operation remain in place. See [TimesFM setup and use limits](timesfm.md).

Decision-only trading validation for Trade Harness.

- Per-asset walk-forward evidence attached to recommendations, with mode and real_execution_enabled=false.
- Purged temporal checks, momentum comparison, doubled-fee/slippage replay, and descriptive fold-bootstrap intervals.
- A standalone validate command and GET /validation endpoint.
- Independent artifact-based evidence; provider-generated validation claims cannot promote a model.

The current models and all assets remain research-only: no positive edge has been established. Passing checks does not enable real orders. Simulated replay is retained for measuring decision quality.

Development builds remain unsigned. Signing/publication continues to require explicit manual workflow dispatch; Apple notarization awaits its configured credentials.
