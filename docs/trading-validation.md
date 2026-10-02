# Validate decision quality

This project returns recommendations and measures them through simulated replay. Default decisions have `mode: decision_only`; simulations use `mode: paper`. `real_execution_enabled` is always false. No validation result changes that capability.

`trade-harness validate` reads the configured model's measured evidence. `GET /validation` exposes the same data; `?symbol=BTCUSDT` selects an asset. Each `/decisions` response includes `trading_validation` with the asset, timeframe, horizon, model version, measured statistics, gate results, and failure reasons. `--input path/to/decision-model.json` lets the validation command inspect another numerical artifact.

The default orchestrator has no independent walk-forward edge evidence and remains research-only. It does not inherit validation from the numerical anchor. Use `trade-harness validate --backend decision` to inspect the standalone numerical results summarized below, or `--backend hybrid` to inspect the hybrid artifact. `GET /validation` always reports the server's configured backend.

## What qualifies as evidence

The trainer uses chronological, shared cross-asset boundaries. Features and normalization use available history; labels crossing fit/calibration/validation boundaries are purged. Validation folds do not overlap. The final 20% cannot choose the model, thresholds, or validation status.

The legacy `walk-forward-edge-v1` measurements require all of:

- At least three purged walk-forward folds.
- Positive mean simulated account return after modeled fees and slippage.
- At least 20 closed simulated trades and positive returns in at least two folds.
- Brier-score improvement over the training-prior baseline.
- Mean net return above the momentum baseline under the same risk/cost policy.
- Positive mean return in a new replay with **double** the original fees and slippage.
- A positive lower bound for a descriptive 95% bootstrap interval over fold mean returns.
- Evidence that fit/calibration/validation boundaries are purged and the final test is excluded.

These historical measurements are insufficient for promotion. Version 0.8 adds mandatory `research-promotion-v2` forward evidence: a policy locked before new observations, five completed chronological windows, 100 closed trades, sufficient directional/calibration-bin observations, positive paired time-block intervals against cash and momentum, doubled-cost survival, and coverage/per-asset checks. Heuristic consensus confidence is not calibrated probability. See the [complete workflow and acceptance criteria](research-workflow.md).

The doubled-cost replay reruns sizing, costs, stops, and eligibility at the higher costs; it is not a subtraction from a headline return. Every asset has separate return, probability-quality, count, stress, and interval checks. A positive pooled average cannot certify an asset with negative returns. Historical evidence without accepted, matching forward promotion evidence remains research-only. Neither pipeline activates artifacts automatically.

Evidence comes from the configured backend's measured artifact through `validation_for(symbol)`. An LLM's generated JSON claiming validation is ignored. Unknown, missing, inconsistent, or out-of-contract evidence cannot promote a recommendation. A high probability score alone is insufficient.

## Current measured outcome

The selected boosted model's mean account return across validation folds is **−0.304%**, compared with **−0.571%** for momentum. Beating a losing benchmark does not establish a positive edge. The doubled-cost replay returns **−0.202%**. The descriptive mean-return interval is approximately **[−1.375%, +0.340%]**. It fails the positive net return, positive stressed return, and positive interval-lower-bound checks.

BTC has a positive mean under this model, but too few closed trades and an interval crossing zero. ETH and SOL also fail validation. Neither model nor any asset is promoted. Exact values are in [reports/trading-validation.json](../reports/trading-validation.json) and the full [walk-forward report](../reports/walk-forward.json).

## Limits and next research

The interval resamples whole non-overlapping folds, preserving each fold as a block. With only three folds and dependent market regimes, it is descriptive, not strong statistical proof. A passing result would still need fresh-period replication and monitoring.

These expanded checks are a retrospective audit of data already examined during earlier model work. Do not describe this dataset as an unseen confirmation of the new policy. Collect new completed periods, predeclare changes, train on earlier data, and evaluate later periods. The final test remains excluded from model promotion, even when its return is positive.

Results use independent funded accounts per asset and a fixed long/cash simulated policy. They do not measure a shared live portfolio, short trading, market impact, or actual fills. A validated policy can still issue losing recommendations. Existing model/risk choices, thresholds, and sampled universes must be tracked when comparing experiments.
