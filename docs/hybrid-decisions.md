# Forecasts → strategies → trained decisions

The `hybrid` backend combines three sources of evidence:

1. Twenty-two observed market features from closed OHLCV and causal indicators.
2. Eight features from versioned trend, breakout, mean-reversion, and no-trade hypotheses.
3. Five TimesFM 3.0 forecast features: horizon return, lower/upper quantiles, interval width, and forecast versus recent momentum.

A numerical classifier chooses BUY / SELL / HOLD; a separate fitted regressor predicts the next-open-to-horizon-close return. Its class scores are temperature-calibrated on a separate chronological calibration segment. Deterministic position/risk checks decide the final allowed action. No real exchange execution exists.

TimesFM remains frozen in this experiment. The trained decision layer consumes its numerical forecasts rather than its heuristic BUY/SELL policy. Forecast evidence and strategy signals remain separately visible in each decision, so forecasts are never represented as observed candles or calibrated action probabilities.

See the [application architecture](architecture.md) for the surrounding consensus, risk, persistence and deployment layers.

## Run

```bash
pip install -e '.[timesfm]'
trade-harness decide --backend hybrid
TRADING_BACKEND=hybrid trade-harness serve
trade-harness validate --backend hybrid
# Inspect the fixed strategies independently:
trade-harness decide --backend strategy
```

The shipped combined artifact is for BTCUSDT, ETHUSDT, SOLUSDT; 1h candles; a 3-candle horizon. Its TimesFM context is fixed to the training contract (100 candles for this experiment), overriding the standalone backend's default 512. Checkpoint/version and strategy-policy mismatches fail closed. Forecast failure produces HOLD/model_failure; there is no silent market-only substitution. Managed updates select the `timesfm` dependencies for `TRADING_BACKEND=hybrid`.

`forecast` describes the fitted decision regressor. `forecast_evidence` describes TimesFM separately, including its model version, input timestamp, horizon, expected return, and uncalibrated 10th–90th quantile return range. `strategy_signals` contains four named hypotheses with direction, heuristic strength, expected return, invalidation price, and holding horizon. These invalidation prices are evidence only; existing deterministic risk policy owns the actual simulated stop/target.

## Strategy hypotheses

- **Trend:** BUY/SELL when 20-candle momentum exceeds ±1%.
- **Breakout:** close above/below the preceding 20 candles' highs/lows, excluding the current candle.
- **Mean reversion:** deviation from the 20-candle mean beyond ±1.5 standard deviations with RSI below 40/above 60.
- **No trade:** directional conflict, no active hypothesis, or realized one-candle volatility above 4%.

The independent `strategy` backend abstains on conflicts. The trained hybrid sees these signals as features and can choose a different action; its final risk checks still apply. These are starting hypotheses, not individually validated profitable strategies. Strategy thresholds were fixed for the experiment rather than searched for the best historical return.

## Reproduce training and comparison

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 trade-harness train-hybrid \
  --data-dir data/markets \
  --forecast-context 100 --forecast-stride 48 \
  --forecast-cache artifacts/hybrid/forecasts.jsonl \
  --output reports/hybrid-comparison.json \
  --model-output artifacts/hybrid/model.json

TRADING_HYBRID_MODEL=artifacts/hybrid/model.json trade-harness decide --backend hybrid
```

First run downloads TimesFM weights and generates historical forecasts. Inference is batched by compatible input shapes; warm-up indicator gaps never backfill from the future. The JSONL cache is resumable and records source-file hashes, checkpoint/context/stride contract, and a digest of each historical input. A changed source/contract needs a new cache file. A malformed or mismatched cache fails closed. The report and model include the completed cache SHA256 for provenance; cache forecasts are not independently signed or proven correct by that hash.

Targets enter the dataset only after forecasts are generated from historical prefixes. Labels use next candle's open to horizon close, with BUY/SELL thresholds of ±0.30% (10 bps fees + 5 bps slippage per side). Training and calibration labels must mature before the next segment begins.

Five logistic/Ridge ablations isolate added evidence: market-only, strategy-only, market+strategy, forecast-only, combined. A boosted market-only reference reuses the current model architecture on the identical sampled cohort. Fixed momentum, strategy, and TimesFM-rule policies are also replayed. Selection among the five ablations uses validation log loss only; the boosted reference is reported separately. The combined model is shipped for experimentation regardless of which ablation wins, and never replaces the default backend automatically.

Three purged chronological validation folds occupy the development period; the final 20% is excluded from selection. Each trained candidate is replayed with normal and doubled fees/slippage. Paper simulations process every intervening hourly bar while an order is pending or a position is open, preserving fills, stops, horizon exits, and account risk checks. Idle cash bars outside the decision schedule are skipped; daily equity statistics restore unchanged cash days. Replay parity is tested against processing every bar. Model entry decisions occur only at the identical sampled timestamps across all candidates. Cash and capped buy-and-hold comparisons are included by the replay engine.

## Initial measured result

The first run generated 1,455 real TimesFM forecasts. The combined model averaged −0.165% net account return across validation folds/assets, with ten closed trades, and failed the positive-edge gates. Forecast-only had the lowest validation log loss among the logistic ablations; the boosted market-only reference scored better still. Combining everything did not improve the measured result. See the [comparison summary](../reports/hybrid-comparison.md) and [full report](../reports/hybrid-comparison.json).

## Interpretation and next experiments

The initial CPU experiment samples one decision every 48 candles from the existing 70,128-candle, three-asset history. Sparse sampling keeps research reproducible on local hardware. It does **not** establish performance for hourly decisions or for other horizons/assets. The available history has already been examined, and foundation-model pretraining overlap is unknown. This is a retrospective comparison, not fresh confirmation.

All hybrid decisions remain `research_only`, even if measured retrospective gates pass. TimesFM 3.0's non-commercial/non-production restrictions apply to its outputs and this research pipeline. Its weights are separately downloaded, never bundled with signed releases. Fine-tuning or using an LLM to interpret forecasts does not remove those restrictions.

Next, use new prospective periods with predeclared settings and denser sampling, compare Kronos/Chronos-2 under the same protocol, and only then consider forecaster fine-tuning. Reviewed LLM strategy explanations and matured feedback can be added later; they are not ground truth or fitted history features in this first combined numerical model. The [default orchestrator](orchestration.md) now connects the numerical anchor to language/typed reviewers, including [Ollaya](ollaya.md). These reviews are not fitted numerical history features or established evidence of improved returns.
