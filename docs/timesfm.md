# TimesFM 3.0 research backend

TimesFM is a numerical time-series forecasting model, not a language model. The optional `timesfm` backend uses Google's pretrained 3.0 checkpoint to produce a BUY / SELL / HOLD proposal and a horizon close-price forecast. It is available through the existing CLI, HTTP API, and paper monitor. The default orchestrator uses TimesFM evidence through the hybrid numerical model and adds language review.

## Run

```bash
pip install -e '.[timesfm]'
trade-harness decide --backend timesfm
# Your own closed, regularly spaced MarketInput JSON:
trade-harness decide --backend timesfm --input snapshot.json
# Start the API and browser monitor with this backend:
TRADING_BACKEND=timesfm trade-harness serve
```

First inference downloads weights from `google/timesfm-3.0-pytorch` at pinned revision `43046b85ec22d584a13f8098c2ed39c889e129c2` into the Hugging Face cache. The harness wheel, repository, and release assets do not contain these weights. The optional package is pinned to `timesfm[torch]==3.0.2`. CPU is the default; set `TRADING_TIMESFM_DEVICE=cuda` for a compatible GPU and PyTorch installation. Allow several GB of RAM for loading and inference. Initial download and CPU inference can take time.

`TRADING_TIMESFM_CONTEXT` defaults to the most recent 512 candles, with a supported range of 21–10000. Set `TRADING_TIMESFM_OFFLINE=1` to require already-cached weights. Initialization is lazy; concurrent requests serialize model loading and inference. Missing dependencies, uncached offline weights, irregular timestamps, and invalid forecasts become the harness's existing HOLD/model_failure response.

## Inputs and decision rule

OHLC becomes four jointly forecast price channels; volume and aligned indicators become past-only covariates. Indicators containing missing values within the selected context are omitted, with no future backfilling. Caller-supplied indicators must themselves be causal. Timeframe defines the required regular timestamp spacing; forecast horizon remains measured in candles. Targets and covariates are normalized by TimesFM using only the supplied historical context.

The final horizon close-channel point forecast supplies projected_close and expected_return. `forecast_interval` contains close-return forecasts at the 10th and 90th quantiles, not prices. A BUY proposal requires the lower forecast return to exceed +0.30%; SELL requires the upper return below −0.30%; otherwise HOLD. This is an initial research policy, not a fitted strategy. Forecast OHLC channels are not constrained to form valid future candles and are not used as simulated fill prices.

Confidence is a fixed policy heuristic (0.80 for directional proposals, 0.65 for HOLD). Quantile forecasts are not calibrated action probabilities or validated financial confidence intervals. This backend ignores decision history; it does not train online or combine its outputs with the separate language/numerical decision models. Future work can train a decision model using causally generated forecast features after separate evaluation.

## Evaluation and use limits

All proposals are `research_only`, with no positive walk-forward edge certification. Existing position/risk checks still apply; real execution is disabled. A forecast smoke check establishes that inference works, not that a strategy makes money. Foundation-model pretraining cutoffs are not proven by pinning a checkpoint: historical tests may overlap unknown pretraining data. Fresh prospective evaluation is required before claiming forecasting or trading improvements.

The downloaded weights use the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE). This integration is for non-commercial, non-production research on historical data or simulation. That license restricts revenue-generating and production use, redistribution of weights/derivatives, and commercial use of outputs. Fine-tuning does not remove those restrictions. Harness source remains under its own license; users download TimesFM separately. Use an appropriately licensed model or authorized Google Cloud service for other uses.

Managed updates automatically install the `timesfm` optional dependencies when `TRADING_BACKEND=timesfm` is set, or explicitly with `TRADING_UPDATE_EXTRAS=timesfm`. Cached weights remain separate from signed harness releases. Signing still runs only through the manual release workflow.
