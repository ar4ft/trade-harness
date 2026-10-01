# Trade Harness

A decision-only CLI, HTTP API, and browser dashboard for trading research. Pass closed OHLC candles, volume, aligned indicators, symbol, timeframe, and decision history. Receive BUY / SELL / HOLD proposals, individual model scores, a return forecast, projected price, and a separate risk-controlled execution decision.

Includes **70,128 real hourly candles across BTCUSDT, ETHUSDT, and SOLUSDT**, trained numerical model weights, and a custom SmolLM2 language-model LoRA adapter. Jev/TypeSafe-style typed fields expose direction, execution, risk level, regime, and rule results with their sources.

The model has **not demonstrated a profitable walk-forward edge**. Default risk policy blocks new entries from research models. `--research` explicitly enables simulated entries. SELL closes a long position; this version supports long/cash paper trading, with no exchange order placement, leverage, or shorts.

## Start

Python 3.11+:

```bash
git clone https://github.com/ar4ft/trade-harness.git
cd trade-harness
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,orchestrator]'
trade-harness decide
uvicorn trade_harness.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** for the paper-trading dashboard, or `/docs` for API documentation. The dashboard shows account equity, exposure, model probabilities, risk reasons, and simulated fills. It is a browser interface, not a native desktop application.

`decide` uses the shipped historical August 2026 snapshot; it does not fetch live prices. The default `orchestrator` backend connects TimesFM forecasts, the hybrid numerical model, and the local LoRA reviewer. Base weights download on first use; no API key is required. The shipped reviewer supports BTCUSDT, 1h candles, a 3-candle horizon. Other contracts require a suitable reviewer. For lightweight numerical inference without model downloads, use `trade-harness decide --backend decision`.

## Shared-evidence consensus and parameter experiments

The default orchestrator connects the hybrid numerical model to language reviewers, sharing forecast, strategy, and engineered feature evidence. Disagreement or reviewer failure produces HOLD, with each vote visible in the API and dashboard.

```bash
pip install -e '.[orchestrator]'
trade-harness decide --backend orchestrator --reviewers local-llm
trade-harness tune-hybrid --model-output artifacts/hybrid/tuned-model.json
```

Consensus is not a calibrated probability or evidence of profitability. Feature/regularization experiments use chronological validation and never activate a model automatically. See [orchestration, configuration, tuning, and fine-tuning data](docs/orchestration.md).

## Combined forecast and strategy decisions

The optional `hybrid` backend feeds TimesFM forecast returns and uncertainty, observed market features, and trend/breakout/mean-reversion hypotheses into a trained numerical decision model. Each response exposes forecast evidence and strategy signals separately.

```bash
pip install -e '.[timesfm]'
trade-harness decide --backend hybrid
trade-harness validate --backend hybrid
trade-harness train-hybrid --data-dir data/markets
```

See [architecture, strategies, training, and comparison limits](docs/hybrid-decisions.md). The initial comparison averaged **−0.165%** validation account return for the combined model, with ten closed trades; it did not improve the baseline. See the [measured comparison](reports/hybrid-comparison.md). The combined model remains research-only and serves as the numerical anchor inside the default orchestrator.

## TimesFM 3.0 forecasts

Optional pretrained research forecasting with OHLC channels, past-only volume/indicators, and horizon quantile ranges:

```bash
pip install -e '.[timesfm]'
trade-harness decide --backend timesfm
```

Weights download separately on first inference. This backend remains research-only and does not establish a profitable edge. See [setup, input mapping, decision policy, and license limits](docs/timesfm.md).

## Live paper trading

```bash
# One tick, using public Binance market data and a persistent simulated account:
trade-harness paper --steps 1

# Continue polling until stopped:
trade-harness paper --steps 0 --poll-seconds 5

# Explicitly permit research-model entries in a separate simulated account:
trade-harness paper --research --run-id research-BTC --steps 0

trade-harness status --run-id research-BTC
```

Run the API in another terminal against the same `paper.sqlite` to see updates in the dashboard. Only completed candles create new model decisions. Fresh quotes still trigger protective exits between candle closes. Future/stale candles, stale quotes, clock skew, gaps, invalid model output, and duplicate ticks are checked. Public endpoints require no exchange credentials.

SQLite persists account state, pending orders, decisions, matured outcomes, equity marks, and fills. Account updates are transactional. Restarting the same run preserves its state; changing its model, market, or risk configuration requires a new run ID. Histories are isolated by run and market. Observed outcomes enter context only after they mature. Outcomes are not automatic retraining.

## Risk and replay

Default policy caps exposure at 20% of equity and budgets 0.5% per trade including estimated exit costs. It uses ATR-based stops, a 2:1 target, three-candle maximum holding period, three-candle reentry cooldown, 2% daily loss limit, 10% maximum drawdown, and a halt after three consecutive losses. Halts persist; use a new run after reviewing the cause. Confidence must be at least 0.55 and projected net edge at least 0.1% after costs/spread. Fees are 10 bps and slippage 5 bps per side.

```bash
trade-harness paper-replay --research --run-id replay-BTC \
  --input data/markets/BTCUSDT-2026-1h.json
```

Replay signals fill at the **next candle's open**, including fees and slippage. Missing next bars or excessive entry gaps cancel pending entries. If stop and target both touch within a candle, the stop wins; gap stops fill at the open when worse. Intrabar event times use the candle close because OHLC cannot reveal the exact touch time. Final pending orders are not filled outside the data boundary.

Supply a JSON `RiskConfig` with `--risk-config path.json`; omitted fields use defaults. `--research` affects simulated entries only. The old `backtest` command remains available as a simple forecast smoke test; use `paper-replay` and `evaluate` for account/risk/cost evaluation.

## Input and output

| Field | Format |
| --- | --- |
| `ohlc` | Chronological `[open, high, low, close]` rows, 21–10,000 candles |
| `volume` | Nonnegative array aligned with candles |
| `timestamps` | Strictly increasing closed-candle end times, Unix milliseconds |
| `indicators` | `{ "name": "rsi_14", "values": [...] }`; aligned, warmup values may be `null` |
| `symbol`, `timeframe` | Such as `BTCUSDT`, `1h` |
| `horizon` | Forecast length in candles; default 3 |
| `position` | `flat` or `long`; default `flat` |

See [the real example](src/trade_harness/assets/latest.json). Custom indicators must be causal; the harness cannot detect future information hidden in caller-supplied values. Standard SMA20, RSI14, MACD, and ATR14 are computed causally when needed.

```bash
trade-harness decide --input your-market.json
curl http://127.0.0.1:8000/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @src/trade_harness/assets/latest.json
```

Responses include `proposed_action`, final `action`, model probabilities/calibration, forecast/return interval where available, `execution` simulated sizing/stops/reasons, `guardrails`, model version, typed `fields`, and per-asset `trading_validation`. Default `mode` is `decision_only` and `real_execution_enabled` is always `false`. Numerical direction does not assert that an account should place a trade. A long account requires its actual snapshot for sizing/exits; `/decisions` alone cannot reconstruct it. Use the persisted paper runner or `POST /paper/tick` for account-aware decisions.

## Jev-style fields and Nimble

The harness exposes `choice`, `noul` (0–1), and ordinal `score` fields. Probability vectors are finite, normalized, and checked against their selected choice. Each field identifies its source. Model direction probabilities are separate from deterministic rule fields; a rule's one-hot result is not a calibrated market probability.

`POST /v1/systemone` accepts a TypeSafe-shaped request with market data in `state` and questions for `direction`, `execution`, `risk_allowed`, `data_valid`, `risk_level`, or `regime`. It supports these fixed contracts, not arbitrary natural-language classification. Candidates must match the documented enums; custom instructions are rejected. Inspect `/docs` for the schema.

[Bespoke Nimble](https://github.com/bespokelabsai/nimble) provides similar typed classifications and can be selected with `--backend nimble`. Its direction probabilities are paired with our independent numerical return forecast and still pass the risk policy. Configure `NIMBLE_BASE_URL` and optional credentials in [.env.example](.env.example). The public service returned 401 during verification; adapter request/response validation and failure-to-HOLD behavior were tested with mocks. Live authenticated integration remains unverified. Nimble's reported generic classification accuracy is not trading accuracy; its scores remain uncalibrated for our market task.

## Data, training, and measured results

[data/markets](data/markets) contains 2024, 2025, and January–August 2026 hourly archives for BTC, ETH, and SOL. Each annual file has archive URLs and SHA256 provenance alongside it. All downloads verified the Binance archive checksums. The older BTC-only dataset is retained for reproducing the language adapter.

```bash
trade-harness fetch --symbol ETHUSDT --timeframe 1h \
  --start 2024-01 --end 2024-12 --output data/markets/ETHUSDT-2024-1h.json
trade-harness evaluate --data-dir data/markets --output reports/walk-forward.json
# Equivalent shipped-model rebuild:
python scripts/train_decision_models.py
```

The expanded trainer uses 22 causal features, including indicators, returns, volatility, candle shape, volume changes, and calendar cycles. It compares logistic/Ridge and boosted classifier/regressor models through three purged chronological walk-forward folds. Temperature calibration uses a later, separately purged calibration partition. Shared timestamp boundaries keep all assets on the same side of splits. The final 20% is reserved for testing and cannot promote a model. Artifacts use portable JSON coefficients/trees, without executable pickle files.

Exact results: [reports/walk-forward.json](reports/walk-forward.json).

| Selected boosted model | Result |
| --- | --- |
| Labeled dataset / final test | 70,059 / 14,013 samples |
| Mean walk-forward account return | **−0.304%**, after costs |
| Walk-forward closed trades | 98 across independent asset accounts/folds |
| Final-test direction accuracy | 44.33% vs 30.16% training-prior baseline |
| Final-test Brier score | 0.6283 vs 0.6728 training-prior baseline |
| Final-test expected calibration error | 0.0119 |
| Final-test mean research account return | +0.675% vs −0.365% momentum |
| Deployment status | **research_only** |

Selection uses lowest mean walk-forward log loss. The `walk-forward-edge-v1` validation policy additionally requires at least three purged folds, positive mean net return, at least 20 closed trades, at least two positive folds, Brier improvement, an advantage over momentum, positive return in a doubled-cost replay, a positive fold-bootstrap interval lower bound, and exclusion of the final test. The pooled model and the requested asset must qualify. Neither candidate nor any individual asset currently passes all checks. The encouraging final-test result does not override that gate. Asset results use independent $10,000 accounts, not a shared portfolio. Costs are a fixed approximation; this is one later market period, not proof of future profit.

## Custom language model and distribution

Actual LoRA weights for `HuggingFaceTB/SmolLM2-135M-Instruct` are included. The adapter was trained for 120 CPU optimizer steps on 764 BTCUSDT hourly records; 480 examples were processed, about 0.63 epochs. It scores full BUY/SELL/HOLD token sequences using market features and previously available action/outcome context. Return projections use training-only class means. Scores are uncalibrated. Held-out accuracy was 44.8% versus a 45.8% majority baseline on 96 sampled records; this adapter is also research-only.

```bash
# Install the appropriate PyTorch build for your machine; CPU Linux example:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[llm-train]'
trade-harness decide --backend local-llm

trade-harness train-language --input data/BTCUSDT-1h.json \
  --output artifacts/my-trading-lora --steps 120
TRADING_LORA_PATH=artifacts/my-trading-lora trade-harness decide --backend local-llm
```

First Python inference downloads the pinned base weights (~270 MB). The [metadata](src/trade_harness/assets/trading_lora/trading_metadata.json) records the base revision, dataset hash, training versions, and evaluation. `scripts/rebuild_models.py --with-language` reproduces the older BTC Ridge/LoRA workflow; it does not rebuild the new multi-asset boosted model.

**Mozilla llamafile can distribute merged language weights and a local inference server in one executable.** The numerical model is already included in the harness's Python package. See [docs/distribution.md](docs/distribution.md) for merging the LoRA, exporting GGUF, building a checksum-pinned executable, and connecting via `--backend llamafile`. The harness still requires Python; llamafile packages the language runtime and weights, not the complete application or training pipeline. The two components can be shipped together. Quantization or changing inference engines requires reevaluation.

An alternative model can use `--backend llm` with an OpenAI-compatible JSON chat server (`LLM_BASE_URL`, `LLM_MODEL`, optional `LLM_API_KEY`). This backend receives raw market arrays and historical decisions. The small shipped adapter needs its dedicated scoring backend rather than generic JSON chat. `baseline` and older `trained` backends are also retained.

## API and verification

Endpoints include `/health`, `/validation`, `/decisions`, `/feedback`, `/backtest`, `/paper/runs`, `/paper/{run_id}/events`, `/paper/tick`, and `/v1/systemone`. Set `TRADING_API_KEY` to require Bearer authentication for data/action endpoints. Dashboard credentials are kept in browser memory. Bind locally by default; terminate HTTPS and manage access separately for remote deployment. Environment variables are listed in `.env.example`; it is not loaded automatically.

```bash
ruff check .
pytest -q
```

Tests cover temporal boundaries, portable model prediction parity, typed probability validation, risk sizing, stale feeds, protective exits despite model failure, next-open fills, conservative stops, account isolation, atomic persistence, replay/tick deduplication, API contracts, and optional-provider failure handling. Training and data download are explicit commands. No automatic live trading or online training process starts when installing the package.

## Signed releases and automatic updates

Automatic builds produce unsigned development artifacts; signing and publication require a manual Actions run with `sign_release` enabled. Stable release manifests are signed with Sigstore using this repository's tagged GitHub Actions identity. `trade-harness update --check` verifies availability; `trade-harness update --apply` stages a verified upgrade. Launch with `trade-harness-managed -- decide`, `-- paper --steps 0`, or `-- serve` to check automatically at startup, at most once per day. Running processes keep their current version until restarted.

A macOS Developer ID Installer signing/notarization pipeline is prepared; it requires Apple Developer credentials before notarized installers can be released. See [release and update documentation](docs/releases-and-updates.md) for trust checks, deployment behavior, credential names, and activation. The portable signed wheel and prepared Apple notarization pipeline are distinct release capabilities.

## Trading validation for recommendations

```bash
trade-harness validate --output reports/trading-validation.json
curl http://127.0.0.1:8000/validation?symbol=BTCUSDT
```

Validation exposes measured returns, the doubled-fee/slippage replay, momentum comparison, sample counts, a descriptive 95% interval, each gate, and failure reasons. Each recommendation carries evidence bound to its asset, timeframe, horizon, and model version. Provider-generated claims of successful validation are not accepted as evidence.

The selected model's mean walk-forward return remains **−0.304%**, and its doubled-cost replay returns **−0.202%**. Its descriptive interval crosses zero. It remains `research_only`. These expanded checks audit an already examined dataset; establish future evidence using fresh periods, without choosing thresholds from the final test. Validation measures a fixed simulated policy and does not guarantee individual decisions or future profitability. Passing never enables real orders. See [docs/trading-validation.md](docs/trading-validation.md).
