# Trade Harness

A working **CLI and HTTP API** for trading decisions from OHLCV candles, indicator arrays, timeframe, and prior decisions. Returns BUY / SELL / HOLD, a return forecast, projected close, rationale, and guardrail results. There is no desktop interface.

This repository includes real BTC/USDT hourly market data, a trained numerical forecast model, and actual trained weights for a small custom language-model adapter. Both models were evaluated chronologically. The current results do **not** establish a profitable trading edge; keep this version in research/paper trading.

## Run immediately

Python 3.11+:

```bash
git clone https://github.com/ar4ft/trade-harness.git
cd trade-harness
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
trade-harness decide
```

This runs the shipped numerical model on the included real-data snapshot. It needs no API key or model download. The snapshot ends August 31, 2026; it is historical, not a live market feed. A HOLD result is an intentional decision when confidence is low.

Use your own closed candles:

```bash
trade-harness decide --input your-market.json --position flat
trade-harness decide --input your-market.json --position long
trade-harness backtest --input data/BTCUSDT-1h.json
```

The shipped models are specific to **BTCUSDT, 1h candles, a 3-candle forecast horizon**. Train new weights for another asset, timeframe, or horizon. The `baseline` and remote `llm` backends are not restricted to that training contract.

## Run the trained custom language model

The included LoRA adapter was trained from `HuggingFaceTB/SmolLM2-135M-Instruct`: 120 optimizer steps, seed 42, 764 records in the training partition, assistant-only loss masking. The run processed 480 samples (0.63 epochs). It is actual trained model weights, not just a system prompt. Training ran on CPU in about six minutes; inference was checked separately.

For a CPU-only Linux installation:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[llm-train]'
trade-harness decide --backend local-llm
```

For other platforms, install the appropriate PyTorch build, then the optional dependencies. First use downloads the pinned upstream base-model weights, approximately 270 MB, from Hugging Face. Later runs reuse the cache. No provider API key is required. The adapter and tokenizer are included in the repository and Python package.

This small language model consumes six OHLCV features, SMA20, RSI14, MACD, ATR14, and the latest prior decision/available realized outcome. Standard indicators are computed causally when missing. It scores the likelihood of BUY, SELL, and HOLD instead of relying on free-form JSON generation. Its return projection is the weighted mean of training-set returns for those three classes. Confidence scores are uncalibrated.

## HTTP API

```bash
uvicorn trade_harness.api:app --host 127.0.0.1 --port 8000
```

Or use the local language model:

```bash
TRADING_BACKEND=local-llm uvicorn trade_harness.api:app --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

```bash
curl http://127.0.0.1:8000/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @src/trade_harness/assets/latest.json
```

Endpoints: `GET /health`, `POST /decisions`, `POST /feedback`, and `POST /backtest`. The API is intended to run locally; add authentication before exposing it.

## Input and output

| Field | Format |
| --- | --- |
| `ohlc` | Chronological array of `[open, high, low, close]` rows, 21–10,000 candles |
| `volume` | Nonnegative array aligned with OHLC |
| `timestamps` | Strictly increasing closed-candle end times, Unix milliseconds |
| `indicators` | Array of `{ "name": "rsi_14", "values": [...] }`, aligned with candles; warmup values may be `null` |
| `timeframe` | Such as `15m`, `1h`, `1d` |
| `symbol` | Such as `BTCUSDT` |
| `position` | `flat` or `long`; default `flat` |
| `horizon` | Forecast length in candles, 1–100; default 3 |

A complete real example is in [src/trade_harness/assets/latest.json](src/trade_harness/assets/latest.json). The older `examples/market.json` fixture is synthetic and used only for tests. Do not pass unfinished candles. Compute custom indicators using historical data only; the API cannot identify future information embedded by the caller.

Output includes `action`, `confidence`, `rationale`, `forecast.horizon`, `forecast.expected_return`, `forecast.projected_close`, `id`, `as_of`, `backend`, and `guardrails`. SELL exits a long position. Short selling, leverage, exchange order placement, and pyramiding are not implemented.

## Real data and reproducibility

Included dataset: **8,760 hourly BTC/USDT candles, September 2025–August 2026**. Downloaded from Binance public monthly spot kline archives. Every archive was checked against its SHA256 checksum; timestamps in microseconds were normalized to milliseconds. Duplicate or missing candle intervals are rejected. SMA20, RSI14, MACD, and ATR14 are computed using only historical prefixes.

Source URLs and checksums: [data/BTCUSDT-1h.provenance.json](data/BTCUSDT-1h.provenance.json).

Download again, or choose another completed month range:

```bash
trade-harness fetch --symbol BTCUSDT --timeframe 1h \
  --start 2025-09 --end 2026-08 --output data/BTCUSDT-1h.json
```

The downloader supports Binance archive intervals from 1 minute through 1 day, subject to the 10,000-candle input limit. Use fewer months for smaller intervals. It filters unfinished candles and requires monthly archives to exist. Future data collection is an explicit CLI command; no background trading process starts automatically.

Reproduce the shipped numerical weights and evaluation:

```bash
python scripts/rebuild_models.py
```

Reproduce both models and evaluation:

```bash
HF_HUB_DISABLE_XET=1 python scripts/rebuild_models.py --with-language --steps 120
```

Floating-point results can differ across PyTorch versions and hardware. The LoRA base revision and training parameters are saved with the weights.

## Train your own models

Numerical model:

```bash
trade-harness train --input data/BTCUSDT-1h.json \
  --with-indicators --output artifacts/forecast.json
TRADING_MODEL_PATH=artifacts/forecast.json trade-harness decide
```

Training uses an 80/20 chronological split within the supplied file, purges labels crossing the split, and fits normalization on training data only. `--with-indicators` adds four standard indicator features to six OHLCV features. Missing standard indicators are computed during inference. Shipped numerical weights were fitted within the first 85% of the dataset, leaving the final 15% outside both training and validation.

Custom local language model:

```bash
trade-harness train-language --input data/BTCUSDT-1h.json \
  --output artifacts/my-trading-lora --steps 120
TRADING_LORA_PATH=artifacts/my-trading-lora trade-harness decide --backend local-llm
```

Language training uses 70% train, 15% validation, 15% test, with boundary labels purged. Direction labels are derived automatically from subsequent horizon returns: BUY above +0.3%, SELL below -0.3%, otherwise HOLD. Future outcomes appear only in target labels. Historical context uses an earlier simulated momentum decision and an outcome that had already matured. This is not the same as training on human-reviewed decisions. The default training stride is 8 candles; validation/test metrics use 96 uniformly spaced records per partition.

The training script saves adapter weights, tokenizer, pinned base revision, class-return means, optimizer-step count, and measured evaluation results. It does not perform automatic online retraining when feedback is recorded.

## Measured performance

See [reports/real-data-evaluation.json](reports/real-data-evaluation.json) for exact held-out metrics and [the adapter metadata](src/trade_harness/assets/trading_lora/trading_metadata.json) for training/evaluation details.

The numerical model's final test MAE was approximately **0.410%**, compared with **0.406%** for a zero-return forecast. Its sign accuracy was **44.2%**. The conservative harness made no trades in that test period and remained in cash. This is a functioning research pipeline, not evidence of an effective trading strategy.

The small language model's validation direction accuracy matched the training-majority baseline. Its final test direction accuracy was **44.8%**, versus **45.8%** for the training-majority baseline, on 96 uniformly sampled later records. Do not confuse sign accuracy for numerical return forecasts with three-class accuracy for language-model direction labels.

The local language model also remained in cash in a separate backtest of the final 168 test candles using actual harness history. No thresholds were loosened to force trades after seeing the results. Further research needs multiple market regimes, additional assets, walk-forward evaluation, calibrated confidence, and an untouched final test period.

## Other backends and customization

```bash
trade-harness decide --backend baseline --input examples/market.json
```

Use any OpenAI-compatible chat-completions server, including a different custom language model:

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=your-model-id
export LLM_API_KEY=your-key
trade-harness decide --backend llm
```

The remote LLM backend receives the supplied raw OHLCV, all named indicator arrays, current position, prior decisions, and available outcomes. `.env.example` lists environment variables; the application does not automatically load `.env`. A provider receives the submitted market data and decision context.

To train another instruction model on manually reviewed decisions, record outcomes using `/feedback`, export with `trade-harness export-finetuning --output artifacts/trading-chat.jsonl`, review the retained forecast/rationale/confidence fields, and use `scripts/train_lora.py`. That general script is separate from the executed market-label training workflow.

Implement `predict(market, history) -> Proposal` and a `name` property to register another backend. The harness checks position, forecast consistency, confidence, and volatility after prediction. Model errors or invalid output produce HOLD. Shipped model inputs outside their asset/timeframe/horizon/training-time contract also fail closed.

## Memory and backtesting

SQLite stores requests, decisions, and feedback. Only prior decisions from the same symbol/timeframe are exposed. A realized outcome enters model context only after its observation timestamp precedes the current decision. Use separate databases for separate accounts or strategies.

`POST /feedback` accepts:

```json
{
  "decision_id": "uuid-from-response",
  "realized_return": 0.012,
  "reviewed_action": "BUY",
  "notes": "Reviewed after horizon close",
  "observed_at": 1788231599999
}
```

`realized_return` is the market close-to-close return over the forecast horizon, not account P&L. Outcome time must follow the forecast horizon; caller-provided results are trusted rather than verified against an exchange.

Backtests signal at candle close and fill at the next open, with 10 bps fee and 5 bps slippage per fill by default. They support long/cash, keep simulated history isolated from live data, add matured outcomes progressively, and report equity, return, maximum drawdown, fills, and open units. Final holdings are marked to market without forced liquidation. `backtest(..., start_at=timestamp)` isolates a later test period while retaining earlier candle context.

They exclude funding, full spread dynamics, market impact, partial fills, and exchange constraints. Remote language-model backtests can incur provider charges. Pretrained language models may already know historical market events; this cannot be eliminated by chronological fine-tuning splits alone.

## Development

```bash
ruff check .
ruff format --check .
pytest -q
```

GitHub CI runs lint and tests. Unit tests mock remote providers; real-data inference and local model training were also executed during project validation. Data/model attribution is in [THIRD_PARTY.md](THIRD_PARTY.md). Original project code uses the MIT license.
