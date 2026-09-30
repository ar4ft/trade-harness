# Trade Harness

A custom trading decision harness and model-training toolkit. Pass OHLC arrays, volume, named indicator arrays, timeframe, symbol, and current position. Receive **BUY / SELL / HOLD**, a forecast return, projected price, rationale, and guardrail results. Designed for research and paper trading.

Includes a transparent momentum baseline, a trainable local numerical model, and a custom trading language-model interface with a LoRA training workflow. It is an independent implementation; no compatibility with projects named Jev, Laya, or OpenJev is claimed.

## Quick start

Python 3.11+:

```bash
git clone https://github.com/ar4ft/trade-harness.git
cd trade-harness
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
trade-harness decide --input examples/market.json
uvicorn trade_harness.api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation: http://127.0.0.1:8000/docs

```bash
curl http://127.0.0.1:8000/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/market.json
```

The checked-in sample contains **synthetic** hourly prices for repeatable demonstrations. They are not market observations or evidence of profitability.

## Input contract

| Field | Format |
| --- | --- |
| `ohlc` | Chronological array of `[open, high, low, close]` rows, 21–10,000 candles |
| `volume` | Nonnegative array, same length as OHLC |
| `timestamps` | Strictly increasing closed-candle end times, Unix milliseconds |
| `indicators` | Array of `{ "name": "rsi_14", "values": [...] }`, each aligned with OHLC; warmup values may be `null` |
| `timeframe` | Such as `1m`, `15m`, `1h`, `1d`, `1w` |
| `symbol` | Such as `BTCUSDT` |
| `position` | `flat` or `long` |
| `horizon` | Forecast length in candles, 1–100; default 3 |

See [examples/market.json](examples/market.json) for a complete valid request. Supply only closed candles and compute indicators using past data. The API cannot determine whether a caller's indicators contain future information. Use regular candle intervals; missing periods require preprocessing, particularly for outcome timestamps.

Response fields:

```json
{
  "action": "BUY",
  "confidence": 0.72,
  "rationale": "Example research signal",
  "forecast": {
    "horizon": 3,
    "expected_return": 0.02,
    "projected_close": 102.0,
    "method": "your-model"
  },
  "id": "generated-uuid",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "as_of": 1700000000000,
  "backend": "your-model",
  "guardrails": []
}
```

Confidence is a heuristic score, not a calibrated success probability. SELL closes an existing long; this version does not model short positions, leverage, or pyramiding.

## Model choices

| Backend | Behavior |
| --- | --- |
| `baseline` (default) | Deterministic five-candle momentum forecast; convenient for testing |
| `trained` | Ridge regression fitted to OHLCV features, portable JSON weights |
| `llm` | Custom prompt, all supplied indicators, prior decisions and available outcomes, strict validated JSON |

The numerical backends use six fixed OHLCV features and do not consume supplied indicators or decision memory. The language-model backend consumes both. Decision memory provides context; it does not automatically update model weights.

### Train your local forecasting model

Replace the example with your historical dataset, in the same input format:

```bash
trade-harness train --input examples/market.json --output artifacts/forecast.json
TRADING_BACKEND=trained trade-harness decide --input examples/market.json
TRADING_BACKEND=trained trade-harness backtest --input examples/market.json
```

Training uses the first 80% chronologically, purges training labels crossing the split, fits normalization on training only, and evaluates on the remaining labeled candles. Metrics include MAE, zero-return baseline MAE, and direction accuracy. Saved models reject predictions that overlap training or change the symbol, timeframe, or horizon. Trained-model backtests start after the training boundary. The initial implementation has one chronological holdout; production research needs additional walk-forward periods and an untouched final test set.

### Connect a language model

```bash
export TRADING_BACKEND=llm
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=your-model-or-fine-tuned-model-id
export LLM_API_KEY=your-key
uvicorn trade_harness.api:app --host 127.0.0.1 --port 8000
```

Use an OpenAI-compatible `/chat/completions` provider or local server that supports JSON response mode. `.env.example` lists configuration variables; the application does not load `.env` automatically. An external provider receives the supplied market data and history. Provider failures and malformed or inconsistent forecasts produce HOLD.

### Train a custom language-model adapter

1. Collect decisions and record realized outcomes plus a human-reviewed action with `POST /feedback`.
2. Export reviewed chat examples:

```bash
trade-harness export-finetuning --output artifacts/trading-chat.jsonl
```

3. Review the exported assistant forecasts, rationales, and confidence scores. The export replaces the action with the reviewed action but retains the model's original rationale and forecast. Correct inconsistent answers before training. Input messages exclude future outcomes; labels may use hindsight. The current exporter uses empty history, so history-aware training examples need to be curated separately.
4. Train a LoRA adapter on an instruction language model:

```bash
pip install -e '.[llm-train]'
python scripts/train_lora.py \
  --data artifacts/trading-chat.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output artifacts/trading-lora
```

This trains actual adapter weights with assistant-only loss masking. It downloads the base model and requires sufficient memory; GPU training is recommended. The script rejects overlong examples instead of silently removing their targets. Use a bounded candle window for training data and keep later time periods out of the training set. Check the selected base model's license before distribution.

5. Serve the base model plus adapter with a compatible inference server and point `LLM_BASE_URL` and `LLM_MODEL` to it. Evaluate on later market periods before paper trading. This repository does not include pretrained trading language-model weights, and the optional GPU training run has not been executed as part of the starter-project validation.

## Decision memory and feedback

SQLite stores requests, final decisions, and feedback. Only earlier decisions from the same symbol and timeframe are exposed to the language model; an outcome is included only when its observation time precedes the current decision time.

```json
{
  "decision_id": "uuid-from-response",
  "realized_return": 0.012,
  "reviewed_action": "BUY",
  "notes": "Reviewed after the horizon closed",
  "observed_at": 1700010800000
}
```

`observed_at` must follow the decision timestamp plus the forecast horizon duration. `realized_return` means the market close-to-close return over that horizon, not account P&L. Caller-provided outcomes are trusted; the service does not fetch market prices to verify them. Use separate databases for independent strategies/accounts.

## Backtesting

```bash
trade-harness backtest --input examples/market.json
```

Signals use candle prefixes; fills occur at the **next candle open**, with defaults of 10 bps fee and 5 bps slippage per fill. Supports one full-capital long position or cash. Reports equity, total return, maximum drawdown, trades, and open units. Final holdings are marked to market without forced liquidation. Decision history is isolated from live SQLite data, and matured simulated outcomes are added progressively. Python callers can customize costs, cash, and candle-window length through `backtest()`.

Backtests exclude funding, market impact, spreads beyond modeled slippage, partial fills, exchange constraints, and intra-candle execution. LLM calls can make backtests slow and incur provider charges. Historical replay cannot remove knowledge already present in a pretrained language model; evaluate this risk separately.

## Extending the harness

Implement `predict(market, history) -> Proposal` and a `name` property, then pass your model to `Harness(model, Store(...))`. Edit `models.py` to register a backend. Configure confidence and volatility thresholds through the Harness constructor. Position and forecast-consistency checks run after every backend prediction.

```text
Input arrays -> Validation -> Prior decision/outcome context -> Model
            -> Forecast/position/confidence/volatility checks -> Stored decision
Recorded outcomes -> Reviewed training examples -> Custom model adapter
Historical candles -> Chronological training / next-open backtesting
```

## Development

```bash
ruff check .
ruff format --check .
pytest -q
```

GitHub Actions runs lint and tests. Tests cover malformed OHLCV, provider failures, model output validation, future-outcome exclusion, next-open execution and costs, training leakage, API behavior, and export structure. Language-model HTTP calls are mocked; live provider behavior and GPU training require configured infrastructure.

No exchange execution is included. The HTTP API has no authentication and is intended to run locally; add authentication and access controls before exposing it. Forecasts and signals are research outputs and do not establish profitable trading performance.
