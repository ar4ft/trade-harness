# Data and model provenance

The real market dataset comes from Binance's public spot monthly kline archives at
https://data.binance.vision/. The exact source URLs, archive SHA256 checksums, and
normalized dataset checksum are recorded in `data/BTCUSDT-1h.provenance.json`.
The repository's MIT license covers its original code; it does not grant rights
beyond the source provider's terms for market data.

The local trading language model is a LoRA adapter trained from
`HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The upstream model declares the
Apache 2.0 license: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct.
Base-model weights are downloaded from the pinned revision on first use and are
not redistributed in this repository. Adapter training provenance, label method,
class-return means, training steps, and held-out metrics are recorded in
`src/trade_harness/assets/trading_lora/trading_metadata.json`.

Language-model targets are generated from subsequent three-candle market returns,
using a 0.3% threshold for directional labels. These are supervised market-outcome
labels, not human-reviewed recommendations. They do not account for trade execution
in the labeling step. Simulated account performance is measured separately.
