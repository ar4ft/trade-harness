# Distribute the harness and local language model

The recommended package has two components: the Python harness wheel (numerical weights, dashboard, risk/paper engine, tokenizer/adapter metadata) and a Mozilla llamafile containing merged language weights plus the local inference server. Llamafile does not package the Python application or train the model. The default orchestrator uses the Python local reviewer and works without llamafile; install the `orchestrator` dependency group.

## Build from the shipped adapter

Install `.[llm-train]` using the appropriate PyTorch build. These commands download the pinned base model if it is not cached, merge the LoRA, and export F16 GGUF. No trading occurs.

```bash
python scripts/export_language_weights.py
```

Obtain llama.cpp at the revision used by Mozilla llamafile 0.10.6, then install its GGUF tooling and the converter's tokenizer dependencies:

```bash
git clone https://github.com/ggml-org/llama.cpp.git artifacts/llama.cpp
git -C artifacts/llama.cpp checkout 7f5ee549683d600ad41db6a295a232cdd2d8eb9f
pip install -e artifacts/llama.cpp/gguf-py
pip install sentencepiece
python artifacts/llama.cpp/convert_hf_to_gguf.py \
  artifacts/merged-trading-language \
  --outfile artifacts/trading-language-f16.gguf --outtype f16
python scripts/build_llamafile.py --gguf artifacts/trading-language-f16.gguf \
  --model-notice docs/licenses/trading-model-NOTICE.txt
```

The builder downloads Mozilla's 0.10.6 thin executable and zipalign, verifies pinned SHA256 checksums, and combines the weights and default arguments into `dist/trading-language.llamafile`. It writes a manifest recording runtime version, upstream digests, input/output hashes, and size. Existing outputs are never overwritten. Conversion source, intermediate weights, and built executables are ignored by Git.

If the build reports an executable format error, supply an APE loader installed following [Mozilla's platform instructions](https://github.com/mozilla-ai/llamafile). Pass its path as `--loader /path/to/ape`. On such hosts, also launch the bundled model as `/path/to/ape dist/trading-language.llamafile`. No system loader is installed automatically.

## Run

```bash
chmod +x dist/trading-language.llamafile
./dist/trading-language.llamafile
```

Defaults bind the inference server to `127.0.0.1:8080`, with a 4,096-token context and four CPU threads. In another terminal:

```bash
trade-harness decide --backend llamafile
# Or serve the harness API/dashboard with this backend:
TRADING_BACKEND=llamafile uvicorn trade_harness.api:app --host 127.0.0.1 --port 8000
```

`LLAMAFILE_BASE_URL` overrides the server URL. `LLAMAFILE_API_KEY` optionally sends Bearer authentication. For a different adapter, point `TRADING_LORA_PATH` to its matching tokenizer/metadata; the server must run its merged weights. The current dedicated scoring backend uses the shipped SmolLM chat template. An arbitrary GGUF requires its own compatible backend/template.

The backend scores **the complete multi-token BUY/SELL/HOLD labels** using teacher-forced token likelihoods from `/tokenize` and `/completion`, then normalizes their sequence scores. It does not treat a sampled word as confidence. Missing candidate likelihoods, truncated prompts, unavailable servers, and invalid responses fail to HOLD. This can require several HTTP requests per decision and is not Nimble's single-token optimization. Scores remain uncalibrated and research-only. Numerical return projections use the same training-class means as the Python adapter.

Generic instruction GGUFs can instead use the `llm` backend with `LLM_BASE_URL=http://127.0.0.1:8080/v1` and an appropriate `LLM_MODEL`. They must generate the harness's JSON proposal schema. The small shipped adapter was trained for direction labels, so use its dedicated `llamafile` backend.

## Share a build

Build the wheel separately with `pip wheel --no-deps . -w dist`. Ship the wheel, `.llamafile`, manifest, instructions, and applicable licenses/notices together. Install the wheel on the target machine, start llamafile, then run the harness. The wheel's runtime dependencies still need installation or a platform-specific offline wheel set. This is not yet a single-file desktop application.

Mozilla supports multiple operating systems and x86/ARM CPUs, but these project build/inference checks run on Linux x86-64. Other platforms require their own checks. Windows requires a bundled executable below 4 GB; rename it with an `.exe` suffix. Larger GGUFs can remain separate from the executable. Some Linux/macOS configurations need Mozilla's documented executable-loader setup; see [upstream installation instructions](https://github.com/mozilla-ai/llamafile).

The shipped SmolLM2 base is Apache 2.0. Preserve upstream model and runtime notices when distributing merged weights or executables. Nimble is a separate 9B model; verify its model license, supported architecture, tokenizer/prompt contract, and trading evaluation before distributing a quantized version. Its complete weights were not downloaded or converted here.

Changing precision, quantization, prompts, or inference implementations can change results. Compare sequence probabilities and rerun chronological market evaluation before promoting any new artifact. Neither portability nor structured classification guarantees profitable trading.

## Verified build

The F16 merged-adapter build was exercised on Linux x86-64 CPU: server health, dedicated scoring, and the harness CLI. The executable was approximately 315 MB. On one real-data snapshot its normalized label probabilities differed from Python inference by at most 0.0007. See [the verification report](../reports/llamafile-verification.json) for hashes and scope. This checks packaging/inference compatibility, not trading performance. Prebuilt binary release upload was unavailable in this environment; source build tooling is included.
