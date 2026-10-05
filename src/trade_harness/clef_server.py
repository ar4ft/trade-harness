"""Run a pinned upstream Clef snapshot in a separate GPU environment.

This module needs the upstream Transformers 5 runtime, independently of the
harness's Transformers 4 environment. Importing it does not load/download weights.
"""

import argparse
import hashlib
import json
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

from .clef_contract import CONTRACT, MAX_REQUEST_BYTES, QUESTIONS, RELEASES

SERVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def create_app(model, processor, manifest, encode_record, systemone, api_key="", startup_metrics=None):
    app = FastAPI(title="Pinned Clef research service")
    lock = threading.Lock()
    metrics_lock = threading.Lock()
    durations = deque(maxlen=512)
    stats = {"attempted": 0, "succeeded": 0, "rejected": 0, "failed": 0}

    def record(status, started):
        with metrics_lock:
            stats[status] += 1
            durations.append((time.monotonic() - started) * 1000)

    def authorized(authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or "", "Bearer " + api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/metadata", dependencies=[Depends(authorized)])
    def metadata():
        return manifest

    @app.get("/metrics", dependencies=[Depends(authorized)])
    def metrics():
        from .operations import quantiles

        with metrics_lock:
            return {"contract": "clef-service-metrics-v1", "startup": startup_metrics or {},
                    "requests": dict(stats), "request_duration": quantiles(list(durations)),
                    "latency_scope": "Queue, validation, tokenization and native scoring; last 512 completed requests"}

    @app.post("/v1/systemone", dependencies=[Depends(authorized)])
    def decide(request: dict):
        started = time.monotonic()
        with metrics_lock:
            stats["attempted"] += 1
        if (
            request.get("model") != manifest["model"]
            or "state" not in request
            or not isinstance(request.get("questions"), dict)
            or not 1 <= len(request["questions"]) <= 64
            or set(request) != {"model", "state", "questions"}
        ):
            record("rejected", started)
            raise HTTPException(
                status_code=422,
                detail="Use the loaded model with text/JSON state and typed questions",
            )
        try:
            raw = json.dumps(request, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        except (ValueError, TypeError) as error:
            record("rejected", started)
            raise HTTPException(status_code=422, detail="State must contain finite JSON values") from error
        if len(raw) > MAX_REQUEST_BYTES:
            record("rejected", started)
            raise HTTPException(status_code=422, detail="Complete-state byte budget exceeded")
        try:
            with lock:
                # A +1 preflight detects even a one-token overflow before the upstream
                # encoder/systemone helper can silently truncate the state.
                encoded = encode_record(
                    processor.tokenizer,
                    request,
                    processor=processor,
                    max_length=manifest["max_length"] + 1,
                )
                if len(encoded.input_ids) > manifest["max_length"]:
                    raise ValueError("State exceeds the complete-input token budget")
                head = manifest.get("custom_head")
                if head:
                    state, market = request["state"], request["state"]["market"]
                    costs = state["reference_costs"]
                    risk = head["risk_config"]
                    if (request["questions"] != QUESTIONS
                            or market["symbol"] not in head["symbols"]
                            or market["timeframe"] != head["timeframe"] or market["horizon"] != head["horizon"]
                            or market["timestamps"][-1] <= head["fine_tuned_until"]
                            or state["input_scope"]["raw_context_candles"] != head["raw_context_candles"]
                            or sorted(state["shared_evidence"]["features"]) != sorted(head["feature_names"])
                            or costs["fee_bps_per_side"] != risk["fee_bps"]
                            or costs["slippage_bps_per_side"] != risk["slippage_bps"]
                            or costs["minimum_net_edge"] != risk["min_net_edge"]):
                        raise ValueError("Native head input differs from its trained scope")
                answer = systemone(model, processor, request, max_length=manifest["max_length"])
        except (ValueError, KeyError, TypeError) as error:
            record("rejected", started)
            raise HTTPException(
                status_code=422, detail="Invalid or over-budget state/schema"
            ) from error
        except Exception as error:  # noqa: BLE001 - native failures counted without exposing internals
            record("failed", started)
            raise HTTPException(status_code=500, detail="Native inference failed") from error
        record("succeeded", started)
        return {**answer, "provenance": manifest, "state_truncated": False}

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Pinned native Clef server; requires a separate GPU runtime"
    )
    parser.add_argument("--model", choices=RELEASES, default="clef-flash")
    parser.add_argument("--head-path", help="Explicit native head artifact; never activated automatically")
    parser.add_argument("--revision", help="Immutable 40-character upstream snapshot revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report fresh-deployment disk/memory planning budgets; do not download or serve",
    )
    parser.add_argument(
        "--cache-dir", help="Snapshot cache directory, separate from the application wheel"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use an already downloaded pinned snapshot"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    import re

    revision = args.revision or RELEASES[args.model][1]
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or not 1 <= args.max_length <= 65536:
        parser.error("Use an immutable snapshot revision and a token budget of 1–65536")
    if not 1 <= args.threads <= 128:
        parser.error("Use 1–128 CPU threads")
    if args.check:
        from .clef_resources import inspect_resources

        try:
            report = inspect_resources(
                args.model, args.quantization, args.device, args.cache_dir, revision
            )
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2))
        return 0 if report["ready_for_fresh_deployment"] else 2
    from .clef_runtime import load_native

    started = time.monotonic()
    model, processor, upstream, runtime = load_native(
        args.model, args.device, args.quantization, args.cache_dir, args.offline,
        revision, args.threads, args.head_path,
    )
    manifest = {
        **runtime, "contract": CONTRACT, "input_truncation": "reject",
        "max_length": args.max_length,
        "server_sha256": SERVER_SHA256,
    }
    import resource

    startup = {"cold_load_ms": (time.monotonic() - started) * 1000,
               "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "device": args.device, "quantization": args.quantization}
    if args.device.startswith("cuda"):
        import torch

        startup["gpu_allocated_bytes"] = torch.cuda.memory_allocated(args.device)
        startup["gpu_reserved_bytes"] = torch.cuda.memory_reserved(args.device)
        startup["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    app = create_app(
        model,
        processor,
        manifest,
        upstream.encode_record,
        upstream.systemone,
        api_key=os.environ.get("CLEF_API_KEY", ""),
        startup_metrics=startup,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
