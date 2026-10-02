"""Run a pinned upstream Clef snapshot in a separate GPU environment.

This module needs the upstream Transformers 5 runtime, independently of the
harness's Transformers 4 environment. Importing it does not load/download weights.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import secrets
import sys
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

from .clef_contract import CONTRACT, MAX_REQUEST_BYTES, RELEASES


def create_app(model, processor, manifest, encode_record, systemone, api_key=""):
    app = FastAPI(title="Pinned Clef research service")
    lock = threading.Lock()

    def authorized(authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or "", "Bearer " + api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/metadata", dependencies=[Depends(authorized)])
    def metadata():
        return manifest

    @app.post("/v1/systemone", dependencies=[Depends(authorized)])
    def decide(request: dict):
        if (
            request.get("model") != manifest["model"]
            or "state" not in request
            or not isinstance(request.get("questions"), dict)
            or not 1 <= len(request["questions"]) <= 64
            or set(request) != {"model", "state", "questions"}
        ):
            raise HTTPException(
                status_code=422,
                detail="Use the loaded model with text/JSON state and typed questions",
            )
        if (
            len(
                json.dumps(
                    request, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                ).encode()
            )
            > MAX_REQUEST_BYTES
        ):
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
                answer = systemone(model, processor, request, max_length=manifest["max_length"])
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(
                status_code=422, detail="Invalid or over-budget state/schema"
            ) from error
        return {**answer, "provenance": manifest, "state_truncated": False}

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Pinned native Clef server; requires a separate GPU runtime"
    )
    parser.add_argument("--model", choices=RELEASES, default="clef-flash")
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
    import torch

    torch.set_num_threads(args.threads)
    from huggingface_hub import snapshot_download

    repo = RELEASES[args.model][0]
    path = Path(
        snapshot_download(
            repo,
            revision=revision,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
            max_workers=2,
        )
    )
    spec = importlib.util.spec_from_file_location(
        "clef_release_" + revision, path / "joint_schema_model.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    kwargs = {}
    packages = ["torch", "transformers"]
    if args.quantization == "nf4":
        from transformers import BitsAndBytesConfig

        # Keep lm_head dense: the joint head reads its lexical embedding rows directly.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"],
        )
        packages.append("bitsandbytes")
        try:
            importlib.metadata.version("kernels")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            packages.append("kernels")
    model, processor = upstream.load_release_model(path, device=args.device, **kwargs)
    if model.language_model.get_output_embeddings().weight.dtype == torch.uint8:
        raise ValueError("The native lexical head needs dense output embedding rows")
    manifest = {
        "model": args.model,
        "repository": repo,
        "revision": revision,
        "contract": CONTRACT,
        "input_truncation": "reject",
        "max_length": args.max_length,
        "server_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {package: importlib.metadata.version(package) for package in packages},
        "quantization": args.quantization,
        "device": args.device,
        "threads": args.threads,
    }
    app = create_app(
        model,
        processor,
        manifest,
        upstream.encode_record,
        upstream.systemone,
        api_key=os.environ.get("CLEF_API_KEY", ""),
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
