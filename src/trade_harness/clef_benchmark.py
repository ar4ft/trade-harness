"""Measure a pinned native service on declared complete requests; never forward labels."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .clef import distribution
from .clef_contract import CONTRACT, QUESTIONS, RELEASES
from .clef_training import load_native_dataset
from .hybrid_training import digest
from .operations import quantiles


def benchmark(dataset, source_dataset, output, base_url, max_requests=3, timeout=600, api_key=""):
    if not 1 <= max_requests <= 100 or not 5 <= timeout <= 600:
        raise ValueError("Declare 1–100 requests and a timeout of 5–600 seconds")
    url = urlsplit(base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.query
        or url.fragment
    ):
        raise ValueError("Use an HTTP service URL without embedded credentials/query")
    rows, native, _ = load_native_dataset(dataset, source_dataset)
    rows = [r for r in rows if r["phase"] in ("validation", "test")]
    # Declare the sample before contacting the model; no outcome-conditioned selection.
    selected = sorted(rows, key=lambda r: r["id"])[:: max(1, len(rows) // max_requests)][
        :max_requests
    ]
    destination = Path(output)
    if destination.exists():
        raise ValueError("Benchmark exists; use a new immutable output")
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    base_url = base_url.rstrip("/")
    report = {
        "contract": "native-service-benchmark-v1",
        "model": native["model"],
        "dataset_sha256": native["sha256"],
        "raw_context_candles": native["raw_context_candles"],
        "sample_ids": [r["id"] for r in selected],
        "request_timeout_seconds": timeout,
        "requests": [],
        "actual_gpu_measurements": False,
        "labels_forwarded": False,
        "trading_edge_measured": False,
        "real_execution_enabled": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout) as client:
        response = client.get(base_url + "/metadata", headers=headers)
        response.raise_for_status()
        manifest = response.json()
        variant = native["model"]
        if (
            manifest.get("model") != variant
            or manifest.get("repository") != RELEASES[variant][0]
            or manifest.get("revision") != RELEASES[variant][1]
            or manifest.get("contract") != CONTRACT
            or manifest.get("input_truncation") != "reject"
        ):
            raise ValueError("Service differs from the pinned native benchmark variant")
        report["manifest"] = manifest
        try:
            metrics = client.get(base_url + "/metrics", headers=headers)
            metrics.raise_for_status()
            report["service_startup"] = metrics.json().get("startup", {})
        except httpx.HTTPError:
            report["service_startup"] = None
        for row in selected:
            result = {
                "id": row["id"],
                "as_of": row["request"]["state"]["market"]["timestamps"][-1],
                "request_sha256": digest(row["request"]),
            }
            start = time.monotonic()
            try:
                current = client.get(base_url + "/metadata", headers=headers)
                current.raise_for_status()
                if current.json() != manifest:
                    raise ValueError("Service identity drift")
                answer = client.post(
                    base_url + "/v1/systemone", headers=headers, json=row["request"]
                )
                answer.raise_for_status()
                body = answer.json()
                if (
                    body.get("model") != variant
                    or body.get("provenance") != manifest
                    or body.get("state_truncated") is not False
                ):
                    raise ValueError("Service returned different provenance or incomplete input")
                direction = body["answers"]["direction"]
                if set(body["answers"]) != set(QUESTIONS) or direction.get("type") != "choice":
                    raise ValueError("Incomplete native question schema")
                enough, risk = body["answers"]["evidence_sufficient"], body["answers"]["risk_level"]
                if (enough.get("type") != "noul" or risk.get("type") != "score"
                        or isinstance(enough.get("noul"), bool) or not isinstance(enough.get("noul"), (int, float))
                        or not math.isfinite(enough["noul"]) or not 0 <= enough["noul"] <= 1):
                    raise ValueError("Invalid native advisory probability")
                levels = distribution(risk, ("0", "1", "2", "3"))
                if not math.isclose(risk["score"], sum(int(k) * p for k, p in levels.items()), abs_tol=.001):
                    raise ValueError("Invalid native ordinal score")
                probability = distribution(direction, ("BUY", "SELL", "HOLD"))
                if direction["choice"] != max(probability, key=probability.get):
                    raise ValueError("Invalid benchmark direction")
                result.update(
                    status="ok",
                    action=direction["choice"],
                    probabilities=probability,
                    input_tokens=body.get("usage", {}).get("input_tokens"),
                )
            except httpx.TimeoutException:
                result["status"] = "client_timeout"
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                result["status"] = "rejected_or_invalid"
            result["roundtrip_ms"] = (time.monotonic() - start) * 1000
            report["requests"].append(result)
            # Retain failures if the process is interrupted after a completed probe.
            destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    successful = [r["roundtrip_ms"] for r in report["requests"] if r["status"] == "ok"]
    report.update(
        {
            "successful_roundtrip": quantiles(successful),
            "succeeded": len(successful),
            "failed": len(selected) - len(successful),
            "actual_gpu_measurements": manifest["device"].startswith("cuda") and bool(successful),
            "limitations": [
                "HTTP timing includes metadata check, queue, encoding and inference.",
                "A client timeout does not cancel computation on the server.",
                "Few historical probes do not establish sustained p95 latency or trading accuracy.",
                "Startup metrics are operational measurements, not part of model identity.",
            ],
        }
    )
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-url", default=os.environ.get("CLEF_BASE_URL", "http://127.0.0.1:11436")
    )
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.dataset,
                args.source_dataset,
                args.output,
                args.base_url,
                args.max_requests,
                args.timeout,
                os.environ.get("CLEF_API_KEY", ""),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
