"""Native direction-head training with frozen backbone and separated future labels."""

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from .clef import review_state
from .clef_contract import CONTRACT, MAX_REQUEST_BYTES, QUESTIONS, RELEASES
from .clef_runtime import load_native, questions_digest, sha256
from .research_data import PHASES, load_research
from .risk import RiskConfig
from .schemas import MarketInput


def load_native_dataset(dataset, source_dataset):
    """Reconstruct every request from verified source, including all three question fields."""
    path = Path(dataset)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if manifest.get("contract") != CONTRACT or sha256(path) != manifest.get("sha256"):
        raise ValueError("Native dataset contract or digest mismatch")
    original, source = load_research(source_dataset)
    if (
        manifest.get("source_dataset_sha256") != source["dataset_sha256"]
        or manifest.get("model") not in RELEASES
    ):
        raise ValueError("Native dataset source or variant mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lookup = {f"{r['symbol']}:{r['as_of']}:{r['position']}": r for r in original}
    seen = set()
    risk = RiskConfig.model_validate(source["risk_config"])
    for row in rows:
        if row["id"] in seen or row["id"] not in lookup:
            raise ValueError("Duplicate or unknown native example")
        seen.add(row["id"])
        source_row = lookup[row["id"]]
        expected = {
            "model": manifest["model"],
            "questions": QUESTIONS,
            "state": review_state(
                MarketInput.model_validate(source_row["market"]),
                source_row["history"],
                source_row["evidence"],
                risk,
                manifest["raw_context_candles"],
            ),
        }
        if (
            row["request"] != expected
            or row["targets"] != {"direction": source_row["label"]}
            or row["phase"] != source_row["phase"]
            or row["outcome"] != source_row["outcome"]
        ):
            raise ValueError(
                "Native example differs from the causal runtime renderer or target contract"
            )
        if (
            len(json.dumps(expected, ensure_ascii=True, separators=(",", ":")).encode())
            > MAX_REQUEST_BYTES
        ):
            raise ValueError("Native example exceeds complete-input byte budget")
    if seen != set(lookup) or any(not any(r["phase"] == phase for r in rows) for phase in PHASES):
        raise ValueError(
            "Native examples must preserve the complete source and all purged partitions"
        )
    return rows, manifest, source


def encode_complete(upstream, processor, request, max_length):
    encoded = upstream.encode_record(
        processor.tokenizer, request, processor=processor, max_length=max_length + 1
    )
    if len(encoded.input_ids) > max_length:
        raise ValueError("Native training input exceeds token budget; truncation is prohibited")
    return encoded


def direction_loss(logits, record, target, weights=None):
    """Only direction is supervised; schema context retains unsupervised advisory questions."""
    import torch

    index = next(i for i, q in enumerate(record.questions) if q.question_id == "direction")
    options = record.questions[index].option_ids
    if set(options) != {"BUY", "SELL", "HOLD"} or target not in options:
        raise ValueError("Invalid native direction target or option mapping")
    label = torch.tensor([options.index(target)], device=logits[index].device)
    weight = torch.tensor([weights[a] for a in options], device=label.device) if weights else None
    return torch.nn.functional.cross_entropy(
        logits[index].float().unsqueeze(0), label, weight=weight, reduction="sum"
    )


def train(
    dataset,
    source_dataset,
    output,
    variant="clef-flash",
    device="cuda",
    quantization="nf4",
    cache_dir=None,
    offline=False,
    steps=120,
    learning_rate=1e-5,
    max_length=16384,
    eval_samples=48,
    threads=4,
    seed=42,
    head_scope="residual",
):
    import numpy as np
    import torch
    from safetensors.torch import save_file

    if not 1 <= steps <= 100000 or not 1e-7 <= learning_rate <= 1e-3:
        raise ValueError("Declare 1–100000 training steps and a learning rate of 1e-7–1e-3")
    if not 1 <= max_length <= 65536 or not 0 <= eval_samples <= 10000 or not 1 <= threads <= 128:
        raise ValueError("Invalid token, evaluation or thread budget")
    rows, native, source = load_native_dataset(dataset, source_dataset)
    if native["model"] != variant:
        raise ValueError("Training variant differs from exported requests")
    destination = Path(output)
    if destination.exists():
        raise ValueError("Head output exists; use a new immutable artifact directory")
    torch.manual_seed(seed)
    groups = {phase: [r for r in rows if r["phase"] == phase] for phase in PHASES}
    counts = Counter(r["targets"]["direction"] for r in groups["train"])
    if set(counts) != {"BUY", "SELL", "HOLD"}:
        raise ValueError("Native training needs all three direction classes")
    weights = {a: (len(groups["train"]) / (3 * counts[a])) ** 0.5 for a in counts}
    order = np.random.default_rng(seed).permutation(len(groups["train"]))
    selected = [groups["train"][int(order[i % len(order)])] for i in range(steps)]
    eval_rows = {p: groups[p][:eval_samples] for p in PHASES[1:]}
    started = time.monotonic()
    model, processor, upstream, runtime = load_native(
        variant, device, quantization, cache_dir, offline, threads=threads, train_head=True
    )
    load_ms = (time.monotonic() - started) * 1000
    encoded = {
        r["id"]: encode_complete(upstream, processor, r["request"], max_length)
        for r in selected + [r for group in eval_rows.values() for r in group]
    }
    destination.mkdir(parents=True)
    plan = {
        "dataset_sha256": native["sha256"],
        "source_dataset_sha256": source["dataset_sha256"],
        "steps": steps,
        "head_scope": head_scope,
        "learning_rate": learning_rate,
        "seed": seed,
        "eval_samples": eval_samples,
        "training_ids_sha256": hashlib.sha256(
            json.dumps([r["id"] for r in selected]).encode()
        ).hexdigest(),
        "runtime": runtime,
        "max_length": max_length,
        "automatic_activation": False,
    }
    (destination / "training_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if head_scope not in ("residual", "full"):
        raise ValueError("Choose residual scorer or full native head training")
    if head_scope == "residual":
        model.head.requires_grad_(False)
        for name, parameter in model.head.named_parameters():
            if name.startswith("residual_scorer.") or name in (
                "prior_logit_scale",
                "joint_logit_scale",
                "residual_gate",
            ):
                parameter.requires_grad_(True)
    trainable = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.language_model.eval()
    model.head.train()
    step_rows = []
    for index, row in enumerate(selected):
        start = time.monotonic()
        record = encoded[row["id"]]
        batch = upstream.collate_records(
            [record], processor.tokenizer.pad_token_id, torch.device(device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss = direction_loss(model(batch)[0], record, row["targets"]["direction"], weights)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite native training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0, error_if_nonfinite=True)
        if any(p.grad is not None for p in model.language_model.parameters()):
            raise ValueError("Backbone unexpectedly received gradients")
        optimizer.step()
        result = {
            "step": index + 1,
            "id": row["id"],
            "loss": float(loss.detach()),
            "tokens": len(record.input_ids),
            "elapsed_ms": (time.monotonic() - start) * 1000,
        }
        step_rows.append(result)
        with (destination / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    model.eval()
    scores = []
    with torch.inference_mode():
        for phase, group in eval_rows.items():
            for row in group:
                record = encoded[row["id"]]
                batch = upstream.collate_records(
                    [record], processor.tokenizer.pad_token_id, torch.device(device)
                )
                logits = model(batch)[0]
                i = next(i for i, q in enumerate(record.questions) if q.question_id == "direction")
                probabilities = dict(
                    zip(record.questions[i].option_ids, logits[i].float().softmax(-1).tolist())
                )
                scores.append(
                    {
                        "id": row["id"],
                        "phase": phase,
                        "probabilities": probabilities,
                        "target": row["targets"]["direction"],
                        "as_of": row["request"]["state"]["market"]["timestamps"][-1],
                        "observed_at": row["outcome"]["observed_at"],
                    }
                )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
        str(destination / "joint_head.safetensors"),
    )
    shutil.copyfile(upstream.license_path, destination / "LICENSE")
    metadata = {
        "contract": CONTRACT,
        "model": variant,
        "revision": RELEASES[variant][1],
        "questions_sha256": questions_digest(),
        "head_sha256": sha256(destination / "joint_head.safetensors"),
        "backbone_frozen": True,
        "head_dtype": "float32",
        "quantization": quantization,
        "raw_context_candles": native["raw_context_candles"],
        "symbols": source["symbols"],
        "timeframe": source["timeframe"],
        "horizon": source["horizon"],
        "feature_names": source["feature_names"],
        "risk_config": source["risk_config"],
        "fine_tuned_until": max(r["outcome"]["observed_at"] for r in selected),
        "foundation_trained_until": None,
        "probability_calibration": "uncalibrated",
        "training_labels": dict(counts),
        "class_weights": weights,
        "plan": plan,
        "load_ms": load_ms,
        "actual_optimizer_steps": len(step_rows),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "head_scope": head_scope,
        "evaluation_samples_by_phase": dict(Counter(r["phase"] for r in scores)),
        "validation_status": "research_only",
        "license": "Apache-2.0",
        "real_execution_enabled": False,
        "trainer_sha256": sha256(__file__),
        "limitations": [
            "Direction-only supervision; advisory evidence/risk outputs remain unvalidated.",
            "Automatic cost-aware targets are not expert or path-dependent policy labels.",
            "Unknown foundation fitting cutoff prevents certified retrospective chronology.",
            "No calibration, positive edge, automatic activation or execution is claimed.",
        ],
    }
    (destination / "training_manifest.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    (destination / "evaluation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=RELEASES, default="clef-flash")
    parser.add_argument("--head-scope", choices=("residual", "full"), default="residual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--check", action="store_true", help="Validate native/source data without loading weights"
    )
    args = parser.parse_args()
    if args.check:
        rows, manifest, _ = load_native_dataset(args.dataset, args.source_dataset)
        if manifest["model"] != args.model:
            parser.error("Model differs from native dataset")
        print(
            json.dumps(
                {
                    "verified_examples": len(rows),
                    "partitions": manifest["partitions"],
                    "training_performed": False,
                },
                indent=2,
            )
        )
        return
    result = train(
        args.dataset,
        args.source_dataset,
        args.output,
        args.model,
        args.device,
        args.quantization,
        args.cache_dir,
        args.offline,
        args.steps,
        args.learning_rate,
        args.max_length,
        args.eval_samples,
        args.threads,
        head_scope=args.head_scope,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
