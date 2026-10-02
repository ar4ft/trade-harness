"""Train and evaluate a CPU-sized trading LoRA adapter from real OHLCV outcomes."""

import argparse
import hashlib
import json
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

from trade_harness.local_language import (
    ACTIONS,
    COMPACT_SYSTEM,
    LocalLanguageModel,
    training_records,
)
from trade_harness.schemas import MarketInput


def main(argv=None):
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/BTCUSDT-1h.json")
    parser.add_argument("--dataset", help="Purged research JSONL and its manifest")
    parser.add_argument("--output", default="src/trade_harness/assets/trading_lora")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--base-revision", help="Immutable Hugging Face commit")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=96)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.stride < 1 or args.eval_samples < 1:
        raise ValueError("Steps, stride, and evaluation sample count must be positive")
    torch.set_num_threads(4)
    set_seed(42)
    output = Path(args.output)
    if (output / "adapter_model.safetensors").exists():
        raise ValueError("Adapter already exists; use a new candidate output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = None
    if args.dataset:
        from .research_data import PHASES, load_research

        examples, manifest = load_research(args.dataset)
        records = {p: [r for r in examples if r["phase"] == p] for p in PHASES}
    else:
        market = MarketInput.model_validate_json(Path(args.input).read_text())
        records = training_records(market, args.stride)
    if any(not any(row["label"] == action for row in records["train"]) for action in ACTIONS):
        raise ValueError("Training partition must contain BUY, SELL, and HOLD targets")
    if len(records["train"]) < 30:
        raise ValueError("Not enough training records")
    revision = args.base_revision or HfApi().model_info(args.base_model).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Base revision must be an immutable 40-character commit")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = []
    for row in records["train"]:
        messages = [
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        tokens = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["label"]}], tokenize=True
        )
        if tokens[: len(prompt)] != prompt:
            raise ValueError("Chat template prefix differs; cannot safely mask labels")
        if len(tokens) > args.max_length:
            raise ValueError("Example exceeds max-length; no silent evidence truncation")
        encoded.append(
            {
                "input_ids": tokens,
                "attention_mask": [1] * len(tokens),
                "labels": [-100] * len(prompt) + tokens[len(prompt) :],
            }
        )
    print(
        json.dumps(
            {
                "train_examples": len(encoded),
                "max_tokens": max(len(r["input_ids"]) for r in encoded),
                "base_revision": revision,
                "class_counts": dict(Counter(r["label"] for r in records["train"])),
            }
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=revision)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )

    def collate(batch):
        n = max(len(row["input_ids"]) for row in batch)
        return {
            key: torch.tensor([r[key] + [pad] * (n - len(r[key])) for r in batch])
            for key, pad in [
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }

    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(encoded),
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"),
            max_steps=args.steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-4,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=42,
            use_cpu=not torch.cuda.is_available(),
        ),
    )
    training = trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    class_returns = {
        a: float(np.mean([r["outcome"]["next_open_return"] if manifest else r["realized_return"]
                          for r in records["train"] if r["label"] == a]))
        for a in ACTIONS
    }
    metadata = {
        "dataset_sha256": manifest["dataset_sha256"] if manifest else
                          hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "samples_processed": args.steps * 4,
        "training_versions": {
            p: version(p) for p in ("torch", "transformers", "peft", "datasets", "accelerate")
        },
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "version": 1,
        "base_model": args.base_model,
        "base_revision": revision,
        "symbols": manifest["symbols"] if manifest else [market.symbol],
        "symbol": None if manifest else market.symbol,
        "timeframe": manifest["timeframe"] if manifest else market.timeframe,
        "horizon": manifest["horizon"] if manifest else market.horizon,
        "trained_until": max(r["outcome"]["observed_at"] if manifest else r["label_observed_at"]
                             for r in records["train"]),
        "class_returns": class_returns,
        "threshold": 0.003,
        "label_source": ("cost-aware position-dependent next-open outcomes; automatic labels, not human judgments"
                         if manifest else "future market returns; automatic directional labels, not human judgments"),
        "train_examples": len(records["train"]),
        "steps": args.steps,
        "seed": 42,
        "stride": args.stride,
        "training_metrics": training.metrics,
        "feature_contract": manifest["prompt_contract"] if manifest else
                            "compact_prompt v1: six OHLCV features, four indicators, prior action/outcome",
        "max_length": args.max_length,
    }
    if manifest:
        metadata.update({"feature_names": manifest["feature_names"],
                         "dataset_manifest": manifest, "label_contract": manifest["label_contract"],
                         "fit_labels_last": metadata["trained_until"]})
    (output / "trading_metadata.json").write_text(json.dumps(metadata, indent=2))
    # Reuse trained model for evaluation instead of loading another copy.
    evaluator = LocalLanguageModel.__new__(LocalLanguageModel)
    evaluator.model = model.eval()
    evaluator.tokenizer = tokenizer
    evaluator.candidates = [tokenizer.encode(a, add_special_tokens=False) for a in ACTIONS]
    temperature = 1.0
    if manifest:
        from scipy.special import softmax
        from sklearn.metrics import log_loss

        group = records["calibration"]
        ids = np.linspace(0, len(group) - 1, min(args.eval_samples, len(group)), dtype=int)
        logits = np.log(np.maximum(np.asarray([evaluator.probabilities(group[i]["prompt"])
                                              for i in ids]), 1e-12))
        y = np.asarray([ACTIONS.index(group[i]["label"]) for i in ids])
        temperature = float(min(np.geomspace(0.5, 10, 40),
                                key=lambda t: log_loss(y, softmax(logits / t, axis=1),
                                                       labels=[0, 1, 2])))
        metadata["probability_calibration"] = {
            "method": "temperature", "temperature": temperature, "samples": len(ids),
            "first_timestamp": min(group[i]["as_of"] for i in ids),
            "labels_last": max(group[i]["outcome"]["observed_at"] for i in ids),
            "raw_log_loss": float(log_loss(y, softmax(logits, axis=1), labels=[0, 1, 2])),
            "calibrated_log_loss": float(log_loss(y, softmax(logits / temperature, axis=1),
                                                   labels=[0, 1, 2])),
        }
        metadata["trained_until"] = metadata["probability_calibration"]["labels_last"]
        print(json.dumps({"calibration": metadata["probability_calibration"]}), flush=True)
    train_majority = Counter(r["label"] for r in records["train"]).most_common(1)[0][0]
    evaluations = {}
    for partition in ("validation", "test"):
        group = records[partition]
        # Uniform deterministic sample of later partition, never selected by outcome.
        ids = np.linspace(0, len(group) - 1, min(args.eval_samples, len(group)), dtype=int)
        truths = []
        predictions = []
        errors = []
        probabilities = []
        for i in ids:
            row = group[i]
            probs = evaluator.probabilities(row["prompt"])
            if manifest:
                probs = softmax(np.log(np.maximum(probs, 1e-12)) / temperature)
            probabilities.append(probs)
            predictions.append(ACTIONS[int(probs.argmax())])
            truths.append(row["label"])
            expected = sum(p * class_returns[a] for p, a in zip(probs, ACTIONS))
            errors.append(abs(expected - (row["outcome"]["next_open_return"] if manifest
                                          else row["realized_return"])))
        evaluations[partition] = {
            "samples": len(ids),
            "direction_accuracy": float(np.mean(np.array(predictions) == np.array(truths))),
            "training_majority_baseline_accuracy": float(
                np.mean(np.array(truths) == train_majority)
            ),
            "forecast_mae": float(np.mean(errors)),
            "zero_return_mae": float(np.mean([abs(group[i]["outcome"]["next_open_return"]
                                                  if manifest else group[i]["realized_return"])
                                               for i in ids])),
            "predicted_counts": dict(Counter(predictions)),
            "label_counts": dict(Counter(truths)),
            "first_timestamp": group[0]["as_of"],
            "last_timestamp": group[-1]["as_of"],
        }
        if manifest:
            from .diagnostics import probability_diagnostics

            evaluations[partition]["probability_quality"] = probability_diagnostics(
                np.asarray(probabilities), np.asarray([ACTIONS.index(t) for t in truths]))
            prior = np.asarray([sum(r["label"] == a for r in records["train"]) /
                                len(records["train"]) for a in ACTIONS])
            evaluations[partition]["training_prior_baseline"] = probability_diagnostics(
                np.tile(prior, (len(truths), 1)), np.asarray([ACTIONS.index(t) for t in truths]))
            evaluations[partition]["target_semantics"] = "Position-aware action; accuracy is not trading edge"
        print(json.dumps({partition: evaluations[partition]}), flush=True)
    metadata["evaluation"] = evaluations
    (output / "trading_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Trained weights saved to {output}", flush=True)


if __name__ == "__main__":
    main()
