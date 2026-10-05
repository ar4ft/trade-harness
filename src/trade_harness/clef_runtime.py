"""Shared pinned native loader for serving, head training and hardware measurements."""

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from .clef_contract import CONTRACT, QUESTIONS, RELEASES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def questions_digest():
    return hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()


def frozen_head_model(model):
    """FP32 native head with a frozen prefill, without copying the full lexical matrix."""
    import torch

    class LexicalRows:
        def __init__(self, weight):
            self.weight = weight

        def __getitem__(self, ids):
            return self.weight[ids].float()

    class FrozenHead(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.language_model = original.language_model
            self.head = original.head.float()
            self.language_model.requires_grad_(False)

        def forward(self, batch):
            base = self.language_model
            text = base.model
            if hasattr(text, "language_model"):
                text = text.language_model
            with torch.no_grad():
                hidden = text(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
            return self.head(
                hidden.float(),
                batch["input_ids"],
                batch["attention_mask"],
                batch["records"],
                LexicalRows(base.get_output_embeddings().weight),
            )

    return FrozenHead(model)


def head_manifest(path, variant, revision, quantization):
    root = Path(path)
    value = json.loads((root / "training_manifest.json").read_text())
    if (
        value.get("contract") != CONTRACT
        or value.get("model") != variant
        or value.get("revision") != revision
        or value.get("quantization") != quantization
        or value.get("questions_sha256") != questions_digest()
        or value.get("head_sha256") != sha256(root / "joint_head.safetensors")
        or value.get("backbone_frozen") is not True
        or value.get("head_dtype") != "float32"
    ):
        raise ValueError("Native head differs from its pinned training/inference contract")
    return value


def load_native(
    variant,
    device="cuda",
    quantization="none",
    cache_dir=None,
    offline=False,
    revision=None,
    threads=4,
    head_path=None,
    train_head=False,
):
    import torch
    from huggingface_hub import snapshot_download

    revision = revision or RELEASES[variant][1]
    torch.set_num_threads(threads)
    path = Path(
        snapshot_download(
            RELEASES[variant][0],
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=offline,
            max_workers=2,
        )
    )
    spec = importlib.util.spec_from_file_location(
        "clef_release_" + revision, path / "joint_schema_model.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    upstream.license_path = path / "LICENSE"
    kwargs, packages = {}, ["torch", "transformers"]
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"],
        )
        packages.append("bitsandbytes")
    model, processor = upstream.load_release_model(path, device=device, **kwargs)
    if model.language_model.get_output_embeddings().weight.dtype == torch.uint8:
        raise ValueError("Native lexical rows must remain dense")
    manifest = {
        "model": variant,
        "repository": RELEASES[variant][0],
        "revision": revision,
        "runtime": {p: importlib.metadata.version(p) for p in packages},
        "quantization": quantization,
        "device": device,
        "threads": threads,
        "native_runtime_sha256": sha256(__file__),
        "upstream_code_sha256": sha256(path / "joint_schema_model.py"),
        "head_dtype": "bfloat16",
    }
    if train_head or head_path:
        model = frozen_head_model(model)
        manifest["head_dtype"] = "float32"
    if head_path:
        from safetensors.torch import load_file

        metadata = head_manifest(head_path, variant, revision, quantization)
        model.head.load_state_dict(
            load_file(str(Path(head_path) / "joint_head.safetensors")), strict=True
        )
        manifest["custom_head"] = {
            "manifest_sha256": sha256(Path(head_path) / "training_manifest.json"),
            "head_sha256": metadata["head_sha256"],
            "fine_tuned_until": metadata["fine_tuned_until"],
            "raw_context_candles": metadata["raw_context_candles"],
            "symbols": metadata["symbols"],
            "timeframe": metadata["timeframe"],
            "horizon": metadata["horizon"],
            "feature_names": metadata["feature_names"],
            "risk_config": metadata["risk_config"],
            "probability_calibration": "uncalibrated",
        }
    return model.eval(), processor, upstream, manifest
