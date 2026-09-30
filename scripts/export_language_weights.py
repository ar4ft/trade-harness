"""Merge the shipped LoRA into Hugging Face weights for conversion to GGUF."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="src/trade_harness/assets/trading_lora")
    parser.add_argument("--output", default="artifacts/merged-trading-language")
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(args.adapter)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output already exists; choose a new directory")
    metadata = json.loads((adapter / "trading_metadata.json").read_text())
    torch.set_num_threads(4)
    base = AutoModelForCausalLM.from_pretrained(
        metadata["base_model"], revision=metadata["base_revision"]
    )
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(output)
    (output / "trading_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Merged weights saved to {output}. Convert with llama.cpp convert_hf_to_gguf.py.")


if __name__ == "__main__":
    main()
