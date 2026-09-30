"""Train a custom language-model adapter on reviewed decision chat JSONL.

Optional training dependencies and sufficient RAM/GPU memory required.
"""

import argparse
import json
from pathlib import Path


def main():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="artifacts/trading-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = []
    for line in Path(args.data).read_text().splitlines():
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        if [m["role"] for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("Expected system/user/assistant training examples")
        tokens = tokenizer.apply_chat_template(messages, tokenize=True)
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        if tokens[: len(prompt)] != prompt:
            raise ValueError("Chat template prefix mismatch; adjust assistant loss masking")
        # Never silently truncate away the target decision or candle context.
        if len(tokens) > args.max_length:
            raise ValueError("Example exceeds max-length; shorten the candle window before export")
        labels = [-100] * len(prompt) + tokens[len(prompt) :]
        if all(t == -100 for t in labels):
            raise ValueError("No assistant tokens available")
        records.append({"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": labels})
    if len(records) < 10:
        raise ValueError(
            "At least 10 reviewed examples required; use substantially more for useful training"
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
        ),
    )

    def collate(batch):
        longest = max(len(row["input_ids"]) for row in batch)
        result = {}
        for field, padding in [
            ("input_ids", tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("labels", -100),
        ]:
            result[field] = torch.tensor(
                [row[field] + [padding] * (longest - len(row[field])) for row in batch],
                dtype=torch.long,
            )
        return result

    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(records),
        data_collator=collate,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}; evaluate on a separate later time period.")


if __name__ == "__main__":
    main()
