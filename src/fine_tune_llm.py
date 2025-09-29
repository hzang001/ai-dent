#!/usr/bin/env python3
"""Fine-tune a causal LLM on the oral-cancer biomarker prompts using LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/llm/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/llm/valid.jsonl"))
    parser.add_argument("--base-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/llm"))
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, args: argparse.Namespace) -> AutoModelForCausalLM:
    quantization_config = None
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=None,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def tokenise_dataset(dataset, tokenizer, max_length: int):
    eos_token_id = tokenizer.eos_token_id

    def _tokenise(example: Dict[str, str]) -> Dict[str, List[int]]:
        prompt_ids = tokenizer(
            example["prompt"],
            add_special_tokens=False,
        )["input_ids"]
        completion_text = example["completion"] + tokenizer.eos_token
        completion_ids = tokenizer(
            completion_text,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = prompt_ids + completion_ids
        input_ids = input_ids[:max_length]
        cutoff = max(0, len(prompt_ids))
        cutoff = min(cutoff, len(input_ids))
        attention_mask = [1] * len(input_ids)
        labels = [-100] * cutoff + input_ids[cutoff:]
        if len(labels) < len(input_ids):
            pad_needed = len(input_ids) - len(labels)
            labels += [-100] * pad_needed
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenised = dataset.map(_tokenise, remove_columns=dataset.column_names)
    tokenised.set_format(type="torch")
    return tokenised


def collate_fn(features: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(feat["input_ids"].shape[0] for feat in features)
    input_batch, mask_batch, label_batch = [], [], []
    for feat in features:
        input_ids = feat["input_ids"].tolist()
        attention_mask = feat["attention_mask"].tolist()
        labels = feat["labels"].tolist()
        pad_len = max_len - len(input_ids)
        input_batch.append(input_ids + [pad_token_id] * pad_len)
        mask_batch.append(attention_mask + [0] * pad_len)
        label_batch.append(labels + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_batch, dtype=torch.long),
        "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
        "labels": torch.tensor(label_batch, dtype=torch.long),
    }


def evaluate_generation(model, tokenizer, dataset, max_new_tokens: int) -> Dict[str, float]:
    device = next(model.parameters()).device
    total = 0
    correct = 0
    for sample in dataset:
        prompt = sample["prompt"]
        expected = sample["completion"].strip().lower()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
        output_tokens = generated[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip().lower()
        prediction = "cancer" if "cancer" in text else "control" if "control" in text else ""
        total += 1
        if prediction == expected:
            correct += 1
    accuracy = correct / total if total else 0.0
    return {"num_samples": total, "accuracy": accuracy}


def main() -> None:
    args = parse_args()
    data_files = {
        "train": str(args.train_file),
        "validation": str(args.valid_file),
    }
    raw_dataset = load_dataset("json", data_files=data_files)

    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.base_model, args)

    tokenised_train = tokenise_dataset(raw_dataset["train"], tokenizer, args.max_length)
    tokenised_valid = tokenise_dataset(raw_dataset["validation"], tokenizer, args.max_length)

    data_collator = lambda features: collate_fn(features, tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=[],
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_valid,
        data_collator=data_collator,
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")

    metrics = evaluate_generation(model, tokenizer, raw_dataset["validation"], args.max_new_tokens)
    metrics_path = args.output_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"adapter_dir": str(adapter_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
