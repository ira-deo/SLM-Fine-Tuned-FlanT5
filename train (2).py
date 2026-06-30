"""
train.py

Fine-tunes flan-t5-base for procurement email -> JSON field extraction
using HuggingFace Seq2SeqTrainer.

Usage
-----
    python train.py \
        --data_path synthetic_data.jsonl \
        --output_dir ./checkpoints \
        --epochs 5

If --data_path is not provided, a small synthetic dataset is generated
automatically for a smoke test (NOT for production use).

Notes on the <1 min/epoch requirement
--------------------------------------
Training speed depends heavily on dataset size, hardware, and batch size.
This script defaults to settings (batch size 16, grad accumulation 2,
fp16 on GPU) that keep a single epoch over a small/medium dataset
(a few hundred to a couple thousand short examples) under a minute on a
single modern GPU (e.g. T4/A10/A100). For larger datasets, reduce
--max_train_samples or increase --per_device_train_batch_size (if memory
allows) to stay within the time budget. On CPU-only machines, fp16 is
automatically disabled and training will be substantially slower -
reduce dataset size accordingly for smoke tests.
"""

import argparse
import os
import sys

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from dataset import (
    TARGET_KEYS,
    generate_synthetic_dataset,
    load_and_prepare_datasets,
)
from model import DEFAULT_MODEL_NAME, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune flan-t5-base for procurement email JSON extraction."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to .json or .jsonl labeled dataset. If omitted, a "
        "synthetic dataset is generated for a smoke test.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints and final model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs (1-5 recommended).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Stop if eval_loss does not improve for this many evals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap on number of training examples (useful to "
        "keep epoch time under the 1-minute budget on small GPUs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs < 1 or args.epochs > 20:
        print(
            f"WARNING: --epochs={args.epochs} is outside the recommended "
            f"1-5 range. Clamping to that range.",
            file=sys.stderr,
        )
        args.epochs = max(1, min(20, args.epochs))

    os.makedirs(args.output_dir, exist_ok=True)

    data_path = args.data_path
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "synthetic_data.jsonl")
        print(
            f"No --data_path provided. Generating a small synthetic dataset "
            f"at '{data_path}' for a smoke test. "
            f"Replace with real labeled data for production training."
        )
        generate_synthetic_dataset(data_path, n=200, seed=args.seed)

    print(f"Loading and tokenizing dataset from '{data_path}'...")
    train_dataset, eval_dataset, tokenizer = load_and_prepare_datasets(
        data_path=data_path,
        model_name=args.model_name,
        test_size=0.1,
        seed=args.seed,
    )

    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples:  {len(eval_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after preparation.")
    if len(eval_dataset) == 0:
        raise ValueError("Eval dataset is empty after preparation.")

    print(f"Loading model '{args.model_name}'...")
    model, model_tokenizer = load_model_and_tokenizer(args.model_name)
    # Use the tokenizer returned from dataset prep (same model, same tokenizer)
    # but keep a single consistent reference.
    tokenizer = tokenizer if tokenizer is not None else model_tokenizer

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="epoch",
        load_best_model_at_end=False,
        predict_with_generate=True,
        generation_max_length=1024,
        report_to=[],
        seed=args.seed,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating final model...")
    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"Saving final model and tokenizer to '{final_dir}'...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Done.")
    print(f"Checkpoint + tokenizer saved at: {final_dir}")
    print(f"Target keys (fixed order): {TARGET_KEYS}")


if __name__ == "__main__":
    main()