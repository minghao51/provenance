"""Training script for fine-tuning RoBERTa on RAID dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_raid_dataset(
    sample_limit: int | None = None,
    cache_dir: str | None = None,
) -> tuple[list[str], list[int]]:
    dataset = load_dataset("liamdugan/raid", cache_dir=cache_dir)

    texts = []
    labels = []

    for split in dataset.values():
        for item in split:
            text = item.get("text", "") or item.get("content", "")
            if not text or len(text) < 50:
                continue

            label = item.get("label", item.get("is_ai_generated", 0))
            if isinstance(label, str):
                label = 1 if label.lower() in {"ai", "ai_generated", "1"} else 0

            texts.append(text)
            labels.append(int(label))

    if sample_limit and len(texts) > sample_limit:
        texts, labels = zip(
            *list(zip(texts, labels, strict=False))[:sample_limit],
            strict=False,
        )
        texts = list(texts)
        labels = list(labels)

    return texts, labels


def train_raid_detector(
    model_name: str = "roberta-base",
    output_dir: str = "models/raid_roberta",
    sample_limit: int | None = None,
    num_train_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    cache_dir: str | None = None,
):
    print("Loading RAID dataset...")
    texts, labels = load_raid_dataset(sample_limit=sample_limit, cache_dir=cache_dir)
    print(f"Loaded {len(texts)} samples")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=512
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=512
    )

    train_dataset = Dataset.from_dict(
        {
            "input_ids": train_encodings["input_ids"],
            "attention_mask": train_encodings["attention_mask"],
            "labels": train_labels,
        }
    )
    val_dataset = Dataset.from_dict(
        {
            "input_ids": val_encodings["input_ids"],
            "attention_mask": val_encodings["attention_mask"],
            "labels": val_labels,
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on RAID dataset")
    parser.add_argument("--model", default="roberta-base", help="Base model name")
    parser.add_argument("--output", default="models/raid_roberta", help="Output directory")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache directory")

    args = parser.parse_args()

    train_raid_detector(
        model_name=args.model,
        output_dir=args.output,
        sample_limit=args.sample_limit,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        cache_dir=args.cache_dir,
    )
