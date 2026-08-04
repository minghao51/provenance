"""Training script for fine-tuning RoBERTa on RAID dataset."""

from __future__ import annotations

import argparse
import json
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


def normalize_raid_label(label: str | int) -> int:
    if isinstance(label, int):
        return label
    label_str = str(label).strip().lower()
    if label_str in {"human", "0"}:
        return 0
    return 1


def compute_class_weights(labels: list[int]) -> np.ndarray:
    import numpy as np

    labels_arr = np.array(labels)
    classes, counts = np.unique(labels_arr, return_counts=True)
    total = len(labels_arr)
    weights = total / (len(classes) * counts)
    result = np.zeros(len(classes), dtype=np.float64)
    for cls, w in zip(classes, weights):
        result[cls] = w
    return result


def augment_training_texts(
    texts: list[str],
    labels: list[int],
    enabled: bool = True,
) -> tuple[list[str], list[int]]:
    import random

    import numpy as np

    if not enabled:
        return texts, labels

    augmented_texts = list(texts)
    augmented_labels = list(labels)

    for text, label in zip(texts, labels):
        words = text.split()
        if len(words) < 30:
            continue

        for _ in range(2):
            aug_words = list(words)
            n_swap = max(1, len(aug_words) // 10)
            for _ in range(n_swap):
                i, j = random.sample(range(len(aug_words)), 2)
                aug_words[i], aug_words[j] = aug_words[j], aug_words[i]
            augmented_texts.append(" ".join(aug_words))
            augmented_labels.append(label)

    return augmented_texts, augmented_labels


def prepare_raid_splits(
    texts: list[str],
    labels: list[int],
    eval_size: float = 0.2,
    seed: int = 42,
    augment: bool = True,
):
    import numpy as np
    from collections import namedtuple

    PreparedSplits = namedtuple(
        "PreparedSplits",
        ["train_texts", "val_texts", "train_labels", "val_labels", "class_weights", "metadata"],
    )

    if augment:
        texts, labels = augment_training_texts(texts, labels, enabled=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=eval_size, random_state=seed, stratify=labels
    )

    class_weights = compute_class_weights(train_labels)

    metadata = {
        "augmentation_enabled": augment,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "total_samples": len(texts),
    }

    return PreparedSplits(
        train_texts=train_texts,
        val_texts=val_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        class_weights=class_weights,
        metadata=metadata,
    )


def compute_classification_metrics(eval_pred: tuple) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
    }


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
        paired = list(zip(texts, labels, strict=False))[:sample_limit]
        texts = [t for t, _ in paired]
        labels = [label for _, label in paired]

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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=512
    )
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

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
        eval_strategy="epoch",
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
    parser.add_argument(
        "--output", default="models/raid_roberta", help="Output directory"
    )
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
