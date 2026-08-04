"""Tests for the RAID transformer training helpers."""

import numpy as np

from provenance.detectors.transformer.train_raid import (
    augment_training_texts,
    compute_class_weights,
    compute_classification_metrics,
    normalize_raid_label,
    prepare_raid_splits,
)


class TestTransformerTrainingHelpers:
    def test_normalize_raid_label_handles_strings(self):
        assert normalize_raid_label("human") == 0
        assert normalize_raid_label("ai") == 1
        assert normalize_raid_label("1") == 1

    def test_compute_class_weights_upweights_minority_class(self):
        weights = compute_class_weights([0, 0, 0, 1])

        assert len(weights) == 2
        assert weights[1] > weights[0]

    def test_augment_training_texts_adds_variants_for_long_text(self):
        text = " ".join(f"token{i}" for i in range(50))

        augmented_texts, augmented_labels = augment_training_texts(
            [text],
            [1],
            enabled=True,
        )

        assert len(augmented_texts) > 1
        assert augmented_labels == [1] * len(augmented_labels)

    def test_prepare_raid_splits_returns_metadata_and_weights(self):
        texts = [
            f"human sample {i} " + ("word " * 30)
            if i % 2 == 0
            else f"ai sample {i} " + ("word " * 30)
            for i in range(20)
        ]
        labels = [0 if i % 2 == 0 else 1 for i in range(20)]

        prepared = prepare_raid_splits(
            texts,
            labels,
            eval_size=0.25,
            seed=7,
            augment=True,
        )

        assert len(prepared.val_texts) > 0
        assert len(prepared.val_texts) < len(texts)
        assert prepared.metadata["augmentation_enabled"] is True
        assert len(prepared.class_weights) == 2
        assert prepared.metadata["train_samples"] > 0

    def test_compute_classification_metrics(self):
        logits = np.array([[3.0, 1.0], [0.1, 2.0], [0.2, 1.2], [1.0, 0.5]])
        labels = np.array([0, 1, 1, 0])

        metrics = compute_classification_metrics((logits, labels))

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
