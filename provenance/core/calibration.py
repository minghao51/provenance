"""Calibration mixin for replacing heuristic scoring with learned models."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import joblib
import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    CalibratedClassifierCV = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


class CalibratedDetectorMixin:
    """Mixin that adds calibration support to any detector.

    Subclasses must implement:
        _extract_features(text) -> list[float]

    The mixin provides:
        calibrate(texts, labels) — train a calibrated classifier
        detect(text) — uses calibrated model if available, falls back to heuristic
        save_calibration(path) / load_calibration(path) — persistence
    """

    _calibrator: Pipeline | None = None
    _feature_names: list[str] | None = None

    @abstractmethod
    def _extract_features(self, text: str) -> list[float]:
        """Extract raw numeric features from text for calibration."""
        ...

    @abstractmethod
    def _extract_feature_names(self) -> list[str]:
        """Return the names of features produced by _extract_features."""
        ...

    def calibrate(
        self,
        texts: list[str],
        labels: list[int],
        method: str = "isotonic",
        cv: int = 5,
    ) -> None:
        """Train a calibrated classifier on labeled data.

        Args:
            texts: Training texts.
            labels: Binary labels (0=human, 1=ai).
            method: Calibration method ('isotonic' or 'sigmoid').
            cv: Number of cross-validation folds.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for calibration. Install with: pip install scikit-learn"
            )

        X = np.array([self._extract_features(t) for t in texts])  # noqa: N806
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            raise ValueError("Training data must contain both classes (0 and 1)")
        class_counts = np.bincount(y)
        min_class_count = int(class_counts[class_counts > 0].min())
        effective_cv = min(cv, min_class_count)
        if effective_cv < 2:
            raise ValueError(
                "Calibration requires at least 2 samples in each class for cross-validation"
            )

        base_estimator = LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        )

        self._calibrator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        base_estimator, method=method, cv=effective_cv
                    ),
                ),
            ]
        )
        self._calibrator.fit(X, y)
        self._feature_names = self._extract_feature_names()

    def _get_calibrated_score(self, text: str) -> tuple[float, float] | None:
        """Get calibrated score and confidence, or None if not calibrated."""
        if self._calibrator is None:
            return None

        features = np.array([self._extract_features(text)])
        proba = self._calibrator.predict_proba(features)[0]
        score = float(proba[1])
        confidence = float(max(proba))
        return score, confidence

    def save_calibration(self, path: str | Path) -> None:
        """Save calibration model to disk."""
        if self._calibrator is None:
            raise ValueError("No calibration model to save. Call calibrate() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "calibrator": self._calibrator,
                "feature_names": self._feature_names,
                "detector_name": getattr(self, "name", "unknown"),
            },
            path,
        )

    def load_calibration(self, path: str | Path) -> None:
        """Load calibration model from disk."""
        data = joblib.load(path)
        self._calibrator = data["calibrator"]
        self._feature_names = data["feature_names"]
