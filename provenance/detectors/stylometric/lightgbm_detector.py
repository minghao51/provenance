"""LightGBM-based classifier with SHAP explainability."""

from __future__ import annotations

from pathlib import Path

import joblib

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.errors import DetectorInitError, ModelNotFoundError

try:
    import lightgbm as lgb
    import shap
except ImportError:
    lgb = None
    shap = None


class LightGBMDetector(BaseDetector):
    name = "lgbm_stylometric"
    latency_tier = "fast"
    domains = ["prose", "academic"]

    def __init__(
        self,
        model_path: str | None = None,
        feature_extractor=None,
    ):
        if lgb is None or shap is None:
            raise DetectorInitError("lightgbm and shap are required for LightGBMDetector")

        self.model = None
        self.explainer = None
        self.feature_names: list[str] = []
        self.feature_extractor = feature_extractor

        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise ModelNotFoundError(
                    f"LightGBM model file not found: {model_path}",
                    model_path=model_path,
                )
            self.load_model(model_path)
        else:
            self._init_default_model()

    def _init_default_model(self) -> None:
        from provenance.detectors.stylometric.feature_extractor import FeatureExtractor

        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()

        self.feature_names = self.feature_extractor.get_feature_names()

        dummy_data = lgb.Dataset(
            [[0.0] * len(self.feature_names)],
            feature_name=self.feature_names,
        )
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
        }
        self.model = lgb.train(params, dummy_data, num_iteration=1)

        self.explainer = shap.Explainer(self.model)

    def load_model(self, model_path: str) -> None:
        from provenance.detectors.stylometric.feature_extractor import FeatureExtractor

        path = Path(model_path)
        if not path.exists():
            raise ModelNotFoundError(
                f"LightGBM model file not found: {model_path}",
                model_path=model_path,
            )

        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()

        model_data = joblib.load(model_path)

        self.model = model_data["model"]
        self.explainer = shap.Explainer(self.model)
        self.feature_names = model_data.get(
            "feature_names", self.feature_extractor.get_feature_names()
        )

    def save_model(self, model_path: str) -> None:
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
        }
        joblib.dump(model_data, model_path)

    def detect(self, text: str) -> DetectorResult:
        if self.model is None or self.feature_extractor is None:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Model not initialized"},
            )

        features = self.feature_extractor.extract(text)
        vector = self.feature_extractor.to_vector(features)

        while len(vector) < len(self.feature_names):
            vector.append(0.0)
        vector = vector[: len(self.feature_names)]

        score = self.model.predict([vector])[0]
        score = float(score)

        shap_values = self.explainer([vector])
        shap_vals = shap_values.values[0].tolist()

        feature_importance = list(zip(self.feature_names, shap_vals, strict=False))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_importance[:10]

        confidence = 0.7 + 0.2 * abs(score - 0.5) * 2

        return DetectorResult(
            score=score,
            confidence=min(1.0, confidence),
            metadata={
                "features": features,
                "shap_values": dict(top_features),
                "top_features": top_features,
            },
        )

    def train(
        self,
        texts: list[str],
        labels: list[int],
        params: dict | None = None,
        use_optuna: bool = True,
        n_trials: int = 50,
    ) -> None:
        if self.feature_extractor is None:
            from provenance.detectors.stylometric.feature_extractor import (
                FeatureExtractor,
            )

            self.feature_extractor = FeatureExtractor()

        self.feature_names = self.feature_extractor.get_feature_names()

        feature_matrix = []
        for text in texts:
            features = self.feature_extractor.extract(text)
            vector = self.feature_extractor.to_vector(features)
            while len(vector) < len(self.feature_names):
                vector.append(0.0)
            feature_matrix.append(vector[: len(self.feature_names)])

        train_data = lgb.Dataset(
            feature_matrix,
            label=labels,
            feature_name=self.feature_names,
        )

        if use_optuna:
            try:
                import optuna

                def objective(trial):
                    optuna_params = {
                        "objective": "binary",
                        "metric": "auc",
                        "boosting_type": "gbdt",
                        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 0.01, 0.2
                        ),
                        "feature_fraction": trial.suggest_float(
                            "feature_fraction", 0.5, 1.0
                        ),
                        "bagging_fraction": trial.suggest_float(
                            "bagging_fraction", 0.5, 1.0
                        ),
                        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                        "min_child_samples": trial.suggest_int(
                            "min_child_samples", 5, 100
                        ),
                        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                        "verbose": -1,
                    }
                    if params:
                        optuna_params.update(
                            {k: v for k, v in params.items() if k not in optuna_params}
                        )

                    cv_results = lgb.cv(
                        optuna_params,
                        train_data,
                        nfold=5,
                        num_boost_round=200,
                        early_stopping_rounds=20,
                        verbose_eval=False,
                    )
                    return cv_results["auc-mean"][-1]

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                best_params = study.best_params
                best_params.update(
                    {
                        "objective": "binary",
                        "metric": "auc",
                        "boosting_type": "gbdt",
                        "verbose": -1,
                    }
                )
                if params:
                    best_params.update(
                        {k: v for k, v in params.items() if k not in best_params}
                    )

                self.model = lgb.train(
                    best_params,
                    train_data,
                    num_boost_round=study.best_trial.n_trials
                    if hasattr(study.best_trial, "n_trials")
                    else 100,
                )
            except ImportError:
                default_params = {
                    "objective": "binary",
                    "metric": "auc",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbose": -1,
                }
                if params:
                    default_params.update(params)
                self.model = lgb.train(default_params, train_data, num_iteration=100)
        else:
            default_params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            }
            if params:
                default_params.update(params)
            self.model = lgb.train(default_params, train_data, num_iteration=100)

        self.explainer = shap.Explainer(self.model)


def register(registry) -> None:
    if lgb is not None and shap is not None:
        registry.register(LightGBMDetector)
