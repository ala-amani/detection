"""Compact, leakage-safe reference implementation of the XAI-NIDS baseline.

Datasets are intentionally external. Supply a CSV plus its label column; the
pipeline trains the seven classifier families discussed in the base paper.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODELS = ("rf", "ada", "dnn", "svm", "knn", "mlp", "lightgbm")


@dataclass(frozen=True)
class ExperimentConfig:
    csv: Path
    label: str
    model: str = "rf"
    test_size: float = 0.30
    seed: int = 42
    max_rows: int | None = None
    explain_rows: int = 20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(config.csv, nrows=config.max_rows)
    if config.label not in frame.columns:
        raise ValueError(f"Label column {config.label!r} is absent")
    frame = frame.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    y = frame.pop(config.label).astype(str)
    return frame, y


def split_data(
    X: pd.DataFrame, y: pd.Series, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(
        X, y, test_size=config.test_size, random_state=config.seed, stratify=stratify
    )


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        verbose_feature_names_out=False,
    )


def make_estimator(name: str, seed: int) -> Any:
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=2, random_state=seed
        )
    if name == "ada":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=seed),
            n_estimators=50,
            learning_rate=1.0,
            random_state=seed,
        )
    if name == "svm":
        return SVC(C=0.5, kernel="linear", gamma=0.5, probability=True, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto")
    if name == "mlp":
        return _KerasDNN(seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier()
    if name == "dnn":
        return _KerasDNN(seed)
    raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")


class _KerasDNN:
    """Small scikit-learn-compatible multiclass dense network."""

    def __init__(self, seed: int = 42, epochs: int = 11, batch_size: int = 1024):
        self.seed, self.epochs, self.batch_size = seed, epochs, batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_KerasDNN":
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        self.classes_ = np.unique(y)
        self.encoder_ = LabelEncoder().fit(self.classes_)
        encoded = self.encoder_.transform(y)
        targets = tf.keras.utils.to_categorical(encoded, num_classes=len(self.classes_))
        self.model_ = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(X.shape[1], activation="relu"),
                tf.keras.layers.Dropout(0.01),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(len(self.classes_), activation="softmax"),
            ]
        )
        self.model_.compile(optimizer="adam", loss="categorical_crossentropy")
        self.model_.fit(X, targets, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.encoder_.inverse_transform(indices)


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predicted = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="weighted", zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, predicted),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "mcc": matthews_corrcoef(y_test, predicted),
        "classes": model.named_steps["classifier"].classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
    }
    if hasattr(model, "predict_proba") and len(metrics["classes"]) > 1:
        probabilities = model.predict_proba(X_test)
        truth = label_binarize(y_test, classes=metrics["classes"])
        try:
            metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                truth, probabilities, average="weighted", multi_class="ovr"
            )
        except ValueError:
            metrics["roc_auc_ovr_weighted"] = None
    return metrics


def explain_shap(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, rows: int) -> Any:
    import shap

    transform = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    background = shap.sample(transform.transform(X_train), min(100, len(X_train)), random_state=0)
    samples = transform.transform(X_test.iloc[:rows])
    if classifier.__class__.__name__ in {"RandomForestClassifier", "LGBMClassifier"}:
        return shap.TreeExplainer(classifier)(samples)
    return shap.Explainer(classifier.predict_proba, background)(samples)


def explain_lime(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, row: int = 0) -> Any:
    from lime.lime_tabular import LimeTabularExplainer

    transform = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    train_array = transform.transform(X_train)
    test_array = transform.transform(X_test)
    feature_names = transform.get_feature_names_out().tolist()
    explainer = LimeTabularExplainer(
        train_array,
        feature_names=feature_names,
        class_names=classifier.classes_.astype(str).tolist(),
        mode="classification",
        random_state=0,
    )
    return explainer.explain_instance(test_array[row], classifier.predict_proba)


def run(config: ExperimentConfig) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    seed_everything(config.seed)
    X, y = load_dataset(config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    model = Pipeline(
        [("preprocess", make_preprocessor(X_train)),
         ("classifier", make_estimator(config.model, config.seed))]
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    metrics["fit_seconds"] = time.perf_counter() - started
    return model, metrics, X_train, X_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", choices=MODELS, default="rf")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()
    config = ExperimentConfig(args.csv, args.label, args.model, args.test_size,
                              args.seed, args.max_rows)
    _, metrics, _, _ = run(config)
    args.output.write_text(json.dumps({"config": asdict(config), "metrics": metrics},
                                     indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
