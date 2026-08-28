"""Controlled IDS retraining from analyst-reviewed feedback.

Feedback must be accumulated in a separate review stream. A candidate model is
trained on the original training partition plus reviewed rows and is promoted
only after evaluation on an untouched holdout set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from src.validation.human_feedback import FeedbackStore


@dataclass(frozen=True)
class CandidateEvaluation:
    class_names: list[str]
    confusion_matrix: list[list[int]]
    false_positives: int
    wrong_attack_classes: int


def reviewed_training_rows(
    store: FeedbackStore,
    feature_names: list[str],
    class_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert reviewed JSONL records into aligned features and one-hot labels."""
    examples = store.training_examples()
    if not examples:
        return pd.DataFrame(columns=feature_names), np.empty((0, len(class_names)), dtype=int)
    rows: list[list[float]] = []
    labels: list[list[int]] = []
    for features, label in examples:
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Reviewed row is missing {len(missing)} model features")
        if label not in class_names:
            raise ValueError(f"Unsupported reviewed label: {label}")
        rows.append([float(features[name]) for name in feature_names])
        labels.append([int(name == label) for name in class_names])
    return pd.DataFrame(rows, columns=feature_names), np.asarray(labels, dtype=int)


def train_candidate(
    fitted_model,
    X_train: pd.DataFrame,
    Y_train,
    store: FeedbackStore,
    class_names: list[str],
):
    """Train a new candidate; never mutate or automatically promote the live IDS."""
    reviewed_X, reviewed_Y = reviewed_training_rows(store, list(X_train.columns), class_names)
    if reviewed_X.empty:
        raise ValueError("No analyst-reviewed examples are available for retraining")
    combined_X = pd.concat([X_train, reviewed_X], ignore_index=True)
    combined_Y = np.vstack([np.asarray(Y_train), reviewed_Y])
    candidate = clone(fitted_model)
    candidate.fit(combined_X, combined_Y)
    return candidate


def evaluate_candidate(model, X_holdout, Y_holdout, class_names: list[str]) -> CandidateEvaluation:
    """Evaluate a candidate on an untouched holdout before any deployment decision."""
    predicted = np.argmax(np.asarray(model.predict(X_holdout)), axis=1)
    actual = np.argmax(np.asarray(Y_holdout), axis=1)
    matrix = confusion_matrix(actual, predicted, labels=range(len(class_names)))
    false_positives = int(np.sum((actual == 0) & (predicted != 0)))
    wrong_classes = int(np.sum((actual != 0) & (predicted != 0) & (actual != predicted)))
    return CandidateEvaluation(
        class_names=class_names,
        confusion_matrix=matrix.astype(int).tolist(),
        false_positives=false_positives,
        wrong_attack_classes=wrong_classes,
    )
