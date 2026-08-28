"""Base-study dataset prediction -> local SHAP -> analyst narrative pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.baseline_reference import make_estimator, make_preprocessor
from src.data.nsl_kdd import load_nsl_kdd
from src.explainability.generative_explainer import (
    AttackEvent, FeatureEvidence, OpenAINarrativeClient, explain_if_attack,
)


def _class_shap_values(classifier: Any, sample: np.ndarray, class_index: int) -> np.ndarray:
    import shap

    explanation = shap.TreeExplainer(classifier)(sample)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        return values[0, :, class_index]
    if values.ndim == 2:
        return values[0]
    raise ValueError(f"Unexpected SHAP value shape: {values.shape}")


def build_event(
    pipeline: Pipeline,
    X_train,
    X_test,
    preferred_attack: str | None = "DoS",
    top_k: int = 5,
) -> tuple[AttackEvent, int, str]:
    predictions = pipeline.predict(X_test)
    candidates = np.flatnonzero(np.char.lower(predictions.astype(str)) != "normal")
    if preferred_attack:
        preferred = np.flatnonzero(predictions.astype(str) == preferred_attack)
        if len(preferred):
            candidates = preferred
    if not len(candidates):
        raise RuntimeError("No non-normal prediction was available for explanation")

    row = int(candidates[0])
    predicted = str(predictions[row])
    probabilities = pipeline.predict_proba(X_test.iloc[[row]])[0]
    classifier = pipeline.named_steps["classifier"]
    class_index = int(np.flatnonzero(classifier.classes_.astype(str) == predicted)[0])
    confidence = float(probabilities[class_index])

    transformer = pipeline.named_steps["preprocess"]
    transformed_train = transformer.transform(X_train)
    transformed_sample = transformer.transform(X_test.iloc[[row]])
    feature_names = transformer.get_feature_names_out().astype(str)
    contributions = _class_shap_values(classifier, transformed_sample, class_index)
    baseline = np.asarray(transformed_train).mean(axis=0)
    observed = np.asarray(transformed_sample)[0]
    ranked = np.argsort(np.abs(contributions))[::-1][:top_k]
    evidence = [
        FeatureEvidence(
            feature=feature_names[i],
            observed_value=float(observed[i]),
            contribution=float(contributions[i]),
            direction="supports prediction" if contributions[i] >= 0 else "opposes prediction",
            baseline=float(baseline[i]),
        )
        for i in ranked
    ]
    return AttackEvent(predicted, confidence, evidence, flow_id=f"nsl-kdd-test-{row}"), row, predicted


def run_nsl_kdd(
    data_root: Path,
    seed: int = 42,
    max_rows_per_file: int | None = None,
    live_ai: bool = False,
) -> dict[str, Any]:
    X, y = load_nsl_kdd(data_root, max_rows_per_file=max_rows_per_file)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y,
    )
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("classifier", make_estimator("rf", seed)),
    ])
    pipeline.fit(X_train, y_train)
    event, row, predicted = build_event(pipeline, X_train, X_test)
    result: dict[str, Any] = {
        "dataset": "NSL-KDD",
        "model": "Random Forest",
        "split": "70/30 stratified",
        "test_row": row,
        "actual_class": str(y_test.iloc[row]),
        "predicted_class": predicted,
        "event": asdict(event),
        "narrative": None,
    }
    if live_ai:
        narrative = explain_if_attack(event, OpenAINarrativeClient())
        result["narrative"] = asdict(narrative) if narrative else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows-per-file", type=int)
    parser.add_argument("--live-ai", action="store_true")
    args = parser.parse_args()
    result = run_nsl_kdd(args.data_root, max_rows_per_file=args.max_rows_per_file, live_ai=args.live_ai)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
