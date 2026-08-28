"""Evaluate documented synthetic local-network flows with the published IDS."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.explainability.generative_explainer import FeatureEvidence, OllamaNarrativeClient, AttackEvent, explain_if_attack
from src.pipeline.upstream_nsl_rf_ai import CLASS_NAMES, execute_published_prediction_block, show_attack_alert
from src.validation.human_feedback import AlertValidator, FeedbackStore


CATEGORICAL_PREFIXES = ("Protocol_type_", "service_", "flag_")


def transform_flow(row: pd.Series, namespace: dict) -> pd.DataFrame:
    """Map a documented raw NSL-KDD-compatible row to the fitted RF feature space."""
    training_raw = namespace["df"]
    columns = list(namespace["X_train"].columns)
    transformed = {name: 0.0 for name in columns}
    for name in training_raw.select_dtypes(include="number").columns:
        if name not in transformed or name not in row:
            continue
        values = training_raw[name].to_numpy(dtype=float)
        scale = float(values.std()) or 1.0
        transformed[name] = (float(row[name]) - float(values.mean())) / scale
    categorical = {
        f"Protocol_type_{row['protocol_type']}",
        f"service_{row['service']}",
        f"flag_{row['flag']}",
    }
    for name in categorical:
        if name in transformed:
            transformed[name] = 1.0
    return pd.DataFrame([[transformed[name] for name in columns]], columns=columns)


def evaluate_flow(row: pd.Series, namespace: dict, top_k: int = 5):
    model = namespace["multi_target_rf"]
    sample = transform_flow(row, namespace)
    binary_predictions = np.asarray(model.predict(sample.values))[0]
    class_index = int(np.argmax(binary_predictions))
    classifier = model.estimators_[class_index]
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    confidence = float(classifier.predict_proba(sample.values)[0][positive_index])
    explanation = shap.TreeExplainer(classifier)(sample)
    values = np.asarray(explanation.values)
    contributions = values[0, :, positive_index] if values.ndim == 3 else values[0]
    observed = sample.iloc[0].to_numpy(dtype=float)
    baseline = namespace["X_train"].mean(axis=0).to_numpy(dtype=float)
    ranked = np.argsort(np.abs(contributions))[::-1][:top_k]
    evidence = [
        FeatureEvidence(str(sample.columns[i]), float(observed[i]), float(contributions[i]),
                        "supports prediction" if contributions[i] >= 0 else "opposes prediction",
                        float(baseline[i]))
        for i in ranked
    ]
    event = AttackEvent(CLASS_NAMES[class_index], confidence, evidence, f"synthetic-{row['scenario']}")
    return event, {str(name): float(value) for name, value in sample.iloc[0].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--upstream-script", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feedback-store", required=True, type=Path)
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--show-alert", action="store_true")
    args = parser.parse_args()
    flows = pd.read_csv(args.flows)
    matches = flows if args.scenario == "all" else flows[flows["scenario"] == args.scenario]
    if matches.empty:
        raise ValueError(f"No scenario named {args.scenario!r}")
    namespace = execute_published_prediction_block(args.upstream_script.resolve(), args.data_root.resolve())
    validator = AlertValidator(FeedbackStore(args.feedback_store))
    client = OllamaNarrativeClient() if args.ollama else None
    results = []
    interactive = None
    for _, row in matches.iterrows():
        event, model_features = evaluate_flow(row, namespace)
        assessment = validator.assess(event)
        narrative = explain_if_attack(event, client, validation=assessment) if client else None
        results.append({
            "data_status": "synthetic_offline_lab",
            "scenario": row["scenario"],
            "synthetic_label": row["synthetic_label"],
            "event": asdict(event),
            "validation": asdict(assessment),
            "narrative": asdict(narrative) if narrative else None,
        })
        if args.show_alert:
            interactive = (event, model_features, assessment, narrative)
    result = results if args.scenario == "all" else results[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if interactive and interactive[3]:
        event, model_features, assessment, narrative = interactive
        show_attack_alert(event, narrative, client, validator, assessment, model_features)


if __name__ == "__main__":
    main()
