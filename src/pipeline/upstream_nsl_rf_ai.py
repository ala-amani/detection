"""Run the published NSL-KDD RF code, then explain an attack with the AI layer.

The upstream script is executed unchanged through its first Random Forest
prediction block. Only notebook-era imports that are unused by that block
(TensorFlow/Keras/seaborn) are skipped so the published preprocessing and model
code remain the code being exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import shap

from src.explainability.generative_explainer import (
    AttackEvent,
    FeatureEvidence,
    OllamaNarrativeClient,
    OpenAINarrativeClient,
    explain_if_attack,
)


CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
SKIPPED_IMPORT_PREFIXES = (
    "import seaborn",
    "import tensorflow",
    "from tensorflow",
    "from keras",
)


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def execute_published_prediction_block(script: Path, data_root: Path) -> dict:
    lines = script.read_text(encoding="utf-8").splitlines()
    # Line 436 is the end of the first published RF train/predict block.
    published_block = lines[:436]
    compatible_block = [
        line for line in published_block
        if not line.strip().startswith(SKIPPED_IMPORT_PREFIXES)
    ]
    namespace: dict = {"__name__": "xai_nids_published_rf_block"}
    with working_directory(data_root):
        exec(compile("\n".join(compatible_block), str(script), "exec"), namespace)
    return namespace


def event_from_published_model(namespace: dict, top_k: int = 5) -> tuple[AttackEvent, int, str]:
    model = namespace["multi_target_rf"]
    X_train = namespace["X_train"]
    X_test = namespace["X_test"]
    predictions = np.asarray(namespace["preds"])

    predicted_indices = np.argmax(predictions, axis=1)
    candidates = np.flatnonzero(predicted_indices != 0)
    if not len(candidates):
        raise RuntimeError("The published model produced no attack prediction")
    row = int(candidates[0])
    class_index = int(predicted_indices[row])
    classifier = model.estimators_[class_index]
    sample = X_test.iloc[[row]]
    probabilities = classifier.predict_proba(sample)[0]
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    confidence = float(probabilities[positive_index])

    explanation = shap.TreeExplainer(classifier)(sample)
    values = np.asarray(explanation.values)
    contributions = values[0, :, positive_index] if values.ndim == 3 else values[0]
    observed = sample.iloc[0].to_numpy(dtype=float)
    baseline = X_train.mean(axis=0).to_numpy(dtype=float)
    ranked = np.argsort(np.abs(contributions))[::-1][:top_k]
    evidence = [
        FeatureEvidence(
            feature=str(X_test.columns[i]),
            observed_value=float(observed[i]),
            contribution=float(contributions[i]),
            direction="supports prediction" if contributions[i] >= 0 else "opposes prediction",
            baseline=float(baseline[i]),
        )
        for i in ranked
    ]
    actual_index = int(np.argmax(np.asarray(namespace["Y_test"])[row]))
    return AttackEvent(
        predicted_class=CLASS_NAMES[class_index],
        confidence=confidence,
        evidence=evidence,
        flow_id=f"published-nsl-kdd-test-{row}",
    ), row, CLASS_NAMES[actual_index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--upstream-script",
        type=Path,
        default=Path("XAI_NIDS/NSL-KDD/RF_ALL_FINAL.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--live-ai", action="store_true", help="Use the OpenAI API")
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama instead of OpenAI")
    parser.add_argument("--ollama-model", default="qwen3:4b")
    args = parser.parse_args()

    namespace = execute_published_prediction_block(args.upstream_script.resolve(), args.data_root.resolve())
    event, row, actual_class = event_from_published_model(namespace)
    result = {
        "source": str(args.upstream_script),
        "published_code_executed_through_line": 436,
        "test_row": row,
        "actual_class": actual_class,
        "event": asdict(event),
        "narrative": None,
    }
    if args.live_ai and args.ollama:
        parser.error("Choose either --live-ai or --ollama, not both")
    if args.live_ai or args.ollama:
        client = OllamaNarrativeClient(model=args.ollama_model) if args.ollama else OpenAINarrativeClient()
        narrative = explain_if_attack(event, client)
        result["narrative"] = asdict(narrative) if narrative else None
        if isinstance(client, OllamaNarrativeClient):
            result["generation"] = client.last_metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
