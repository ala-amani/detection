"""Generate a confusion matrix and two human-feedback examples from NSL-KDD."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.pipeline.upstream_nsl_rf_ai import (
    CLASS_NAMES,
    event_from_published_model,
    execute_published_prediction_block,
)
from src.validation.human_feedback import AlertValidator, FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--upstream-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feedback-store", required=True, type=Path)
    args = parser.parse_args()

    namespace = execute_published_prediction_block(
        args.upstream_script.resolve(), args.data_root.resolve()
    )
    predicted = np.argmax(np.asarray(namespace["preds"]), axis=1)
    actual = np.argmax(np.asarray(namespace["Y_test"]), axis=1)
    matrix = confusion_matrix(actual, predicted, labels=range(len(CLASS_NAMES)))

    false_positive_rows = np.flatnonzero((actual == 0) & (predicted != 0))
    wrong_class_rows = np.flatnonzero(
        (actual != 0) & (predicted != 0) & (actual != predicted)
    )
    if not len(false_positive_rows) or not len(wrong_class_rows):
        raise RuntimeError("The run did not produce both required error examples")

    validator = AlertValidator(FeedbackStore(args.feedback_store))
    examples = []
    for kind, row in (
        ("false_positive", int(false_positive_rows[0])),
        ("wrong_attack_class", int(wrong_class_rows[0])),
    ):
        event, _, actual_class, features = event_from_published_model(namespace, row=row)
        before = validator.assess(event)
        corrected = actual_class if kind == "wrong_attack_class" else None
        reason = (
            "The held-out benchmark label is Normal, so the attack alert is a false positive."
            if kind == "false_positive"
            else f"The held-out benchmark label is {actual_class}, not {event.predicted_class}."
        )
        feedback = validator.record_feedback(
            event,
            before,
            kind,
            analyst="benchmark-reviewer",
            analyst_reason=reason,
            corrected_class=corrected,
            model_features=features,
        )
        after = validator.assess(event)
        examples.append({
            "error_type": kind,
            "test_row": row,
            "actual_class": actual_class,
            "event": asdict(event),
            "validation_before_feedback": asdict(before),
            "human_feedback": asdict(feedback),
            "validation_after_feedback": asdict(after),
        })

    payload = {
        "class_order": CLASS_NAMES,
        "confusion_matrix": matrix.tolist(),
        "total_test_rows": int(len(actual)),
        "false_positive_count": int(len(false_positive_rows)),
        "wrong_attack_class_count": int(len(wrong_class_rows)),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
