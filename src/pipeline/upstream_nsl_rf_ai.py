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
from src.validation.human_feedback import AlertValidator, FeedbackStore, ValidationAssessment


CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
SKIPPED_IMPORT_PREFIXES = (
    "import seaborn",
    "import tensorflow",
    "from tensorflow",
    "from keras",
)


def _legacy_show_attack_alert(event: AttackEvent, narrative) -> None:
    """Show a readable analyst alert only after the AI response is received."""
    import tkinter as tk
    from tkinter import messagebox

    reasons = "\n".join(f"- {item}" for item in narrative.observed_signs)
    message = (
        f"Detected attack class: {event.predicted_class}\n"
        f"Model confidence: {event.confidence * 100:.2f} percent\n\n"
        f"AI explanation:\n{narrative.summary}\n\n"
        f"Relevant evidence:\n{reasons}\n\n"
        f"Confidence assessment:\n{narrative.confidence_note}\n\n"
        f"Limitation:\n{narrative.limitations}"
    )
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showwarning("Intrusion Detection Alert - AI Explanation", message, parent=root)
    root.destroy()


def show_attack_alert(
    event: AttackEvent,
    narrative,
    client,
    validator: AlertValidator,
    assessment: ValidationAssessment,
    model_features: dict[str, float],
) -> None:
    """Show validation, collect human feedback, and support grounded chat."""
    import tkinter as tk
    import threading

    attack_meanings = {
        "DoS": "an attempt to disrupt or exhaust the availability of a service",
        "Probe": "network reconnaissance or scanning for hosts, ports, or services",
        "R2L": "an attempt to obtain local access from a remote system",
        "U2R": "an attempt by a local user to gain administrator privileges",
    }
    feature_meanings = {
        "dst_host_srv_count": "recent connections to the same service on the destination host",
        "dst_host_rerror_rate": "rate of rejected-connection errors at the destination host",
        "dst_host_diff_srv_rate": "rate of connections to different services on the destination host",
        "diff_srv_rate": "rate of connections to different services in the recent traffic window",
        "dst_host_same_src_port_rate": "rate at which the same source port is reused",
    }
    supporting, opposing = [], []
    for item in event.evidence:
        meaning = feature_meanings.get(item.feature, item.feature.replace("_", " "))
        relative = "above" if item.observed_value > item.baseline else "below"
        line = (
            f"- {item.feature}: {meaning}. Its standardized value was "
            f"{item.observed_value:.3f}, {relative} the training average "
            f"({item.baseline:.3f}); SHAP impact {item.contribution:+.4f}."
        )
        (supporting if item.contribution >= 0 else opposing).append(line)

    confidence = event.confidence * 100
    certainty = "low and borderline" if confidence < 60 else "moderate" if confidence < 80 else "high"
    message = (
        "AI-ASSISTED INTRUSION ALERT\n\n"
        f"Detected class: {event.predicted_class}\n"
        f"Meaning: {attack_meanings.get(event.predicted_class, 'potential malicious network activity')}\n"
        f"Model confidence: {confidence:.2f}% ({certainty})\n\n"
        "INDEPENDENT ALERT VALIDATION\n"
        f"Reliability score: {assessment.reliability_score:.3f}\n"
        f"Validation verdict: {assessment.verdict.replace('_', ' ')}\n"
        f"Evidence agreement: {assessment.evidence_agreement:.1%}\n"
        f"Historical precision: {assessment.historical_precision:.1%} "
        f"from {assessment.reviewed_alerts} reviewed alerts\n"
        f"Human-review priority: {assessment.review_priority}\n\n"
        "WHY THE MODEL MADE THIS DECISION\n"
        "Evidence supporting the prediction:\n"
        + ("\n".join(supporting) if supporting else "- No strong supporting feature was found.")
        + "\n\nEvidence opposing the prediction:\n"
        + ("\n".join(opposing) if opposing else "- No opposing feature was found.")
        + "\n\nINTERPRETATION\n"
        f"The classifier labeled this flow as {event.predicted_class}, but the {confidence:.2f}% "
        "confidence means the result is uncertain. Positive SHAP values pushed the model toward "
        "this attack class; negative values pushed it away. This is an alert for analyst review, "
        "not proof of an attack.\n\n"
        "RECOMMENDED CHECKS\n"
        "- Inspect the source and destination IPs, ports, timestamps, and packet capture.\n"
        "- Check whether the same source contacted many ports, hosts, or services in a short period.\n"
        "- Compare the event with firewall, router, and authentication logs.\n"
        "- Do not automatically disconnect or block the system based only on this prediction.\n\n"
        "The LLM interpretation shown above is advisory. The analyst must read it together with "
        "the displayed XAI evidence and operational records, decide whether the IDS prediction is "
        "correct, a false positive, or the wrong attack class, and record the reviewed outcome below. "
        "Validated feedback is stored for controlled IDS retraining; it does not update firewall policy."
    )
    root = tk.Tk()
    root.title("AI-Assisted Intrusion Detection Chat")
    root.geometry("960x760")
    text_widget = tk.Text(root, wrap="word", font=("Segoe UI", 11), padx=20, pady=20)
    text_widget.insert("1.0", "AI:\n" + message + "\n\n")
    text_widget.configure(state="disabled")
    text_widget.pack(fill="both", expand=True, padx=12, pady=(12, 6))

    controls = tk.Frame(root)
    controls.pack(fill="x", padx=12, pady=(6, 12))
    entry = tk.Entry(controls, font=("Segoe UI", 11))
    entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=7)
    history: list[dict[str, str]] = []

    def append_chat(speaker: str, content: str) -> None:
        text_widget.configure(state="normal")
        text_widget.insert("end", f"{speaker}:\n{content}\n\n")
        text_widget.configure(state="disabled")
        text_widget.see("end")

    def finish_answer(question: str, answer: str) -> None:
        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        append_chat("AI", answer)
        send_button.configure(state="normal", text="Send")
        entry.configure(state="normal")
        entry.focus_set()

    def report_error(error: Exception) -> None:
        append_chat("System", f"The AI request failed: {error}")
        send_button.configure(state="normal", text="Send")
        entry.configure(state="normal")

    def ask_ai(question: str) -> None:
        try:
            if not hasattr(client, "chat"):
                raise RuntimeError("Interactive chat is currently available with --ollama.")
            answer = client.chat(event, assessment, history, question)
            root.after(0, finish_answer, question, answer)
        except Exception as exc:
            root.after(0, report_error, exc)

    def send_question(event_arg=None) -> None:
        question = entry.get().strip()
        if not question:
            return
        entry.delete(0, "end")
        append_chat("You", question)
        append_chat("System", "AI is analyzing the event evidence...")
        send_button.configure(state="disabled", text="Thinking...")
        entry.configure(state="disabled")
        threading.Thread(target=ask_ai, args=(question,), daemon=True).start()

    send_button = tk.Button(controls, text="Send", command=send_question, width=12)
    send_button.pack(side="left")
    tk.Button(controls, text="Close", command=root.destroy, width=12).pack(side="left", padx=(8, 0))

    feedback_controls = tk.Frame(root)
    feedback_controls.pack(fill="x", padx=12, pady=(0, 12))

    feedback_buttons: list[tk.Button] = []

    def ask_analyst_reason() -> str | None:
        """Collect a readable multi-line justification from the analyst."""
        dialog = tk.Toplevel(root)
        dialog.title("Analyst reason")
        dialog.geometry("720x300")
        dialog.transient(root)
        dialog.grab_set()
        tk.Label(
            dialog,
            text="Explain why the IDS prediction is correct or incorrect:",
            font=("Segoe UI", 11),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(18, 8))
        reason_box = tk.Text(dialog, wrap="word", font=("Segoe UI", 11), height=8)
        reason_box.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        result: dict[str, str | None] = {"value": None}

        def accept() -> None:
            value = reason_box.get("1.0", "end").strip()
            if value:
                result["value"] = value
                dialog.destroy()

        buttons = tk.Frame(dialog)
        buttons.pack(fill="x", padx=18, pady=(0, 16))
        tk.Button(buttons, text="Save feedback", command=accept, width=16).pack(side="right")
        tk.Button(buttons, text="Cancel", command=dialog.destroy, width=12).pack(
            side="right", padx=(0, 8)
        )
        reason_box.focus_set()
        dialog.bind("<Control-Return>", lambda _event: accept())
        root.wait_window(dialog)
        return result["value"]

    def save_feedback(outcome: str) -> None:
        nonlocal assessment
        from tkinter import messagebox, simpledialog

        corrected_class = None
        if outcome == "wrong_attack_class":
            corrected_class = simpledialog.askstring(
                "Correct attack class", "Enter the analyst-confirmed attack class:", parent=root
            )
            if not corrected_class:
                return
        reason = ask_analyst_reason()
        if not reason:
            return
        validator.record_feedback(
            event,
            assessment,
            outcome,
            analyst_reason=reason,
            corrected_class=corrected_class,
            model_features=model_features,
        )
        updated = validator.assess(event)
        assessment = updated
        for button in feedback_buttons:
            button.configure(state="disabled")
        append_chat(
            "Feedback",
            f"Recorded outcome: {outcome.replace('_', ' ')}. "
            f"Updated class history: {updated.reviewed_alerts} reviewed alerts; "
            f"historical precision {updated.historical_precision:.1%}. "
            "The reviewed feature row and label were added to the IDS feedback store for controlled "
            "candidate retraining and holdout evaluation. No firewall rule was changed.",
        )
        messagebox.showinfo("Feedback saved", "Analyst feedback was saved successfully.", parent=root)

    tk.Label(feedback_controls, text="Analyst-confirmed outcome:").pack(side="left", padx=(0, 8))
    correct_button = tk.Button(
        feedback_controls, text="Correct", command=lambda: save_feedback("correct")
    )
    correct_button.pack(side="left")
    false_positive_button = tk.Button(
        feedback_controls, text="False positive", command=lambda: save_feedback("false_positive")
    )
    false_positive_button.pack(side="left", padx=(8, 0))
    wrong_class_button = tk.Button(
        feedback_controls,
        text="Wrong attack class",
        command=lambda: save_feedback("wrong_attack_class"),
    )
    wrong_class_button.pack(side="left", padx=(8, 0))
    feedback_buttons.extend([correct_button, false_positive_button, wrong_class_button])
    entry.bind("<Return>", send_question)
    entry.focus_set()
    root.mainloop()


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


def event_from_published_model(
    namespace: dict, top_k: int = 5, row: int | None = None
) -> tuple[AttackEvent, int, str, dict[str, float]]:
    model = namespace["multi_target_rf"]
    X_train = namespace["X_train"]
    X_test = namespace["X_test"]
    predictions = np.asarray(namespace["preds"])

    predicted_indices = np.argmax(predictions, axis=1)
    if row is None:
        candidates = np.flatnonzero(predicted_indices != 0)
        if not len(candidates):
            raise RuntimeError("The published model produced no attack prediction")
        row = int(candidates[0])
    if row < 0 or row >= len(predicted_indices):
        raise IndexError(f"Test row {row} is outside the available prediction range")
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
    event = AttackEvent(
        predicted_class=CLASS_NAMES[class_index],
        confidence=confidence,
        evidence=evidence,
        flow_id=f"published-nsl-kdd-test-{row}",
    )
    model_features = {
        str(name): float(value) for name, value in sample.iloc[0].items()
    }
    return event, row, CLASS_NAMES[actual_index], model_features


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
    parser.add_argument(
        "--test-row", type=int,
        help="Use a specific NSL-KDD test row for a reproducible analyst-review demonstration",
    )
    parser.add_argument("--show-alert", action="store_true", help="Show the received AI explanation in a popup")
    parser.add_argument(
        "--feedback-store", type=Path, default=Path("feedback/analyst_feedback.jsonl"),
        help="Append-only analyst feedback file used by the alert validator",
    )
    args = parser.parse_args()

    namespace = execute_published_prediction_block(args.upstream_script.resolve(), args.data_root.resolve())
    event, row, actual_class, model_features = event_from_published_model(
        namespace, row=args.test_row
    )
    validator = AlertValidator(FeedbackStore(args.feedback_store))
    assessment = validator.assess(event)
    result = {
        "source": str(args.upstream_script),
        "published_code_executed_through_line": 436,
        "test_row": row,
        "actual_class": actual_class,
        "event": asdict(event),
        "validation": asdict(assessment),
        "narrative": None,
    }
    if args.live_ai and args.ollama:
        parser.error("Choose either --live-ai or --ollama, not both")
    if args.live_ai or args.ollama:
        client = OllamaNarrativeClient(model=args.ollama_model) if args.ollama else OpenAINarrativeClient()
        narrative = explain_if_attack(event, client, validation=assessment)
        result["narrative"] = asdict(narrative) if narrative else None
        if isinstance(client, OllamaNarrativeClient):
            result["generation"] = client.last_metrics
        if args.show_alert and narrative is not None:
            show_attack_alert(event, narrative, client, validator, assessment, model_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
