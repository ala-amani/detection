"""Run a fresh Ollama explanation and save a readable English proof report."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.explainability.generative_explainer import (
    AttackEvent,
    FeatureEvidence,
    OllamaNarrativeClient,
    explain_if_attack,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_json", type=Path)
    parser.add_argument("output_txt", type=Path)
    parser.add_argument("--model", default="qwen3:4b")
    args = parser.parse_args()

    saved = json.loads(args.event_json.read_text(encoding="utf-8"))
    raw = saved["event"]
    event = AttackEvent(
        predicted_class=raw["predicted_class"],
        confidence=raw["confidence"],
        evidence=[FeatureEvidence(**item) for item in raw["evidence"]],
        flow_id=raw.get("flow_id"),
    )

    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    print("LIVE AI CALL STARTED - waiting for Ollama...", flush=True)
    client = OllamaNarrativeClient(model=args.model)
    narrative = explain_if_attack(event, client)
    elapsed = time.perf_counter() - started
    finished_at = datetime.now().astimezone()

    lines = [
        "LIVE LOCAL AI TEST REPORT",
        "=================================",
        "",
        "Status: the live AI call completed successfully.",
        f"Started: {started_at.isoformat(timespec='seconds')}",
        f"Response received: {finished_at.isoformat(timespec='seconds')}",
        f"Call duration: {elapsed:.2f} seconds",
        f"Provider: {client.last_metrics['provider']}",
        f"Model: {client.last_metrics['model']}",
        f"Prompt tokens: {client.last_metrics['prompt_tokens']}",
        f"Generated tokens: {client.last_metrics['generated_tokens']}",
        "",
        "INTRUSION DETECTION OUTPUT",
        "----------------------",
        f"Predicted attack class: {event.predicted_class}",
        f"Model confidence: {event.confidence * 100:.2f} percent",
        f"Ground-truth class: {saved.get('actual_class', 'not recorded')}",
        "",
        "LIVE AI RESPONSE",
        "--------------",
        f"Title: {narrative.title}",
        f"Summary: {narrative.summary}",
        "",
        "Observed signs:",
        *[f"- {item}" for item in narrative.observed_signs],
        "",
        f"Reasoning: {narrative.reasoning}",
        f"Confidence note: {narrative.confidence_note}",
        "",
        "Recommended checks:",
        *[f"- {item}" for item in narrative.recommended_checks],
        "",
        f"Limitation: {narrative.limitations}",
        "",
        "Scientific note: this is raw local-model output and must be checked for numerical and technical consistency.",
    ]
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"LIVE AI CALL FINISHED in {elapsed:.2f}s", flush=True)
    print(str(args.output_txt), flush=True)


if __name__ == "__main__":
    main()
