"""Make a fresh, visible Ollama call from a previously detected attack event."""

from __future__ import annotations

import argparse
import json
import time
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

    print("LIVE CALL: sending SHAP evidence to Ollama at http://127.0.0.1:11434 ...", flush=True)
    started = time.perf_counter()
    client = OllamaNarrativeClient(model=args.model)
    narrative = explain_if_attack(event, client)
    elapsed = time.perf_counter() - started
    print(f"LIVE RESPONSE RECEIVED in {elapsed:.2f} seconds", flush=True)
    print(json.dumps({"narrative": narrative.__dict__, "generation": client.last_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
