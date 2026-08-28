"""Open interactive Ollama chat for a previously generated attack result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.explainability.generative_explainer import (
    AnalystNarrative,
    AttackEvent,
    FeatureEvidence,
    OllamaNarrativeClient,
)
from src.pipeline.upstream_nsl_rf_ai import show_attack_alert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--model", default="qwen3:4b")
    args = parser.parse_args()

    saved = json.loads(args.result.read_text(encoding="utf-8"))
    event_data = saved["event"]
    event = AttackEvent(
        predicted_class=event_data["predicted_class"],
        confidence=event_data["confidence"],
        flow_id=event_data.get("flow_id"),
        evidence=[FeatureEvidence(**item) for item in event_data["evidence"]],
    )
    narrative = AnalystNarrative(**saved["narrative"])
    show_attack_alert(event, narrative, OllamaNarrativeClient(model=args.model))


if __name__ == "__main__":
    main()
