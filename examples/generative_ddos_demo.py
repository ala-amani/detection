"""Run a synthetic DDoS-alert narrative, offline or through the OpenAI API."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from src.explainability.generative_explainer import (
    AnalystNarrative,
    AttackEvent,
    FeatureEvidence,
    OpenAINarrativeClient,
    explain_if_attack,
)


class OfflineDemoClient:
    """Deterministic stand-in for demonstrations without an API key."""

    def explain(self, event: AttackEvent, language: str = "Persian") -> AnalystNarrative:
        return AnalystNarrative(
            title="Potential DDoS Alert",
            summary=(
                "The model classified this flow as DDoS with 92 percent confidence. "
                "This analytical alert alone does not prove that an attack occurred."
            ),
            observed_signs=[
                "Packet length was 1514 bytes, compared with a reference value of 420 bytes.",
                "Packet rate was 18,500 packets per second and positively influenced the DDoS prediction.",
                "Flow duration was 0.18 seconds and influenced the prediction toward DDoS.",
            ],
            reasoning=(
                "The combination of a high packet rate, packet length above the reference, and a short flow "
                "moved the model toward the DDoS class. This explanation reflects feature attribution and "
                "does not establish a causal relationship."
            ),
            confidence_note="The 92 percent value is model confidence, not confirmation of a real attack.",
            recommended_checks=[
                "Review source connection and packet rates around the alert time.",
                "Compare destinations, ports, and source addresses with the network baseline.",
                "Correlate the alert with router, firewall, and destination-host logs before blocking.",
            ],
            limitations="This example uses synthetic data and requires validation with real traffic before operational use.",
        )


def synthetic_event() -> AttackEvent:
    return AttackEvent(
        predicted_class="DDoS",
        confidence=0.92,
        flow_id="synthetic-flow-001",
        evidence=[
            FeatureEvidence("packet_length", 1514, 0.41, "supports DDoS", 420),
            FeatureEvidence("packets_per_second", 18500, 0.34, "supports DDoS", 850),
            FeatureEvidence("flow_duration_seconds", 0.18, 0.19, "supports DDoS", 4.7),
        ],
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call OpenAI instead of the offline demo")
    args = parser.parse_args()
    client = OpenAINarrativeClient() if args.live else OfflineDemoClient()
    narrative = explain_if_attack(synthetic_event(), client)
    print(json.dumps(asdict(narrative), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
