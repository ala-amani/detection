import json

import pytest

from src.explainability.generative_explainer import (
    AnalystNarrative, AttackEvent, FeatureEvidence, OpenAINarrativeClient,
    explain_if_attack,
)


class RecordingClient:
    def __init__(self) -> None:
        self.calls = []

    def explain(self, event, language="English"):
        self.calls.append((event, language))
        return AnalystNarrative(
            "DDoS Alert", "Summary", ["Packet length is unusual"], "Reasoning",
            "Model confidence is 92 percent", ["Review the flow rate"],
            "This explanation does not prove that an attack occurred",
        )


def test_ai_is_not_called_for_benign_traffic() -> None:
    client = RecordingClient()
    assert explain_if_attack(AttackEvent("BENIGN", 0.99, []), client) is None
    assert client.calls == []


def test_ai_receives_attack_and_xai_evidence() -> None:
    client = RecordingClient()
    event = AttackEvent(
        "DDoS", 0.92,
        [FeatureEvidence("packet_length", 1514, 0.41, "supports DDoS", 420)],
        "flow-17",
    )
    narrative = explain_if_attack(event, client)
    assert narrative and narrative.title == "DDoS Alert"
    assert client.calls == [(event, "English")]


class FakeResponses:
    def create(self, **kwargs):
        self.kwargs = kwargs
        payload = {
            "title": "Alert", "summary": "Summary", "observed_signs": ["Sign"],
            "reasoning": "Reasoning", "confidence_note": "Confidence",
            "recommended_checks": ["Review"], "limitations": "Limitation",
        }
        return type("Response", (), {"output_text": json.dumps(payload)})()


def test_openai_request_is_strict_and_grounded() -> None:
    responses = FakeResponses()
    api = type("Client", (), {"responses": responses})()
    client = OpenAINarrativeClient("test-model", api)
    event = AttackEvent(
        "DDoS", 0.92,
        [FeatureEvidence("packet_length", 1514, 0.41, "supports DDoS", 420)],
    )
    client.explain(event)
    assert responses.kwargs["text"]["format"]["strict"] is True
    assert "packet_length" in responses.kwargs["input"]
    assert "1514" in responses.kwargs["input"]


def test_attack_without_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHAP/LIME"):
        explain_if_attack(AttackEvent("DDoS", 0.92, []), RecordingClient())
