import json

import pytest

from src.explainability.generative_explainer import (
    AnalystNarrative, AttackEvent, FeatureEvidence, OpenAINarrativeClient,
    explain_if_attack,
)
from src.validation.human_feedback import AlertValidator, FeedbackStore


class RecordingClient:
    def __init__(self) -> None:
        self.calls = []

    def explain(self, event, validation=None, language="English"):
        self.calls.append((event, validation, language))
        return AnalystNarrative(
            "DDoS Alert", "Summary", ["Packet length is unusual"], "Reasoning",
            "Model confidence is 92 percent", ["Review the flow rate"],
            "This explanation does not prove that an attack occurred",
            "likely correct", "prioritize analyst review", "Is this alert correct?",
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
    assert client.calls == [(event, None, "English")]


class FakeResponses:
    def create(self, **kwargs):
        self.kwargs = kwargs
        payload = {
            "title": "Alert", "summary": "Summary", "observed_signs": ["Sign"],
            "reasoning": "Reasoning", "confidence_note": "Confidence",
            "recommended_checks": ["Review"], "limitations": "Limitation",
            "validation_verdict": "uncertain",
            "review_recommendation": "analyst review",
            "human_review_question": "Is this alert correct?",
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


def test_conflicting_low_confidence_alert_requires_review(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    validator = AlertValidator(store)
    event = AttackEvent(
        "Probe", 0.51,
        [
            FeatureEvidence("support", 1.0, 0.08, "supports prediction", 0.0),
            FeatureEvidence("oppose", -1.0, -0.15, "opposes prediction", 0.0),
        ],
        "flow-1",
    )
    assessment = validator.assess(event)
    assert assessment.verdict in {"uncertain", "likely_incorrect"}
    assert assessment.review_priority == "high"


def test_false_positive_feedback_is_routed_to_ids_retraining(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    validator = AlertValidator(store)
    event = AttackEvent(
        "Probe", 0.70,
        [FeatureEvidence("scan_rate", 1.0, 0.20, "supports prediction", 0.0)],
        "flow-fp",
    )
    before = validator.assess(event)
    validator.record_feedback(
        event, before, "false_positive", analyst_reason="No matching activity in firewall logs",
        model_features={"scan_rate": 1.0},
    )
    after = validator.assess(event)
    assert after.reviewed_alerts == 1
    assert after.historical_precision < before.historical_precision
    assert store.training_examples() == [({"scan_rate": 1.0}, "Normal")]


def test_wrong_attack_class_requires_corrected_label(tmp_path) -> None:
    validator = AlertValidator(FeedbackStore(tmp_path / "feedback.jsonl"))
    event = AttackEvent(
        "Probe", 0.51,
        [FeatureEvidence("rerror", 1.0, 0.10, "supports prediction", 0.0)],
        "flow-wrong-class",
    )
    assessment = validator.assess(event)
    with pytest.raises(ValueError, match="corrected class"):
        validator.record_feedback(
            event, assessment, "wrong_attack_class", analyst_reason="Service exhaustion was confirmed"
        )
    feedback = validator.record_feedback(
        event, assessment, "wrong_attack_class", corrected_class="DoS",
        analyst_reason="Service exhaustion was confirmed", model_features={"rerror": 1.0},
    )
    assert feedback.corrected_class == "DoS"
    assert validator.feedback_store.training_examples() == [({"rerror": 1.0}, "DoS")]


def test_feedback_requires_analyst_reason(tmp_path) -> None:
    validator = AlertValidator(FeedbackStore(tmp_path / "feedback.jsonl"))
    event = AttackEvent(
        "Probe", 0.70,
        [FeatureEvidence("scan_rate", 1.0, 0.20, "supports prediction", 0.0)],
    )
    with pytest.raises(ValueError, match="analyst reason"):
        validator.record_feedback(event, validator.assess(event), "correct")
