"""Evidence-based alert validation and persistent analyst feedback.

The validator estimates alert reliability; it does not claim access to runtime
ground truth. A human analyst supplies the final outcome, which is retained for
future thresholding and model-retraining decisions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.explainability.generative_explainer import AttackEvent


FeedbackOutcome = Literal["correct", "false_positive", "wrong_attack_class"]


@dataclass(frozen=True)
class ValidationAssessment:
    reliability_score: float
    verdict: Literal["likely_correct", "uncertain", "likely_incorrect"]
    evidence_agreement: float
    historical_precision: float
    reviewed_alerts: int
    recommended_threshold: float
    firewall_action: Literal["analyst_review", "candidate_block"]
    reasons: list[str]


@dataclass(frozen=True)
class HumanFeedback:
    flow_id: str | None
    predicted_class: str
    outcome: FeedbackOutcome
    corrected_class: str | None
    analyst: str
    analyst_reason: str
    reviewed_label: str
    model_features: dict[str, float]
    reliability_score: float
    created_at: str


class FeedbackStore:
    """Append-only JSONL storage for analyst-confirmed alert outcomes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> list[HumanFeedback]:
        if not self.path.exists():
            return []
        records: list[HumanFeedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(HumanFeedback(**json.loads(line)))
        return records

    def append(self, feedback: HumanFeedback) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(feedback), ensure_ascii=False) + "\n")

    def class_summary(self, predicted_class: str) -> dict[str, float | int]:
        relevant = [item for item in self.records() if item.predicted_class == predicted_class]
        correct = sum(item.outcome == "correct" for item in relevant)
        false_positive = sum(item.outcome == "false_positive" for item in relevant)
        wrong_class = sum(item.outcome == "wrong_attack_class" for item in relevant)
        reviewed = len(relevant)
        # Beta(1, 1) smoothing prevents extreme estimates from one review.
        precision = (correct + 1) / (reviewed + 2)
        error_rate = (false_positive + wrong_class + 1) / (reviewed + 2)
        return {
            "reviewed": reviewed,
            "correct": correct,
            "false_positive": false_positive,
            "wrong_attack_class": wrong_class,
            "historical_precision": precision,
            "historical_error_rate": error_rate,
        }

    def training_examples(self) -> list[tuple[dict[str, float], str]]:
        """Return human-reviewed feature rows and labels for controlled retraining."""
        return [
            (item.model_features, item.reviewed_label)
            for item in self.records()
            if item.model_features
        ]


class AlertValidator:
    """Estimate reliability from confidence, attribution agreement, and feedback."""

    def __init__(self, feedback_store: FeedbackStore, minimum_reviews_for_block: int = 5) -> None:
        self.feedback_store = feedback_store
        self.minimum_reviews_for_block = minimum_reviews_for_block

    def assess(self, event: AttackEvent) -> ValidationAssessment:
        positive = sum(max(float(item.contribution), 0.0) for item in event.evidence)
        negative = sum(abs(min(float(item.contribution), 0.0)) for item in event.evidence)
        attribution_mass = positive + negative
        agreement = positive / attribution_mass if attribution_mass else 0.5
        history = self.feedback_store.class_summary(event.predicted_class)
        historical_precision = float(history["historical_precision"])
        reviewed = int(history["reviewed"])

        reliability = (
            0.45 * min(max(event.confidence, 0.0), 1.0)
            + 0.35 * agreement
            + 0.20 * historical_precision
        )
        reliability = min(max(reliability, 0.0), 1.0)
        if reliability >= 0.75:
            verdict = "likely_correct"
        elif reliability <= 0.45:
            verdict = "likely_incorrect"
        else:
            verdict = "uncertain"

        error_rate = float(history["historical_error_rate"])
        threshold = min(0.95, max(0.75, 0.75 + 0.20 * error_rate))
        can_block = (
            reliability >= threshold
            and reviewed >= self.minimum_reviews_for_block
            and historical_precision >= 0.80
        )
        action = "candidate_block" if can_block else "analyst_review"
        reasons = [
            f"Classifier confidence is {event.confidence:.3f}.",
            f"Supporting SHAP mass represents {agreement:.1%} of the selected attribution mass.",
            f"Historical analyst-confirmed precision for {event.predicted_class} is "
            f"{historical_precision:.1%} across {reviewed} reviewed alerts.",
            f"The current feedback-aware firewall threshold is {threshold:.3f}.",
        ]
        return ValidationAssessment(
            reliability_score=reliability,
            verdict=verdict,
            evidence_agreement=agreement,
            historical_precision=historical_precision,
            reviewed_alerts=reviewed,
            recommended_threshold=threshold,
            firewall_action=action,
            reasons=reasons,
        )

    def record_feedback(
        self,
        event: AttackEvent,
        assessment: ValidationAssessment,
        outcome: FeedbackOutcome,
        analyst: str = "analyst",
        analyst_reason: str = "",
        corrected_class: str | None = None,
        model_features: dict[str, float] | None = None,
    ) -> HumanFeedback:
        if not analyst_reason.strip():
            raise ValueError("An analyst reason is required for every feedback outcome")
        if outcome == "wrong_attack_class" and not corrected_class:
            raise ValueError("A corrected class is required for wrong_attack_class feedback")
        if outcome != "wrong_attack_class" and corrected_class:
            raise ValueError("A corrected class is only valid for wrong_attack_class feedback")
        reviewed_label = (
            event.predicted_class if outcome == "correct"
            else "Normal" if outcome == "false_positive"
            else str(corrected_class)
        )
        feedback = HumanFeedback(
            flow_id=event.flow_id,
            predicted_class=event.predicted_class,
            outcome=outcome,
            corrected_class=corrected_class,
            analyst=analyst,
            analyst_reason=analyst_reason.strip(),
            reviewed_label=reviewed_label,
            model_features=model_features or {},
            reliability_score=assessment.reliability_score,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.feedback_store.append(feedback)
        return feedback
