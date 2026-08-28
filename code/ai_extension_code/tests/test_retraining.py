import numpy as np

from src.explainability.generative_explainer import AttackEvent, FeatureEvidence
from src.validation.human_feedback import AlertValidator, FeedbackStore
from src.validation.retraining import reviewed_training_rows


def test_reviewed_feedback_is_aligned_for_retraining(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    validator = AlertValidator(store)
    event = AttackEvent(
        "Probe", 0.51,
        [FeatureEvidence("f1", 1.0, 0.10, "supports prediction", 0.0)],
        "flow-1",
    )
    validator.record_feedback(
        event,
        validator.assess(event),
        "wrong_attack_class",
        corrected_class="DoS",
        analyst_reason="Service disruption was confirmed",
        model_features={"f1": 1.0, "f2": 2.0},
    )
    X, Y = reviewed_training_rows(store, ["f1", "f2"], ["Normal", "DoS", "Probe"])
    assert X.to_numpy().tolist() == [[1.0, 2.0]]
    assert np.array_equal(Y, np.array([[0, 1, 0]]))
