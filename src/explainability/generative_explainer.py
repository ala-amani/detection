"""Grounded analyst narratives for attacks detected by an IDS model.

The language model is downstream of the classifier and XAI layer: it explains
supplied evidence, but it never decides whether traffic is an attack.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class FeatureEvidence:
    feature: str
    observed_value: float | int | str
    contribution: float
    direction: str
    baseline: float | int | str | None = None


@dataclass(frozen=True)
class AttackEvent:
    predicted_class: str
    confidence: float
    evidence: Sequence[FeatureEvidence]
    flow_id: str | None = None


@dataclass(frozen=True)
class AnalystNarrative:
    title: str
    summary: str
    observed_signs: list[str]
    reasoning: str
    confidence_note: str
    recommended_checks: list[str]
    limitations: str


class NarrativeClient(Protocol):
    def explain(self, event: AttackEvent, language: str = "English") -> AnalystNarrative: ...


NARRATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "observed_signs": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "confidence_note": {"type": "string"},
        "recommended_checks": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "string"},
    },
    "required": [
        "title", "summary", "observed_signs", "reasoning", "confidence_note",
        "recommended_checks", "limitations",
    ],
}


class OpenAINarrativeClient:
    """Generate a schema-constrained narrative using the OpenAI Responses API."""

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

    def explain(self, event: AttackEvent, language: str = "English") -> AnalystNarrative:
        evidence_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You explain network IDS alerts to a security analyst. The classifier has "
                "already made the decision. Use only the supplied prediction and feature "
                "evidence; never invent packets, thresholds, causes, or certainty. Explain "
                "that feature attribution is evidence about the model, not proof of causality. "
                "Recommend verification steps, not automatic shutdown or blocking."
            ),
            input=(
                f"Write the analyst narrative in {language}. Explain why the model labelled "
                f"this flow as {event.predicted_class}, including observed feature values, "
                "direction, and contribution. If a baseline is missing, say that no baseline "
                f"comparison is available.\n\nEvent evidence:\n{evidence_json}"
            ),
            text={"format": {
                "type": "json_schema", "name": "analyst_narrative", "strict": True,
                "schema": NARRATIVE_SCHEMA,
            }},
        )
        return AnalystNarrative(**json.loads(response.output_text))


class OllamaNarrativeClient:
    """Generate a grounded narrative through a local Ollama chat endpoint."""

    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://127.0.0.1:11434", timeout: float = 180.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_metrics: dict[str, float | int | str] = {}

    def explain(self, event: AttackEvent, language: str = "English") -> AnalystNarrative:
        evidence_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        schema_json = json.dumps(NARRATIVE_SCHEMA, ensure_ascii=False)
        payload = {
            "model": self.model, "stream": False, "think": False,
            "format": NARRATIVE_SCHEMA, "options": {"temperature": 0, "seed": 42},
            "messages": [
                {"role": "system", "content": (
                    "You explain network IDS alerts to a security analyst. The classifier already made the prediction. "
                    "Use only supplied values and attribution evidence. Never invent packets, thresholds, causal claims, "
                    "or certainty. Clearly distinguish evidence supporting the prediction from evidence opposing it. "
                    "Recommend verification, never automatic shutdown. Use clear, professional English security "
                    "terminology and avoid ambiguous wording. Never use confirmation language to "
                    "describe a model prediction. A probability is confidence, not confirmation.")},
                {"role": "user", "content": (
                    f"Write the narrative in {language}. Return only JSON matching this schema: {schema_json}"
                    f"\nEvent evidence:\n{evidence_json}")},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}: {exc}") from exc
        self.last_metrics = {
            "provider": "Ollama", "model": str(body.get("model", self.model)),
            "total_duration_seconds": float(body.get("total_duration", 0)) / 1_000_000_000,
            "load_duration_seconds": float(body.get("load_duration", 0)) / 1_000_000_000,
            "prompt_tokens": int(body.get("prompt_eval_count", 0)),
            "generated_tokens": int(body.get("eval_count", 0)),
        }
        return AnalystNarrative(**json.loads(body["message"]["content"]))


def explain_if_attack(
    event: AttackEvent,
    client: NarrativeClient,
    benign_labels: Sequence[str] = ("normal", "benign"),
    language: str = "English",
) -> AnalystNarrative | None:
    """Call the narrative AI only after a non-benign prediction."""

    normalized = event.predicted_class.strip().casefold()
    if normalized in {label.strip().casefold() for label in benign_labels}:
        return None
    if not event.evidence:
        raise ValueError("An attack narrative requires SHAP/LIME feature evidence")
    return client.explain(event, language=language)
