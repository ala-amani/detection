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
    validation_verdict: str
    firewall_recommendation: str
    human_review_question: str


class NarrativeClient(Protocol):
    def explain(
        self, event: AttackEvent, validation: Any | None = None, language: str = "English"
    ) -> AnalystNarrative: ...


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
        "validation_verdict": {"type": "string"},
        "firewall_recommendation": {"type": "string"},
        "human_review_question": {"type": "string"},
    },
    "required": [
        "title", "summary", "observed_signs", "reasoning", "confidence_note",
        "recommended_checks", "limitations", "validation_verdict",
        "firewall_recommendation", "human_review_question",
    ],
}

NSL_KDD_FEATURE_GLOSSARY = {
    "dst_host_srv_count": "count of recent connections to the destination host that used the same service",
    "dst_host_rerror_rate": "rate of rejected-connection errors in recent traffic to the destination host",
    "dst_host_diff_srv_rate": "rate of different services used in recent traffic to the destination host",
    "diff_srv_rate": "rate of connections to different services in the short-term traffic window",
    "dst_host_same_src_port_rate": "rate of recent destination-host connections using the same source port",
}

ATTACK_GLOSSARY = {
    "Probe": "network reconnaissance or scanning intended to discover reachable hosts, ports, or services",
    "DoS": "an attempt to disrupt service availability through abnormal load or requests",
    "R2L": "a remote user attempting to obtain unauthorized local access",
    "U2R": "a normal user attempting to gain administrative privileges",
}


class OpenAINarrativeClient:
    """Generate a schema-constrained narrative using the OpenAI Responses API."""

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

    def explain(
        self, event: AttackEvent, validation: Any | None = None, language: str = "English"
    ) -> AnalystNarrative:
        evidence_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        validation_json = json.dumps(asdict(validation) if validation else None, ensure_ascii=False, indent=2)
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You explain network IDS alerts to a security analyst. The classifier has "
                "already made the decision. Use only the supplied prediction and feature "
                "evidence; never invent packets, thresholds, causes, or certainty. Explain "
                "that feature attribution is evidence about the model, not proof of causality. "
                "Assess whether the alert is likely reliable using the supplied validation result. "
                "The validation score is an estimate, not ground truth. Recommend verification steps, "
                "request an explicit human outcome, and never authorize automatic blocking by yourself."
            ),
            input=(
                f"Write the analyst narrative in {language}. Explain why the model labelled "
                f"this flow as {event.predicted_class}, including observed feature values, "
                "direction, and contribution. If a baseline is missing, say that no baseline "
                f"comparison is available.\n\nEvent evidence:\n{evidence_json}"
                f"\n\nIndependent validation assessment:\n{validation_json}"
            ),
            text={"format": {
                "type": "json_schema", "name": "analyst_narrative", "strict": True,
                "schema": NARRATIVE_SCHEMA,
            }},
        )
        return AnalystNarrative(**json.loads(response.output_text))


class OllamaNarrativeClient:
    """Generate a grounded narrative through a local Ollama chat endpoint."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_metrics: dict[str, float | int | str] = {}

    def explain(
        self, event: AttackEvent, validation: Any | None = None, language: str = "English"
    ) -> AnalystNarrative:
        evidence_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        schema_json = json.dumps(NARRATIVE_SCHEMA, ensure_ascii=False)
        context_json = json.dumps(
            {
                "predicted_attack_meaning": ATTACK_GLOSSARY.get(event.predicted_class, event.predicted_class),
                "feature_meanings": {
                    item.feature: NSL_KDD_FEATURE_GLOSSARY.get(item.feature, item.feature)
                    for item in event.evidence
                },
                "value_scale": (
                    "Observed values are standardized model inputs. Positive means above the training average; "
                    "negative means below it. They are not raw packet counts or percentages."
                ),
            },
            ensure_ascii=False,
        )
        validation_json = json.dumps(asdict(validation) if validation else None, ensure_ascii=False, indent=2)
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": NARRATIVE_SCHEMA,
            "options": {"temperature": 0, "seed": 42},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain network IDS alerts to a security analyst. The classifier "
                        "already made the prediction. Use only supplied values and attribution "
                        "evidence. Never invent packets, thresholds, causal claims, or certainty. "
                        "Clearly distinguish evidence supporting the prediction from evidence "
                        "opposing it. Recommend verification, never automatic shutdown. Write fluent, plain, "
                        "professional English for a general IT analyst. Avoid ambiguous or unusual terminology. "
                        "Describe Probe as network reconnaissance or scanning. Explain each "
                        "technical feature using the supplied glossary. State which signs support and which oppose "
                        "the prediction. Treat confidence near 51 percent as low and borderline. A probability is "
                        "confidence, never confirmation. Explain the independent validation verdict and whether the "
                        "firewall should hold the alert for human review. Ask the analyst to label the alert as correct, "
                        "false positive, or wrong attack class. Never describe the validation estimate as ground truth."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write the narrative in {language}. Return only JSON matching this schema: "
                        f"{schema_json}\nPlain-language context:\n{context_json}\nEvent evidence:\n{evidence_json}"
                        f"\nIndependent validation assessment:\n{validation_json}"
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}: {exc}") from exc
        content = body["message"]["content"]
        self.last_metrics = {
            "provider": "Ollama",
            "model": str(body.get("model", self.model)),
            "total_duration_seconds": float(body.get("total_duration", 0)) / 1_000_000_000,
            "load_duration_seconds": float(body.get("load_duration", 0)) / 1_000_000_000,
            "prompt_tokens": int(body.get("prompt_eval_count", 0)),
            "generated_tokens": int(body.get("eval_count", 0)),
        }
        return AnalystNarrative(**json.loads(content))

    def chat(
        self, event: AttackEvent, validation: Any, history: list[dict[str, str]], question: str
    ) -> str:
        """Continue a grounded English conversation about one detected event."""
        event_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        validation_json = json.dumps(asdict(validation), ensure_ascii=False, indent=2)
        messages = [{
            "role": "system",
            "content": (
                "You are assisting a network security analyst with one IDS alert. Answer in clear, "
                "concise English. Use only the supplied classifier prediction and SHAP evidence. "
                "Do not invent packet contents, IP addresses, thresholds, or facts that are absent. "
                "Explain technical terms plainly. Say when the available evidence cannot answer a "
                "question. SHAP explains model influence, not causality. Recommend verification; "
                "never claim the attack is proven and never order an automatic shutdown. Explain that the validator "
                "estimates reliability and only human feedback establishes the reviewed outcome.\n\n"
                f"Fixed event evidence:\n{event_json}\n\nValidation assessment:\n{validation_json}"
            ),
        }, *history[-8:], {"role": "user", "content": question}]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "seed": 42},
            "messages": messages,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}: {exc}") from exc
        return str(body["message"]["content"]).strip()


def explain_if_attack(
    event: AttackEvent,
    client: NarrativeClient,
    validation: Any | None = None,
    benign_labels: Sequence[str] = ("normal", "benign"),
    language: str = "English",
) -> AnalystNarrative | None:
    """Call the narrative AI only after a non-benign prediction."""

    normalized = event.predicted_class.strip().casefold()
    if normalized in {label.strip().casefold() for label in benign_labels}:
        return None
    if not event.evidence:
        raise ValueError("An attack narrative requires SHAP/LIME feature evidence")
    return client.explain(event, validation=validation, language=language)
