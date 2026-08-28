# Human-in-the-Loop XAI-NIDS Validation

This package extends the published XAI_NIDS NSL-KDD Random Forest workflow. After a non-normal prediction, it ranks local SHAP evidence, estimates alert reliability, invokes a local Ollama model, and asks an analyst to record `Correct`, `False positive`, or `Wrong attack class` with a reason. Feedback changes later validation and firewall thresholds and supplies reviewed rows for controlled IDS retraining.

## Requirements

- Python 3.10 or newer
- Ollama
- The `qwen3:4b` model
- NSL-KDD files `KDDTrain+.txt` and `KDDTest+.txt`

## Installation

Run these commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
ollama pull qwen3:4b
```

## Automated tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Full NSL-KDD execution

```powershell
.\.venv\Scripts\python -m src.pipeline.upstream_nsl_rf_ai `
  --data-root "C:\path\to\NSL-KDD" `
  --upstream-script "..\base_article_code\NSL-KDD\RF_ALL_FINAL.py" `
  --output "result.json" `
  --feedback-store "feedback\analyst_feedback.jsonl" `
  --ollama --ollama-model qwen3:4b --show-alert
```

Use `--test-row 1` for the reproducible wrong-class example (actual DoS, predicted Probe) or `--test-row 33` for the false-positive example (actual Normal, predicted Probe).

## Feedback and retraining policy

- The language model is advisory and receives only structured IDS, validation, and top-k SHAP evidence.
- Human feedback is append-only JSONL and includes the complete transformed feature row.
- Thresholds react immediately to reviewed class history, but the live IDS is never silently modified.
- `src.validation.retraining` creates a cloned candidate model and evaluates it on an untouched holdout set before any deployment decision.
- Datasets, feedback files, generated results, packet captures, model artifacts, secrets, and caches are intentionally excluded from source control.

## Synthetic local-network experiment

`examples/generate_synthetic_mikrotik_lab.py` creates an offline PCAP and a documented NSL-KDD-compatible flow table for a private laboratory topology containing a MikroTik router, Wi-Fi access point, laptop, and phone. No packet is transmitted. The four scenarios are normal web use, a router-port Probe, a Wi-Fi access-point SYN burst, and an authorized router health check that is operationally benign but scan-like.

`src.pipeline.synthetic_lab_evaluation` aligns the documented numeric and categorical fields with the published Random Forest feature space, then applies the same IDS, SHAP, reliability, local-LLM, and human-feedback path. The companion flow table is generated from scenario definitions; it is not claimed to be a general-purpose automatic NSL-KDD extractor from arbitrary PCAP files.
