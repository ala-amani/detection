# Human-in-the-Loop Explainable Intrusion Detection

Research implementation of an explainable network intrusion-detection workflow. The repository combines the published XAI-NIDS reference implementation with alert validation, grounded LLM interpretation, explicit analyst adjudication, and controlled IDS retraining from validated feedback.

## Repository layout

- `src/` — maintained implementation
  - `src/explainability/` — SHAP evidence records and the schema-constrained LLM interpretation
  - `src/validation/` — reliability estimation, append-only analyst feedback, and controlled candidate retraining
  - `src/pipeline/` — the NSL-KDD Random Forest pipeline, the analyst interface, and the synthetic local-network evaluation
  - `src/data/` — NSL-KDD loading and attack-family mapping
- `examples/` — runnable scripts, including the deterministic synthetic MikroTik packet generator
- `tests/` — automated tests (`python -m pytest`)
- `XAI_NIDS/` — upstream reference implementation of the base study
- `code/base_article_code/` — preserved baseline implementation
- `code/ai_extension_code/` — earlier standalone copy of the extension, kept for reference; `src/` is the maintained version
- `docs/datasets/` — dataset provenance and schemas (no datasets are committed)

## Workflow

1. The IDS classifies a network-flow record.
2. SHAP ranks the features supporting and opposing the prediction.
3. An independent validation score estimates whether the alert should be reviewed.
4. A grounded language model converts the selected evidence into an English analyst narrative.
5. The analyst records `Correct`, `False positive`, or `Wrong attack class` with a reason.
6. Reviewed feedback updates the class-history term of later reliability estimates and supplies labelled rows for controlled IDS retraining; a candidate model is promoted only after holdout evaluation and explicit approval.

## Quick start

See [`code/README.md`](code/README.md) for installation and execution instructions. The interactive runs reported in the manuscript used a local Ollama model (`qwen3:4b`), so no traffic data left the host; the OpenAI client in `src/explainability/` is an optional alternative. Credentials, datasets, packet captures, generated results, and manuscript files are intentionally excluded from this repository.

## Reproducibility

Dataset provenance and expected schemas are documented under `docs/datasets/`. The synthetic local-network generator uses fixed seeds and reproducible scenario definitions. Full experiment data and generated artifacts must be produced locally and are not versioned.
