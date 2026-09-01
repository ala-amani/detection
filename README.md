# Human-in-the-Loop Explainable Intrusion Detection

Research implementation of an explainable network intrusion-detection workflow. The repository combines the published XAI-NIDS reference implementation with alert validation, grounded LLM interpretation, explicit analyst adjudication, and controlled IDS retraining from validated feedback.

## Repository layout

- `src/` — implementation
- `tests/` — automated tests
- `docs/datasets/` — dataset documentation only
- `XAI_NIDS/` — upstream reference implementation
- `code/base_article_code/` — preserved baseline implementation
- `code/ai_extension_code/` — complete human-feedback extension and runnable examples
- `src/` — integration modules
- `tests/` — automated tests
- `docs/datasets/` — dataset provenance and schemas (no datasets are committed)

## Workflow

1. The IDS classifies a network-flow record.
2. SHAP ranks the features supporting and opposing the prediction.
3. An independent validation score estimates whether the alert should be reviewed.
4. A grounded language model converts the selected evidence into an English analyst narrative.
5. The analyst records `Correct`, `False positive`, or `Wrong attack class` with a reason.
6. Reviewed feedback updates subsequent validation and can supply labelled rows for controlled retraining.

## Quick start

See [`code/README.md`](code/README.md) for installation and execution instructions. The demonstrated interactive workflow uses a local Ollama model; API-based execution is also supported. Credentials, datasets, packet captures, generated results, and manuscript files are intentionally excluded from this repository.

## Reproducibility

Dataset provenance and expected schemas are documented under `docs/datasets/`. The synthetic local-network generator uses fixed seeds and reproducible scenario definitions. Full experiment data and generated artifacts must be produced locally and are not versioned.
