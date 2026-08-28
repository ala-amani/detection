# Source Code

This directory contains the two implementation parts submitted with the manuscript.

- `base_article_code/`: the published XAI_NIDS reference implementation.
- `ai_extension_code/`: the evidence-grounded generative explanation and interactive analyst chat extension.

Datasets, generated results, packet captures, credentials, caches, and local environment files are excluded.

## Main extension entry point

The extension requires Python 3.10 or newer and Ollama. From the extracted ZIP:

```powershell
cd ai_extension_code
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
ollama pull qwen3:4b
```

Copy `KDDTrain+.txt` and `KDDTest+.txt` into a local data directory, then run:

```powershell
.\.venv\Scripts\python -m src.pipeline.upstream_nsl_rf_ai `
  --data-root "C:\path\to\NSL-KDD" `
  --upstream-script "..\base_article_code\NSL-KDD\RF_ALL_FINAL.py" `
  --output "result.json" `
  --ollama --ollama-model qwen3:4b --show-alert
```

Run the automated tests with:

```powershell
.\.venv\Scripts\python -m pytest -q
```

The local language model is invoked only after a non-benign prediction.
