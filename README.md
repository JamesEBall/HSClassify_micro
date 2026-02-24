# HSClassify_micro 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Machine learning model for multilingual HS/HTS classification** for trade finance and customs workflows, built with FastAPI + OCR.

Classifies product descriptions into [Harmonized System (HS) codes](https://en.wikipedia.org/wiki/Harmonized_System) using sentence embeddings and k-NN search, with an interactive latent space visualization.

## Live Demo

- Hugging Face Space: [https://huggingface.co/spaces/Troglobyte/MicroHS/](https://huggingface.co/spaces/Mead0w1ark/MicroHS)
## Features

- 🌍 **Multilingual** — example supports English, Thai, Vietnamese, and Chinese product descriptions
- ⚡ **Real-time classification** — top-3 HS code predictions with confidence scores
- 📊 **Latent space visualization** — interactive UMAP plot showing embedding clusters
- 🎯 **KNN-based** — simple, interpretable nearest-neighbor approach using `paraphrase-multilingual-MiniLM-L12-v2`
- 🧾 **Official HS coverage** — training generation incorporates the [datasets/harmonized-system](https://github.com/datasets/harmonized-system) 6-digit nomenclature

## Dataset Attribution

This project includes HS nomenclature content sourced from:

- [datasets/harmonized-system](https://github.com/datasets/harmonized-system)
- Upstream references listed by that dataset:
  - WCO HS nomenclature documentation
  - UN Comtrade data extraction API

Related datasets (evaluated during development):

- [Customs-Declaration-Datasets](https://github.com/Seondong/Customs-Declaration-Datasets) — 54,000 synthetic customs declaration records derived from 24.7M real Korean customs entries. Provides structured trade metadata (HS codes, country of origin, price, weight, fraud labels) but does not include free-text product descriptions. Cited as a reference for customs data research. See: *S. Kim et al., "DATE: Dual Attentive Tree-aware Embedding for Customs Fraud Detection," KDD 2020.*

Licensing:

- Upstream HS source data: **ODC Public Domain Dedication and License (PDDL) v1.0**
- Project-added synthetic multilingual examples and labels: **MIT** (this repo)

## Quick Start

```bash
# Clone
git clone https://github.com/JamesEBall/HSClassify_micro.git
cd HSClassify_micro

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate training data & train model
python scripts/generate_training_data.py
python scripts/train_model.py

# Run the web app
uvicorn app:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) to classify products.

## Deployment

- The Space runs in Docker (`sdk: docker`, `app_port: 7860`).
- OCR endpoints require OS packages; `Dockerfile` installs:
  - `tesseract-ocr`
  - `poppler-utils` (for PDF conversion via `pdf2image`)
- Model loading is resilient in hosted environments:
  - if local `models/sentence_model` includes weights/tokenizer, it is used
  - otherwise the app falls back to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - optional override: set `SENTENCE_MODEL_NAME`

### Auto Sync (GitHub -> Hugging Face Space)

This repo includes a GitHub Action at `.github/workflows/sync_to_hf_space.yml` that syncs `main` to:

- `spaces/Troglobyte/MicroHS`

Required GitHub secret:

- `HF_TOKEN`: Hugging Face token with write access to the Space

## Publish Dataset to Hugging Face Datasets

Use the included publish helper:

```bash
bash scripts/publish_dataset_to_hf.sh <namespace>/<dataset-repo>
# Example:
bash scripts/publish_dataset_to_hf.sh Troglobyte/hsclassify-micro-dataset
```

The script creates/updates a Dataset repo and uploads:

- `training_data_indexed.csv`
- `harmonized-system.csv` (attributed source snapshot)
- `hs_codes_reference.json`
- Dataset card + attribution notes

## How It Works

1. **Embedding**: Product descriptions are encoded using `paraphrase-multilingual-MiniLM-L12-v2` (384-dim sentence embeddings)
2. **Classification**: K-nearest neighbors (k=5) over pre-computed embeddings of HS-coded training examples
3. **Visualization**: UMAP reduction to 2D for interactive cluster exploration via Plotly

## Project Structure

```
├── app.py                  # FastAPI web application
├── dataset/
│   ├── README.md           # HF dataset card (attribution + schema)
│   └── ATTRIBUTION.md      # Source and license attribution details
├── requirements.txt        # Python dependencies
├── scripts/
│   ├── generate_training_data.py   # Synthetic training data generator
│   ├── train_model.py              # Model training (embeddings + KNN)
│   └── publish_dataset_to_hf.sh    # Publish dataset artifacts to HF Datasets
├── data/
│   ├── hs_codes_reference.json     # HS code definitions
│   ├── harmonized-system/harmonized-system.csv  # Upstream HS source snapshot
│   ├── training_data.csv           # Generated training examples
│   └── training_data_indexed.csv   # App/latent-ready training examples
├── models/                 # Trained artifacts (generated)
│   ├── sentence_model/     # Cached sentence transformer
│   ├── embeddings.npy      # Pre-computed embeddings
│   ├── knn_classifier.pkl  # Trained KNN model
│   └── label_encoder.pkl   # Label encoder
└── templates/
    └── index.html          # Web UI
```

## Context

Built as a rapid POC exploring whether multilingual sentence embeddings can simplify HS code classification for customs authorities.
## License

MIT — see [LICENSE](LICENSE)
