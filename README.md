---
title: HS Code Classifier Micro
emoji: ⚡
colorFrom: pink
colorTo: blue
sdk: docker
app_port: 7860
---

# HSClassify_micro 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Lightweight HS code classifier using multilingual embeddings** — a proof-of-concept for ADB customs digitization.

Classifies product descriptions into [Harmonized System (HS) codes](https://en.wikipedia.org/wiki/Harmonized_System) using sentence embeddings and k-NN search, with an interactive latent space visualization.

## Features

- 🌍 **Multilingual** — supports English, Thai, Vietnamese, and Chinese product descriptions
- ⚡ **Real-time classification** — top-3 HS code predictions with confidence scores
- 📊 **Latent space visualization** — interactive UMAP plot showing embedding clusters
- 🎯 **KNN-based** — simple, interpretable nearest-neighbor approach using `paraphrase-multilingual-MiniLM-L12-v2`

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

## How It Works

1. **Embedding**: Product descriptions are encoded using `paraphrase-multilingual-MiniLM-L12-v2` (384-dim sentence embeddings)
2. **Classification**: K-nearest neighbors (k=5) over pre-computed embeddings of HS-coded training examples
3. **Visualization**: UMAP reduction to 2D for interactive cluster exploration via Plotly

## Project Structure

```
├── app.py                  # FastAPI web application
├── requirements.txt        # Python dependencies
├── scripts/
│   ├── generate_training_data.py   # Synthetic training data generator
│   └── train_model.py              # Model training (embeddings + KNN)
├── data/
│   ├── hs_codes_reference.json     # HS code definitions
│   └── training_data.csv           # Generated training examples
├── models/                 # Trained artifacts (generated)
│   ├── sentence_model/     # Cached sentence transformer
│   ├── embeddings.npy      # Pre-computed embeddings
│   ├── knn_classifier.pkl  # Trained KNN model
│   └── label_encoder.pkl   # Label encoder
└── templates/
    └── index.html          # Web UI
```

## Context

Built as a rapid POC exploring whether multilingual sentence embeddings can simplify HS code classification for customs authorities in developing Asian economies — part of broader digital public goods work with the Asian Development Bank (ADB).

## License

MIT — see [LICENSE](LICENSE)
