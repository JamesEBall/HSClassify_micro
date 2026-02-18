# HS Code Classifier — POC v1.0

A multilingual HS (Harmonized System) code classifier with interactive latent space visualization. Accepts product descriptions in English, Thai, Vietnamese, and Chinese, and predicts the most likely 6-digit HS code.

## 🚀 Quick Start

```bash
# Activate the conda environment
conda activate hs-classifier

# Run the web app
cd projects/hs-code-classifier
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Open in browser
open http://localhost:8000
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 86.7% |
| **Weighted F1** | 0.8625 |
| **Macro F1** | 0.8544 |
| **Inference Time** | 60-150ms per query |
| **Model Size** | ~118MB (sentence-transformers) |
| **Total Project** | ~470MB (incl. cached HF model) |

## 🏗 Architecture

```
Product Description → [multilingual-e5-small] → 384-dim embedding → [KNN classifier] → HS Code
                                                        ↓
                                                   [UMAP 2D] → Interactive Visualization
```

- **Embedding Model**: `intfloat/multilingual-e5-small` — a compact multilingual sentence transformer (118MB)
- **Classifier**: K-Nearest Neighbors (k=5, cosine distance, distance-weighted)
- **Dimensionality Reduction**: UMAP (n_neighbors=15, min_dist=0.1)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/JS + Plotly.js for visualization

## 📁 Project Structure

```
hs-code-classifier/
├── app.py                          # FastAPI web app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── training_data.csv           # 1,280 labeled examples
│   ├── training_data.json          # Same data in JSON
│   ├── training_data_indexed.csv   # With index for app
│   └── hs_codes_reference.json     # HS code descriptions
├── models/
│   ├── sentence_model/             # Fine-tuned model weights
│   ├── knn_classifier.pkl          # Trained KNN classifier
│   ├── label_encoder.pkl           # Label encoder
│   ├── embeddings.npy              # Pre-computed embeddings
│   ├── umap_data.json              # UMAP projections cache
│   └── metadata.json               # Model metadata
├── scripts/
│   ├── generate_training_data.py   # Synthetic data generation
│   └── train_model.py              # Model training pipeline
├── templates/
│   └── index.html                  # Web UI
└── static/                         # Static assets
```

## 📦 Data Sources

Training data is **synthetically generated** covering:
- **1,280 labeled examples** across **118 unique HS codes**
- **39 HS chapters** (from Live Animals to Toys)
- **4 languages**: English (1,180), Thai (39), Vietnamese (22), Chinese (39)

HS codes cover major trade categories:
- Food & Agriculture (meat, fish, dairy, vegetables, fruits, cereals, sugar, cocoa, beverages)
- Mineral Products (petroleum, natural gas, cement)
- Chemical Products (pharmaceuticals, cosmetics, fertilizers)
- Plastics & Rubber (HDPE, tyres)
- Textiles & Garments (cotton, T-shirts, trousers, footwear)
- Base Metals (steel coils, aluminium, copper)
- Machinery & Electronics (laptops, smartphones, TVs, IC chips, batteries)
- Vehicles (petrol cars, electric cars, motorcycles)
- Furniture, Toys, Medical Instruments

## 🧪 Example Queries to Test

| Query | Expected HS Code | Language |
|-------|------------------|----------|
| Fresh boneless beef for restaurant supply | 020130 | English |
| Laptop computer 14 inch 16GB RAM | 847130 | English |
| ข้าวหอมมะลิไทย ขัดสี 5% หัก | 100630 | Thai |
| 冷冻虾仁 去头去壳 | 030617 | Chinese |
| Tôm đông lạnh xuất khẩu | 030617 | Vietnamese |
| Cotton T-shirt men printed knitted | 610910 | English |
| Lithium-ion battery pack for electric vehicles | 850760 | English |
| White refined cane sugar ICUMSA 45 | 170199 | English |
| New radial tyres for passenger cars 205/55R16 | 401110 | English |
| 智能手机 安卓系统 6.7英寸 | 851712 | Chinese |
| Electric passenger car battery powered Tesla | 870380 | English |
| สมาร์ทโฟน แอนดรอยด์ จอ 6.7 นิ้ว | 851712 | Thai |
| Cà phê nhân xanh chưa rang Robusta | 090111 | Vietnamese |
| Hot rolled steel coil width 600mm | 720839 | English |

## 🔧 Setup from Scratch

```bash
# Create conda environment
conda create -n hs-classifier python=3.11 -y
conda activate hs-classifier

# Install dependencies
pip install -r requirements.txt

# Generate training data
python scripts/generate_training_data.py

# Train model (downloads multilingual-e5-small ~118MB)
python scripts/train_model.py

# Run web app
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🎯 Features

1. **Real-time Classification**: Type a product description and get top HS code predictions with confidence scores
2. **Multilingual**: Works with English, Thai, Vietnamese, and Chinese input
3. **Confidence Visualization**: Color-coded confidence bars (green >70%, yellow >30%, red <30%)
4. **Similar Examples**: Shows nearest training examples for explainability
5. **Interactive UMAP Visualization**: 
   - 1,280 data points colored by HS chapter
   - Hover to see product descriptions
   - Query points appear as red stars
   - Auto-zooms to relevant region
6. **Fast Inference**: <200ms per query on Mac Mini M4

## ⚠️ Limitations (POC)

- Training data is synthetic — real customs declarations may differ
- Only 118 HS codes covered (full HS system has ~5,000+ at 6-digit level)
- KNN classifier is simple — production would use fine-tuned classification head
- No continuous learning / feedback loop
- UMAP projection is pre-computed, query point is approximated

## 🔮 Next Steps for Production

1. **Real training data**: Partner with customs agencies for real declarations
2. **More HS codes**: Expand to full 5,300+ 6-digit codes
3. **Fine-tuned classifier**: Replace KNN with neural classification head
4. **Active learning**: User feedback to improve predictions
5. **API integration**: REST API for customs systems
6. **Deployment**: Docker container, cloud hosting
