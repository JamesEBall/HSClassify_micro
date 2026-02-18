"""
HS Code Classifier Web App

FastAPI backend with:
- Real-time HS code prediction from text input
- Top-3 suggestions with confidence scores
- Latent space visualization with UMAP
- Multilingual support (EN, TH, VI, ZH)
"""

import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Paths
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"

# Initialize FastAPI
app = FastAPI(title="HS Code Classifier", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(PROJECT_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))

# Global model state
model = None
classifier = None
label_encoder = None
hs_reference = None
training_data = None
embeddings = None
umap_data = None


def load_models():
    """Load all model artifacts on startup."""
    global model, classifier, label_encoder, hs_reference, training_data, embeddings, umap_data
    
    print("Loading models...")
    start = time.time()
    
    # Load sentence transformer
    model = SentenceTransformer(str(MODEL_DIR / "sentence_model"))
    
    # Load classifier
    with open(MODEL_DIR / "knn_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    
    # Load label encoder
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load HS code reference
    with open(DATA_DIR / "hs_codes_reference.json") as f:
        hs_reference = json.load(f)
    
    # Load training data
    training_data = pd.read_csv(DATA_DIR / "training_data_indexed.csv")
    
    # Load embeddings
    embeddings = np.load(MODEL_DIR / "embeddings.npy")
    
    # Compute UMAP projection
    print("Computing UMAP projection...")
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        umap_coords = reducer.fit_transform(embeddings)
        
        # Build visualization data
        umap_data = []
        for i, row in training_data.iterrows():
            hs_code = str(row["hs_code"]).zfill(6)
            chapter = row["hs_chapter"]
            desc = hs_reference.get(hs_code, {}).get("desc", "Unknown")
            umap_data.append({
                "x": float(umap_coords[i, 0]),
                "y": float(umap_coords[i, 1]),
                "text": row["text"][:80],
                "hs_code": hs_code,
                "chapter": chapter,
                "hs_desc": desc,
                "language": row["language"]
            })
        
        # Save UMAP data for caching
        with open(MODEL_DIR / "umap_data.json", "w", encoding="utf-8") as f:
            json.dump(umap_data, f, ensure_ascii=False)
        
        print(f"UMAP projection computed for {len(umap_data)} points")
    except Exception as e:
        print(f"UMAP computation failed: {e}")
        umap_data = []
    
    elapsed = time.time() - start
    print(f"All models loaded in {elapsed:.1f}s")


@app.on_event("startup")
async def startup():
    load_models()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page."""
    metadata = {}
    try:
        with open(MODEL_DIR / "metadata.json") as f:
            metadata = json.load(f)
    except:
        pass
    return templates.TemplateResponse("index.html", {
        "request": request,
        "metadata": metadata,
    })


@app.post("/predict")
async def predict(request: Request):
    """Predict HS code for a product description."""
    body = await request.json()
    query_text = body.get("text", "").strip()
    
    if not query_text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    start = time.time()
    
    # Encode query with e5 prefix
    query_emb = model.encode(
        [f"query: {query_text}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Get predictions with probabilities
    probs = classifier.predict_proba(query_emb)[0]
    top_k = 5
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        hs_code = label_encoder.classes_[idx]
        hs_code_padded = str(hs_code).zfill(6)
        confidence = float(probs[idx])
        if confidence < 0.01:
            continue
        
        info = hs_reference.get(hs_code_padded, {})
        chapter_code = hs_code_padded[:2]
        heading_code = hs_code_padded[:4]
        
        predictions.append({
            "hs_code": hs_code_padded,
            "confidence": confidence,
            "description": info.get("desc", "No description available"),
            "chapter": info.get("chapter", "Unknown"),
            "chapter_code": chapter_code,
            "heading_code": heading_code,
        })
    
    # Find nearest training examples
    distances, indices = classifier.kneighbors(query_emb, n_neighbors=3)
    similar_examples = []
    for dist, idx in zip(distances[0], indices[0]):
        row = training_data.iloc[classifier._tree.data[idx] if hasattr(classifier, '_tree') else idx]
        # Get training data by index
        similar_examples.append({
            "text": training_data.iloc[idx]["text"] if idx < len(training_data) else "N/A",
            "hs_code": str(training_data.iloc[idx]["hs_code"]).zfill(6) if idx < len(training_data) else "N/A",
            "distance": float(dist),
        })
    
    elapsed = time.time() - start
    
    return JSONResponse({
        "query": query_text,
        "predictions": predictions,
        "similar_examples": similar_examples,
        "inference_time_ms": round(elapsed * 1000, 1),
    })


@app.get("/visualization-data")
async def get_visualization_data():
    """Return UMAP projection data for visualization."""
    if umap_data:
        return JSONResponse({"points": umap_data})
    
    # Try to load cached data
    cache_path = MODEL_DIR / "umap_data.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse({"points": data})
    
    return JSONResponse({"points": [], "error": "No UMAP data available"})


@app.post("/embed-query")
async def embed_query(request: Request):
    """Get UMAP coordinates for a query (project into existing space)."""
    body = await request.json()
    query_text = body.get("text", "").strip()
    
    if not query_text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    # Encode query
    query_emb = model.encode(
        [f"query: {query_text}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Find nearest neighbors in the embedding space
    distances, indices = classifier.kneighbors(query_emb, n_neighbors=5)
    
    # Approximate query position as weighted average of nearest neighbors' UMAP positions
    if umap_data and len(umap_data) > 0:
        weights = 1.0 / (distances[0] + 1e-6)
        weights = weights / weights.sum()
        
        x = sum(umap_data[idx]["x"] * w for idx, w in zip(indices[0], weights) if idx < len(umap_data))
        y = sum(umap_data[idx]["y"] * w for idx, w in zip(indices[0], weights) if idx < len(umap_data))
        
        neighbors = []
        for idx in indices[0]:
            if idx < len(umap_data):
                neighbors.append(umap_data[idx])
        
        return JSONResponse({
            "x": float(x),
            "y": float(y),
            "neighbors": neighbors,
        })
    
    return JSONResponse({"error": "No UMAP data for projection"})


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
