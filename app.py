"""
HS Code Classifier Web App

FastAPI backend with:
- Real-time HS code prediction from text input
- Document upload with OCR (Tesseract) support
- Structured field extraction from trade documents
- HS (6-digit) and HTS (7-10 digit) code support
- Top-5 suggestions with confidence scores
- Latent space visualization with UMAP
- Multilingual support (EN, TH, VI, ZH)
"""

import json
import os
import time
import pickle
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from field_extractor import extract_fields, get_all_countries, get_all_currencies
from hs_dataset import get_dataset, get_hts_extensions, get_available_hts_countries

# Paths
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"
UPLOAD_DIR = PROJECT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Upload config
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".pdf"}

# Initialize FastAPI
app = FastAPI(title="HS Code Classifier", version="2.0.0")
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
hs_dataset = None


def load_models():
    """Load all model artifacts on startup."""
    global model, classifier, label_encoder, hs_reference, training_data, embeddings, umap_data, hs_dataset
    
    print("Loading models...")
    start = time.time()
    
    # Load sentence transformer:
    # prefer local bundled model, fall back to Hub model when large files are not in repo.
    local_model_dir = MODEL_DIR / "sentence_model"
    has_local_weights = (
        (local_model_dir / "model.safetensors").exists()
        or (local_model_dir / "pytorch_model.bin").exists()
    )
    has_local_tokenizer = (local_model_dir / "tokenizer.json").exists()

    if local_model_dir.exists() and has_local_weights and has_local_tokenizer:
        model = SentenceTransformer(str(local_model_dir))
        print("Loaded local sentence model from models/sentence_model")
    else:
        fallback_model = os.getenv(
            "SENTENCE_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        model = SentenceTransformer(fallback_model)
        print(f"Loaded sentence model from Hugging Face Hub: {fallback_model}")
    
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
    
    # Load HS dataset (official harmonized-system data)
    hs_dataset = get_dataset()
    
    # Load UMAP data from cache (or compute if needed)
    cache_path = MODEL_DIR / "umap_data.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            umap_data = json.load(f)
        print(f"Loaded cached UMAP data: {len(umap_data)} points")
    else:
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
            
            with open(cache_path, "w", encoding="utf-8") as f:
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
    
    countries = get_all_countries()
    currencies = get_all_currencies()
    hts_countries = get_available_hts_countries()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "metadata": metadata,
        "countries": countries,
        "currencies": currencies,
        "hts_countries": hts_countries,
    })


@app.post("/predict")
async def predict(request: Request):
    """Predict HS code for a product description with optional structured context."""
    body = await request.json()
    query_text = body.get("text", "").strip()
    made_in = body.get("made_in", "")
    ship_to = body.get("ship_to", "")
    item_price = body.get("item_price", None)
    currency = body.get("currency", "")
    
    if not query_text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    start = time.time()
    
    # Build enriched query using structured fields
    enriched_query = query_text
    context_parts = []
    if made_in:
        context_parts.append(f"origin: {made_in}")
    if ship_to:
        context_parts.append(f"destination: {ship_to}")
    if item_price and currency:
        context_parts.append(f"value: {currency} {item_price}")
    
    if context_parts:
        enriched_query = f"{query_text} ({', '.join(context_parts)})"
    
    # Encode query with e5 prefix
    query_emb = model.encode(
        [f"query: {enriched_query}"],
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
        
        # Get official description from HS dataset if available
        official = hs_dataset.lookup(hs_code_padded) if hs_dataset else None
        official_desc = official['description'] if official else None
        
        # Validate against official dataset
        validation = hs_dataset.validate_hs_code(hs_code_padded) if hs_dataset else None
        
        predictions.append({
            "hs_code": hs_code_padded,
            "confidence": confidence,
            "description": info.get("desc", official_desc or "No description available"),
            "official_description": official_desc,
            "chapter": info.get("chapter", "Unknown"),
            "chapter_code": chapter_code,
            "heading_code": heading_code,
            "validated": validation['valid'] if validation else None,
        })
    
    # Find nearest training examples
    sims = embeddings @ query_emb.T
    top_sim_idx = np.argsort(sims.flatten())[-3:][::-1]
    similar_examples = []
    for idx in top_sim_idx:
        if idx < len(training_data):
            similar_examples.append({
                "text": training_data.iloc[idx]["text"],
                "hs_code": str(training_data.iloc[idx]["hs_code"]).zfill(6),
                "similarity": float(sims[idx][0]),
            })
    
    elapsed = time.time() - start
    
    return JSONResponse({
        "query": query_text,
        "enriched_query": enriched_query,
        "predictions": predictions,
        "similar_examples": similar_examples,
        "inference_time_ms": round(elapsed * 1000, 1),
    })


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (image/PDF) and extract text via OCR + structured fields."""
    # Validate file
    if not file.filename:
        return JSONResponse({"error": "No file provided"}, status_code=400)
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"},
            status_code=400
        )
    
    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(
            {"error": f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB"},
            status_code=400
        )
    
    # Save to temp file
    file_id = str(uuid.uuid4())[:8]
    temp_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(temp_path, "wb") as f:
        f.write(content)
    
    try:
        import pytesseract
        from PIL import Image
        
        ocr_text = ""
        
        if ext == ".pdf":
            # Convert PDF to images, then OCR
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(str(temp_path), dpi=300)
                texts = []
                for img in images:
                    texts.append(pytesseract.image_to_string(img))
                ocr_text = "\n\n".join(texts)
            except ImportError:
                return JSONResponse(
                    {"error": "PDF support requires pdf2image and poppler. Install with: pip install pdf2image"},
                    status_code=500
                )
            except Exception as e:
                return JSONResponse(
                    {"error": f"PDF processing error: {str(e)}"},
                    status_code=500
                )
        else:
            # Image OCR
            img = Image.open(temp_path)
            ocr_text = pytesseract.image_to_string(img)
        
        if not ocr_text.strip():
            return JSONResponse({
                "error": "OCR could not extract any text from this document. Please try a clearer image.",
                "raw_text": "",
                "fields": {},
            })
        
        # Extract structured fields
        fields = extract_fields(ocr_text)
        
        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "raw_text": ocr_text.strip(),
            "fields": fields,
        })
    
    except Exception as e:
        return JSONResponse(
            {"error": f"OCR processing failed: {str(e)}"},
            status_code=500
        )
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/extract-fields")
async def extract_fields_endpoint(request: Request):
    """Extract structured fields from arbitrary text (no OCR needed)."""
    body = await request.json()
    text = body.get("text", "").strip()
    
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    fields = extract_fields(text)
    return JSONResponse({"fields": fields})


@app.get("/hts-extensions/{hs_code}")
async def get_hts(hs_code: str, country: str = "US"):
    """Get HTS (country-specific) extensions for a 6-digit HS code."""
    result = get_hts_extensions(hs_code, country)
    return JSONResponse(result)


@app.get("/hs-lookup/{hs_code}")
async def hs_lookup(hs_code: str):
    """Look up an HS code in the official dataset."""
    if not hs_dataset:
        return JSONResponse({"error": "HS dataset not loaded"}, status_code=500)
    
    result = hs_dataset.lookup(hs_code)
    if not result:
        # Try search instead
        search_results = hs_dataset.search(hs_code, max_results=5)
        return JSONResponse({
            "found": False,
            "message": f"Code {hs_code} not found. Did you mean one of these?",
            "suggestions": search_results,
        })
    
    return JSONResponse({"found": True, **result})


@app.get("/hs-search")
async def hs_search(q: str = "", limit: int = 20):
    """Search HS codes by description."""
    if not q:
        return JSONResponse({"error": "No query provided"}, status_code=400)
    
    results = hs_dataset.search(q, max_results=limit)
    return JSONResponse({"results": results, "query": q})


@app.get("/hs-validate/{hs_code}")
async def hs_validate(hs_code: str):
    """Validate whether an HS code exists."""
    result = hs_dataset.validate_hs_code(hs_code)
    return JSONResponse(result)


@app.get("/hts-countries")
async def hts_countries():
    """Get list of countries with HTS extensions available."""
    return JSONResponse({"countries": get_available_hts_countries()})


@app.get("/visualization-data")
async def get_visualization_data():
    """Return UMAP projection data for visualization."""
    if umap_data:
        return JSONResponse({"points": umap_data})
    
    cache_path = MODEL_DIR / "umap_data.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse({"points": data})
    
    return JSONResponse({"points": [], "error": "No UMAP data available"})


@app.post("/embed-query")
async def embed_query(request: Request):
    """Get UMAP coordinates for a query."""
    body = await request.json()
    query_text = body.get("text", "").strip()
    
    if not query_text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    query_emb = model.encode(
        [f"query: {query_text}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    distances, indices = classifier.kneighbors(query_emb, n_neighbors=5)
    
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
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "hs_dataset_loaded": hs_dataset._loaded if hs_dataset else False,
        "hs_codes_count": len(hs_dataset.subheadings) if hs_dataset else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
