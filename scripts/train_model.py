"""
Train an HS code classifier using sentence-transformers.

Fine-tunes a multilingual model to produce embeddings that cluster by HS code,
then trains a lightweight classifier head on top.
"""

import json
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    """Load training data."""
    df = pd.read_csv(DATA_DIR / "training_data.csv")
    print(f"Loaded {len(df)} examples with {df['hs_code'].nunique()} unique HS codes")
    return df


def compute_embeddings(model, texts, batch_size=64):
    """Compute embeddings for a list of texts."""
    print(f"Computing embeddings for {len(texts)} texts...")
    start = time.time()
    embeddings = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    elapsed = time.time() - start
    print(f"Embeddings computed in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/sec)")
    return embeddings


def train_classifier(embeddings, labels, n_neighbors=5):
    """Train a KNN classifier on embeddings."""
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_samples = len(y)
    n_classes = len(le.classes_)
    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    
    # Split data
    test_size = 0.2
    n_test = math.ceil(n_samples * test_size)
    n_train = n_samples - n_test
    can_stratify = (
        min_class_count >= 2
        and n_test >= n_classes
        and n_train >= n_classes
    )
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        print(
            "Warning: skipping stratified split "
            f"(samples={n_samples}, classes={n_classes}, min_class_count={min_class_count}, "
            f"train={n_train}, test={n_test})."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=test_size, random_state=42, stratify=None
        )
    
    print(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train KNN classifier (fast, works well with good embeddings)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', weights='distance')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*60}")
    
    # Detailed report (top codes only for readability)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    
    # Print summary
    print(f"\nWeighted avg F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"Macro avg F1: {report['macro avg']['f1-score']:.4f}")
    
    return clf, le, accuracy, report, X_train, X_test, y_train, y_test


def save_artifacts(model, clf, le, embeddings_all, df, accuracy, report):
    """Save all model artifacts."""
    # Save sentence transformer model
    model_path = MODEL_DIR / "sentence_model"
    model.save(str(model_path))
    print(f"Saved sentence model to {model_path}")
    
    # Save classifier
    with open(MODEL_DIR / "knn_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    
    # Save label encoder
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # Save embeddings for visualization
    np.save(MODEL_DIR / "embeddings.npy", embeddings_all)
    
    # Save metadata
    metadata = {
        "accuracy": accuracy,
        "n_examples": len(df),
        "n_codes": df["hs_code"].nunique(),
        "n_chapters": df["hs_chapter"].nunique(),
        "languages": df["language"].unique().tolist(),
        "model_name": "intfloat/multilingual-e5-small",
        "embedding_dim": embeddings_all.shape[1],
        "classifier": "KNN (k=5, cosine distance)",
        "report_summary": {
            "weighted_f1": report["weighted avg"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
        }
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save the full training data with embeddings index for the app
    df_out = df.copy()
    df_out.to_csv(DATA_DIR / "training_data_indexed.csv", index=False)
    
    print(f"\nAll artifacts saved to {MODEL_DIR}")
    print(f"Total model directory size: ", end="")
    os.system(f"du -sh {MODEL_DIR}")


def main():
    print("=" * 60)
    print("HS Code Classifier Training")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Load pre-trained multilingual model
    print("\nLoading multilingual-e5-small model...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Prepend query prefix for e5 models
    # e5 models expect "query: " or "passage: " prefix
    df["text_prefixed"] = df["text"].apply(lambda x: f"query: {x}")
    
    # Compute embeddings
    embeddings = compute_embeddings(model, df["text_prefixed"])
    
    # Train classifier
    clf, le, accuracy, report, X_train, X_test, y_train, y_test = train_classifier(
        embeddings, df["hs_code"].astype(str)
    )
    
    # Save everything
    save_artifacts(model, clf, le, embeddings, df, accuracy, report)
    
    # Print example predictions
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)
    
    test_queries = [
        "Fresh boneless beef for restaurant",
        "Laptop computer 14 inch for office",
        "ข้าวหอมมะลิไทย",  # Thai jasmine rice
        "冷冻虾仁",  # Frozen shrimp
        "Tôm đông lạnh xuất khẩu",  # Vietnamese frozen shrimp
        "Cotton T-shirt, men's, printed",
        "Lithium battery for electric car",
        "Refined white sugar 50kg bags",
        "New car tyres 205/55R16",
        "智能手机 5G",  # Smartphone 5G
    ]
    
    for query in test_queries:
        q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
        # Get top 3 predictions with distances
        distances, indices = clf.kneighbors(q_emb, n_neighbors=5)
        
        # Predict probabilities
        probs = clf.predict_proba(q_emb)[0]
        top_3_idx = np.argsort(probs)[-3:][::-1]
        top_3_codes = le.classes_[top_3_idx]
        top_3_probs = probs[top_3_idx]
        
        # Load HS code reference for descriptions
        with open(DATA_DIR / "hs_codes_reference.json") as f:
            hs_ref = json.load(f)
        
        print(f"\nQuery: {query}")
        for code, prob in zip(top_3_codes, top_3_probs):
            desc = hs_ref.get(code, {}).get("desc", "Unknown")
            print(f"  {code} ({prob*100:.1f}%) - {desc}")


if __name__ == "__main__":
    main()
