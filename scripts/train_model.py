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
    df = pd.read_csv(DATA_DIR / "training_data.csv", dtype={"hs_code": str})
    df["hs_code"] = df["hs_code"].astype(str).str.zfill(6)
    print(f"Loaded {len(df)} examples with {df['hs_code'].nunique()} unique HS codes")
    return df


def select_training_subset(df):
    """Select label space for classifier training/eval."""
    label_space = os.getenv("TRAIN_LABEL_SPACE", "curated").strip().lower()
    if label_space == "all":
        print("Training label space: all HS codes from training_data.csv")
        return df.copy()

    # Default: curated codes from hs_codes_reference.json (matches app behavior).
    with open(DATA_DIR / "hs_codes_reference.json", "r", encoding="utf-8") as f:
        hs_ref = json.load(f)
    curated_codes = {str(c).zfill(6) for c in hs_ref.keys()}
    subset = df[df["hs_code"].isin(curated_codes)].copy()
    if subset.empty:
        print("Warning: curated subset empty; falling back to all codes.")
        return df.copy()

    print(
        f"Training label space: curated ({subset['hs_code'].nunique()} codes, {len(subset)} rows)"
    )
    return subset


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


def finetune_model(model, df_train, epochs=2, batch_size=64, lr=2e-5, warmup=0.1):
    """Fine-tune sentence model with contrastive loss on curated training pairs."""
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from datasets import Dataset
    import tempfile

    print(f"\n{'='*60}")
    print("Fine-tuning embedding model (contrastive learning)")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}, Warmup: {warmup}")

    # Build pairs: for each sample, pair with a random sample from the same HS code.
    grouped = df_train.groupby("hs_code")["text"].apply(list).to_dict()
    anchors, positives = [], []
    rng = np.random.RandomState(42)
    for code, texts in grouped.items():
        if len(texts) < 2:
            continue
        for text in texts:
            # Pick a random different text from the same code
            candidates = [t for t in texts if t != text]
            if not candidates:
                candidates = texts  # fallback if all identical
            pos = candidates[rng.randint(len(candidates))]
            anchors.append(f"query: {text}")
            positives.append(f"passage: {pos}")

    print(f"  Built {len(anchors)} contrastive pairs from {len(grouped)} HS codes")

    train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})

    loss = MultipleNegativesRankingLoss(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use smaller micro-batches with gradient accumulation on MPS to avoid OOM.
        device = str(model.device)
        if "mps" in device or torch.backends.mps.is_available():
            micro_batch = min(batch_size, 4)
            grad_accum = max(1, batch_size // micro_batch)
        else:
            micro_batch = batch_size
            grad_accum = 1

        training_args = SentenceTransformerTrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=epochs,
            per_device_train_batch_size=micro_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            warmup_ratio=warmup,
            fp16=False,  # MPS doesn't support fp16 training
            logging_steps=10,
            save_strategy="no",
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
        )

        trainer.train()

    print("Fine-tuning complete.")
    return model


def save_artifacts(model, clf, le, embeddings_all, train_df, full_df, accuracy, report, ft_config=None):
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
        "n_examples": len(train_df),
        "n_codes": train_df["hs_code"].nunique(),
        "n_chapters": train_df["hs_chapter"].nunique(),
        "languages": train_df["language"].unique().tolist(),
        "label_space": os.getenv("TRAIN_LABEL_SPACE", "curated").strip().lower(),
        "n_examples_full_dataset": len(full_df),
        "n_codes_full_dataset": full_df["hs_code"].nunique(),
        "model_name": "intfloat/multilingual-e5-small",
        "embedding_dim": embeddings_all.shape[1],
        "classifier": "KNN (k=5, cosine distance)",
        "report_summary": {
            "weighted_f1": report["weighted avg"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
        },
        "fine_tuned": ft_config is not None,
    }
    if ft_config:
        metadata.update({
            "ft_epochs": ft_config["epochs"],
            "ft_batch_size": ft_config["batch_size"],
            "ft_lr": ft_config["lr"],
        })
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save the full training data with embeddings index for the app
    df_out = full_df.copy()
    df_out.to_csv(DATA_DIR / "training_data_indexed.csv", index=False)
    
    print(f"\nAll artifacts saved to {MODEL_DIR}")
    print(f"Total model directory size: ", end="")
    os.system(f"du -sh {MODEL_DIR}")


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Train HS code classifier")
    parser.add_argument("--finetune", action="store_true",
                        help="Enable contrastive fine-tuning before classifier training")
    parser.add_argument("--ft-epochs", type=int, default=2,
                        help="Number of fine-tuning epochs (default: 2)")
    parser.add_argument("--ft-batch-size", type=int, default=64,
                        help="Batch size for fine-tuning (default: 64)")
    parser.add_argument("--ft-lr", type=float, default=2e-5,
                        help="Learning rate for fine-tuning (default: 2e-5)")
    parser.add_argument("--ft-warmup", type=float, default=0.1,
                        help="Warmup ratio for fine-tuning (default: 0.1)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("HS Code Classifier Training")
    print("=" * 60)

    # Load full data and choose training subset.
    df_full = load_data()
    df_train = select_training_subset(df_full)

    # Load pre-trained multilingual model
    print("\nLoading multilingual-e5-small model...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Fine-tune if requested (before computing embeddings).
    ft_config = None
    if args.finetune:
        ft_config = {
            "epochs": args.ft_epochs,
            "batch_size": args.ft_batch_size,
            "lr": args.ft_lr,
        }
        model = finetune_model(
            model, df_train,
            epochs=args.ft_epochs,
            batch_size=args.ft_batch_size,
            lr=args.ft_lr,
            warmup=args.ft_warmup,
        )

    # e5 models should use passage prefix for index/training documents.
    df_full["text_prefixed"] = df_full["text"].apply(lambda x: f"passage: {x}")

    # Compute embeddings for the full dataset (used by app latent visualization).
    embeddings_full = compute_embeddings(model, df_full["text_prefixed"])

    # Subset embeddings for classifier training/eval.
    train_idx = df_train.index.to_numpy()
    embeddings_train = embeddings_full[train_idx]

    # Train classifier
    clf, le, accuracy, report, X_train, X_test, y_train, y_test = train_classifier(
        embeddings_train, df_train["hs_code"].astype(str).str.zfill(6)
    )

    # Save everything
    save_artifacts(model, clf, le, embeddings_full, df_train, df_full, accuracy, report, ft_config=ft_config)
    
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
            code6 = str(code).zfill(6)
            desc = hs_ref.get(code6, {}).get("desc", "Unknown")
            print(f"  {code6} ({prob*100:.1f}%) - {desc}")


if __name__ == "__main__":
    main()
