"""
Benchmark evaluation for the HS code classifier.

Runs hand-crafted test cases through the model and reports accuracy metrics.
Optionally performs split-analysis on the training data for per-class diagnostics.

Usage:
    python scripts/benchmark.py                              # basic benchmark
    python scripts/benchmark.py --output results/out.json    # custom output path
    python scripts/benchmark.py --split-analysis             # + per-class analysis
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"


# ---------------------------------------------------------------------------
# Model loading (mirrors app.py:77-97)
# ---------------------------------------------------------------------------

def load_model():
    """Load sentence transformer, classifier, label encoder, and HS reference."""
    from sentence_transformers import SentenceTransformer

    # Prefer local bundled model, fall back to Hub
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
            "intfloat/multilingual-e5-small",
        )
        model = SentenceTransformer(fallback_model)
        print(f"Loaded sentence model from Hugging Face Hub: {fallback_model}")

    with open(MODEL_DIR / "knn_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)

    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open(DATA_DIR / "hs_codes_reference.json") as f:
        hs_reference = json.load(f)

    return model, classifier, label_encoder, hs_reference


# ---------------------------------------------------------------------------
# Prediction (mirrors app.py:346-365)
# ---------------------------------------------------------------------------

def predict_top_k(model, classifier, label_encoder, text, k=5):
    """Return top-k (hs_code, confidence) pairs for a query string."""
    query_emb = model.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    probs = classifier.predict_proba(query_emb)[0]
    top_indices = np.argsort(probs)[-k:][::-1]

    results = []
    for idx in top_indices:
        hs_code = str(label_encoder.classes_[idx]).zfill(6)
        results.append((hs_code, float(probs[idx])))
    return results


# ---------------------------------------------------------------------------
# Basic benchmark
# ---------------------------------------------------------------------------

def run_benchmark(model, classifier, label_encoder, hs_reference):
    """Run benchmark cases and return detailed results."""
    bench_path = DATA_DIR / "benchmark_cases.csv"
    if not bench_path.exists():
        print(f"ERROR: {bench_path} not found")
        sys.exit(1)

    df = pd.read_csv(bench_path, dtype={"expected_hs_code": str})
    df["expected_hs_code"] = df["expected_hs_code"].str.zfill(6)

    curated_codes = set(hs_reference.keys())

    results = []
    for _, row in df.iterrows():
        text = row["text"]
        expected = row["expected_hs_code"]
        category = row["category"]
        language = row.get("language", "en")
        notes = row.get("notes", "")

        preds = predict_top_k(model, classifier, label_encoder, text, k=5)
        pred_codes = [code for code, _ in preds]
        top1_code = pred_codes[0]
        top1_conf = preds[0][1]

        hit_at_1 = top1_code == expected
        hit_at_3 = expected in pred_codes[:3]
        hit_at_5 = expected in pred_codes[:5]

        # Chapter-level accuracy (first 2 digits)
        chapter_hit = top1_code[:2] == expected[:2]

        # Is the expected code in our label space?
        in_label_space = expected in curated_codes

        results.append({
            "text": text,
            "expected": expected,
            "predicted": top1_code,
            "confidence": top1_conf,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "chapter_hit": chapter_hit,
            "in_label_space": in_label_space,
            "category": category,
            "language": language,
            "notes": notes,
            "top5": pred_codes,
        })

    return results


def compute_metrics(results):
    """Compute aggregate metrics from benchmark results."""
    total = len(results)
    in_space = [r for r in results if r["in_label_space"]]
    n_in_space = len(in_space)

    # Overall (all cases)
    top1 = sum(r["hit_at_1"] for r in results) / total if total else 0
    top3 = sum(r["hit_at_3"] for r in results) / total if total else 0
    top5 = sum(r["hit_at_5"] for r in results) / total if total else 0
    chapter = sum(r["chapter_hit"] for r in results) / total if total else 0

    # In-label-space only (excludes known_failure with out-of-space codes)
    top1_ls = sum(r["hit_at_1"] for r in in_space) / n_in_space if n_in_space else 0
    top3_ls = sum(r["hit_at_3"] for r in in_space) / n_in_space if n_in_space else 0
    top5_ls = sum(r["hit_at_5"] for r in in_space) / n_in_space if n_in_space else 0
    chapter_ls = sum(r["chapter_hit"] for r in in_space) / n_in_space if n_in_space else 0

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "top1": 0, "top3": 0, "top5": 0, "chapter": 0}
        categories[cat]["total"] += 1
        categories[cat]["top1"] += r["hit_at_1"]
        categories[cat]["top3"] += r["hit_at_3"]
        categories[cat]["top5"] += r["hit_at_5"]
        categories[cat]["chapter"] += r["chapter_hit"]

    for cat in categories:
        n = categories[cat]["total"]
        categories[cat]["top1_acc"] = categories[cat]["top1"] / n
        categories[cat]["top3_acc"] = categories[cat]["top3"] / n
        categories[cat]["top5_acc"] = categories[cat]["top5"] / n
        categories[cat]["chapter_acc"] = categories[cat]["chapter"] / n

    # Per-language breakdown
    languages = {}
    for r in results:
        lang = r["language"]
        if lang not in languages:
            languages[lang] = {"total": 0, "top1": 0, "top3": 0, "top5": 0}
        languages[lang]["total"] += 1
        languages[lang]["top1"] += r["hit_at_1"]
        languages[lang]["top3"] += r["hit_at_3"]
        languages[lang]["top5"] += r["hit_at_5"]

    for lang in languages:
        n = languages[lang]["total"]
        languages[lang]["top1_acc"] = languages[lang]["top1"] / n
        languages[lang]["top3_acc"] = languages[lang]["top3"] / n
        languages[lang]["top5_acc"] = languages[lang]["top5"] / n

    # Failures list
    failures = [
        {
            "text": r["text"],
            "expected": r["expected"],
            "predicted": r["predicted"],
            "confidence": round(r["confidence"], 4),
            "category": r["category"],
            "language": r["language"],
            "top5": r["top5"],
            "notes": r["notes"],
        }
        for r in results
        if not r["hit_at_1"]
    ]

    return {
        "total_cases": total,
        "in_label_space_cases": n_in_space,
        "overall": {
            "top1_accuracy": round(top1, 4),
            "top3_accuracy": round(top3, 4),
            "top5_accuracy": round(top5, 4),
            "chapter_accuracy": round(chapter, 4),
        },
        "in_label_space": {
            "top1_accuracy": round(top1_ls, 4),
            "top3_accuracy": round(top3_ls, 4),
            "top5_accuracy": round(top5_ls, 4),
            "chapter_accuracy": round(chapter_ls, 4),
        },
        "by_category": categories,
        "by_language": languages,
        "failures": failures,
        "n_failures": len(failures),
    }


# ---------------------------------------------------------------------------
# Split analysis (mirrors train_model.py:98-136)
# ---------------------------------------------------------------------------

def run_split_analysis(model, hs_reference):
    """Replicate 80/20 stratified split and report per-class metrics."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "=" * 60)
    print("Split Analysis (replicating training 80/20 split)")
    print("=" * 60)

    # Load training data
    df = pd.read_csv(DATA_DIR / "training_data.csv", dtype={"hs_code": str})
    df["hs_code"] = df["hs_code"].astype(str).str.zfill(6)

    # Filter to curated codes (same as train_model.py:select_training_subset)
    curated_codes = {str(c).zfill(6) for c in hs_reference.keys()}
    df = df[df["hs_code"].isin(curated_codes)].copy()
    print(f"Training data: {len(df)} rows, {df['hs_code'].nunique()} codes")

    # Load pre-computed embeddings
    embeddings_path = MODEL_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found. Run train_model.py first.")
        return None

    embeddings_full = np.load(embeddings_path)

    # Subset embeddings to curated rows
    embeddings = embeddings_full[df.index.to_numpy()]
    labels = df["hs_code"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_samples = len(y)
    n_classes = len(le.classes_)
    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0

    # Replicate exact split from train_model.py:98-109
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
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=test_size, random_state=42, stratify=None
        )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train fresh KNN (same params as train_model.py:114)
    clf = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    overall_acc = float(np.mean(y_test == y_pred))
    print(f"\nTest accuracy: {overall_acc:.4f} ({overall_acc * 100:.1f}%)")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")

    # Worst 15 codes by F1
    code_metrics = []
    for code in le.classes_:
        if code in report and isinstance(report[code], dict):
            m = report[code]
            code_metrics.append({
                "hs_code": code,
                "desc": hs_reference.get(code, {}).get("desc", ""),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1-score"], 4),
                "support": int(m["support"]),
            })

    code_metrics.sort(key=lambda x: x["f1"])
    worst_15 = code_metrics[:15]

    print("\nWorst 15 codes by F1:")
    print(f"{'HS Code':<10} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Sup':>5}  Description")
    print("-" * 75)
    for m in worst_15:
        print(f"{m['hs_code']:<10} {m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['support']:>5}  {m['desc'][:40]}")

    # Top 20 cross-chapter confusions
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    confusions = []
    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count == 0:
                continue
            true_code = le.classes_[true_idx]
            pred_code = le.classes_[pred_idx]
            true_chapter = true_code[:2]
            pred_chapter = pred_code[:2]
            if true_chapter == pred_chapter:
                continue  # same chapter, not cross-chapter
            confusions.append({
                "true_code": true_code,
                "pred_code": pred_code,
                "true_chapter": hs_reference.get(true_code, {}).get("chapter", true_chapter),
                "pred_chapter": hs_reference.get(pred_code, {}).get("chapter", pred_chapter),
                "count": count,
            })

    confusions.sort(key=lambda x: x["count"], reverse=True)
    top_20_confusions = confusions[:20]

    print(f"\nTop 20 cross-chapter confusions:")
    print(f"{'True Code':<10} {'Pred Code':<10} {'Count':>5}  True Chapter → Pred Chapter")
    print("-" * 70)
    for c in top_20_confusions:
        print(f"{c['true_code']:<10} {c['pred_code']:<10} {c['count']:>5}  {c['true_chapter']} → {c['pred_chapter']}")

    return {
        "test_accuracy": round(overall_acc, 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "worst_15_by_f1": worst_15,
        "top_20_cross_chapter_confusions": top_20_confusions,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics):
    """Print a human-readable benchmark report."""
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)

    o = metrics["overall"]
    print(f"\nTotal cases: {metrics['total_cases']}  (in label space: {metrics['in_label_space_cases']})")
    print(f"\n{'Metric':<25} {'All Cases':>10} {'In-Space':>10}")
    print("-" * 47)
    ls = metrics["in_label_space"]
    print(f"{'Top-1 Accuracy':<25} {o['top1_accuracy']:>10.1%} {ls['top1_accuracy']:>10.1%}")
    print(f"{'Top-3 Accuracy':<25} {o['top3_accuracy']:>10.1%} {ls['top3_accuracy']:>10.1%}")
    print(f"{'Top-5 Accuracy':<25} {o['top5_accuracy']:>10.1%} {ls['top5_accuracy']:>10.1%}")
    print(f"{'Chapter Accuracy':<25} {o['chapter_accuracy']:>10.1%} {ls['chapter_accuracy']:>10.1%}")

    print(f"\nPer-Category Breakdown:")
    print(f"{'Category':<15} {'N':>4} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'Chapter':>8}")
    print("-" * 52)
    for cat, m in sorted(metrics["by_category"].items()):
        print(f"{cat:<15} {m['total']:>4} {m['top1_acc']:>7.1%} {m['top3_acc']:>7.1%} {m['top5_acc']:>7.1%} {m['chapter_acc']:>8.1%}")

    print(f"\nPer-Language Breakdown:")
    print(f"{'Language':<10} {'N':>4} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7}")
    print("-" * 40)
    for lang, m in sorted(metrics["by_language"].items()):
        print(f"{lang:<10} {m['total']:>4} {m['top1_acc']:>7.1%} {m['top3_acc']:>7.1%} {m['top5_acc']:>7.1%}")

    n_fail = metrics["n_failures"]
    print(f"\nFailures ({n_fail}):")
    print("-" * 90)
    for f in metrics["failures"]:
        marker = " [OUT-OF-SPACE]" if f["expected"] not in {c.zfill(6) for c in f.get("_curated", [])} else ""
        print(f"  {f['text'][:45]:<45} expected={f['expected']}  got={f['predicted']}  conf={f['confidence']:.2f}  [{f['category']}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark HS code classifier")
    parser.add_argument("--output", "-o", default="benchmark_results.json",
                        help="Path for JSON results (default: benchmark_results.json)")
    parser.add_argument("--split-analysis", action="store_true",
                        help="Run per-class analysis on 80/20 training split")
    args = parser.parse_args()

    start = time.time()

    print("Loading models...")
    model, classifier, label_encoder, hs_reference = load_model()
    load_time = time.time() - start
    print(f"Models loaded in {load_time:.1f}s")

    # Basic benchmark
    print("\nRunning benchmark cases...")
    bench_start = time.time()
    results = run_benchmark(model, classifier, label_encoder, hs_reference)
    metrics = compute_metrics(results)
    bench_time = time.time() - bench_start
    print(f"Benchmark completed in {bench_time:.1f}s")

    print_report(metrics)

    # Optional split analysis
    split_metrics = None
    if args.split_analysis:
        split_metrics = run_split_analysis(model, hs_reference)

    # Save JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": metrics,
        "timing": {
            "model_load_s": round(load_time, 2),
            "benchmark_s": round(bench_time, 2),
            "total_s": round(time.time() - start, 2),
        },
    }
    if split_metrics:
        report["split_analysis"] = split_metrics

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
