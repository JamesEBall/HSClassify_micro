#!/usr/bin/env python3
"""Pre-compute UMAP projection and save to models/umap_data.json.

Run this locally (where it's fast) then upload the resulting
``models/umap_data.json`` to the HF model repo so the Space can
download it at startup instead of computing UMAP on a free-tier CPU.

Usage:
    python scripts/precompute_umap.py
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"


def main():
    start = time.time()

    # Load HS reference
    hs_ref_path = DATA_DIR / "hs_codes_reference.json"
    if not hs_ref_path.exists():
        sys.exit(f"Missing {hs_ref_path}")
    with open(hs_ref_path) as f:
        hs_reference = json.load(f)

    # Load training data
    training_data_path = DATA_DIR / "training_data_indexed.csv"
    if not training_data_path.exists():
        training_data_path = DATA_DIR / "training_data.csv"
    if not training_data_path.exists():
        sys.exit(f"Missing training data at {training_data_path}")
    training_data = pd.read_csv(training_data_path)
    training_data["hs_code"] = training_data["hs_code"].astype(str).str.zfill(6)
    print(f"Loaded {len(training_data)} training rows")

    # Load embeddings
    embeddings_path = MODEL_DIR / "embeddings.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    else:
        part_paths = sorted(MODEL_DIR.glob("embeddings_part*.npy"))
        if part_paths:
            embeddings = np.concatenate([np.load(p) for p in part_paths], axis=0)
        else:
            sys.exit(f"Missing embeddings at {embeddings_path}")
    print(f"Loaded embeddings: {embeddings.shape}")

    if len(embeddings) != len(training_data):
        sys.exit(
            f"Embeddings ({len(embeddings)}) and training data ({len(training_data)}) "
            "have different lengths"
        )

    # Compute UMAP
    print("Computing UMAP projection...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    umap_coords = reducer.fit_transform(embeddings)

    # Build output
    umap_data = []
    for i, row in training_data.iterrows():
        hs_code = str(row["hs_code"]).zfill(6)
        chapter = row["hs_chapter"]
        chapter_name = str(row.get("hs_chapter_name", "")).strip()
        if not chapter_name or re.match(r"^HS\s\d{2}$", chapter_name):
            chapter_name = str(chapter).split(";")[0].strip()
        desc = hs_reference.get(hs_code, {}).get("desc", "Unknown")
        umap_data.append({
            "x": float(umap_coords[i, 0]),
            "y": float(umap_coords[i, 1]),
            "text": row["text"][:80],
            "hs_code": hs_code,
            "chapter": chapter,
            "chapter_name": chapter_name,
            "hs_desc": desc,
            "language": row["language"],
        })

    # Save
    out_path = MODEL_DIR / "umap_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(umap_data, f, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"Saved {len(umap_data)} points to {out_path} in {elapsed:.1f}s")
    print(
        "Upload this file to your HF model repo so the Space can "
        "download it instead of computing UMAP at runtime."
    )


if __name__ == "__main__":
    main()
