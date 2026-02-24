"""
Download and normalize real product description datasets from Hugging Face.

Fetches three datasets with real HS/HTS-labeled product descriptions:
  1. flexifyai/cross_rulings_hts_dataset_for_tariffs  (CBP CROSS rulings)
  2. samabos/product_hscode                           (e-commerce products)
  3. AIDC-AI/HSCodeComp                               (expert-annotated products)

All records are normalized to the 7-column training format and saved to
data/hf_real_data.csv.  Only records whose 6-digit HS code exists in the
official nomenclature (data/harmonized-system/harmonized-system.csv) are kept.
"""

import csv
import json
import os
import re
import sys

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
HS_CSV = os.path.join(PROJECT_DIR, "data", "harmonized-system", "harmonized-system.csv")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "hf_real_data.csv")

FIELDNAMES = [
    "text",
    "hs_code",
    "hs_chapter",
    "hs_chapter_code",
    "hs_chapter_name",
    "hs_desc",
    "language",
    "source",
]


# ---------------------------------------------------------------------------
# Load official HS nomenclature for validation
# ---------------------------------------------------------------------------
def load_valid_hs6():
    """Return dict {hs6_code: description} for all official 6-digit codes."""
    codes = {}
    if not os.path.exists(HS_CSV):
        print(f"WARNING: {HS_CSV} not found — skipping HS validation")
        return codes
    with open(HS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level") != "6":
                continue
            hscode = row.get("hscode", "").strip()
            desc = row.get("description", "").strip()
            if len(hscode) == 6 and hscode.isdigit() and desc:
                codes[hscode] = desc
    return codes


def load_chapter_labels():
    """Return dict {chapter_2digit: description} from official nomenclature."""
    labels = {}
    if not os.path.exists(HS_CSV):
        return labels
    with open(HS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level") != "2":
                continue
            code = row.get("hscode", "").strip()
            desc = row.get("description", "").strip()
            if len(code) == 2 and code.isdigit() and desc:
                labels[code] = desc
    return labels


def make_record(text, hs_code, hs_desc, language, source, chapter_labels):
    """Create a normalized training row."""
    hs_code = str(hs_code).zfill(6)
    chapter_2 = hs_code[:2]
    chapter_code = f"HS {chapter_2}"
    chapter_name = chapter_labels.get(chapter_2, f"Chapter {chapter_2}")
    return {
        "text": text,
        "hs_code": hs_code,
        "hs_chapter": chapter_name,
        "hs_chapter_code": chapter_code,
        "hs_chapter_name": chapter_name,
        "hs_desc": hs_desc,
        "language": language,
        "source": source,
    }


def truncate_to_hs6(code_str):
    """Extract leading 6 digits from an HTS/HS code string."""
    digits = re.sub(r"[^0-9]", "", str(code_str))
    if len(digits) >= 6:
        return digits[:6]
    return None


# ---------------------------------------------------------------------------
# Dataset fetchers
# ---------------------------------------------------------------------------
def fetch_cbp_cross(valid_codes, chapter_labels):
    """Fetch CBP CROSS rulings dataset and extract product descriptions + HTS codes."""
    print("Fetching flexifyai/cross_rulings_hts_dataset_for_tariffs ...")
    ds = load_dataset("flexifyai/cross_rulings_hts_dataset_for_tariffs", split="train")
    print(f"  Raw records: {len(ds)}")

    records = []
    for row in ds:
        # The dataset has a conversational 'messages' format
        messages = row.get("messages", [])
        if not messages:
            continue

        # Extract product description from user message(s)
        user_text = ""
        hts_code = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                user_text = content.strip()
            elif role == "assistant" and content:
                # Assistant message contains HTS code — extract digits
                # Look for patterns like "8471.30.0100" or "8471300100"
                code_match = re.search(r"(\d{4})[.\s]?(\d{2})[.\s]?(\d{2,4})?", content)
                if code_match:
                    digits = code_match.group(1) + code_match.group(2)
                    if code_match.group(3):
                        digits += code_match.group(3)
                    hts_code = digits

        if not user_text or not hts_code:
            continue

        hs6 = truncate_to_hs6(hts_code)
        if not hs6 or hs6 not in valid_codes:
            continue

        # Clean up description — remove question/classification prefixes
        desc = user_text
        desc = re.sub(r"^What\s+is\s+the\s+HTS?\s+(?:US\s+)?(?:Code|code)\s+for\s+", "", desc, flags=re.IGNORECASE)
        desc = re.sub(r"^(?:Classify|Please classify)\s+(?:the\s+following\s+)?(?:product|item)\s*:\s*", "", desc, flags=re.IGNORECASE)
        # Remove trailing question mark
        desc = desc.rstrip("?").strip()

        if len(desc) < 10:
            continue

        records.append(make_record(
            text=desc,
            hs_code=hs6,
            hs_desc=valid_codes[hs6],
            language="en",
            source="cbp_cross",
            chapter_labels=chapter_labels,
        ))

    print(f"  Valid records: {len(records)}")
    return records


def fetch_product_hscode(valid_codes, chapter_labels):
    """Fetch samabos/product_hscode dataset."""
    print("Fetching samabos/product_hscode ...")
    ds_dict = load_dataset("samabos/product_hscode")
    # Concatenate all splits (train + test)
    from datasets import concatenate_datasets
    ds = concatenate_datasets([ds_dict[s] for s in ds_dict])
    print(f"  Raw records: {len(ds)}")

    records = []
    for row in ds:
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip()

        if not text or not label:
            continue

        hs6 = truncate_to_hs6(label)
        if not hs6 or hs6 not in valid_codes:
            continue

        if len(text) < 5:
            continue

        records.append(make_record(
            text=text,
            hs_code=hs6,
            hs_desc=valid_codes[hs6],
            language="en",
            source="product_hscode",
            chapter_labels=chapter_labels,
        ))

    print(f"  Valid records: {len(records)}")
    return records


def fetch_hscomp(valid_codes, chapter_labels):
    """Fetch AIDC-AI/HSCodeComp dataset."""
    print("Fetching AIDC-AI/HSCodeComp ...")
    # This dataset only has a 'test' split
    ds = load_dataset("AIDC-AI/HSCodeComp", split="test")
    print(f"  Raw records: {len(ds)}")

    records = []
    for row in ds:
        # Combine product_name + selected product_attributes for richer description
        name = str(row.get("product_name", "")).strip()
        attrs_raw = str(row.get("product_attributes", "")).strip()
        hs_code_raw = str(row.get("hs_code", "")).strip()

        if not name or not hs_code_raw:
            continue

        # product_attributes is a JSON string — extract useful fields
        attr_parts = []
        if attrs_raw and attrs_raw not in ("nan", "None", ""):
            try:
                attrs = json.loads(attrs_raw)
                for key in ("Material", "Metals Type", "Origin", "Gender"):
                    val = attrs.get(key, "")
                    if val and val not in ("None", "NoEnName_Null"):
                        attr_parts.append(f"{key}: {val}")
            except (json.JSONDecodeError, TypeError):
                pass

        if attr_parts:
            text = f"{name}, {', '.join(attr_parts)}"
        else:
            text = name

        hs6 = truncate_to_hs6(hs_code_raw)
        if not hs6 or hs6 not in valid_codes:
            continue

        if len(text) < 5:
            continue

        records.append(make_record(
            text=text,
            hs_code=hs6,
            hs_desc=valid_codes[hs6],
            language="en",
            source="hscomp",
            chapter_labels=chapter_labels,
        ))

    print(f"  Valid records: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    valid_codes = load_valid_hs6()
    chapter_labels = load_chapter_labels()
    print(f"Official HS 6-digit codes loaded: {len(valid_codes)}")

    all_records = []

    # Fetch all three datasets
    all_records.extend(fetch_cbp_cross(valid_codes, chapter_labels))
    all_records.extend(fetch_product_hscode(valid_codes, chapter_labels))
    all_records.extend(fetch_hscomp(valid_codes, chapter_labels))

    # Deduplicate by (text, hs_code)
    seen = set()
    unique = []
    for rec in all_records:
        key = (rec["text"].lower(), rec["hs_code"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    print(f"\nTotal unique records: {len(unique)}")

    # Source breakdown
    source_counts = {}
    for rec in unique:
        source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    # HS code coverage
    hs_codes_covered = set(rec["hs_code"] for rec in unique)
    print(f"Unique HS6 codes covered: {len(hs_codes_covered)}")
    print(f"Overlap with official nomenclature: {len(hs_codes_covered)}/{len(valid_codes)} "
          f"({100 * len(hs_codes_covered) / len(valid_codes):.1f}%)")

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(unique)

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
