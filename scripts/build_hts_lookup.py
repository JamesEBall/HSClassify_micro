#!/usr/bin/env python3
"""
Build US HTS lookup table from official USITC CSV files (2019-2026).

Parses 8- and 10-digit HTS codes with descriptions and duty rates.
Uses 2026 as primary source, older years as fallback for retired codes.

Output: data/hts/us_hts_lookup.json
  Keyed by 6-digit HS code, values are arrays of HTS extensions.
"""

import csv
import json
import os
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
HTS_DIR = PROJECT_DIR / "data" / "hts"

# Ordered newest-first so 2026 takes priority
HTS_FILES = [
    "hts_2026_revision_2_csv.csv",
    "hts_2025_revision_31_csv.csv",
    "hts_2024_revision_9_csv.csv",
    "hts_2023_revision_10_csv.csv",
    "hts_2022_revision_11_csv.csv",
    "hts_2020_revision_17_csv.csv",
    "hts_2019_rev_19_data_csv.csv",
]


def clean_hts_number(raw: str) -> str:
    """Strip dots and whitespace from HTS number."""
    return raw.strip().replace(".", "").replace(" ", "")


def clean_unit(raw: str) -> str:
    """Clean unit of quantity field (may have JSON-like brackets)."""
    raw = raw.strip()
    if not raw:
        return ""
    # Remove JSON-ish brackets and quotes: ["No."] -> No.
    raw = raw.strip("[]\"' ")
    # Handle multiple units separated by commas
    raw = re.sub(r'"\s*,\s*"', ", ", raw)
    raw = raw.strip("\"' ")
    return raw


def build_description_context(rows_by_indent, current_indent, current_idx):
    """Build full hierarchical description by walking up indent levels.

    HTS CSVs use indentation to show hierarchy. Rows with empty HTS numbers
    are continuation/parent descriptions. We need to prepend parent context
    to make descriptions meaningful (e.g., "Other" becomes "Cattle: Other").
    """
    context_parts = []
    target_indent = current_indent - 1
    # Walk backwards looking for parent descriptions
    for i in range(current_idx - 1, -1, -1):
        row = rows_by_indent[i]
        indent = int(row.get("Indent", "0") or "0")
        if indent == target_indent:
            desc = row.get("Description", "").strip().rstrip(":")
            if desc and desc.lower() not in ("", "other"):
                context_parts.insert(0, desc)
            target_indent -= 1
            if target_indent < 0:
                break
    return context_parts


def parse_hts_csv(filepath: Path) -> dict:
    """Parse a single HTS CSV file, returning {hts_code_clean: record}."""
    records = {}
    all_rows = []

    # Handle BOM
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)

    for idx, row in enumerate(all_rows):
        hts_raw = row.get("HTS Number", "").strip()
        if not hts_raw:
            continue

        hts_clean = clean_hts_number(hts_raw)
        if not re.match(r"^\d{8,10}$", hts_clean):
            continue

        desc = row.get("Description", "").strip()
        indent = int(row.get("Indent", "0") or "0")

        # Build fuller description for vague entries like "Other"
        if desc.lower() in ("other", "other:", "other,", "others"):
            parents = build_description_context(all_rows, indent, idx)
            if parents:
                desc = ": ".join(parents) + ": " + desc

        general_duty = row.get("General Rate of Duty", "").strip()
        special_duty = row.get("Special Rate of Duty", "").strip()
        unit = clean_unit(row.get("Unit of Quantity", ""))

        records[hts_clean] = {
            "hts_code": hts_clean,
            "description": desc,
            "general_duty": general_duty,
            "special_duty": special_duty,
            "unit": unit,
        }

    return records


def build_lookup():
    """Build the lookup table: {hs6: [extensions]}."""
    # Merge all years, newest first (so 2026 overwrites older entries)
    all_codes = {}

    for filename in reversed(HTS_FILES):  # Load oldest first, newest overwrites
        filepath = HTS_DIR / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue

        year = re.search(r"(\d{4})", filename).group(1)
        records = parse_hts_csv(filepath)
        print(f"  {filename}: {len(records)} codes (8-10 digit)")

        for code, record in records.items():
            all_codes[code] = record

    print(f"\nTotal unique HTS codes (merged): {len(all_codes)}")

    # Group by 6-digit HS code
    lookup = {}
    for code, record in sorted(all_codes.items()):
        hs6 = code[:6]
        if hs6 not in lookup:
            lookup[hs6] = []
        lookup[hs6].append(record)

    print(f"Unique 6-digit HS codes with extensions: {len(lookup)}")

    # Stats
    total_extensions = sum(len(v) for v in lookup.values())
    avg_ext = total_extensions / len(lookup) if lookup else 0
    print(f"Total extensions: {total_extensions}")
    print(f"Average extensions per HS6: {avg_ext:.1f}")

    return lookup


def main():
    print("Building US HTS lookup table...")
    print(f"Source directory: {HTS_DIR}\n")

    lookup = build_lookup()

    output_path = HTS_DIR / "us_hts_lookup.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=1)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {output_path} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
