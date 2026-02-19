---
pretty_name: HSClassify Micro Training Dataset
license: pddl
language:
  - en
  - th
  - vi
  - zh
task_categories:
  - text-classification
task_ids:
  - multi-class-classification
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: training_data_indexed.csv
---

# Dataset Card for HSClassify Micro Training Dataset

## Dataset Summary

This dataset supports multilingual HS code classification for customs and trade workflows.
It combines:

- HS nomenclature records (6-digit level and hierarchy context)
- Synthetic product descriptions mapped to HS codes
- Human-readable chapter/category labels for UI and latent-space analysis

## Included Files

- `training_data_indexed.csv`: training rows with text, HS code, chapter metadata, and language.
- `harmonized-system.csv`: source HS table snapshot used for data generation and indexing.
- `hs_codes_reference.json`: curated HS reference used by the app and training pipeline.
- `ATTRIBUTION.md`: explicit source and license attribution.

## Data Fields (`training_data_indexed.csv`)

- `text`: product description text used for embedding/classification.
- `hs_code`: 6-digit HS code target.
- `hs_chapter`: chapter description text.
- `hs_chapter_code`: chapter ID (e.g., `HS 08`).
- `hs_chapter_name`: normalized human-readable category label.
- `hs_desc`: HS description aligned to `hs_code`.
- `language`: language code (`en`, `th`, `vi`, `zh`).

## Source Attribution

Core HS nomenclature content is sourced from the `datasets/harmonized-system` project:

- Repository: <https://github.com/datasets/harmonized-system>
- Declared source chain in upstream metadata:
  - WCO HS nomenclature documentation
  - UN Comtrade data extraction API
- Upstream data license: ODC Public Domain Dedication and License (PDDL) v1.0

Project-added synthetic texts and normalized labels are released under this project's MIT license.

## Limitations

- Language balance is intentionally skewed toward English in the current snapshot.
- Synthetic text patterns may not cover all commercial phrasing edge cases.
- This dataset is for research/prototyping and is not legal customs advice.
