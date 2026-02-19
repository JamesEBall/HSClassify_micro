#!/usr/bin/env bash
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: hf CLI not found. Install with: pip install -U huggingface_hub"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <namespace/dataset-repo> [--private]"
  echo "Example: $0 Troglobyte/hsclassify-micro-dataset"
  exit 1
fi

REPO_ID="$1"
VISIBILITY_FLAG="--no-private"
if [[ "${2:-}" == "--private" ]]; then
  VISIBILITY_FLAG="--private"
fi
TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--token "${HF_TOKEN}")
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${ROOT_DIR}/.hf_dataset_release"

echo "Preparing dataset release directory at ${RELEASE_DIR}"
rm -rf "${RELEASE_DIR}"
mkdir -p "${RELEASE_DIR}"

cp "${ROOT_DIR}/dataset/README.md" "${RELEASE_DIR}/README.md"
cp "${ROOT_DIR}/dataset/ATTRIBUTION.md" "${RELEASE_DIR}/ATTRIBUTION.md"
cp "${ROOT_DIR}/data/training_data_indexed.csv" "${RELEASE_DIR}/training_data_indexed.csv"
cp "${ROOT_DIR}/data/hs_codes_reference.json" "${RELEASE_DIR}/hs_codes_reference.json"
cp "${ROOT_DIR}/data/harmonized-system/harmonized-system.csv" "${RELEASE_DIR}/harmonized-system.csv"

echo "Creating dataset repo if needed: ${REPO_ID}"
hf repo create "${REPO_ID}" --repo-type dataset --exist-ok ${VISIBILITY_FLAG} "${TOKEN_ARGS[@]}"

echo "Uploading dataset snapshot to Hugging Face..."
hf upload "${REPO_ID}" "${RELEASE_DIR}" . \
  --repo-type dataset \
  "${TOKEN_ARGS[@]}" \
  --commit-message "Publish dataset snapshot $(date +%F)"

echo "Done. Dataset available at: https://huggingface.co/datasets/${REPO_ID}"
