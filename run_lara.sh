#!/usr/bin/env bash
# =============================================================================
# run_lara.sh — one-click full-LARa pipeline for HiST-VQ:
#   preprocess raw LARa CSVs -> train -> evaluate.
#
# Usage:
#   bash run_lara.sh /path/to/raw_lara [EPOCHS]
#
#   /path/to/raw_lara : folder with the LARa OMoCap CSV pairs
#                       (L0X_S0X_R0X.csv + L0X_S0X_R0X_labels.csv).
#                       Download the full set from:
#                       https://zenodo.org/records/8189341
#   EPOCHS            : number of training epochs (default 30).
#
# GPU is used automatically if available (recommended; CPU is very slow).
# =============================================================================
set -euo pipefail

RAW_DIR="${1:-}"
EPOCHS="${2:-30}"
DATA_ROOT="data"
DATASET="lara"

if [[ -z "${RAW_DIR}" ]]; then
  echo "ERROR: missing raw LARa directory."
  echo "Usage: bash run_lara.sh /path/to/raw_lara [EPOCHS]"
  echo "Download LARa v3 (OMoCap) from https://zenodo.org/records/8189341"
  exit 1
fi

echo "=================================================================="
echo "[HiST-VQ] LARa pipeline"
echo "  raw data : ${RAW_DIR}"
echo "  epochs   : ${EPOCHS}"
echo "=================================================================="

# --- 1. Preprocess (skip if already done) -----------------------------------
if [[ -d "${DATA_ROOT}/${DATASET}/features" ]] && \
   [[ -n "$(ls -A "${DATA_ROOT}/${DATASET}/features" 2>/dev/null)" ]]; then
  echo "[1/3] Preprocessed data already exists at ${DATA_ROOT}/${DATASET}/ — skipping."
else
  echo "[1/3] Preprocessing raw LARa CSVs -> ${DATA_ROOT}/${DATASET}/ ..."
  python src/data/lara.py -i "${RAW_DIR}" -o "${DATA_ROOT}/"
fi

NSEQ=$(ls -1 "${DATA_ROOT}/${DATASET}/features" | wc -l)
echo "      sequences ready: ${NSEQ}"
if [[ "${NSEQ}" -lt 10 ]]; then
  echo "      WARNING: only ${NSEQ} sequences found. Use the FULL LARa set to"
  echo "               approach the paper numbers (too few -> codebook collapse)."
fi

# --- 2. Train ----------------------------------------------------------------
echo "[2/3] Training HiST-VQ on ${DATASET} for ${EPOCHS} epochs ..."
python main.py --action=train --dataset="${DATASET}" --epoch="${EPOCHS}"

# --- 3. Evaluate the final checkpoint ----------------------------------------
CKPT="models/${DATASET}/epoch-${EPOCHS}.model"
echo "[3/3] Evaluating ${CKPT} ..."
python main.py --action=eval --dataset="${DATASET}" --ckpt "${CKPT}" --epoch "${EPOCHS}"

echo "=================================================================="
echo "[HiST-VQ] Done. Paper-level LARa target: MoF ~45.9."
echo "  Tuning knobs to get closer: --lambda_temp (try 0.05/0.1/0.2),"
echo "  --epoch (try 50), and the ablation self-checks in REPRODUCE.md."
echo "=================================================================="
