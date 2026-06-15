# HiST-VQ (Reproduction)

Unofficial reproduction of **"Unsupervised Skeleton-Based Action Segmentation via
Hierarchical Spatiotemporal Vector Quantization" (HiST-VQ)**, built on top of the
**SMQ** codebase (Gökay et al., ICCV 2025, `github.com/bachlab/SMQ`), which the
HiST-VQ paper states it extends.

> Status: **runs end-to-end (train + eval + Hungarian metrics)** on LARa.
> This is a faithful-to-spec reproduction aimed at correctness of the pipeline,
> not at matching the paper's headline numbers.

## What HiST-VQ changes vs SMQ

| Module | SMQ | HiST-VQ |
|---|---|---|
| Input | skeleton `S` | skeleton `S` + patch-level timestamps `T` |
| Encoder | joint-decoupled MS-TCN | same (reused) |
| Patching | 1-second non-overlapping patches | same (reused) |
| Quantization | **single** codebook | **two-level cascade**: sub-action `Z` (αK) → action `A` (K) |
| Commitment loss | one | **two** (`L_commitZ`, `L_commitA`) |
| Decoding | one spatial decoder → skeleton | **spatial decoder (Q_A→skeleton) + temporal decoder (Q_Z→timestamps)** |
| Reconstruction loss | inter-joint MSE | inter-joint MSE **+ timestamp MSE** |
| Codebook update | EMA | two-level EMA + dead-code revival |
| Segmentation export | code index | **second-level (action) index** |

The encoder, joint-decoupling, inter-joint distance loss, EMA mechanism and the
Hungarian evaluation protocol are reused unchanged from SMQ.

## Key files

- `src/model/hist_quantizer.py` — **[new]** two-level cascaded patch VQ (`p → z → a`),
  dual commitment, two-level EMA, dead-code revival.
- `src/model/histvq.py` — **[new]** HiST-VQ model: encoder + hierarchical VQ +
  spatial decoder (from `Q_A`) + temporal-decoder MLP (from `Q_Z`).
- `model.py` — Trainer with the four losses and CTE-style timestamp construction.
- `main.py` — CLI/config (adds `--alpha`, `--lambda_commit/spat/temp`, `--nu_z/nu_a`).
- Reused from SMQ: `src/model/ms_tcn.py`, `src/model/utils.py`,
  `src/model/eval_utils.py`, `src/model/motion_quantizer.py` (helpers),
  `batch_gen.py`, `src/data/*` (preprocessing).

See `REPRODUCE.md` for the full implementation notes, hyper-parameters, and the
smoke-test commands/results.

## Quick start (LARa)

```bash
pip install torch numpy scipy scikit-learn tslearn matplotlib tqdm pandas

# train
python main.py --action=train --dataset=lara --epoch=30

# eval (Hungarian MoF / Edit / F1@{10,25,50})
python main.py --action=eval --dataset=lara --ckpt models/lara/epoch-30.model --epoch 30
```

### One-click full pipeline

Download the full LARa v3 (OMoCap) CSVs from
https://zenodo.org/records/8189341, then:

```bash
bash run_lara.sh /path/to/raw_lara 30     # preprocess -> train -> eval
```

GPU is used automatically if available (recommended). Use the **full** LARa set
(not a few sequences) to approach the paper's MoF ~45.9; too few sequences cause
codebook collapse.
