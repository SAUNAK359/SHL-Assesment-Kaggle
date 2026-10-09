# SHL Grammar Scoring - V8

This repository contains the V8 solution for the private SHL grammar-scoring challenge. The model predicts a continuous grammar score from spoken English recordings on a 0 to 5 scale.

## Submission status

| Field | Value |
| --- | --- |
| Kaggle username | `saunakdaschaudhuri` |
| Kaggle kernel | `saunakdaschaudhuri/shl-embedded` |
| Current Kaggle score | **0.3353** (provided by the author) |
| Final submission | [`submissions/submission.csv`](submissions/submission.csv) |
| Submission rows | 216 |
| Recorded V7 public score | 0.34 |
| GitHub repository | https://github.com/SAUNAK359/SHL-Assesment-Kaggle |

The score `0.3353` is the current leaderboard value supplied by the author. It is not present in the exported V8 metrics JSON, whose `v8_leaderboard_score` field is `null`. The V8 local metrics below are validation diagnostics, not leaderboard scores.

## Results

| Evaluation | RMSE | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Selected OOF development | 0.484167 | 0.364305 | 0.920491 | 0.883383 |
| Held-out audit, 116 recordings | 0.512449 | 0.392706 | 0.906029 | 0.866285 |
| Partly in-sample 5-fold ensemble | 0.217282 | 0.166448 | 0.985328 | 0.977715 |

The detailed technical report is in [`REPORT.md`](REPORT.md). The source notebook is [`notebooks/shl-embedded-v8.ipynb`](notebooks/shl-embedded-v8.ipynb).

## Approach

- Decode recordings to mono 16 kHz audio and cover each full recording with overlapping windows.
- Extract multi-level WavLM audio features, speaker features, acoustic/fluency features, Whisper transcripts, and a second CTC ASR view.
- Encode transcript chunks with DeBERTa and retain ASR confidence, entropy, blank-rate, and transcript-agreement features.
- Train fold-local Ridge, RBF, CatBoost, residual, ordinal, ranking, cross-modal, and supervised fine-tuning candidates.
- Blend out-of-fold predictions with non-negative weights that sum to one, using only training-fold information for preprocessing and tuning.
- Clip final predictions to the valid 0 to 5 range and preserve the test file order in the submission.

## Repository layout

```text
README.md
REPORT.md
LICENSE
notebooks/shl-embedded-v8.ipynb
reports/                 # metrics, plots, training history, and manifests
submissions/submission.csv
scripts/validate_submission.py
data/README.md           # dataset handling note
```

The original 3.1 GB all-artifacts archive and the generated feature/cache files are kept locally under `Version8/` and `artifacts/v8/`. They are excluded from Git because they are oversized generated files and include competition data derivatives. The compact report artifacts required to review the work are tracked under `reports/`.

## Validation

Run the dependency-free submission check from the repository root:

```bash
python scripts/validate_submission.py submissions/submission.csv
```

This checks the `filename,label` schema, row count, duplicate IDs, numeric values, finite values, and the valid score range.

## Reproduction

The notebook was executed on Kaggle with Python 3.13, Torch 2.11, Transformers 4.57, scikit-learn 1.6, and NumPy 2.1. Re-running it requires the private SHL competition dataset and the pinned pretrained model downloads described in the notebook and report.

