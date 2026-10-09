# SHL Grammar Scoring - V8 Technical Report

## 1. Executive summary

The solution predicts a continuous spoken-grammar score from audio recordings. V8 combines acoustic, speaker, speech-recognition, and language-model representations, then blends several leakage-controlled candidate models.

The current Kaggle submission score is **0.3353**, supplied by the author for this report. The checked-in artifact bundle does not contain an official V8 leaderboard response; its `v8_leaderboard_score` value is `null`. The previously recorded V7 public score was `0.34`.

The strongest artifact-backed local diagnostics are:

- Selected development OOF RMSE: `0.484167`
- Held-out audit RMSE: `0.512449`
- Held-out audit MAE: `0.392706`
- Held-out audit Pearson: `0.906029`
- Held-out audit Spearman: `0.866285`

RMSE is lower-is-better. Correlation metrics are higher-is-better. Local validation numbers must not be presented as Kaggle leaderboard scores.

## 2. Submission metadata

| Field | Value |
| --- | --- |
| Kaggle username | `saunakdaschaudhuri` |
| Kernel | `saunakdaschaudhuri/shl-embedded` |
| Current Kaggle score | **0.3353**, author-provided |
| Submission file | [`submissions/submission.csv`](submissions/submission.csv) |
| Submission schema | `filename,label` |
| Submission rows | 216 |
| GitHub | https://github.com/SAUNAK359/SHL-Assesment-Kaggle |

## 3. Dataset and integrity checks

The executed notebook found 769 labeled training recordings, 216 test recordings, and 204 rows in the original sample template. The test set was kept unlabeled during model selection.

Integrity checks recorded by the notebook:

- Train and test audio were resolved within their own split directories.
- Exact audio-content grouping produced 769 unique groups and no repeated training content.
- No exact audio content was shared between train and test.
- No real speaker or candidate ID was supplied, so unseen-speaker generalization is not directly measurable.
- The original template contained incompatible IDs: 179 unknown IDs and only 25 of 204 IDs covered by the current test set.
- The primary submission therefore reconstructs the accepted schema from `test.csv` and contains all 216 current test IDs.

The source archive contains 6,341 entries, including generated caches and feature tensors. The reviewable compact result files were unpacked locally; large generated caches are excluded from the public Git history.

## 4. Feature and modeling pipeline

### Audio processing

Audio is converted to mono 16 kHz without deleting pauses or applying grammar-correction heuristics. Each complete recording is represented with overlapping 15-second windows. WavLM contributes multi-level audio representations and a separate speaker encoder contributes speaker-style features.

### Speech recognition and text

Whisper provides the primary transcription. A CTC recognizer provides a second ASR view using overlapping 20-second core regions with context. Frame overlap is trimmed before joining. The pipeline records transcript confidence, entropy, blank rate, and agreement between the two ASR views. DeBERTa encodes transcript chunks with overlap-aware tokenization.

### Candidate models

The V8 ensemble evaluates linear and nonlinear multimodal candidates, including Ridge, RBF regression, CatBoost, residual, soft-ordinal, attention-ranking, cross-modal, DeBERTa fine-tuning, and WavLM fine-tuning. Fold-local scaling and model tuning prevent validation information from leaking into the training features.

### Blend and calibration

Non-negative ensemble weights were fitted on development OOF predictions with a small regularization penalty. The selected calibration was the raw blend. The largest weights were:

| Candidate | Weight |
| --- | ---: |
| attention_rank | 0.238063 |
| speaker_retrieval | 0.169725 |
| fusion_rbf | 0.159081 |
| residual | 0.141402 |
| cross_modal | 0.137620 |
| deberta_dual_view | 0.107134 |
| wavlm_finetuned | 0.046975 |

The complete weight table is in [`reports/ensemble_weights_v8.csv`](reports/ensemble_weights_v8.csv).

## 5. Training and validation metrics

| Evaluation | N | RMSE | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected OOF development | 769 | 0.484167 | 0.364305 | 0.920491 | 0.883383 |
| Partly in-sample 5-fold ensemble | 769 | 0.217282 | 0.166448 | 0.985328 | 0.977715 |
| Held-out audit | 116 | 0.512449 | 0.392706 | 0.906029 | 0.866285 |
| Reference-family OOF | 769 | 0.485934 | 0.366716 | 0.919871 | 0.883027 |

The audit was fixed before viewing final model results, used 653 development recordings and 116 audit recordings, and was scored once after the recipe was fixed. Its descriptive group-bootstrap 95% RMSE interval was `[0.444255, 0.578254]`.

The partly in-sample result is optimistic and is included only as a training-fit diagnostic. The selected OOF and held-out audit are the appropriate local generalization references.

### Error by target band

| Score band | N | RMSE | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-1 | 37 | 0.139785 | 0.062350 | - | - |
| 1-2 | 4 | 0.945809 | 0.867403 | -0.889873 | -0.774597 |
| 2-3 | 181 | 0.536052 | 0.428790 | 0.208135 | 0.191249 |
| 3-4 | 238 | 0.475676 | 0.377906 | 0.245107 | 0.243719 |
| 4-5 | 309 | 0.476311 | 0.345701 | 0.666104 | 0.687918 |

The 1-2 band contains only four examples, so its correlation values are unstable and should not be over-interpreted.

## 6. Submission construction

The final file is [`submissions/submission.csv`](submissions/submission.csv). It has exactly two columns, `filename` and `label`, with 216 rows, unique test filenames, finite numeric predictions, and labels clipped to `[0, 5]`. The notebook also wrote `submission_all_test.csv` and `submission_sample_format.csv`; all three contain the current 216-row test schema.

## 7. Reproducibility artifacts

- Notebook: [`notebooks/shl-embedded-v8.ipynb`](notebooks/shl-embedded-v8.ipynb)
- Metrics JSON: [`reports/metrics_v8.json`](reports/metrics_v8.json)
- Audit metrics: [`reports/audit_metrics_v8.json`](reports/audit_metrics_v8.json)
- Training history: [`reports/training_history_v8.csv`](reports/training_history_v8.csv)
- Training curves: [`reports/training_curves_v8.png`](reports/training_curves_v8.png)
- Diagnostics: [`reports/diagnostics_v8.png`](reports/diagnostics_v8.png)
- Feature/model configuration: [`reports/experiment_manifest_v8.json`](reports/experiment_manifest_v8.json)
- Compact notebook report: [`reports/assessment_report_v8.md`](reports/assessment_report_v8.md)

The notebook recorded the following core environment versions: Python 3.13.15, Torch 2.11.0+cu128, Transformers 4.57.1, scikit-learn 1.6.1, and NumPy 2.1.3.

## 8. Limitations and honest interpretation

- The current score `0.3353` is supplied by the author and is not independently retrievable in this environment.
- The V8 exported metrics record no leaderboard score, so the public score must be documented separately from local metrics.
- Only 769 labeled recordings are available, which limits the reliability of deeper fine-tuned models.
- No real speaker IDs were available; exact audio identities do not establish unseen-speaker accuracy.
- The original sample template IDs were incompatible with the current test IDs, so the submission schema was reconstructed from the test set.
- The previous `0.34` value belongs to V7 and should not be claimed as the V8 score.

