# SHL 2026 — V8 assessment report

## Approach and preprocessing
Only train.csv provides labels. WAV recordings are decoded to mono 16 kHz; the full recording is covered by audio windows.
WavLM multi-layer statistics, a separate speaker encoder, acoustic/fluency features and CTC speech representations
provide complementary information. Whisper and CTC generate independent complete transcriptions. CTC overlap is trimmed
at frame level. Confidence, entropy, blank rate and transcript agreement expose ASR uncertainty. No grammatical correction
prompts or invented grammar annotations are used. Raw continuous MOS labels remain in [0,5].

## Pipeline architecture and training
V8 learns SHL MOS by fine-tuning the last 2 WavLM transformer blocks and last 4
DeBERTa blocks. Frame/token attention, recording/chunk attention, learned positions, text-layer mixing and hierarchical
transformers support recording-level prediction. A separate cross-modal transformer reads audio, two text views and
fluency features. Existing residual, ordinal and ranking heads and tuned Ridge/Kernel Ridge/CatBoost baselines remain
candidates. A training-only confidence-gated retrieval candidate uses no filename or test label transfer.

Feature scaling, baseline tuning and epoch selection use only each fold's training data. Inner splits select epochs;
models then restart from the pretrained initialization and refit on the full outer training partition. Separate learning
rates, staged encoder warmup, accumulation, clipping, warmup/cosine scheduling and EMA stabilize training. Epoch checkpoints
store optimizer, scaler, RNG, EMA and history; completed learners retain compact tunable weights plus immutable model commits.

## Required training RMSE and evaluation
- **Training-data selected OOF RMSE (development diagnostic): 0.484167**
- **Training RMSE (5-fold ensemble, partly in-sample): 0.217282**
- Selected OOF Pearson: 0.9204911564465872; MAE: 0.364305; Spearman: 0.883383055492771.
- Held-out audit RMSE: **0.512449**; Pearson: **0.9060288947661722**; 116 audit recordings. Descriptive group-bootstrap RMSE interval: [0.44425470912180337, 0.5782539761163651].
- Same-run reference-family OOF RMSE: 0.485934.

Ensemble weights and calibration are fitted on training OOF predictions; calibration selection: **raw**.
Selected OOF metrics are affected by stacking/calibration selection and are not a fully nested validation of stacking.
The held-out audit is excluded from all fitted choices for this recipe, scored once after those choices are fixed,
and then the unchanged procedure is refitted on all labels. It uses a smaller development training set. The partly
in-sample training RMSE is optimistic. Neither the reference-family result nor the OOF result replays the prior Kaggle
submission exactly. Your **reported V7 public score was 0.34**; V8 public/private scores and leaderboard rank are **unmeasured**.
The provided competition description mentions Pearson and RMSE without an exact custom formula.

## Interpretability
The notebook plots actual/predicted MOS, residuals, per-score errors, prediction distributions, candidate ablations,
internal epoch curves and the audit. High-error transcripts and ensemble weights are displayed. Fold disagreement is
reported as a diagnostic, not a calibrated confidence interval or causal explanation.

## Submission and reproducibility
Train rows: 769; test rows: 216; original supplied template rows: 204.
Template status: **incompatible IDs; schema reconstructed from test.csv**. Primary submission rows: 216, matched by normalized filename and clipped
to [0,5]. Incompatible template IDs are not assigned fabricated scores; a reconstructed schema uses current test IDs.
A separate ID audit records which original template IDs can be predicted. Submit the schema Kaggle currently accepts.
Feature key: 8f156ae3551657ca; training key: f2e67ed5d8d81480; profile: deep; seed: 42.
Implementation SHA: de9a837b4f872dbe1c3527eca1f650aeddf0b9ed524c770adb355aa67fa863c2. Immutable pretrained commits: {'microsoft/wavlm-base-plus': '4c66d4806a428f2e922ccfa1a962776e232d487b', 'openai/whisper-large-v3-turbo': '41f01f3fe87f28c78e2fbf8b568835947dd65ed9', 'microsoft/deberta-v3-base': '8ccc9b6f36199bec6961081d44eb72fb3f7353f3', 'microsoft/wavlm-base-plus-sv': 'feb593a6c23c1cc3d9510425c29b0a14d2b07b1e', 'facebook/wav2vec2-large-960h-lv60-self': '54074b1c16f4de6a5ad59affb4caa8f2ea03a119'}.
Runtime: Python 3.13.15, Torch 2.11.0+cu128, Transformers 4.57.1,
scikit-learn 1.6.1, NumPy 2.1.3.

## Limitations
Only 769 labeled recordings are available. Deeper learners can overfit, alternate ASR can be worse, and speaker/style
features may transfer poorly. Real group column: None; without real candidate IDs, exact-duplicate grouping does
not establish unseen-speaker accuracy. No target-distribution stretching, test-label tuning, pseudo-labeling or public-score
fitting is used. Larger models and more epochs do not guarantee a lower score. The full deep profile may exceed a single
Kaggle session; resume from preserved caches/checkpoints. Inference requires the pinned pretrained weights and fold artifacts.
