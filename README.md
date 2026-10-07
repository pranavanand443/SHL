SHL Grammar Scoring - research assignment
========================================

The editable notebook is SHL_Grammar_Scoring.ipynb.
Run it privately on Kaggle with SHL Hiring Assessment 2026 attached.
The acoustic stage works on CPU. The transcript experiment needs internet
and is much faster on a GPU; enable free GPU T4 x2 after phone verification.

Files
-----
baseline.py: complete-recording audio features and five-fold regression comparison.
transcript_stage.py: Whisper transcription, MiniLM embeddings, linguistic
descriptors, fold-fitted TF-IDF and model comparison. Run after baseline.py.
speech_stage.py: fixed Whisper speech representations and comparison.
Run after transcript_stage.py in the same notebook session.
delivery_stage.py: final measured report, environment versions and private CSV export.
SHL_Grammar_Scoring.ipynb: these stages separated into documented notebook cells.

Measured first run, 7 October 2026
--------------------------------
Training rows: 769. Current test rows: 216.
Selected acoustic model: standardized support-vector regression.
Fitted-training RMSE: 0.1400988633.
Five-fold out-of-fold RMSE: 0.6964201784.
Out-of-fold Pearson correlation: 0.8269087297.
Constant-mean reference RMSE: 1.2382789488.
These are development validation results, not a Kaggle leaderboard score.
Final completed comparison
--------------------------
Selected model: SVR combining acoustic, text and fixed speech representations.
Fitted-training RMSE: 0.1365453254.
Five-fold out-of-fold RMSE: 0.5617009970.
Out-of-fold Pearson correlation: 0.8994185092.
Speech-only SVR RMSE: 0.5653678913.
Text-only Ridge RMSE: 1.0392754314 (alpha=100).
Text additions alone did not improve the acoustic reference. Fixed speech
representations provided the main improvement. The combined result is only
slightly better than speech-only SVR; these are development selection results.
All 985 recordings were transcribed, with no empty transcripts.
The final CSV has 216 unique identifiers and finite predictions in 0-5.
Kaggle leaderboard acceptance and score are separate from these CV results.

Important data contract
-----------------------
train.csv supplies the labels. test.csv labels are -1 placeholders and must
never be used for fitting. The real target range includes zero and extends
to five; predictions are continuous, not rounded score classes.
The supplied sample CSV contains 204 filenames, with only 25 matching the
current test set. Retain filename,label columns but use all 216 test.csv
identifiers in their original order. Validate acceptance on Kaggle.

Generated private outputs
-------------------------
/kaggle/working/submission.csv: current selected model's predictions.
/kaggle/working/shl_results/: audit, validation tables, summaries, plots,
models, feature caches and (after transcription) transcripts.
The transcript stage replaces submission.csv only after model evaluation.

Validation and limitations
--------------------------
Scaling and supervised fitting take place inside each training fold.
Exact duplicate training recordings are grouped together when found.
The first run found no exact duplicates and stratified five folds by broad
score intervals. Speaker and prompt groups are not provided in the CSV,
so this cannot establish speaker-independent or prompt-independent results.
The same CV results guide model selection and are development estimates.
Acoustic features may measure fluency or recording conditions instead of
grammar. ASR may omit errors or hallucinate words. Semantic embeddings may
encode topic. Compare additions against the acoustic reference honestly.
No qualifying threshold, interview outcome or leaderboard score is promised.

Delivery
--------
SHL requests a commented notebook with a brief report and TRAINING RMSE.
Keep training and out-of-fold RMSE clearly distinguished in that report.
The hiring email also requests a public code repository, the output file,
and completion of the supplied SHL form. Prepare and review those before
the final submission. Do not publish competition recordings, transcripts,
CSV labels, cached data, predictions or executed outputs in the public repo.
The source files and an output-cleared notebook can be published as code.

Pretrained-model references
----------------------------
Whisper: https://huggingface.co/openai/whisper-small.en
MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Pretrained models are allowed by this competition; no external labeled
grammar-scoring dataset is added, and private audio stays in Kaggle.
