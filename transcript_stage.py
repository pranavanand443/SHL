"""Run after baseline.py in the same private Kaggle notebook.

Requires pretrained Whisper and MiniLM downloads (or local model directories).
All speech is processed inside the notebook; no inference API receives audio.
"""
# %% Transcribe complete recordings
import gc
import re
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

ASR_MODEL = 'openai/whisper-small.en'
TEXT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
gpu = torch.cuda.is_available()
torch.manual_seed(SEED)
torch.set_num_threads(2)
transcript_file = OUT / 'transcripts_private.json'
records = json.loads(transcript_file.read_text()) if transcript_file.exists() else {}
asr = None
started = time.time()
for i, ((split, name), path) in enumerate(zip(manifest, paths), 1):
    key = split + '/' + name
    if key in records and records[key].get('model') == ASR_MODEL:
        continue
    if asr is None:
        asr = pipeline('automatic-speech-recognition', model=ASR_MODEL,
                       device=0 if gpu else -1,
                       torch_dtype=torch.float16 if gpu else torch.float32,
                       chunk_length_s=30, stride_length_s=(5, 5))
    # Preserve what was spoken. Do not grammar-correct the transcript.
    result = asr({'array': read_audio(path), 'sampling_rate': SR},
                 batch_size=8 if gpu else 1,
                 generate_kwargs={'do_sample': False, 'num_beams': 1})
    records[key] = {'model': ASR_MODEL, 'text': result['text'].strip()}
    # Atomic checkpoints allow interruption and continuation.
    temporary = transcript_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, ensure_ascii=False))
    temporary.replace(transcript_file)
    if i % 10 == 0 or i == 1:
        print(f'Transcribed {i}/{len(paths)}; {time.time()-started:.0f}s', flush=True)
del asr
gc.collect()
if gpu:
    torch.cuda.empty_cache()
texts = [records[split+'/'+name]['text'] for split, name in manifest]
print('Transcript count:', len(texts), '; empty:', sum(not t for t in texts), flush=True)

# %% Fixed pretrained text embeddings and speech descriptors
encoder = SentenceTransformer(TEXT_MODEL, device='cuda' if gpu else 'cpu')
chunks, owner, weights = [], [], []
for i, text in enumerate(texts):
    tokens = encoder.tokenizer.encode(text, add_special_tokens=False)
    # MiniLM truncates long inputs by default. Cover the entire transcript.
    for start in range(0, max(1, len(tokens)), 220):
        part = tokens[start:start+220]
        chunks.append(encoder.tokenizer.decode(part) if part else '')
        owner.append(i)
        weights.append(max(1, len(part)))
encoded = encoder.encode(chunks, batch_size=32 if gpu else 8,
                         normalize_embeddings=True, show_progress_bar=True)
embeddings = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
totals = np.zeros(len(texts), dtype=np.float32)
for vector, i, weight in zip(encoded, owner, weights):
    embeddings[i] += vector * weight
    totals[i] += weight
embeddings /= totals[:, None]
del encoder, encoded
gc.collect()
if gpu:
    torch.cuda.empty_cache()

def text_descriptors(text, duration):
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
    n = len(words)
    filler = sum(w in {'um', 'uh', 'erm', 'hmm'} for w in words)
    repeats = sum(a == b for a, b in zip(words, words[1:]))
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    return [n, n/max(duration, 1e-6)*60, len(set(words))/max(n, 1),
            np.mean([len(w) for w in words]) if words else 0,
            n/sentences, filler/max(n, 1), repeats/max(n, 1), float(n == 0)]

descriptors = np.array([text_descriptors(t, float(f[-5]))
                        for t, f in zip(texts, all_features)], dtype=np.float32)
fixed_text = np.hstack([embeddings, descriptors])
fixed_fusion = np.hstack([all_features, fixed_text])
assert np.isfinite(fixed_fusion).all()
np.savez_compressed(OUT/'text_features_private.npz',
                    embeddings=embeddings, descriptors=descriptors,
                    manifest_id=manifest_id)

# %% Compare acoustic, text and combined models on the same held-out folds
# The encoder is fixed, so extracting embeddings before CV does not learn labels.
# Scaling, TF-IDF vocabulary/IDF and supervised regression are learned in-fold.
candidates = {
    'Text Ridge 10': (fixed_text, make_pipeline(StandardScaler(), Ridge(alpha=10))),
    'Text Ridge 100': (fixed_text, make_pipeline(StandardScaler(), Ridge(alpha=100))),
    'Combined Ridge 100': (fixed_fusion, make_pipeline(StandardScaler(), Ridge(alpha=100))),
    'Combined SVR': (fixed_fusion, make_pipeline(StandardScaler(), SVR(C=10, epsilon=.15))),
}
extended = comparison.copy()
full_models = {}
all_oof = dict(oof_predictions)
with threadpool_limits(limits=1):
    for name, (features, estimator) in candidates.items():
        oof = np.full(len(y), np.nan)
        fold_errors = []
        for fit_ids, valid_ids in folds:
            fitted = clone(estimator).fit(features[fit_ids], y[fit_ids])
            p = np.clip(fitted.predict(features[valid_ids]), 0, 5)
            oof[valid_ids] = p
            fold_errors.append(metrics(y[valid_ids], p)['RMSE'])
        row = {'model': name, **metrics(y, oof),
               'fold_RMSE_sd': float(np.std(fold_errors))}
        print(row, flush=True)
        extended.append(row)
        all_oof[name] = oof
        full_models[name] = (features, clone(estimator).fit(features[:len(y)], y))

def fit_lexical(fit_ids, predict_ids):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500,
                           sublinear_tf=True)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                           max_features=5000, sublinear_tf=True)
    scale = StandardScaler()
    fit_texts = [texts[i] for i in fit_ids]
    pred_texts = [texts[i] for i in predict_ids]
    a = hstack([word.fit_transform(fit_texts), char.fit_transform(fit_texts),
                csr_matrix(scale.fit_transform(fixed_fusion[fit_ids]))], format='csr')
    b = hstack([word.transform(pred_texts), char.transform(pred_texts),
                csr_matrix(scale.transform(fixed_fusion[predict_ids]))], format='csr')
    fitted = Ridge(alpha=100, solver='lsqr').fit(a, y[fit_ids])
    return fitted.predict(b), (word, char, scale, fitted), fitted.predict(a)

with threadpool_limits(limits=1):
    lexical_oof = np.full(len(y), np.nan)
    fold_errors = []
    for fit_ids, valid_ids in folds:
        p, _, _ = fit_lexical(fit_ids, valid_ids)
        p = np.clip(p, 0, 5)
        lexical_oof[valid_ids] = p
        fold_errors.append(metrics(y[valid_ids], p)['RMSE'])
    row = {'model': 'Combined lexical Ridge', **metrics(y, lexical_oof),
           'fold_RMSE_sd': float(np.std(fold_errors))}
    extended.append(row)
    all_oof[row['model']] = lexical_oof
    print(row, flush=True)
all_scores = pd.DataFrame(extended).sort_values('RMSE').reset_index(drop=True)
all_scores.to_csv(OUT/'all_validation_metrics.csv', index=False)
selected_name = all_scores.iloc[0]['model']
with threadpool_limits(limits=1):
    if selected_name == 'Combined lexical Ridge':
        final_test, selected_model, final_train = fit_lexical(
            np.arange(len(y)), np.arange(len(y), len(texts)))
    elif selected_name in full_models:
        features, selected_model = full_models[selected_name]
        final_train = selected_model.predict(features[:len(y)])
        final_test = selected_model.predict(features[len(y):])
    else:
        selected_model = clone(models[selected_name]).fit(X, y)
        final_train = selected_model.predict(X)
        final_test = selected_model.predict(X_test)
final_train, final_test = np.clip(final_train, 0, 5), np.clip(final_test, 0, 5)
final_submission = make_submission(test, final_test)
final_submission.to_csv('/kaggle/working/submission.csv', index=False)
joblib.dump(selected_model, OUT/'selected_model.joblib')
pd.DataFrame({'target': y, 'prediction': all_oof[selected_name]}).to_csv(
    OUT/'oof_predictions_private.csv', index=False)
final_summary = {
    'selected_model': selected_name, 'asr_model': ASR_MODEL, 'text_model': TEXT_MODEL,
    'training_metrics': metrics(y, final_train),
    'validation_metrics': metrics(y, all_oof[selected_name]),
    'validation_design': validation_type, 'test_rows': len(test),
    'limitations': ['ASR may omit errors or hallucinate words.',
                   'Semantic features may encode topic instead of grammar.',
                   'Model selection uses development CV; it is not an untouched test.',
                   'Unavailable speaker/prompt identifiers limit leakage checks.']}
(OUT/'final_summary.json').write_text(json.dumps(final_summary, indent=2))
print('FINAL MEASURED REPORT:', json.dumps(final_summary, indent=2), flush=True)
display(all_scores)
plt.figure(figsize=(5, 4))
plt.scatter(y, all_oof[selected_name], alpha=.4, s=15)
plt.plot([0, 5], [0, 5], '--', color='black')
plt.xlabel('True score'); plt.ylabel('Out-of-fold prediction')
plt.title(selected_name); plt.tight_layout()
plt.savefig(OUT/'final_validation_plot.png', dpi=160); plt.show()
print('Validated submission.csv:', len(final_submission), 'rows.', flush=True)
