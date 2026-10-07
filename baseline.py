"""Reproducible SHL acoustic baseline. Run on the private Kaggle data.

This establishes the data contract and honest validation before adding ASR.
It uses no filenames as predictive features and no external training data.
"""
# %% Setup
from pathlib import Path
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import resample_poly
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
import joblib
import matplotlib.pyplot as plt

SEED = 42
OUT = Path('/kaggle/working/shl_results')
OUT.mkdir(parents=True, exist_ok=True)
SR = 16000

# %% Data audit
roots = [p.parent for p in Path('/kaggle/input').rglob('train.csv')
         if (p.parent / 'test.csv').is_file()]
if len(roots) != 1:
    raise ValueError(f'Attach exactly one SHL dataset. Found {len(roots)} candidates.')
DATA_ROOT = roots[0]
train = pd.read_csv(DATA_ROOT / 'train.csv')
test = pd.read_csv(DATA_ROOT / 'test.csv')
sample = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
for split, frame in [('train', train), ('test', test)]:
    if list(frame.columns) != ['filename', 'label']:
        raise ValueError(f'Unexpected {split} columns: {list(frame.columns)}')
    if frame.filename.isna().any() or frame.filename.duplicated().any():
        raise ValueError(f'Missing or duplicate identifiers in {split}.')
    if any(Path(name).name != name for name in frame.filename):
        raise ValueError('Audio identifiers must be filenames, not paths.')
    missing = [name for name in frame.filename
               if not (DATA_ROOT / split / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing {split} recordings.')
y = pd.to_numeric(train.label, errors='raise').to_numpy(dtype=np.float64)
assert np.isfinite(y).all() and ((y >= 0) & (y <= 5)).all()
audit = {
    'training_rows': len(train), 'test_rows': len(test), 'sample_rows': len(sample),
    'sample_test_id_overlap': len(set(sample.filename) & set(test.filename)),
    'sample_ids_absent_from_test': len(set(sample.filename) - set(test.filename)),
    'training_label_counts': {str(k): int(v) for k, v in
                              train.label.value_counts().sort_index().items()},
    'submission_identifiers': 'test.csv, in original order',
}
print(json.dumps(audit, indent=2), flush=True)
if set(sample.filename) != set(test.filename):
    print('The supplied sample has incompatible identifiers; retain its columns '
          'and use the current test.csv identifiers.', flush=True)
(OUT / 'data_audit.json').write_text(json.dumps(audit, indent=2))

# %% Audio features
def read_audio(path):
    """Read the entire recording as mono float32 and resample to 16 kHz."""
    audio, original_sr = sf.read(str(path), dtype='float32', always_2d=True)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f'Empty/nonfinite audio: {path.name}')
    audio = audio.mean(axis=1)
    if original_sr != SR:
        divisor = math.gcd(original_sr, SR)
        audio = resample_poly(audio, SR // divisor, original_sr // divisor)
    return np.ascontiguousarray(audio, dtype=np.float32)


def summarize(matrix):
    """Describe variation through time, rather than taking one short segment."""
    return np.concatenate([
        matrix.mean(axis=1), matrix.std(axis=1),
        np.quantile(matrix, .10, axis=1), np.median(matrix, axis=1),
        np.quantile(matrix, .90, axis=1),
    ])


def acoustic_features(path):
    audio = read_audio(path)
    duration = len(audio) / SR
    audio = np.pad(audio, (0, max(0, 1024 - len(audio))))
    spectrum = np.abs(librosa.stft(audio, n_fft=512, hop_length=256))
    power = spectrum ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=SR, n_fft=512,
                                        n_mels=40, fmin=40, fmax=7800)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0), n_mfcc=20)
    delta = librosa.feature.delta(mfcc, mode='nearest')
    rms = librosa.feature.rms(S=spectrum, frame_length=512)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=512,
                                            hop_length=256)
    spectral = np.vstack([
        librosa.feature.spectral_centroid(S=spectrum, sr=SR),
        librosa.feature.spectral_bandwidth(S=spectrum, sr=SR),
        librosa.feature.spectral_rolloff(S=spectrum, sr=SR, roll_percent=.85),
        librosa.feature.spectral_flatness(S=spectrum), rms, zcr,
    ])
    speech = rms[0] > max(1e-5, .10 * float(np.quantile(rms, .95)))
    pauses = ~speech
    starts = int(np.sum(pauses[1:] & ~pauses[:-1]))
    extra = [duration, float(np.max(np.abs(audio))),
             float(np.sqrt(np.mean(audio ** 2))), float(speech.mean()),
             starts / max(duration, 1e-6)]
    features = np.concatenate([summarize(mfcc), summarize(delta),
                               summarize(spectral), extra])
    if not np.isfinite(features).all():
        raise ValueError(f'Nonfinite extracted features: {path.name}')
    return features.astype(np.float32)


manifest = [(split, name) for split, frame in [('train', train), ('test', test)]
            for name in frame.filename]
paths = [DATA_ROOT / split / name for split, name in manifest]
cache = OUT / 'acoustic_features.npz'
manifest_id = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
started = time.time()
with threadpool_limits(limits=1):
    if cache.exists():
        cached = np.load(cache, allow_pickle=False)
        if str(cached['manifest_id']) != manifest_id:
            raise ValueError('Feature cache belongs to a different file manifest.')
        all_features = cached['features']
    else:
        print('Extracting full-recording audio features...', flush=True)
        # Warm up librosa before concurrent calls; restrict BLAS oversubscription.
        first = acoustic_features(paths[0])
        rows = [first]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i, features in enumerate(pool.map(acoustic_features, paths[1:]), 2):
                rows.append(features)
                if i % 50 == 0:
                    print(f'Audio features {i}/{len(paths)}; '
                          f'{time.time() - started:.0f}s elapsed', flush=True)
        all_features = np.vstack(rows)
        np.savez_compressed(cache, features=all_features, manifest_id=manifest_id)
print('Feature matrix:', all_features.shape, flush=True)
X, X_test = all_features[:len(train)], all_features[len(train):]

# %% Validation and model comparison
def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    correlation = (float(np.corrcoef(target, prediction)[0, 1])
                   if target.std() > 1e-12 and prediction.std() > 1e-12 else None)
    return {'RMSE': float(np.sqrt(mean_squared_error(target, prediction))),
            'Pearson': correlation}


# Exact duplicate training files must never appear on opposite sides of a fold.
groups = np.array([hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths[:len(train)]])
duplicate_count = len(groups) - len(set(groups))
if duplicate_count:
    splitter = GroupKFold(n_splits=5)
    folds = list(splitter.split(X, y, groups))
    validation_type = '5-fold validation grouped by exact recording hash'
else:
    bins = np.digitize(y, [.5, 2.5, 3.5, 4.5])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, bins))
    validation_type = '5-fold validation stratified by broad score intervals'
print('Validation:', validation_type, '; duplicate training files:', duplicate_count,
      flush=True)

models = {
    'Mean baseline': DummyRegressor(strategy='mean'),
    'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
    'SVR': make_pipeline(StandardScaler(), SVR(C=10, epsilon=.15)),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                      max_features=.8, random_state=SEED, n_jobs=2),
    'HistogramBoosting': HistGradientBoostingRegressor(
        max_iter=250, max_leaf_nodes=15, l2_regularization=5,
        learning_rate=.05, random_state=SEED),
}
comparison, oof_predictions = [], {}
with threadpool_limits(limits=1):
    for name, estimator in models.items():
        oof = np.full(len(y), np.nan)
        fold_rmse = []
        for fit_ids, valid_ids in folds:
            fitted = clone(estimator).fit(X[fit_ids], y[fit_ids])
            prediction = np.clip(fitted.predict(X[valid_ids]), 0, 5)
            oof[valid_ids] = prediction
            fold_rmse.append(metrics(y[valid_ids], prediction)['RMSE'])
        assert np.isfinite(oof).all()
        row = {'model': name, **metrics(y, oof),
               'fold_RMSE_sd': float(np.std(fold_rmse))}
        comparison.append(row)
        oof_predictions[name] = oof
        print(row, flush=True)
scores = pd.DataFrame(comparison).sort_values('RMSE').reset_index(drop=True)
scores.to_csv(OUT / 'validation_metrics.csv', index=False)
best_name = scores.iloc[0]['model']
best_oof = oof_predictions[best_name]
with threadpool_limits(limits=1):
    best_model = clone(models[best_name]).fit(X, y)
    train_prediction = np.clip(best_model.predict(X), 0, 5)
    test_prediction = np.clip(best_model.predict(X_test), 0, 5)
training_metrics = metrics(y, train_prediction)
validation_metrics = metrics(y, best_oof)
print('Selected acoustic model:', best_name, flush=True)
print('TRAINING RMSE (required, fitted-data estimate):', training_metrics['RMSE'],
      flush=True)
print('OUT-OF-FOLD VALIDATION:', validation_metrics, flush=True)

# %% Submission contract and report
def make_submission(test_frame, prediction):
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(test_frame),) or not np.isfinite(prediction).all():
        raise ValueError('Need one finite prediction per current test identifier.')
    if test_frame.filename.duplicated().any():
        raise ValueError('Test identifiers must be unique.')
    result = pd.DataFrame({'filename': test_frame.filename.to_numpy(),
                           'label': np.clip(prediction, 0, 5)})
    assert list(result.columns) == ['filename', 'label']
    assert result.filename.tolist() == test_frame.filename.tolist()
    return result


submission = make_submission(test, test_prediction)
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.to_csv(OUT / 'submission_acoustic.csv', index=False)
joblib.dump(best_model, OUT / 'acoustic_model.joblib')
summary = {'stage': 'acoustic baseline', 'selected_model': best_name,
           'training_metrics': training_metrics, 'validation_metrics': validation_metrics,
           'validation_design': validation_type, 'seed': SEED,
           'submission_rows': len(submission), 'sample_identifiers_are_stale': True,
           'limitations': ['Acoustic cues alone do not directly measure syntax.',
                          'Model selection uses these validation folds; results '
                          'are development estimates, not an untouched final test.',
                          'Speaker/prompt groups are unavailable in the CSV; '
                          'exact-file grouping does not rule out speaker overlap.']}
(OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(y, bins=np.arange(-.25, 5.51, .5), color='#3b82f6')
axes[0].set(title='Training score distribution', xlabel='SHL grammar score', ylabel='Count')
axes[1].scatter(y, best_oof, alpha=.4, s=15)
axes[1].plot([0, 5], [0, 5], '--', color='black')
axes[1].set(title='Predictions on held-out folds', xlabel='True score', ylabel='Prediction')
axes[2].barh(scores.model, scores.RMSE, color='#10b981')
axes[2].invert_yaxis()
axes[2].set(title='Model comparison', xlabel='Out-of-fold RMSE (lower is better)')
fig.tight_layout()
fig.savefig(OUT / 'validation_plots.png', dpi=160)
plt.show()
print('Saved submission.csv:', len(submission), 'rows; columns filename,label.', flush=True)
print('This is a measured baseline. Add and validate transcript features next.', flush=True)
