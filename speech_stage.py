"""Additional fixed speech-representation experiment after transcript_stage.py."""
# %% Pretrained speech representations
from transformers import AutoFeatureExtractor, WhisperForConditionalGeneration

speech_cache = OUT/'whisper_features_private.npz'
speech_cache_id = hashlib.sha256((manifest_id + ASR_MODEL + ':masked-moments-v1').encode()).hexdigest()
speech_rows = []
if speech_cache.exists():
    saved = np.load(speech_cache, allow_pickle=False)
    if str(saved['cache_id']) != speech_cache_id:
        raise ValueError('Speech cache configuration does not match.')
    speech_rows = list(saved['features'])
assert len(speech_rows) <= len(paths)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
if len(speech_rows) < len(paths):
    extractor = AutoFeatureExtractor.from_pretrained(ASR_MODEL)
    speech_encoder = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL, torch_dtype=dtype).get_encoder().to(device).eval()
    started = time.time()
    for i in range(len(speech_rows), len(paths)):
        waveform = read_audio(paths[i])
        segments = [waveform[start:start+30*SR]
                    for start in range(0, len(waveform), 30*SR)]
        total, sum_vector, sum_squares = 0, None, None
        for start in range(0, len(segments), 4):
            batch = segments[start:start+4]
            inputs = extractor(batch, sampling_rate=SR, return_tensors='pt').input_features
            with torch.inference_mode():
                hidden = speech_encoder(inputs.to(device=device, dtype=dtype)).last_hidden_state
            # Whisper outputs one frame per 20 ms. Exclude padded time positions.
            hidden = hidden.float().cpu().numpy()
            for segment, frames in zip(batch, hidden):
                n_valid = min(len(frames), max(1, math.ceil(len(segment)/(2*extractor.hop_length))))
                frames = frames[:n_valid].astype(np.float64)
                if sum_vector is None:
                    sum_vector = np.zeros(frames.shape[1], dtype=np.float64)
                    sum_squares = np.zeros_like(sum_vector)
                total += len(frames)
                sum_vector += frames.sum(axis=0)
                sum_squares += (frames**2).sum(axis=0)
        mean = sum_vector/total
        standard_deviation = np.sqrt(np.maximum(0, sum_squares/total-mean**2))
        speech_rows.append(np.concatenate([mean, standard_deviation]).astype(np.float32))
        if (i+1) % 50 == 0 or i+1 == len(paths):
            np.savez_compressed(speech_cache, features=np.vstack(speech_rows),
                                cache_id=speech_cache_id)
            print(f'Speech representations {i+1}/{len(paths)}; '
                  f'{time.time()-started:.0f}s', flush=True)
    del speech_encoder, extractor
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
speech_features = np.vstack(speech_rows)
assert np.isfinite(speech_features).all()
speech_fusion = np.hstack([speech_features, fixed_fusion])

# %% Evaluate speech representations and retain the best measured model
speech_candidates = {
    'Speech Ridge': (speech_features, make_pipeline(StandardScaler(), Ridge(alpha=100))),
    'Speech SVR': (speech_features, make_pipeline(StandardScaler(), SVR(C=10, epsilon=.15))),
    'All features SVR': (speech_fusion, make_pipeline(StandardScaler(), SVR(C=10, epsilon=.15))),
}
speech_comparison = all_scores.to_dict('records')
speech_models = {}
with threadpool_limits(limits=1):
    for name, (features, estimator) in speech_candidates.items():
        oof = np.full(len(y), np.nan)
        fold_errors = []
        for fit_ids, valid_ids in folds:
            fitted = clone(estimator).fit(features[fit_ids], y[fit_ids])
            p = np.clip(fitted.predict(features[valid_ids]), 0, 5)
            oof[valid_ids] = p
            fold_errors.append(metrics(y[valid_ids], p)['RMSE'])
        assert np.isfinite(oof).all()
        row = {'model': name, **metrics(y, oof),
               'fold_RMSE_sd': float(np.std(fold_errors))}
        speech_comparison.append(row)
        all_oof[name] = oof
        speech_models[name] = (features, clone(estimator).fit(features[:len(y)], y))
        print(row, flush=True)
all_scores = pd.DataFrame(speech_comparison).sort_values('RMSE').reset_index(drop=True)
all_scores.to_csv(OUT/'all_validation_metrics.csv', index=False)
selected_name = all_scores.iloc[0]['model']
if selected_name in speech_models:
    features, selected_model = speech_models[selected_name]
    final_train = np.clip(selected_model.predict(features[:len(y)]), 0, 5)
    final_test = np.clip(selected_model.predict(features[len(y):]), 0, 5)
    final_submission = make_submission(test, final_test)
    final_submission.to_csv('/kaggle/working/submission.csv', index=False)
    joblib.dump(selected_model, OUT/'selected_model.joblib')
    final_summary.update(selected_model=selected_name,
                         training_metrics=metrics(y, final_train),
                         validation_metrics=metrics(y, all_oof[selected_name]),
                         speech_feature_model=ASR_MODEL)
    (OUT/'final_summary.json').write_text(json.dumps(final_summary, indent=2))
pd.DataFrame({'target': y, 'prediction': all_oof[selected_name]}).to_csv(
    OUT/'oof_predictions_private.csv', index=False)
print('FINAL MEASURED REPORT:', json.dumps(final_summary, indent=2), flush=True)
display(all_scores)
plt.figure(figsize=(5, 4))
plt.scatter(y, all_oof[selected_name], alpha=.4, s=15)
plt.plot([0, 5], [0, 5], '--', color='black')
plt.xlabel('True score'); plt.ylabel('Out-of-fold prediction')
plt.title(selected_name); plt.tight_layout()
plt.savefig(OUT/'final_validation_plot.png', dpi=160); plt.show()
print('Validated submission.csv:', len(final_submission), 'rows.', flush=True)
