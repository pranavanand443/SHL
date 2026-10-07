"""Private final report and CSV export, after all model experiments."""
import importlib.metadata
import zipfile

checked = pd.read_csv('/kaggle/working/submission.csv')
assert list(checked.columns) == ['filename', 'label']
assert checked.filename.tolist() == test.filename.tolist()
assert checked.filename.is_unique
assert np.isfinite(checked.label).all() and checked.label.between(0, 5).all()
versions = {name: importlib.metadata.version(name) for name in [
    'numpy', 'pandas', 'scipy', 'scikit-learn', 'librosa', 'soundfile',
    'torch', 'transformers', 'sentence-transformers']}
(OUT/'environment_versions.json').write_text(json.dumps(versions, indent=2))
report = (
    'SHL Grammar Scoring: measured development report\n\n'
    f"Selected model: {final_summary['selected_model']}\n"
    f"Training RMSE (fitted training records): {final_summary['training_metrics']['RMSE']:.6f}\n"
    f"Out-of-fold RMSE: {final_summary['validation_metrics']['RMSE']:.6f}\n"
    f"Out-of-fold Pearson: {final_summary['validation_metrics']['Pearson']:.6f}\n"
    f"Validation: {validation_type}.\n"
    f"Training records: {len(train)}. Test predictions: {len(checked)}.\n\n"
    'Method: complete-recording acoustic statistics; fixed pretrained Whisper\n'
    'transcription and speech representations; MiniLM text representations;\n'
    'linguistic descriptors and in-fold TF-IDF. Supervised regressors use only\n'
    'the supplied training labels. Scaling and learned text vocabulary are\n'
    'fitted inside folds. The selected model is refitted on all training labels.\n\n'
    'The supplied sample contains stale identifiers. The output instead\n'
    'matches all current test.csv rows and retains filename,label columns.\n\n'
    'Limitations: model selection uses development folds, not an independent\n'
    'test. Speaker/prompt identifiers are unavailable. ASR can omit errors or\n'
    'hallucinate words, semantics can encode topic, and acoustic cues may\n'
    'capture recording conditions. No private hiring threshold is assumed.\n'
    'Private recordings, transcripts, labels and caches are not public code.\n')
(OUT/'final_report.txt').write_text(report)
with zipfile.ZipFile('/kaggle/working/shl_final_results_private.zip', 'w',
                     compression=zipfile.ZIP_DEFLATED) as bundle:
    bundle.write('/kaggle/working/submission.csv', 'submission.csv')
    for name in ['final_summary.json', 'all_validation_metrics.csv',
                 'final_report.txt', 'environment_versions.json',
                 'final_validation_plot.png']:
        bundle.write(OUT/name, name)
print(report)
print('SHL_EXPORT_SUMMARY_BEGIN')
print(json.dumps(final_summary))
print('SHL_EXPORT_SUMMARY_END')
print('SHL_EXPORT_CSV_BEGIN')
print(checked.to_csv(index=False), end='')
print('SHL_EXPORT_CSV_END')
