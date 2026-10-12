"""Compare full-test C4/C5 results against the completed C3 metric sweep."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/stage_comparison_epoch17'
new = json.loads((REPORT / 'status.json').read_text())
old = json.loads((ROOT / 'reports/metric_comparison_epoch17/status.json').read_text())
assert new['status'] == old['status'] == 'DONE'
assert len(new['runs']) == 12
rows = []
reference = None
reference_objects = None
identity_keys = ['checkpoint_sha256', 'resolved_detector_sha256', 'dataset_fingerprint', 'split', 'limit', 'code_fingerprint']
for run in old['runs'] + new['runs']:
    folder = ROOT / run
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['metric'] not in ('cosine', 'l1'):
        continue
    identity = {key: manifest[key] for key in identity_keys}
    identity['preprocess'] = manifest['config']['preprocess']
    identity['degradation'] = manifest['config']['degradation']
    if reference is None:
        reference = identity
    assert identity == reference, f'Inconsistent experiment inputs: {run}'
    state = json.loads((folder / 'state.json').read_text())
    assert state['status'] == 'DONE'
    stats = json.loads((folder / 'statistics.json').read_text())
    assert stats['n_images'] == 1610 and stats['n_objects'] == 75102
    objects = pd.read_csv(folder / 'objects.csv')
    keys = objects[['image_id', 'object_id', 'size_bin']]
    if reference_objects is None:
        reference_objects = keys
    assert keys.equals(reference_objects), f'Object mismatch: {run}'
    assert np.isfinite(objects.score).all(), f'Invalid object scores: {run}'
    summary = pd.read_csv(folder / 'summary.csv').set_index('region')
    bg = stats['global_bg']['mean']
    row = dict(stage=manifest['stage'], metric=manifest['metric'], ratio=manifest['ratio'],
               n_images=stats['n_images'], n_objects=stats['n_objects'], BG=bg,
               spearman_original=stats['correlations']['original_size_spearman']['coefficient'],
               spearman_input=stats['correlations']['input_size_spearman']['coefficient'], run=run)
    for size in ['VT', 'T', 'S', 'M', 'L']:
        row[size] = float(summary.loc[size, 'mean'])
        row[size + '_over_BG'] = row[size] / bg
    rows.append(row)
table = pd.DataFrame(rows)
assert len(table) == 18 and not table.duplicated(['stage', 'metric', 'ratio']).any()
table.to_csv(REPORT / 'comparison_with_C3.csv', index=False)
table[table.stage != 'C3'].to_csv(REPORT / 'comparison.csv', index=False)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey='row')
sizes = ['VT', 'T', 'S', 'M', 'L']
for i, metric in enumerate(['cosine', 'l1']):
    for j, ratio in enumerate([.75, .5, .25]):
        ax = axes[i, j]
        for stage in ['C3', 'C4', 'C5']:
            row = table[(table.ratio == ratio) & (table.metric == metric) & (table.stage == stage)].iloc[0]
            ax.plot(sizes, [row[s + '_over_BG'] for s in sizes], marker='o', label=stage)
        ax.axhline(1, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'{metric} / ratio = {ratio}')
        ax.set_xlabel('GT size bin (original image)')
        ax.grid(alpha=.2)
        if j == 0:
            ax.set_ylabel('Object mean / global BG mean')
        if j == 2:
            ax.legend()
fig.suptitle('Epoch 17 / full test: 1,610 images, 75,102 objects')
fig.tight_layout()
fig.savefig(REPORT / 'comparison.png', dpi=180)
fig.savefig(REPORT / 'comparison.svg')
plt.close(fig)
verification = dict(status='PASS', conditions=18, new_conditions=12,
                    n_images_per_condition=1610, n_objects_per_condition=75102,
                    matching_identity=reference, identical_object_keys=True, finite_object_scores=True)
(REPORT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
columns = ['stage', 'metric', 'ratio', 'BG', *sizes, 'spearman_original']
notes = ['# Epoch 17 C4/C5 metric comparison', '',
         'New: C4/C5 x ratios 0.75/0.50/0.25 x cosine/L1 = 12 conditions; C3 reused for comparison.',
         'Fixed checkpoint, full test, short edge 800 / maximum long edge 1333.',
         'Cosine = 1 - cosine similarity; L1 = mean absolute channel difference.',
         'Objects are equally weighted; global BG is weighted by valid input pixels.',
         'Backbone C3/C4/C5 have strides 8/16/32. Maps are bilinearly projected to input coordinates before fractional box pooling.',
         'Metric magnitudes across stages/metrics are not directly comparable; BG ratios are descriptive, not information-loss percentages.',
         'Small boxes can share coarse feature support, especially at C4/C5. Pooled size correlations do not control for scenes, classes or feature magnitude.',
         'Checkpoint was selected on test: exploratory results, not independent holdout evidence.', '',
         '```csv', table[columns].to_csv(index=False).strip(), '```', '',
         'verification.json checks input identity, object identity, counts, completion, and finite scores for all 18 conditions.']
(REPORT / 'README.md').write_text('\n'.join(notes) + '\n')
print(table[columns + ['VT_over_BG']].to_string(index=False))
