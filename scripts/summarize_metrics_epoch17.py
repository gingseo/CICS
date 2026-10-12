"""Export cross-metric tables and a scale-normalized comparison figure."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/metric_comparison_epoch17'
state = json.loads((REPORT / 'status.json').read_text())
if state['status'] != 'DONE':
    raise RuntimeError('Wait for the complete metric sweep before summarizing')
rows = []
for run in state['runs']:
    folder = ROOT / run
    manifest = json.loads((folder / 'manifest.json').read_text())
    stats = json.loads((folder / 'statistics.json').read_text())
    summary = pd.read_csv(folder / 'summary.csv').set_index('region')
    assert stats['n_images'] == 1610 and stats['n_objects'] == 75102
    bg = stats['global_bg']['mean']
    row = dict(metric=manifest['metric'], ratio=manifest['ratio'], stage=manifest['stage'],
               n_images=stats['n_images'], n_objects=stats['n_objects'], BG=bg,
               spearman_original=stats['correlations']['original_size_spearman']['coefficient'],
               spearman_input=stats['correlations']['input_size_spearman']['coefficient'],
               run=run)
    for size in ['VT', 'T', 'S', 'M', 'L']:
        row[size] = float(summary.loc[size, 'mean'])
        row[size + '_over_BG'] = row[size] / bg
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(REPORT / 'comparison.csv', index=False)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
sizes = ['VT', 'T', 'S', 'M', 'L']
for ax, ratio in zip(axes, [.75, .5, .25]):
    for metric in ['cosine', 'l1', 'l2']:
        row = table[(table.ratio == ratio) & (table.metric == metric)].iloc[0]
        ax.plot(sizes, [row[s + '_over_BG'] for s in sizes], marker='o', label=metric)
    ax.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'Downsample ratio = {ratio}')
    ax.set_xlabel('GT size bin (original image)')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Mean object discrepancy / global BG mean')
axes[-1].legend()
fig.suptitle('Epoch 17 / C3 / full test: 1,610 images, 75,102 objects')
fig.tight_layout()
fig.savefig(REPORT / 'comparison.png', dpi=180)
fig.savefig(REPORT / 'comparison.svg')
plt.close(fig)
notes = ['# Epoch 17 C3 metric comparison', '',
         'Full test split; short edge 800, maximum long edge 1333; fixed epoch 17 weights.',
         'Cosine = 1 - cosine similarity; L1 = mean absolute channel difference; L2 = root mean squared channel difference.',
         'Objects are equally weighted instances. Global BG is weighted by valid input pixels.',
         'Metric magnitudes have different scales. Ratios to BG are descriptive contrasts, not percentages of information loss.',
         'Correlations pool objects across images and do not control for scene/class/feature magnitude.',
         'Test was also used for checkpoint selection; these are exploratory statistics, not independent holdout results.', '',
         '```csv', table[['metric','ratio','BG',*sizes,'spearman_original','spearman_input']].to_csv(index=False).strip(), '```', '',
         'See comparison.csv for run paths and normalized contrasts; original per-run CSV/plots are retained.']
(REPORT / 'README.md').write_text('\n'.join(notes) + '\n')
print(table[['metric','ratio','BG',*sizes,'spearman_original','VT_over_BG']].to_string(index=False))
