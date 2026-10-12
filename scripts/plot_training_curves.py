#!/usr/bin/env python3
"""Plot actual MMEngine log epochs (the old monitor history is offset by one)."""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/training_curves'))
    args = parser.parse_args()
    validation = {}
    training = defaultdict(list)
    for line in args.log.read_text().splitlines():
        match = re.search(r'Epoch\((train|val)\)\s*\[(\d+)\]', line)
        if not match:
            continue
        kind, epoch = match.group(1), int(match.group(2))
        values = {key: float(value) for key, value in re.findall(
            r'([\w/]+):\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)}
        if kind == 'val' and 'coco/bbox_mAP' in values:
            validation[epoch] = values
        elif kind == 'train' and 'loss' in values:
            training[epoch].append(values)
    if not validation:
        parser.error('No completed validation epochs found in the log')
    epochs = sorted(validation)
    if epochs != list(range(1, epochs[-1] + 1)):
        parser.error('Validation epochs are incomplete; use a complete run log')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l']
    loss_keys = ['loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
    rows = []
    for epoch in epochs:
        samples = training[epoch]
        if not samples:
            parser.error(f'No training loss samples at epoch {epoch}')
        row = {'epoch': epoch, **{key: validation[epoch]['coco/' + key] for key in metrics}}
        row.update({key + '_logged_mean': sum(s[key] for s in samples) / len(samples) for key in loss_keys})
        row['lr_last_logged'] = samples[-1]['lr']
        row['logged_train_samples'] = len(samples)
        rows.append(row)
    csv_path = args.output_dir / 'epoch_metrics.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
    best = max(rows, key=lambda row: row['bbox_mAP'])
    for key, label in zip(metrics[:3], ['mAP (IoU .50:.95)', 'AP50', 'AP75']):
        axes[0, 0].plot(epochs, [100*r[key] for r in rows], label=label, linewidth=2)
    axes[0, 0].scatter(best['epoch'], best['bbox_mAP']*100, c='black', zorder=5)
    axes[0, 0].annotate(f"Best mAP: {best['bbox_mAP']*100:.1f}% (epoch {best['epoch']})",
                        (best['epoch'], best['bbox_mAP']*100), xytext=(15, -32),
                        textcoords='offset points', arrowprops={'arrowstyle': '->'})
    axes[0, 0].set(title='Validation detection performance', ylabel='AP (%)')
    for key, label in zip(metrics[3:], ['Small', 'Medium', 'Large']):
        axes[0, 1].plot(epochs, [100*r[key] for r in rows], label=label, linewidth=2)
    axes[0, 1].set(title='Validation AP by COCO object size', ylabel='AP (%)')
    for key, label in zip(loss_keys, ['Total', 'RPN classification', 'RPN bbox', 'RoI classification', 'RoI bbox']):
        axes[1, 0].plot(epochs, [r[key+'_logged_mean'] for r in rows], label=label,
                        linewidth=2 if key == 'loss' else 1.3)
    axes[1, 0].set(title='Training loss: mean of logged window averages', ylabel='Loss')
    axes[1, 1].step(epochs, [r['lr_last_logged'] for r in rows], where='post', color='#8054b3', label='End-of-epoch LR')
    axes[1, 1].set(title='Learning rate (within-epoch warmup omitted)', ylabel='Learning rate', yscale='log')
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.set_xlim(1, epochs[-1])
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle(f'VisDrone validation | Faster R-CNN R50-FPN | {len(epochs)} epochs', fontsize=16)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(args.output_dir / f'training_curves.{extension}', dpi=180)
    plt.close(fig)
    notes = (f'Source: {args.log.resolve()}\n'
             'Epoch numbers come from MMEngine Epoch(val) log lines, not the offset monitor history.\n'
             'CSV AP values are fractions; chart AP values are percentages.\n'
             'Loss is the unweighted mean of logged smoothed windows, not an exact per-image epoch mean.\n'
             'Object sizes use COCO evaluation bins, not the analysis tiny-object bins.\n'
             f"Best mAP: epoch {best['epoch']}, {best['bbox_mAP']:.3f}; ties choose the first epoch.\n"
             f"Final mAP: epoch {rows[-1]['epoch']}, {rows[-1]['bbox_mAP']:.3f}.\n")
    (args.output_dir / 'README.txt').write_text(notes)
    print(notes)
    print(f'Saved PNG, PDF, SVG, CSV to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
