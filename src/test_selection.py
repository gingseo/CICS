"""Retain completed epoch checkpoints and select by explicitly labeled test AP."""
import os
import re
from pathlib import Path
from .common import read_json


def retain_completed(root):
    root = Path(root)
    bank = root / 'test_candidates'
    bank.mkdir(exist_ok=True)
    completed = read_json(root/'completed_epoch.json', {}).get('epoch', 0)
    for source in root.glob('epoch_*.pth'):
        match = re.fullmatch(r'epoch_(\d+)\.pth', source.name)
        if not match or int(match[1]) > completed:
            continue
        destination = bank/source.name
        try:
            os.link(source, destination)  # Same filesystem; retained when normal retention unlinks source.
        except FileExistsError:
            pass
        except FileNotFoundError:
            pass  # Checkpoint rotation won the race; report missing epochs explicitly.
    return sorted(bank.glob('epoch_*.pth'), key=lambda p: int(p.stem.split('_')[-1]))


def select_best(results):
    if not results:
        raise ValueError('No completed test evaluations')
    return max(results, key=lambda row: (row['metrics']['coco/bbox_mAP'], -row['epoch']))
