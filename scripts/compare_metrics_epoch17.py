"""Compare selected backbone stages/metrics on full test with epoch 17."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import argparse
import yaml
from src.common import read_json, write_json, sha256, timestamp
from src.analysis_runner import run_analysis
from src.feature_extractor import FeatureExtractor
from src.job_manager import gpu_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', choices=['C3', 'C4', 'C5'], default=['C3'])
    parser.add_argument('--metrics', nargs='+', choices=['cosine', 'l1', 'l2'], default=['cosine', 'l1', 'l2'])
    parser.add_argument('--report-dir', default='reports/metric_comparison_epoch17')
    args = parser.parse_args()
    os.chdir(ROOT)
    report = ROOT / args.report_dir
    report.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path('configs/analysis_epoch17.yaml').read_text())
    detector = 'runs/reproduction_r50_ms20/eval_test/config.py'
    checkpoint = str(ROOT / 'runs/reproduction_r50_ms20/test_candidates/epoch_17.pth')
    validation = 'runs/reproduction_r50_ms20/eval_test/report.json'
    conditions = [(r, s, m) for s in dict.fromkeys(args.stages)
                  for r in (.75, .5, .25) for m in dict.fromkeys(args.metrics)]
    config['detector_config_sha256'] = sha256(detector)
    config['checkpoint_provenance'] = read_json(validation)['provenance']
    config['selection_snapshot'] = {}
    state = dict(status='RUNNING', started_at=timestamp(), conditions=conditions,
                 split='test', limit=None, checkpoint=checkpoint)
    def progress(value):
        state['progress'] = value
        write_json(report / 'status.json', state)
        if value['done'] % 100 == 0 or value['done'] == value['total']:
            print(value, flush=True)
    try:
        with gpu_lease():
            write_json(report / 'status.json', state)
            extractor = FeatureExtractor(detector, checkpoint, 'cuda:0', validation)
            extractor.provenance = config['checkpoint_provenance']
            try:
                state['runs'] = run_analysis(config, extractor, checkpoint, conditions,
                                             split='test', progress=progress)
            finally:
                extractor.close()
        state.update(status='DONE', finished_at=timestamp())
        if 'progress' in state:
            state['progress']['status'] = 'DONE'
    except BaseException as exc:
        state.update(status='FAILED', error=str(exc), finished_at=timestamp())
        raise
    finally:
        write_json(report / 'status.json', state)
    print(state, flush=True)


if __name__ == '__main__':
    main()
