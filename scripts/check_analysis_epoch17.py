"""GPU verification of the selected model and high-resolution analysis pipeline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import numpy as np
import torch
import yaml
from src.common import write_json
from src.job_manager import gpu_lease
from src.feature_extractor import FeatureExtractor
from src.data_audit import load_records
from src.analysis_runner import analyze_image,run_analysis,combinations


def main():
    root=Path(__file__).resolve().parents[1]
    import os
    os.chdir(root)
    config=yaml.safe_load(Path('configs/analysis_epoch17.yaml').read_text())
    config.update(purpose='integration_test_only',runs_dir='reports/analysis_epoch17_smoke')
    checkpoint='runs/reproduction_r50_ms20/test_candidates/epoch_17.pth'
    validation='runs/reproduction_r50_ms20/eval_test/report.json'
    detector='runs/reproduction_r50_ms20/eval_test/config.py'
    with gpu_lease():
        extractor=FeatureExtractor(detector,checkpoint,'cuda:0',validation)
        from src.common import read_json
        extractor.provenance=copy.deepcopy(read_json(validation)['provenance'])
        extractor.provenance['purpose']='integration_test_only'
        buffers={k:v.clone() for k,v in extractor.model.named_buffers()}
        try:
            records=load_records(config['dataset_root'],'test',config['format'])
            result=analyze_image(extractor,records[0],config,[(1.,'C3','cosine')])[(1.,'C3','cosine')]
            finite=result['values'][np.isfinite(result['values'])]
            assert len(finite) and abs(finite).max()<2e-6
            paths=run_analysis(config,extractor,checkpoint,combinations('A'),split='test',limit=2)
            assert all(torch.equal(v,dict(extractor.model.named_buffers())[k]) for k,v in buffers.items())
            report=dict(status='PASSED',same_input_max_absolute_discrepancy=float(abs(finite).max()),
                        geometry=result['geometry'],module_mapping=extractor.mapping,batchnorm_buffers_unchanged=True,
                        n_smoke_images=2,runs=paths)
            write_json('reports/analysis_epoch17_smoke/checks.json',report)
            print(report)
        finally:extractor.close()


if __name__=='__main__':main()
