from pathlib import Path
import hashlib
import json
import os
import tempfile
import datetime
import importlib.metadata
import subprocess
import random
import numpy as np
import torch


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.tmp-')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def environment():
    result = {}
    for name in ('torch', 'torchvision', 'mmengine', 'mmcv', 'mmcv-lite', 'mmdet', 'numpy', 'scipy', 'streamlit'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    result['cuda'] = torch.version.cuda
    result['gpu'] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    p = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    result['git_commit'] = p.stdout.strip() if p.returncode == 0 else None
    return result


def seed_all(seed=128):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
