from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from experiments.io_utils import write_json
from .config import ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def source_identity():
    paths = [*sorted((ROOT / 'experiments/phase1').glob('*.py')),
             ROOT / 'experiments/run_phase1.py', ROOT / 'experiments/predictability_estimators.py',
             ROOT / 'experiments/io_utils.py', ROOT / 'preprocessing/run_preprocessing.py',
             ROOT / 'preprocessing/PTBXL_preprocessing.ipynb',
             ROOT / 'archive/20261028_legacy/experiments/mae_cause_model.py',
             ROOT / 'archive/20261028_legacy/experiments/Masking_pilot.ipynb']
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}


def new_run(output_root, kind, run_id=None):
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run-id must be a single directory name')
    root = Path(output_root) / f'{kind}_{run_id}'
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'status.json', dict(status='partial', stage=kind))
    return root


def finish(root, status='completed', **details):
    write_json(Path(root) / 'status.json', dict(status=status, **details))
