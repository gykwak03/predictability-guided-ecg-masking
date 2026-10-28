"""JSON output shared by the active estimator runners."""
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    """Write standards-compliant JSON: undefined correlations become null."""
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding="utf-8")

