"""Provenance and cell stores for X9--X14, on top of the shared common module.

Campaign loading, the original preprocessing, LayerStudy and the batched GPU
readouts come unchanged from ``experiments/common.py``. This module records
which code is hashed into manifests and finished-cell stores (the experiment
package and the frozen library), and adds small caches for fitted erasers and
MLP readouts so that an interrupted run resumes without refitting them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NEW = REPO  # NEW / "experiments" is this package
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import common  # noqa: E402  (experiments/common.py)

RESULTS = common.RESULTS
X7_RESULTS = RESULTS / "X7_alignment"
LIBRARY = ["studies.py", "geometry.py", "fastprobe.py", "data.py", "labels.py", "extraction.py", "config.py"]


def code_files():
    files = sorted(HERE.glob("*.py")) + [REPO / "run_experiments.py"]
    files += [REPO / "idiom_subspace" / name for name in LIBRARY]
    return [f for f in files if f.exists()]


def code_hashes():
    return {f.relative_to(REPO).as_posix(): common.text_sha256(f) for f in code_files()}


def manifest(ctx, extra=None):
    info = common.manifest(ctx, extra)
    info["code_sha256"] = code_hashes()
    return info


def finish(ctx, tables, extra=None):
    """Write aggregates (dict name -> DataFrame) and the manifest to results/<experiment>/."""
    ctx.portable.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, table in tables.items():
        if table is None:
            continue
        common.write_csv(ctx.portable / name, table)
        counts[name] = int(len(table))
    info = manifest(ctx, dict(rows=counts, **(extra or {})))
    common.write_json(ctx.portable / "MANIFEST.json", info)
    common.write_json(ctx.work / "completed.json", dict(completed_utc=common.utc(), rows=counts))
    return info


_CELL_KEYS = {}


def cells_root(ctx):
    """Finished-cell store keyed by the result-relevant options and by the experiment code."""
    relevant = {k: v for k, v in sorted(ctx.options.items()) if k not in {"n_jobs"}}
    if "code" not in _CELL_KEYS:
        files = sorted(HERE.glob("*.py"))
        _CELL_KEYS["code"] = {f.relative_to(REPO).as_posix(): common.text_sha256(f) for f in files}
    payload = dict(options=relevant, code=_CELL_KEYS["code"])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    root = ctx.work / "cells" / f"run_{digest}"
    if not (root / "key.json").exists():
        common.write_json(root / "key.json", payload)
    return root


def cell_tables(ctx, cell_key, tables=None):
    """Persist (or restore) one model x split cell's tables so finished cells survive restarts."""
    root = cells_root(ctx)
    marker = root / f"{cell_key}.done.json"
    if tables is not None:
        for name, table in tables.items():
            common.write_csv(root / cell_key / name, table)
        common.write_json(marker, dict(completed_utc=common.utc(), tables=sorted(tables)))
        return tables
    if not marker.exists():
        return None
    out = {}
    for name in common.read_json(marker).get("tables", []):
        try:
            out[name] = pd.read_csv(root / cell_key / name)
        except pd.errors.EmptyDataError:
            out[name] = pd.DataFrame()
    return out


def digest(*parts):
    """Stable short hash of strings, numbers, JSON-able values and arrays."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            h.update(str((part.shape, part.dtype.str)).encode())
            h.update(np.ascontiguousarray(part).tobytes())
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode())
        h.update(b"|")
    return h.hexdigest()[:24]


class ArrayCache:
    """Numeric-only (no pickle) cache of fitted erasers and MLP readouts below the work directory."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def load(self, key):
        arrays, meta = self.folder / f"{key}.npz", self.folder / f"{key}.json"
        if not (arrays.exists() and meta.exists()):
            return None
        with np.load(arrays, allow_pickle=False) as values:
            out = {name: values[name] for name in values.files}
        out.update(common.read_json(meta))
        return out

    def save(self, key, arrays, meta):
        arrays_path = self.folder / f"{key}.npz"
        temporary = self.folder / f".{key}.tmp.npz"
        np.savez(temporary, **{k: np.asarray(v) for k, v in arrays.items()})
        temporary.replace(arrays_path)
        common.write_json(self.folder / f"{key}.json", meta)


def source_hash(name):
    """Hash of one module of this package (cache keys change when its code changes)."""
    return common.text_sha256(HERE / name)
