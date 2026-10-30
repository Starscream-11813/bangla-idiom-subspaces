"""Shared loading, readouts and bookkeeping for the follow-up experiments X1--X19.

Original campaign files (caches, items, layer selections) are opened read-only.
Item-level evidence and checkpoints go below ``--out``; small portable aggregates
go to ``results/experiments/<experiment>/`` so they can be committed.

Split 0 is the discovery study (campaign/primary, MLR-selected layers); split 1
is the prospective follow-up (paper_followup_seed1_20261021, COV-selected
layers). Each split rebuilds exactly the features, labels and LayerStudy seeds
of the original analysis, so RAND/PERM null draws reproduce the published ones.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CODE = REPO  # the idiom_subspace package sits at the repository root
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from idiom_subspace.config import Config, model_slug  # noqa: E402
from idiom_subspace.geometry import fitted_preprocessor  # noqa: E402
from idiom_subspace.studies import LayerStudy  # noqa: E402

RESULTS = REPO / "results" / "experiments"
CORPUS_NAME = "bagdhara_export_done_2026-03-01_010356.json"
FOLLOWUP_NAME = "paper_followup_seed1_20261021"
RANK = {"F": 11, "S": 3}
LABEL_COLUMNS = ["split", "component_id", "F", "S", "cp_transparency", "transparent", "opaque"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(path):
    """Line-ending-insensitive digest, so Windows and Linux checkouts agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_clean(serial), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temporary, path)


class Log:
    """Timestamped console lines plus a status.json that shows the current stage."""

    def __init__(self, directory, experiment):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.started = time.time()

    def __call__(self, message, **state):
        elapsed = time.time() - self.started
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.experiment} +{elapsed/60:.1f}min {message}", flush=True)
        write_json(self.directory / "status.json", dict(experiment=self.experiment, message=message,
                                                         updated_utc=utc(), elapsed_minutes=elapsed / 60, **state))


@dataclass
class Context:
    """Command-line settings shared by every experiment."""
    experiment: str
    campaign: Path
    out: Path
    followup: Path
    results: Path
    models: list
    splits: list
    cfg: Config
    options: dict = field(default_factory=dict)

    @property
    def work(self):
        """Item-level evidence and checkpoints for this experiment (not committed)."""
        return self.out / self.experiment

    @property
    def portable(self):
        """Committed aggregates for this experiment."""
        return self.results / self.experiment

    def option(self, name, default=None):
        value = self.options.get(name)
        return default if value is None else value


def base_config(campaign):
    """The campaign configuration with the analysis settings every experiment shares."""
    path = Path(campaign) / "configs" / "full_3090.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; pass the original campaign directory as --campaign")
    cfg = Config.load(path)
    cfg.cache_only = True
    cfg.resume = True
    cfg.compute_leakage = False
    # Predicted labels (for balanced accuracy/macro-F1) come from the batched torch readout.
    cfg.probe_backend = "torch"
    cfg.validate()
    return cfg


def configured_models(cfg, requested=None):
    """Configured model names, optionally filtered by full name, slug or last name component."""
    names = [m["name"] for m in cfg.models]
    if not requested:
        return names
    chosen = set()
    for token in [m.strip() for m in requested.split(",") if m.strip()]:
        hits = [n for n in names if token in (n, model_slug(n), n.replace("\\", "/").rstrip("/").split("/")[-1])]
        if len(hits) != 1:
            raise ValueError(f"Model {token!r} matches {len(hits)} configured models; choose from {names}")
        chosen.add(hits[0])
    return [n for n in names if n in chosen]


def model_spec(cfg, name):
    return copy.deepcopy(next(m for m in cfg.models if m["name"] == name))


@dataclass
class Cell:
    """One model on one split at its selected layer (or a requested layer)."""
    name: str
    slug: str
    split: int
    layer: int
    selected_layer: int
    frame: pd.DataFrame
    X: np.ndarray
    meta: dict
    manifest: dict
    cache: Path
    mean: np.ndarray
    scale: float

    @property
    def label(self):
        return f"{self.slug}/split{self.split}/layer{self.layer}"


def discovery_paths(campaign, name):
    slug = model_slug(name)
    prepared = Path(campaign) / "primary" / slug / "split_0" / "labels_0"
    return prepared, prepared / slug


def cache_leaf(campaign, name):
    _, dest = discovery_paths(campaign, name)
    revision = read_json(dest / "model_revision.json")
    leaf = revision.get("cache_reused_from") or str(Path(campaign) / "primary" / model_slug(name) / "cache"
                                                    / revision["fingerprint"])
    return Path(leaf), revision


def followup_layer(followup, name):
    path = Path(followup) / model_slug(name) / "layer_selection.json"
    if path.exists():
        return int(read_json(path)["layer"])
    portable = REPO / "results" / "replication" / "layer_selection.json"
    return int(read_json(portable)[name]["layer"])


_ITEMS = {}
_CORPUS = {}


def discovery_items(campaign, name):
    prepared, _ = discovery_paths(campaign, name)
    key = str(prepared)
    if key not in _ITEMS:
        _ITEMS.clear()
        _ITEMS[key] = pd.read_json(prepared / "items.json")
    return _ITEMS[key].copy()


def split_corpus_frame(ctx, name, split):
    """Prepared corpus (records with split, components and labels) of one split."""
    if split == 0:
        prepared, _ = discovery_paths(ctx.campaign, name)
        path = prepared / "corpus_prepared.json"
    elif split == 1:
        path = ctx.followup / "prepared" / "corpus_prepared.json"
    else:
        raise ValueError("Only split 0 (discovery) and split 1 (follow-up) exist")
    key = str(path)
    if key not in _CORPUS:
        if len(_CORPUS) > 1:
            _CORPUS.clear()
        _CORPUS[key] = pd.read_json(path)
    return _CORPUS[key].copy()


def split_items(ctx, name, split):
    """All original items with this split's partition, components and labels."""
    items = discovery_items(ctx.campaign, name)
    if split == 1:
        corpus = split_corpus_frame(ctx, name, 1).set_index("idiom_id")
        assert set(items.idiom_id) <= set(corpus.index), "follow-up corpus lacks discovery records"
        for column in LABEL_COLUMNS:
            items[column] = items.idiom_id.map(corpus[column])
    return items


def study_meta(ctx, name, split, layer):
    """LayerStudy meta of the original analysis; its seed fixes RAND/PERM draws."""
    if split == 0:
        return dict(model=name, seed=0, label_seed=0, layer=layer, split_mode=ctx.cfg.split_mode,
                    ambient_space="original_hidden")
    return dict(model=name, seed=1, label_seed=0, layer=layer)


def selected_layer(ctx, name, split):
    if split == 0:
        _, dest = discovery_paths(ctx.campaign, name)
        return int(read_json(dest / "layer_selection.json")["layer"])
    return followup_layer(ctx.followup, name)


def original_cache(ctx, name, split):
    """Items (with this split's labels), read-only feature memmap, validity and manifest."""
    leaf, _ = cache_leaf(ctx.campaign, name)
    items = split_items(ctx, name, split)
    pooling = pd.read_csv(leaf / "pooling.csv", keep_default_na=False, dtype={"item_id": str})
    assert pooling.item_id.tolist() == items.item_id.astype(str).tolist(), "cache/items order mismatch"
    manifest = read_json(leaf / "manifest.json")
    features = np.load(leaf / "features.npy", mmap_mode="r", allow_pickle=False)
    assert list(features.shape) == manifest["shape"], "cache manifest shape mismatch"
    valid = pooling.valid.astype(str).str.lower().eq("true").to_numpy()
    items["token_count"] = pooling.token_count.to_numpy()
    return items, features, valid, manifest, leaf


def original_preprocessor(ctx, name, split, layer, cache=None):
    """SPACE-only mean and global RMS of all original conditions, as in the original studies."""
    items, features, valid, manifest, _ = cache or original_cache(ctx, name, split)
    li = manifest["layers"].index(layer)
    space = np.flatnonzero(valid & items.split.eq("space").to_numpy())
    raw = np.asarray(features[space, li, :], dtype=float)
    assert np.isfinite(raw).all(), "nonfinite cached SPACE features"
    pre = fitted_preprocessor(raw)
    mean, scale = pre.mean_, float(pre.scale_)
    if split == 0:
        _, dest = discovery_paths(ctx.campaign, name)
        saved = dest / f"preprocess_layer_{layer}.npz"
        if saved.exists():
            with np.load(saved, allow_pickle=False) as values:
                assert np.allclose(values["mean"], mean, rtol=0, atol=1e-9), "preprocessor mean differs"
                assert abs(float(values["scale"]) - scale) <= 1e-9 * max(scale, 1.), "preprocessor scale differs"
    return mean, scale


def load_cell(ctx, name, split, layer=None, conditions=None, substitute=None, preprocessor=None):
    """Rebuild the original preprocessed features for one model, split and layer.

    The preprocessor (mean and global RMS) is fitted on SPACE rows of all original
    conditions, exactly as in the discovery and follow-up studies, and then
    applied to the requested ``conditions`` only (default: all). ``substitute``
    = (items, features, valid, manifest) swaps in another feature set (the
    pooling-matched extraction); it requires ``preprocessor=(mean, scale)``.
    """
    slug = model_slug(name)
    chosen = selected_layer(ctx, name, split)
    layer = chosen if layer is None else int(layer)
    if substitute is None:
        items, features, valid, manifest, leaf = original_cache(ctx, name, split)
        if preprocessor is None:
            preprocessor = original_preprocessor(ctx, name, split, layer, (items, features, valid, manifest, leaf))
    else:
        if preprocessor is None:
            raise ValueError("A substitute feature set needs the original preprocessor")
        items, features, valid, manifest = substitute
        leaf = None
    mean, scale = preprocessor
    li = manifest["layers"].index(layer)
    keep = valid.copy()
    if conditions is not None:
        keep &= items.condition.isin(conditions).to_numpy()
    rows = np.flatnonzero(keep)
    raw = np.asarray(features[rows, li, :], dtype=float)
    assert np.isfinite(raw).all(), "nonfinite cached features in valid rows"
    X = (raw - mean) / scale
    del raw
    frame = items.iloc[rows].reset_index(drop=True)
    return Cell(name=name, slug=slug, split=split, layer=layer, selected_layer=chosen, frame=frame, X=X,
                meta=study_meta(ctx, name, split, layer), manifest=manifest, cache=leaf, mean=mean, scale=scale)


def frame_digest(frame, columns):
    payload = frame[[c for c in columns if c in frame]].to_json(orient="split", force_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_study(ctx, cell, tag, X=None, frame=None, extra_columns=(), cfg=None, progress=None):
    """A LayerStudy with its own checkpoint folder below the experiment's work dir.

    LayerStudy's checkpoint fingerprint ignores label columns it does not know
    (e.g. new concepts), so ``extra_columns`` are hashed into the folder name.
    """
    frame = cell.frame if frame is None else frame
    X = cell.X if X is None else X
    folder = ctx.work / "checkpoints" / cell.slug / f"split{cell.split}" / f"layer{cell.layer}" / tag
    if extra_columns:
        folder = folder / ("labels_" + frame_digest(frame, ["item_id"] + list(extra_columns)))
    return LayerStudy(X, frame, cfg or ctx.cfg, cell.meta, checkpoint_dir=folder, progress=progress)


def rank_for(study, concept, requested=None):
    """Highest configured rank supported by the concept's SPACE classes (11 for F, 3 for S)."""
    ranks = sorted(study.cfg.ranks)
    if requested is not None:
        return int(requested)
    classes = pd.to_numeric(study.items[concept], errors="coerce")
    k = int(classes[classes.ge(0) & study.items.split.eq("space")].nunique())
    supported = [r for r in ranks if r <= k - 1]
    return max(supported) if supported else None


def compact_readouts(study, train_list, val_list, test_list, y, indices, Cs=None, need_margin=False, batch=256):
    """Batched L2 logistic readouts on per-problem features (fastprobe), C tuned on validation."""
    from idiom_subspace.fastprobe import compact_features, tune_readouts, default_dtype
    dtype = default_dtype(study._device)
    out = []
    for start in range(0, len(train_list), batch):
        stop = start + batch
        feats = compact_features(train_list[start:stop], val_list[start:stop], test_list[start:stop],
                                 study._device, dtype)
        out += tune_readouts(feats, y, indices, Cs or study.cfg.probe_Cs, need_margin=need_margin,
                             max_iter=max(study.probe_max_iter, 1000))
        del feats
    return out


def erase_readouts(study, Xtr, Xva, Xte, factors, bases, y, indices, Cs=None, batch=16):
    """Batched readouts of X - (X F) G^T for shared X (orthogonal or oblique erasure)."""
    from idiom_subspace.fastprobe import erase_features, tune_readouts, default_dtype
    dtype = default_dtype(study._device)
    out = []
    for start in range(0, len(factors), batch):
        feats = erase_features(Xtr, Xva, Xte, factors[start:start + batch], bases[start:start + batch],
                               study._device, dtype)
        out += tune_readouts(feats, y, indices, Cs or study.cfg.probe_Cs, need_margin=False,
                             max_iter=max(study.probe_max_iter, 1000))
        del feats
    return out


def readout_indices(study, target, concept, train=None):
    """(train, validation, test) row indices; ``train`` swaps the training condition (zero-shot)."""
    source = target if train is None else train
    return (study.idx(source, "probe", concept), study.idx(source, "validation", concept),
            study.idx(target, "test", concept))


def labels(study, concept):
    return pd.to_numeric(study.items[concept], errors="coerce").fillna(-1).astype(int).to_numpy()


def test_labels(study, ev, concept):
    return labels(study, concept)[ev["test_idx"]]


def classes_in(y):
    return np.unique(np.asarray(y)[np.asarray(y) >= 0])


def item_frame(study, test_idx, concept, predictions):
    """Wide item-level table: identifiers, true label and one predicted-label column per readout."""
    f = study.items.iloc[np.asarray(test_idx)]
    table = f[["item_id", "idiom_id", "component_id"]].reset_index(drop=True)
    table.insert(0, "concept", concept)
    table["y_true"] = labels(study, concept)[np.asarray(test_idx)]
    for name, values in predictions.items():
        table[name] = np.asarray(values)
    return table


def code_hashes():
    files = sorted(HERE.glob("*.py")) + [REPO / "run_experiments.py"]
    files += [CODE / "idiom_subspace" / n for n in ["studies.py", "geometry.py", "fastprobe.py", "data.py",
                                                    "labels.py", "extraction.py", "config.py"]]
    return {str(f.relative_to(REPO)).replace(os.sep, "/"): text_sha256(f) for f in files if f.exists()}


def environment():
    from idiom_subspace.runner import environment as base
    info = base()
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover - environment report only
        info["cuda_available"] = None
    info.pop("source_files_sha256", None)
    return info


def manifest(ctx, extra=None):
    """Provenance written next to every experiment's aggregates."""
    return dict(experiment=ctx.experiment, created_utc=utc(), campaign=str(ctx.campaign),
                followup=str(ctx.followup), out=str(ctx.out), models=ctx.models, splits=ctx.splits,
                options=ctx.options, code_sha256=code_hashes(), environment=environment(),
                config={k: getattr(ctx.cfg, k) for k in ["probe_Cs", "ranks", "null_draws", "bootstrap",
                                                          "estimator_C", "estimator_ridge", "mlr_variant",
                                                          "probe_max_iter", "min_den", "split_mode"]},
                **(extra or {}))


def finish(ctx, tables, extra=None):
    """Write aggregates (dict name -> DataFrame) and the manifest to results/<experiment>/."""
    ctx.portable.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, table in tables.items():
        if table is None:
            continue
        write_csv(ctx.portable / name, table)
        counts[name] = int(len(table))
    info = manifest(ctx, dict(rows=counts, **(extra or {})))
    write_json(ctx.portable / "MANIFEST.json", info)
    write_json(ctx.work / "completed.json", dict(completed_utc=utc(), rows=counts))
    return info


_CELL_KEYS = {}


def cells_root(ctx):
    """Finished-cell store, keyed by the options that can change results and by the experiment code.

    A budget/backend change or any edit of experiments/*.py starts a
    fresh store; readout checkpoints are keyed separately by LayerStudy (pipeline
    source, data and labels), so their expensive fits are still reused.
    """
    relevant = {k: v for k, v in sorted(ctx.options.items()) if k not in {"n_jobs"}}
    if "code" not in _CELL_KEYS:
        _CELL_KEYS["code"] = {p.name: text_sha256(p) for p in sorted(HERE.glob("*.py"))}
    payload = dict(options=relevant, code=_CELL_KEYS["code"])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    root = ctx.work / "cells" / f"run_{digest}"
    if not (root / "key.json").exists():
        write_json(root / "key.json", payload)
    return root


def cell_done(ctx, cell_key):
    return (cells_root(ctx) / f"{cell_key}.done.json").exists()


def mark_cell(ctx, cell_key, **info):
    write_json(cells_root(ctx) / f"{cell_key}.done.json", dict(completed_utc=utc(), **info))


def cell_tables(ctx, cell_key, tables=None):
    """Persist (or restore) one cell's result tables so finished cells survive restarts."""
    folder = cells_root(ctx) / cell_key
    if tables is not None:
        for name, table in tables.items():
            write_csv(folder / name, table)
        mark_cell(ctx, cell_key, tables=sorted(tables))
        return tables
    if not cell_done(ctx, cell_key):
        return None
    names = read_json(cells_root(ctx) / f"{cell_key}.done.json").get("tables", [])
    return {name: pd.read_csv(folder / name) for name in names}


def concat(frames):
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def device_info(study):
    return dict(backend=study.backend, device=str(study._device))
