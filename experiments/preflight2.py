"""Check that the campaign, caches, layer selections, X7 reference and GPU are in place for X9--X10.

    python run_experiments.py preflight --campaign CAMPAIGN --out OUT [--splits 0,1]

Reads metadata only (no features are loaded; nothing is written except
OUT/preflight.json). Exits nonzero if a required item is missing.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from experiments.common import (FOLLOWUP_NAME, REPO, base_config, cache_leaf, configured_models, discovery_paths,
                                followup_layer, read_json, utc, write_json)
from experiments.preflight import _ok

from .common2 import X7_RESULTS


def preflight(campaign, out, followup=None, models=None, splits=(0,)):
    campaign = campaign.resolve()
    followup = (followup or campaign / FOLLOWUP_NAME).resolve()
    checks = []
    print(f"campaign {campaign}\nfollow-up {followup}\nsplits {list(splits)}")
    _ok(checks, (campaign / "configs" / "full_3090.json").exists(), "configs/full_3090.json")
    cfg = base_config(campaign)
    names = configured_models(cfg, models)
    if 1 in splits:
        _ok(checks, (followup / "prepared" / "corpus_prepared.json").exists(), "follow-up prepared corpus")
    for name in names:
        print(name)
        prepared, dest = discovery_paths(campaign, name)
        for path in [prepared / "items.json", prepared / "corpus_prepared.json", dest / "model_revision.json",
                     dest / "layer_selection.json"]:
            _ok(checks, path.exists(), str(path.relative_to(campaign)))
        if not (dest / "model_revision.json").exists():
            continue
        leaf, _ = cache_leaf(campaign, name)
        for file in ["features.npy", "manifest.json", "pooling.csv"]:
            _ok(checks, (leaf / file).exists(), f"cache {file}", leaf)
        if not (leaf / "manifest.json").exists():
            continue
        layers = read_json(leaf / "manifest.json")["layers"]
        try:
            layer = int(read_json(dest / "layer_selection.json")["layer"])
            _ok(checks, layer in layers, "discovery layer cached", f"layer {layer} in {layers}")
        except Exception as error:
            _ok(checks, False, "discovery layer", repr(error))
        if 1 in splits:
            try:
                layer = followup_layer(followup, name)
                _ok(checks, layer in layers, "follow-up layer cached", f"layer {layer}")
            except Exception as error:
                _ok(checks, False, "follow-up layer", repr(error))
    if cfg.label_backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(cfg.label_encoder, revision=cfg.label_revision, trust_remote_code=False)
            _ok(checks, True, "label encoder loadable (X11 re-seeded F and TAG)", cfg.label_encoder, required=False)
        except Exception as error:
            _ok(checks, False, "label encoder loadable (X11 re-seeded F and TAG)", f"{cfg.label_encoder}: {error}",
                required=False)
    from idiom_subspace.config import model_slug
    from .x14_decoder import ROOT, decoder_models, decoder_paths
    for name in decoder_models(campaign):
        prepared, dest = decoder_paths(campaign, name)
        files = [prepared / "items.json", prepared / "corpus_prepared.json", dest / "model_revision.json",
                 dest / "layer_selection.json"]
        missing = [str(p.relative_to(campaign)) for p in files if not p.exists()]
        if not missing:
            revision = read_json(dest / "model_revision.json")
            leaf = Path(revision.get("cache_reused_from") or campaign / ROOT / model_slug(name) / "cache"
                        / revision["fingerprint"])
            missing += [str(leaf / f) for f in ["features.npy", "manifest.json", "pooling.csv"]
                        if not (leaf / f).exists()]
        _ok(checks, not missing, f"decoder tree for {name} (X14, optional)", "; ".join(missing) or "complete",
            required=False)
    for file in ["aligned.csv", "retrieval.csv", "MANIFEST.json"]:
        _ok(checks, (X7_RESULTS / file).exists(), f"X7 reference results/experiments/X7_alignment/{file} "
                                                  "(reproduction gate)")
    _ok(checks, (REPO / "results" / "discovery" / "S3_baselines.csv").exists(),
        "reference table results/discovery/S3_baselines.csv", required=False)
    try:
        import joblib  # noqa: F401
        import sklearn
        _ok(checks, True, "scikit-learn and joblib (X9 MLP probes)", sklearn.__version__)
    except ImportError as error:
        _ok(checks, False, "scikit-learn and joblib (X9 MLP probes)", error)
    try:
        import torch
        cuda = torch.cuda.is_available()
        _ok(checks, cuda, "CUDA", torch.cuda.get_device_name(0) if cuda else "not available")
    except ImportError as error:
        _ok(checks, False, "torch", error)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 2 ** 30
    _ok(checks, free >= 5, "free disk in OUT >= 5 GB (well under 2 GB needed)", f"{free:.0f} GB")
    failed = [c for c in checks if not c["passed"] and c["required"]]
    optional = [c for c in checks if not c["passed"] and not c["required"]]
    write_json(out / "preflight.json", dict(created_utc=utc(), campaign=str(campaign), models=names,
                                            splits=list(splits), checks=checks))
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed; "
          f"{len(failed)} required missing, {len(optional)} optional missing")
    return 1 if failed else 0
