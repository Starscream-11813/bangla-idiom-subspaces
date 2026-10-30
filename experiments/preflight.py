"""Check that the campaign, caches, layer selections, model revisions and GPU are all in place.

    python run_experiments.py preflight --campaign CAMPAIGN --out OUT

Reads metadata only (no features are loaded and nothing is written except
OUT/preflight.json). Exits nonzero if a required item is missing. Items needed by
a single experiment (the NLLB model for X2, pinned revisions and the normalizer
for X4's new forward passes) are optional: they print OPTIONAL-MISSING, the run
continues, and only that experiment fails until the item is installed (docs/RUNNING.md).
"""
from __future__ import annotations

import shutil

from .common import (FOLLOWUP_NAME, REPO, base_config, cache_leaf, configured_models, discovery_paths,
                     followup_layer, read_json, utc, write_json)


def _ok(checks, passed, name, detail="", required=True):
    checks.append(dict(check=name, passed=bool(passed), detail=str(detail), required=required))
    status = "ok" if passed else ("MISSING" if required else "OPTIONAL-MISSING")
    print(f"  [{status}] {name} {detail}")
    return passed


def preflight(campaign, out, followup=None, models=None):
    campaign = campaign.resolve()
    followup = (followup or campaign / FOLLOWUP_NAME).resolve()
    checks = []
    print(f"campaign {campaign}\nfollow-up {followup}")
    _ok(checks, (campaign / "configs" / "full_3090.json").exists(), "configs/full_3090.json")
    cfg = base_config(campaign)
    names = configured_models(cfg, models)
    _ok(checks, (followup / "prepared" / "corpus_prepared.json").exists(), "follow-up prepared corpus")
    for name in names:
        print(name)
        prepared, dest = discovery_paths(campaign, name)
        for path in [prepared / "items.json", prepared / "corpus_prepared.json", dest / "model_revision.json",
                     dest / "layer_selection.json"]:
            _ok(checks, path.exists(), str(path.relative_to(campaign)))
        if not (dest / "model_revision.json").exists():
            continue
        leaf, revision = cache_leaf(campaign, name)
        for file in ["features.npy", "manifest.json", "pooling.csv"]:
            _ok(checks, (leaf / file).exists(), f"cache {file}", leaf)
        if not (leaf / "manifest.json").exists():
            continue
        manifest = read_json(leaf / "manifest.json")
        layers = manifest["layers"]
        try:
            l0 = int(read_json(dest / "layer_selection.json")["layer"])
            _ok(checks, l0 in layers, "discovery layer cached", f"layer {l0} in {layers}")
        except Exception as error:
            _ok(checks, False, "discovery layer", repr(error))
        try:
            l1 = followup_layer(followup, name)
            _ok(checks, l1 in layers, "follow-up layer cached", f"layer {l1}")
        except Exception as error:
            _ok(checks, False, "follow-up layer", repr(error))
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(name, revision=manifest["resolved_revision"])
            _ok(checks, True, "pinned model revision loadable (X4)", manifest["resolved_revision"], required=False)
        except Exception as error:
            _ok(checks, False, "pinned model revision loadable (X4)", f"{manifest['resolved_revision']}: {error}",
                required=False)
        if manifest.get("normalization") == "csebuetnlp":
            try:
                import normalizer  # noqa: F401
                _ok(checks, True, "csebuetnlp normalizer importable (X4)", required=False)
            except ImportError as error:
                _ok(checks, False, "csebuetnlp normalizer importable (X4)", error, required=False)
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("facebook/nllb-200-distilled-600M")
        _ok(checks, True, "NLLB-200-distilled-600M available (X2 FBN)", required=False)
    except Exception as error:
        _ok(checks, False, "NLLB-200-distilled-600M available (X2 FBN)", error, required=False)
    try:
        import torch
        cuda = torch.cuda.is_available()
        _ok(checks, cuda, "CUDA", torch.cuda.get_device_name(0) if cuda else "not available")
    except ImportError as error:
        _ok(checks, False, "torch", error)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 2 ** 30
    _ok(checks, free >= 20, "free disk in OUT >= 20 GB (about 17 GB needed)", f"{free:.0f} GB")
    for name in ["results/discovery/primary_confirmatory.csv", "results/discovery/S3_baselines.csv",
                 "results/replication/hypotheses.csv", "results/replication/scores.csv"]:
        _ok(checks, (REPO / name).exists(), f"reference table {name}")
    failed = [c for c in checks if not c["passed"] and c["required"]]
    optional = [c for c in checks if not c["passed"] and not c["required"]]
    write_json(out / "preflight.json", dict(created_utc=utc(), campaign=str(campaign), models=names, checks=checks))
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed; "
          f"{len(failed)} required missing, {len(optional)} optional missing")
    if optional:
        print("Optional items missing: the run continues; install them (docs/RUNNING.md) and rerun "
              "X2 (NLLB) or X4 (pinned revision, normalizer) afterwards.")
    return 1 if failed else 0
