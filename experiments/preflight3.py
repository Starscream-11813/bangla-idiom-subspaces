"""Check that the campaign, caches, layers, reference tables, the frozen X16 plan and the judgments are in place.

    python run_experiments.py preflight --campaign CAMPAIGN --out OUT

Reads metadata only (no features are loaded; nothing is written except OUT/preflight.json). Exits nonzero if a
required item is missing.
"""
from __future__ import annotations

import shutil

import pandas as pd

from experiments.common import (FOLLOWUP_NAME, base_config, cache_leaf, configured_models, discovery_paths,
                                followup_layer, read_json, text_sha256, utc, write_json)
from experiments.preflight import _ok

from .common3 import ESTIMATORS, FOLLOWUP_SCORES, HUMAN_JUDGMENTS, PLAN_X16, PUBLISHED_PAIRS, REPO
from .x16_idiom_prospective import PLAN_SHA256

JUDGMENT_COLUMNS = ["sample_id", "assigned_F"] + [f"{q}_{a}" for q in ["f_plausibility", "function", "counterpart_type",
                                                                    "counterpart_equivalence"] for a in "AB"]


def preflight(campaign, out, followup=None, models=None, human=None):
    campaign = campaign.resolve()
    followup = (followup or campaign / FOLLOWUP_NAME).resolve()
    checks = []
    print(f"campaign {campaign}\nfollow-up {followup}")
    _ok(checks, (campaign / "configs" / "full_3090.json").exists(), "configs/full_3090.json")
    cfg = base_config(campaign)
    names = configured_models(cfg, models)
    _ok(checks, (followup / "prepared" / "corpus_prepared.json").exists(), "follow-up prepared corpus (X16)")
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
            l0 = int(read_json(dest / "layer_selection.json")["layer"])
            _ok(checks, l0 in layers, "discovery layer cached", f"layer {l0}")
        except Exception as error:
            _ok(checks, False, "discovery layer", repr(error))
        try:
            l1 = followup_layer(followup, name)
            _ok(checks, l1 in layers, "follow-up layer cached (X16)", f"layer {l1}")
        except Exception as error:
            _ok(checks, False, "follow-up layer (X16)", repr(error))
    for path in [PUBLISHED_PAIRS, ESTIMATORS, FOLLOWUP_SCORES]:
        _ok(checks, path.exists(), f"reference table {path.relative_to(REPO).as_posix()}")
    digest = text_sha256(PLAN_X16) if PLAN_X16.exists() else None
    _ok(checks, digest == PLAN_SHA256, "frozen X16 plan unchanged", f"sha256 {digest}")
    human = human or HUMAN_JUDGMENTS
    try:
        j = pd.read_csv(human, keep_default_na=False)
        missing = [c for c in JUDGMENT_COLUMNS if c not in j]
        _ok(checks, len(j) == 300 and not missing, "volunteers' judgments (X17)",
            f"{len(j)} rows; missing columns {missing}")
    except FileNotFoundError:
        _ok(checks, False, "volunteers' judgments (X17)", str(human))
    try:
        import torch
        cuda = torch.cuda.is_available()
        _ok(checks, cuda, "CUDA", torch.cuda.get_device_name(0) if cuda else "not available")
    except ImportError as error:
        _ok(checks, False, "torch", error)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 2 ** 30
    _ok(checks, free >= 10, "free disk in OUT >= 10 GB", f"{free:.0f} GB")
    failed = [c for c in checks if not c["passed"] and c["required"]]
    write_json(out / "preflight.json", dict(created_utc=utc(), campaign=str(campaign), models=names, checks=checks))
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed; {len(failed)} required missing")
    return 1 if failed else 0
