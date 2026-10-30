"""X9: other erasers (INLP, R-LACE), rotated erasers, and nonlinear leakage after erasure.

Motivation: erasure was run with LEACE and orthogonal projections only, so
the read/erase bound covers INLP (Ravfogel et al., 2020) and R-LACE (Ravfogel et al.,
2022) only in theory; and erasure was scored by linear probes, so nonlinear
leakage after cross-lingual erasure is unmeasured.

Design: the planned pair (en-idiom -> bn-fig), F at M = 11 and S at M = 3, every
encoder.

erasers  fitted on English (en-idiom) SPACE (cross) and on Bangla (bn-fig) SPACE
         (native): the COV and MLR images (orthogonal) and LEACE (oblique), as
         published; INLP, whose path is kept up to ``inlp_max_rank`` directions
         and whose operating point is the first step that guards the source
         language's VALIDATION rows; and R-LACE at rank M, selected on source
         VALIDATION. Every eraser's guard statistic on source VALIDATION is
         recorded. Random orthogonal subspaces of each removed rank calibrate
         what removing that many unrelated directions costs.
rotated  English erasers carried into Bangla coordinates by the Procrustes maps
         of X10's primary fit (held-out PROBE+VALIDATION pairs; full and top-128
         plane), never by a map fitted on the SPACE records that define the
         class means.
linear   Bangla probes retrained after erasure (the published protocol) and read
         through the removed subspace. Erase share = (A - a_e)/(A - a_b) against
         the Bangla-fitted eraser of the same kind and step; read share = SSI;
         omega = share of Bangla's class-mean (COV) image inside the removed
         subspace (chance: removed rank / D), so the read/erase bound (erase share <=
         omega <= read share) is checked for every eraser. English-on-English
         rows give each eraser's in-distribution erase share above the floor.
mlp      one-hidden-layer MLP probes (X6's grid and selection) retrained after
         COV, LEACE, INLP and R-LACE erasure: nonlinear erase shares and the
         nonlinear signal left above the floor after native and English erasure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug
from idiom_subspace.geometry import Subspace

from experiments import common
from experiments.common import Log, compact_readouts, concat, erase_readouts, labels, load_cell, make_study, rank_for
from experiments.metrics import component_weights, metric_values, paired, paired_two_sided, ssi
from experiments.nulls import majority_label
from experiments.x6_nonlinear import mlp_readout

from . import erasers as er
from .alignment import MAPS, PRIMARY_FIT, condition_mean, fit_maps, overlap, rotate
from .common2 import NEW, ArrayCache, cell_tables, digest, finish, source_hash

SPLITS = ("probe", "validation", "test")
SOURCE, TARGET = "en-idiom", "bn-fig"
FIXED = ["COV", "LEACE", "MLR"]
MLP_ERASERS = ["COV", "LEACE", "INLP", "RLACE"]
PATH_STEPS = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class Entry:
    """One eraser to evaluate: kind, fitted condition, raw or rotated, INLP step (0 otherwise)."""
    eraser: str
    source: str
    condition: str
    step: int
    sub: Subspace
    primary: bool = True
    draw: int = 0

    @property
    def rank(self):
        return int(self.sub.effective_rank)

    @property
    def key(self):
        # Random references are distinguished by rank; every other eraser's rank follows from its step.
        return (self.eraser, self.source, self.condition, self.step, self.draw,
                self.rank if self.eraser == "RAND" else 0)

    @property
    def native_key(self):
        """The Bangla-fitted eraser of the same kind and step (the erase and read shares' reference)."""
        return (self.eraser, TARGET, "raw", self.step, 0, 0)


def erased(X, sub):
    """x - (x F) G^T: oblique for LEACE, orthogonal otherwise (the readouts' erase map)."""
    return X - (X @ sub.input_factor) @ sub.basis.T


def share_ci(base, value, native, groups, n_boot, seed, min_den):
    """(mean base - mean value) / (mean base - mean native) with a paired component-bootstrap interval."""
    den = base.mean() - native.mean()
    estimate = float((base.mean() - value.mean()) / den) if den > min_den else np.nan
    if not np.isfinite(estimate):
        return estimate, np.nan, np.nan
    w = component_weights(groups, n_boot, seed)
    mean = lambda v: (w @ v) / w.sum(1)
    b_den = mean(base) - mean(native)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(b_den > min_den, (mean(base) - mean(value)) / b_den, np.nan)
    values = values[np.isfinite(values)]
    if not len(values):
        return estimate, np.nan, np.nan
    lo, hi = np.quantile(values, [.025, .975])
    return estimate, float(lo), float(hi)


def ssi_ci(cross, native, floor, groups, n_boot, seed, min_den):
    """Accuracy SSI of per-item correctness vectors with a paired component-bootstrap interval."""
    estimate = float(ssi(cross.mean(), native.mean(), floor.mean(), min_den))
    if not np.isfinite(estimate):
        return estimate, np.nan, np.nan
    w = component_weights(groups, n_boot, seed)
    mean = lambda v: (w @ v) / w.sum(1)
    values = ssi(mean(cross), mean(native), mean(floor), min_den)
    values = values[np.isfinite(values)]
    if not len(values):
        return estimate, np.nan, np.nan
    lo, hi = np.quantile(values, [.025, .975])
    return estimate, float(lo), float(hi)


# ----------------------------------------------------------------------------- fitting
def _cached(cache, key, fit, arrays, meta):
    found = cache.load(key)
    if found is not None:
        return found
    out = fit()
    cache.save(key, {k: out[k] for k in arrays}, {k: out[k] for k in meta})
    return {k: out[k] for k in arrays + meta}


def fit_erasers(ctx, study, concept, rank, cache, log):
    """Every eraser of one concept for both sources, with fit diagnostics and the INLP/R-LACE paths."""
    y = labels(study, concept)
    D = study.X.shape[1]
    eps = float(ctx.option("guard_eps", er.EPS))
    max_rank = int(ctx.option("inlp_max_rank", er.INLP_MAX_RANK))
    rl = {k: type(v)(ctx.option(f"rlace_{k}", v)) for k, v in er.RLACE.items()}
    lrs = rlace_rates(ctx)
    device = study._device
    seed = int(study.meta["seed"])
    cov = {s: study.sub(s, concept, "COV", rank) for s in (SOURCE, TARGET)}
    subs, diagnostics, paths = {}, [], []
    meta = dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept, M=rank, D=D)

    def describe(eraser, source, sub, extra):
        diagnostics.append(dict(meta, eraser=eraser, source=source, rank=sub.effective_rank,
                                omega_bn=overlap(cov[TARGET].basis, sub.basis),
                                omega_en=overlap(cov[SOURCE].basis, sub.basis), chance_omega=sub.effective_rank / D,
                                orthonormal_error=er.orthonormal_error(sub.basis), **extra))

    for source in (SOURCE, TARGET):
        space, val = study.idx(source, "space", concept), study.idx(source, "validation", concept)
        X, ys, Xv, yv = study.X[space], y[space], study.X[val], y[val]
        for method in FIXED:
            sub = study.sub(source, concept, method, rank)
            subs[(method, source)] = sub
            describe(method, source, sub, er.guard(X, ys, Xv, yv, sub.input_factor, sub.basis, device=device))
        log(f"{study.meta['model']} {concept}: INLP on {source}")
        key = digest("INLP", study._checkpoint_stamp, concept, source, max_rank, eps, er.INLP_C,
                     source_hash("erasers.py"))
        inlp = _cached(cache, key, lambda: er.inlp(X, ys, Xv, yv, max_rank, er.INLP_C, eps, device),
                       ["basis", "center", "ranks"], ["path", "stop", "guarded", "majority", "per_step", "seconds"])
        ranks = [int(r) for r in np.asarray(inlp["ranks"])]
        basis, center = np.asarray(inlp["basis"]), np.asarray(inlp["center"])
        for p in inlp["path"]:
            B = basis[:, :ranks[p["step"]]] if p["step"] < len(ranks) else basis
            paths.append(dict(meta, eraser="INLP", source=source, **p, omega_bn=overlap(cov[TARGET].basis, B),
                              omega_en=overlap(cov[SOURCE].basis, B), chance_omega=B.shape[1] / D,
                              is_stop=p["step"] == int(inlp["stop"])))
        subs[("INLP", source)] = dict(basis=basis, center=center, ranks=ranks, stop=int(inlp["stop"]),
                                      guarded=bool(inlp["guarded"]), path=inlp["path"])
        stop_sub = Subspace(center, basis[:, :ranks[int(inlp["stop"])]], ranks[int(inlp["stop"])], "INLP",
                            ranks[int(inlp["stop"])])
        stop_path = inlp["path"][int(inlp["stop"])]
        describe("INLP", source, stop_sub, dict(guard_val_acc=stop_path["val_acc"],
                                                guard_val_loss=stop_path["val_loss"],
                                                guard_val_majority=stop_path["val_majority"],
                                                guard_gap=stop_path["val_acc"] - stop_path["val_majority"],
                                                guard_probe_converged=stop_path["probe_converged"],
                                                inlp_stop=int(inlp["stop"]), inlp_guarded=bool(inlp["guarded"]),
                                                inlp_steps_run=len(inlp["path"]) - 1, fit_seconds=inlp["seconds"]))
        log(f"{study.meta['model']} {concept}: R-LACE on {source} (rank {rank})")
        key = digest("RLACE", study._checkpoint_stamp, concept, source, rank, eps, rl, lrs, seed,
                     source_hash("erasers.py"))

        def progress(phase, step, acc, majority, source=source):
            if step % (10 * rl["eval_every"]) == 0:
                log(f"{study.meta['model']} {concept}: R-LACE {source} {phase} step {step}: validation {acc:.3f} "
                    f"(majority {majority:.3f})")

        fitted = _cached(cache, key, lambda: er.rlace(X, ys, Xv, yv, rank, lrs=lrs, eps=eps, seed=seed, device=device,
                                                      progress=progress, **rl),
                         ["basis", "center"], ["history", "best_step", "best_val_acc", "best_val_loss", "majority",
                                               "steps_run", "stopped_early", "seconds", "lr_used", "pilot"])
        sub = Subspace(np.asarray(fitted["center"]), np.asarray(fitted["basis"]), rank, "RLACE", rank)
        subs[("RLACE", source)] = sub
        for p in fitted["pilot"]:
            paths.append(dict(meta, eraser="RLACE", source=source, phase="pilot", lr=p["lr"], removed_rank=rank,
                              val_acc=p["best_val_acc"], val_loss=p["best_val_loss"], steps_run=p["steps_run"],
                              diverged=p["diverged"], stopped_early=p["stopped_early"],
                              is_stop=p["lr"] == fitted["lr_used"]))
        for h in fitted["history"]:
            paths.append(dict(meta, eraser="RLACE", source=source, phase="final", lr=fitted["lr_used"], step=h["step"],
                              removed_rank=rank, val_acc=h["val_acc"], val_loss=h["val_loss"],
                              val_majority=h["val_majority"], guarded=bool(h["val_acc"] <= h["val_majority"] + eps),
                              probe_converged=h["probe_converged"], adversary_loss=h["adversary_loss"],
                              relaxed_mass_top_rank=h["relaxed_mass_top_rank"],
                              is_stop=h["step"] == fitted["best_step"]))
        g = er.guard(X, ys, Xv, yv, sub.basis, sub.basis, device=device)
        describe("RLACE", source, sub, dict(g, rlace_best_step=fitted["best_step"], rlace_steps_run=fitted["steps_run"],
                                            rlace_stopped_early=bool(fitted["stopped_early"]),
                                            rlace_selection_val_acc=fitted["best_val_acc"],
                                            fit_seconds=fitted["seconds"], rlace_lr_used=fitted["lr_used"],
                                            rlace_pilot_diverged=int(sum(p["diverged"] for p in fitted["pilot"])),
                                            **{f"rlace_{k}": v for k, v in rl.items()}))
    return subs, cov, diagnostics, paths


def rlace_rates(ctx):
    """R-LACE learning-rate grid for the pilot (``--rlace-lrs``, comma-separated)."""
    value = ctx.option("rlace_lrs")
    return tuple(float(v) for v in str(value).split(",") if v) if value else tuple(er.RLACE_LRS)


def inlp_sub(fit, step):
    step = min(int(step), len(fit["ranks"]) - 1)
    r = fit["ranks"][step]
    return Subspace(fit["center"], fit["basis"][:, :r], r, "INLP", r), step


def entries_for(study, subs, maps, rand_draws):
    """Raw erasers of both sources, rotated English erasers, and random references of every removed rank."""
    en, bn = subs[("INLP", SOURCE)], subs[("INLP", TARGET)]
    last = min(len(en["ranks"]), len(bn["ranks"])) - 1
    steps = sorted({s for s in PATH_STEPS if s <= last} | {last, min(en["stop"], last), min(bn["stop"], last)} - {0})
    cb = condition_mean(study, TARGET)
    entries = []
    for source in (SOURCE, TARGET):
        for method in FIXED + ["RLACE"]:
            entries.append(Entry(method, source, "raw", 0, subs[(method, source)]))
        for step in steps:
            sub, step = inlp_sub(subs[("INLP", source)], step)
            entries.append(Entry("INLP", source, "raw", step, sub, primary=step == min(en["stop"], last)))
    primary = [e for e in entries if e.source == SOURCE and e.primary]
    for name, W in maps.items():
        for e in primary:
            entries.append(Entry(e.eraser, SOURCE, name, e.step, rotate(e.sub, W, cb)))
    D = study.X.shape[1]
    for rank in sorted({e.rank for e in entries}):
        for draw in range(1, rand_draws + 1):
            rng = np.random.default_rng([int(study.meta["seed"]), 9, rank, draw])
            basis = np.linalg.qr(rng.normal(size=(D, rank)))[0]
            entries.append(Entry("RAND", SOURCE, "raw", 0, Subspace(cb, basis, rank, "RAND", rank), primary=False,
                                 draw=draw))
    return entries


# ----------------------------------------------------------------------------- linear readouts
def _target_setup(study, target, concept):
    y = labels(study, concept)
    idx = tuple(study.idx(target, s, concept) for s in SPLITS)
    te = idx[2]
    y_te = y[te]
    floor_pred = np.full(len(y_te), majority_label(study, target, concept))
    groups = study.items.iloc[te].component_id.astype(str).to_numpy()
    return y, idx, y_te, floor_pred, groups


def linear_part(ctx, study, concept, rank, entries, cov, guard_gaps, log):
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    sign_draws = int(ctx.option("sign_draws", 19999))
    seed = int(study.meta["seed"])
    min_den = study.cfg.min_den
    D = study.X.shape[1]
    specs = [(t, concept, t, concept, "AMBIENT", D, "project", False, 0, None, True) for t in (TARGET, SOURCE)]
    study._prefetch(specs, f"X9 ambient {concept}")
    rows, results = [], {}
    for target in (TARGET, SOURCE):
        y, idx, y_te, floor_pred, groups = _target_setup(study, target, concept)
        tr, va, te = idx
        classes = np.unique(y_te)
        chosen = [e for e in entries if target == TARGET or (e.source == SOURCE and e.condition == "raw")]
        log(f"{study.meta['model']} {concept}: linear readouts on {target} ({len(chosen)} erasers)")
        ambient = study._evaluate_sub(target, concept, target, concept, "AMBIENT", D)
        base = (np.asarray(ambient["predicted"]) == y_te).astype(float)
        floor = (floor_pred == y_te).astype(float)
        amb_m, floor_m = metric_values(y_te, ambient["predicted"], classes), metric_values(y_te, floor_pred, classes)
        # The published estimators' raw erasures are fitted in X7's call pattern (English and Bangla
        # eraser together), so they reproduce X7 exactly; every other eraser is batched.
        paired_raw = {}
        if target == TARGET:
            for method in FIXED:
                pair = [next(e for e in chosen if e.key == (method, s, "raw", 0, 0, 0)) for s in (SOURCE, TARGET)]
                found = erase_readouts(study, study.X[tr], study.X[va], study.X[te], [e.sub.input_factor for e in pair],
                                       [e.sub.basis for e in pair], y, idx)
                paired_raw.update({e.key: ev for e, ev in zip(pair, found)})
        rest = [e for e in chosen if e.key not in paired_raw]
        batched = erase_readouts(study, study.X[tr], study.X[va], study.X[te], [e.sub.input_factor for e in rest],
                                 [e.sub.basis for e in rest], y, idx)
        by_key = dict(paired_raw)
        by_key.update({e.key: ev for e, ev in zip(rest, batched)})
        erase_ev = [by_key[e.key] for e in chosen]
        readable = [e for e in chosen if e.eraser != "RAND"]
        read_ev = compact_readouts(study, [e.sub.project(study.X[tr]) for e in readable],
                                   [e.sub.project(study.X[va]) for e in readable],
                                   [e.sub.project(study.X[te]) for e in readable], y, idx)
        got = {e.key: dict(erase=ev) for e, ev in zip(chosen, erase_ev)}
        for e, ev in zip(readable, read_ev):
            got[e.key]["read"] = ev
        results[target] = dict(got=got, base=base, floor=floor, ambient=ambient, y_te=y_te)
        correct = {k: (np.asarray(v["erase"]["predicted"]) == y_te).astype(float) for k, v in got.items()}
        reads = {k: (np.asarray(v["read"]["predicted"]) == y_te).astype(float) for k, v in got.items() if "read" in v}
        rand = {}
        for e in chosen:
            if e.eraser == "RAND":
                rand.setdefault(e.rank, []).append(100 * (base.mean() - correct[e.key].mean()))
        reference = cov[target].basis
        other = cov[SOURCE if target == TARGET else TARGET].basis
        for e in chosen:
            a = correct[e.key]
            m = metric_values(y_te, got[e.key]["erase"]["predicted"], classes)
            drop = paired(base - a, groups, seed=41, n_boot=n_boot, sign_draws=sign_draws)
            omega = overlap(reference, e.sub.basis)
            row = dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept, target=target,
                       eraser=e.eraser, source=e.source, condition=e.condition, step=e.step, draw=e.draw,
                       primary=e.primary, rank=e.rank, M=rank, D=D, n=len(y_te),
                       ambient_acc=float(base.mean()), ambient_bal=amb_m["bal"], floor_acc=float(floor.mean()),
                       floor_bal=floor_m["bal"], erased_acc=float(a.mean()), erased_bal=m["bal"],
                       drop_pp=100 * drop["estimate"], drop_ci_lo=100 * drop["ci_lo"], drop_ci_hi=100 * drop["ci_hi"],
                       drop_p=drop["p"], drop_bal_pp=100 * (amb_m["bal"] - m["bal"]),
                       omega=omega, omega_other=overlap(other, e.sub.basis), chance_omega=e.rank / D,
                       rand_drop_pp=float(np.mean(rand[e.rank])) if e.rank in rand else np.nan,
                       probe_converged_erase=bool(got[e.key]["erase"].get("probe_converged", True)),
                       source_guard_gap=guard_gaps.get((e.eraser, e.source, e.step), np.nan))
            native_key = e.native_key
            if target == TARGET and e.eraser != "RAND" and native_key in correct:
                share, lo, hi = share_ci(base, a, correct[native_key], groups, n_boot, seed, min_den)
                row.update(native_erased_acc=float(correct[native_key].mean()),
                           native_drop_pp=100 * (base.mean() - correct[native_key].mean()),
                           erase_share=share, erase_share_ci_lo=lo, erase_share_ci_hi=hi)
            if target == SOURCE:
                share, lo, hi = share_ci(base, a, floor, groups, n_boot, seed, min_den)
                row.update(share_above_floor=share, share_above_floor_ci_lo=lo, share_above_floor_ci_hi=hi)
            if e.key in reads:
                r = reads[e.key]
                row.update(read_acc=float(r.mean()),
                           probe_converged_read=bool(got[e.key]["read"].get("probe_converged", True)))
                if target == TARGET and native_key in reads:
                    est, lo, hi = ssi_ci(r, reads[native_key], floor, groups, n_boot, seed, min_den)
                    row.update(native_read_acc=float(reads[native_key].mean()), read_share=est,
                               read_share_ci_lo=lo, read_share_ci_hi=hi)
            if target == TARGET and np.isfinite(row.get("read_share", np.nan)):
                row["read_ok"] = bool(row["read_share"] >= omega)
            if target == TARGET and np.isfinite(row.get("erase_share", np.nan)):
                row["erase_ok"] = bool(row["erase_share"] <= omega)
            rows.append(row)
    return rows, results


# ----------------------------------------------------------------------------- MLP readouts
def mlp_part(ctx, study, concept, entries, linear, cache, log):
    n_jobs = int(ctx.option("n_jobs", 8))
    small = bool(ctx.option("mlp_small", False))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    sign_draws = int(ctx.option("sign_draws", 19999))
    seed = int(study.meta["seed"])
    min_den = study.cfg.min_den
    x6_hash = common.text_sha256(NEW / "experiments" / "x6_nonlinear.py")
    rows = []
    for target in (TARGET, SOURCE):
        y, idx, y_te, floor_pred, groups = _target_setup(study, target, concept)
        classes = np.unique(y_te)
        floor = (floor_pred == y_te).astype(float)
        lin = linear[target]
        if target == TARGET:
            chosen = [e for e in entries if e.eraser in MLP_ERASERS and e.primary
                      and (e.condition in ("raw", "procrustes"))]
        else:
            chosen = [e for e in entries if e.eraser in MLP_ERASERS and e.primary and e.source == SOURCE
                      and e.condition == "raw"]
        fits = {}
        for e in [None] + chosen:
            label = "AMBIENT" if e is None else f"{e.eraser} {e.source} {e.condition} step {e.step}"
            arrays = [] if e is None else [e.sub.input_factor, e.sub.basis]
            key = digest("MLP", study._checkpoint_stamp, concept, target, label, small, x6_hash, *arrays)
            found = cache.load(key)
            if found is None:
                log(f"{study.meta['model']} {concept}: MLP on {target} after {label}")
                parts = tuple(study.X[i] if e is None else erased(study.X[i], e.sub) for i in idx)
                out = mlp_readout(parts, y, idx, n_jobs, small=small)
                cache.save(key, dict(predictions=out["predictions"]),
                           dict(hidden=out["hidden"], alpha=out["alpha"], validation=out["validation"]))
                found = out
            fits[None if e is None else e.key] = found
        amb = fits[None]
        amb_item = (np.asarray(amb["predictions"]) == y_te).mean(0)
        base_lin = lin["base"]
        for e in chosen:
            fit = fits[e.key]
            preds = np.asarray(fit["predictions"])
            per_seed = [metric_values(y_te, p, classes) for p in preds]
            item = (preds == y_te).mean(0)
            lin_item = (np.asarray(lin["got"][e.key]["erase"]["predicted"]) == y_te).astype(float)
            drop = paired(amb_item - item, groups, seed=41, n_boot=n_boot, sign_draws=sign_draws)
            versus = paired_two_sided(item - lin_item, groups, seed=43, n_boot=n_boot, sign_draws=sign_draws)
            row = dict(model=study.meta["model"], split=seed, layer=study.meta["layer"], concept=concept,
                       target=target, eraser=e.eraser, source=e.source, condition=e.condition, step=e.step,
                       rank=e.rank, n=len(y_te), mlp_hidden=int(fit["hidden"]), mlp_alpha=float(fit["alpha"]),
                       mlp_validation_acc=float(fit["validation"]),
                       mlp_acc=float(np.mean([m["acc"] for m in per_seed])),
                       mlp_acc_seed_sd=float(np.std([m["acc"] for m in per_seed])),
                       mlp_bal=float(np.mean([m["bal"] for m in per_seed])),
                       linear_acc=float(lin_item.mean()), ambient_mlp_acc=float(amb_item.mean()),
                       ambient_mlp_hidden=int(amb["hidden"]), ambient_mlp_alpha=float(amb["alpha"]),
                       ambient_linear_acc=float(base_lin.mean()), floor_acc=float(floor.mean()),
                       mlp_drop_pp=100 * drop["estimate"], mlp_drop_ci_lo=100 * drop["ci_lo"],
                       mlp_drop_ci_hi=100 * drop["ci_hi"], mlp_drop_p=drop["p"],
                       linear_drop_pp=100 * float(base_lin.mean() - lin_item.mean()),
                       mlp_above_floor_pp=100 * float(item.mean() - floor.mean()),
                       linear_above_floor_pp=100 * float(lin_item.mean() - floor.mean()),
                       mlp_minus_linear_pp=100 * versus["estimate"], mlp_minus_linear_ci_lo=100 * versus["ci_lo"],
                       mlp_minus_linear_ci_hi=100 * versus["ci_hi"], mlp_minus_linear_p=versus["p_two_sided"])
            native_key = e.native_key
            if target == TARGET and native_key in fits:
                native_item = (np.asarray(fits[native_key]["predictions"]) == y_te).mean(0)
                share, lo, hi = share_ci(amb_item, item, native_item, groups, n_boot, seed, min_den)
                row.update(mlp_native_drop_pp=100 * float(amb_item.mean() - native_item.mean()), mlp_erase_share=share,
                           mlp_erase_share_ci_lo=lo, mlp_erase_share_ci_hi=hi)
            if target == SOURCE:
                share, lo, hi = share_ci(amb_item, item, floor, groups, n_boot, seed, min_den)
                row.update(mlp_share_above_floor=share, mlp_share_above_floor_ci_lo=lo,
                           mlp_share_above_floor_ci_hi=hi)
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- driver
def run_cell(ctx, study, log, parts):
    cache = ArrayCache(ctx.work / "fitted")
    rand_draws = int(ctx.option("rand_draws", 3))
    maps, map_info = fit_maps(study, SOURCE, TARGET, PRIMARY_FIT)
    tables = {k: [] for k in ["erasers.csv", "path.csv", "linear.csv", "mlp.csv"]}
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        if rank is None:
            continue
        subs, cov, diagnostics, paths = fit_erasers(ctx, study, concept, rank, cache, log)
        for d in diagnostics:
            d.update(map_fit=PRIMARY_FIT, map_fit_pairs=map_info["fit_pairs"])
        tables["erasers.csv"] += diagnostics
        tables["path.csv"] += paths
        entries = entries_for(study, subs, maps, rand_draws)
        gaps = {(d["eraser"], d["source"], 0): d["guard_gap"] for d in diagnostics if d["eraser"] != "INLP"}
        gaps.update({("INLP", p["source"], p["step"]): p["val_acc"] - p["val_majority"]
                     for p in paths if p["eraser"] == "INLP"})
        lin_rows, linear = linear_part(ctx, study, concept, rank, entries, cov, gaps, log)
        tables["linear.csv"] += lin_rows
        if "mlp" in parts:
            tables["mlp.csv"] += mlp_part(ctx, study, concept, entries, linear, cache, log)
    return {k: pd.DataFrame(v) for k, v in tables.items()}


def summarize(linear, mlp):
    """One row per encoder x concept x eraser x condition at the English eraser's operating point.

    The native (Bangla-fitted) reference is the eraser of the same kind and step, so
    ``native_guard_gap`` is that eraser's guard gap on Bangla VALIDATION at that step.
    """
    if linear.empty:
        return pd.DataFrame()
    keys = ["model", "split", "concept", "eraser"]
    main = linear[linear.target.eq(TARGET) & linear.source.eq(SOURCE) & linear.primary.astype(bool)
                  & linear.eraser.ne("RAND")].copy()
    cols = keys + ["layer", "condition", "step", "rank", "M", "D", "ambient_acc", "floor_acc", "erased_acc", "drop_pp",
                   "drop_ci_lo", "drop_ci_hi", "drop_p", "native_drop_pp", "erase_share", "erase_share_ci_lo",
                   "erase_share_ci_hi", "omega", "chance_omega", "read_share", "read_share_ci_lo",
                   "read_share_ci_hi", "read_ok", "erase_ok", "rand_drop_pp", "source_guard_gap"]
    out = main[[c for c in cols if c in main]]
    english = linear[linear.target.eq(SOURCE) & linear.primary.astype(bool) & linear.condition.eq("raw")
                     & linear.eraser.ne("RAND")]
    if "share_above_floor" in english:
        out = out.merge(english[keys + ["share_above_floor", "omega"]].rename(
            columns={"share_above_floor": "english_erase_share_above_floor", "omega": "english_omega"}),
            on=keys, how="left")
    native = linear[linear.target.eq(TARGET) & linear.source.eq(TARGET) & linear.condition.eq("raw")
                    & linear.eraser.ne("RAND")]
    native = native[keys + ["step", "source_guard_gap"]].rename(columns={"source_guard_gap": "native_guard_gap"})
    out = out.merge(native, on=keys + ["step"], how="left")
    if mlp is not None and not mlp.empty:
        m = mlp[mlp.target.eq(TARGET) & mlp.source.eq(SOURCE)]
        out = out.merge(m[keys + ["condition", "mlp_acc", "mlp_drop_pp", "mlp_native_drop_pp", "mlp_erase_share",
                                  "mlp_erase_share_ci_lo", "mlp_erase_share_ci_hi", "mlp_above_floor_pp",
                                  "ambient_mlp_acc"]], on=keys + ["condition"], how="left")
        nat = mlp[mlp.target.eq(TARGET) & mlp.source.eq(TARGET)][keys + ["mlp_above_floor_pp", "linear_above_floor_pp"]]
        out = out.merge(nat.rename(columns={"mlp_above_floor_pp": "native_erased_mlp_above_floor_pp",
                                            "linear_above_floor_pp": "native_erased_linear_above_floor_pp"}),
                        on=keys, how="left")
        eng = mlp[mlp.target.eq(SOURCE)][keys + ["mlp_share_above_floor", "mlp_above_floor_pp"]]
        out = out.merge(eng.rename(columns={"mlp_share_above_floor": "english_mlp_erase_share_above_floor",
                                            "mlp_above_floor_pp": "english_erased_mlp_above_floor_pp"}),
                        on=keys, how="left")
    return out


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    parts = [p for p in ctx.option("parts", "linear,mlp").split(",") if p]
    tables = {k: [] for k in ["erasers.csv", "path.csv", "linear.csv", "mlp.csv"]}
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=[SOURCE, TARGET])
                study = make_study(ctx, cell, "erasers")
                saved = cell_tables(ctx, key, run_cell(ctx, study, log, parts))
                del study, cell
            else:
                log(f"restored {key}")
            for k in tables:
                tables[k].append(saved.get(k))
    out = {k: concat(v) for k, v in tables.items()}
    if "mlp" not in parts:
        out.pop("mlp.csv")
    out["summary.csv"] = summarize(out["linear.csv"], out.get("mlp.csv"))
    finish(ctx, out, dict(parts=parts, source=SOURCE, target=TARGET, fixed=FIXED, mlp_erasers=MLP_ERASERS,
                          path_steps=PATH_STEPS, guard_eps=float(ctx.option("guard_eps", er.EPS)),
                          inlp_C=er.INLP_C, inlp_max_rank=int(ctx.option("inlp_max_rank", er.INLP_MAX_RANK)),
                          rlace={k: ctx.option(f"rlace_{k}", v) for k, v in er.RLACE.items()},
                          rlace_lrs=list(rlace_rates(ctx)),
                          rotation_fit=PRIMARY_FIT, rotation_maps=list(MAPS)))
    log("finished")
