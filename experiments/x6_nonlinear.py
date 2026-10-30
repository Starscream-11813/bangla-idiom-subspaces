"""X6: nonlinear (MLP) readouts through the subspaces, and a corrected online-code MDL.

Motivation: readouts are linear; MLP probes gain 0.2-5.0 pp within Bangla,
but their effect on cross-lingual transfer is unmeasured, and the published MDL
diagnostic was worse than a uniform code.

Part A (``mlp``): for English-fitted (cross) and Bangla-fitted (native) MLR, COV
and LEACE subspaces, plus PCA and RAND controls, Bangla PROBE projections train a
one-hidden-layer MLP (hidden 64 or 256, L2 alpha in {1e-4, 1e-2, 1}, chosen on
VALIDATION by mean accuracy over three seeds; inputs standardized on PROBE).
SSI_MLP is compared with the linear SSI of the same projections.

Part B (``mdl``): online codelength (Voita & Titov, 2020) on PROBE+VALIDATION
rows in a component-respecting random order with blocks at 0.1, 0.2, 0.4, 0.8,
1.6, 3.2, 6.25, 12.5, 25, 50 and 100%. The first block is sent with a uniform
code; each later block with an L2 softmax probe over *all* K classes trained on
the preceding items, mixed with a Krichevsky-Trofimov label-frequency code
(weight eps). (C, eps) is chosen per block from a 5 x 4 grid, and the index is
charged (log2 20 bits per block), so the code stays decodable. The published
version gave probability 1e-12 to classes unseen in the prefix and had no prior
mixture, which is why it was worse than uniform. The KT code alone is the
label-frequency baseline; compression is reported against both.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from idiom_subspace.config import model_slug

from .common import Log, cell_tables, concat, finish, labels, load_cell, make_study, rank_for
from .metrics import component_weights, metric_values, ssi
from .nulls import majority_label

VOITA = [.001, .002, .004, .008, .016, .032, .0625, .125, .25, .5, 1.]
MDL_CS = [1e-3, 1e-2, 1e-1, 1., 10.]
EPSILONS = [0., 1e-3, 1e-2, 1e-1]
HIDDEN = [64, 256]
ALPHAS = [1e-4, 1e-2, 1.]
SEEDS = [0, 1, 2]
METHODS = ["MLR", "COV", "LEACE"]


# ----------------------------------------------------------------------------- MLP readouts
def _mlp_one(Xtr, ytr, Xva, Xte, hidden, alpha, seed):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    model = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=500, n_iter_no_change=20, tol=1e-4,
                          random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(scaler.transform(Xtr), ytr)
    return model.predict(scaler.transform(Xva)), model.predict(scaler.transform(Xte))


def mlp_readout(parts, y, rows, n_jobs, small=False):
    """Grid over (hidden, alpha) x seeds; returns the selected config's per-seed TEST predictions."""
    from joblib import Parallel, delayed
    try:
        from joblib import parallel_config as backend_context
    except ImportError:  # joblib < 1.3
        from joblib import parallel_backend as backend_context
    (Xtr, Xva, Xte), (tr, va, te) = parts, rows
    hidden, alphas, seeds = ([64], [1e-2], [0]) if small else (HIDDEN, ALPHAS, SEEDS)
    grid = [(h, a, s) for h in hidden for a in alphas for s in seeds]
    # One BLAS thread per worker: n_jobs processes x inherited thread pools would oversubscribe.
    with backend_context("loky", inner_max_num_threads=1):
        fits = Parallel(n_jobs=n_jobs)(delayed(_mlp_one)(Xtr, y[tr], Xva, Xte, h, a, s) for h, a, s in grid)
    scores = {}
    for (h, a, s), (pva, pte) in zip(grid, fits):
        scores.setdefault((h, a), []).append((float(np.mean(pva == y[va])), pte))
    best = max(scores, key=lambda k: (np.mean([v for v, _ in scores[k]]), -k[0], k[1]))
    return dict(hidden=best[0], alpha=best[1], validation=float(np.mean([v for v, _ in scores[best]])),
                predictions=np.stack([p for _, p in scores[best]]))


def _ssi_difference(cross_mlp, native_mlp, cross_lin, native_lin, floor, groups, n_boot, seed, min_den):
    """Bootstrap CI of SSI_MLP - SSI_linear (accuracy), all on the same TEST items."""
    w = component_weights(groups, n_boot, seed)
    mean = lambda v: (w @ v) / w.sum(1)
    a = ssi(mean(cross_mlp), mean(native_mlp), mean(floor), min_den)
    b = ssi(mean(cross_lin), mean(native_lin), mean(floor), min_den)
    d = (a - b)[np.isfinite(a - b)]
    return (float(np.quantile(d, .025)), float(np.quantile(d, .975))) if len(d) else (np.nan, np.nan)


def mlp_part(ctx, study, log):
    n_jobs = int(ctx.option("n_jobs", 8))
    rand_draws = int(ctx.option("mlp_rand_draws", 3))
    n_boot = int(ctx.option("bootstrap", ctx.cfg.bootstrap))
    rows = []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        y = labels(study, concept)
        idx = tuple(study.idx("bn-fig", s, concept) for s in ["probe", "validation", "test"])
        test_y = y[idx[2]]
        classes = np.unique(test_y)
        floor = (np.full(len(test_y), majority_label(study, "bn-fig", concept)) == test_y).astype(float)
        groups = study.items.iloc[idx[2]].component_id.astype(str).to_numpy()
        D = study.X.shape[1]
        specs = [("bn-fig", concept, "bn-fig", concept, "AMBIENT", D, "project", False, 0, None, True)]
        for method in METHODS + ["PCA"]:
            for source in ["en-idiom", "bn-fig"]:
                specs.append(("bn-fig", concept, source, concept, method, rank, "project", False, 0, None, True))
        specs += [("bn-fig", concept, "en-idiom", concept, "RAND", rank, "project", False, d, None, True)
                  for d in range(1, rand_draws + 1)]
        study._prefetch(specs, f"X6 linear {concept}")
        entries = [("AMBIENT", "native", "bn-fig", 0)]
        entries += [(m, d, s, 0) for m in METHODS + ["PCA"] for d, s in [("cross", "en-idiom"), ("native", "bn-fig")]]
        entries += [("RAND", "cross", "en-idiom", d) for d in range(1, rand_draws + 1)]
        results = {}
        for method, direction, source, draw in entries:
            log(f"{study.meta['model']} {concept}: MLP readout {method} {direction} draw {draw}")
            if method == "AMBIENT":
                parts = tuple(study.X[i] for i in idx)
                linear = study._evaluate_sub("bn-fig", concept, "bn-fig", concept, "AMBIENT", D)
            else:
                sub = study.sub(source, concept, method, rank, draw=draw)
                parts = tuple(sub.project(study.X[i]) for i in idx)
                linear = study._evaluate_sub("bn-fig", concept, source, concept, method, rank, draw=draw)
            mlp = mlp_readout(parts, y, idx, n_jobs, small=bool(ctx.option("mlp_small", False)))
            results[(method, direction, draw)] = (mlp, linear)
            per_seed = [metric_values(test_y, p, classes) for p in mlp["predictions"]]
            lin = metric_values(test_y, linear["predicted"], classes)
            rows.append(dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                             concept=concept, method=method, direction=direction, draw=draw, M=rank if method != "AMBIENT" else D,
                             n=len(test_y), mlp_hidden=mlp["hidden"], mlp_alpha=mlp["alpha"],
                             mlp_validation_acc=mlp["validation"],
                             mlp_acc=float(np.mean([m["acc"] for m in per_seed])),
                             mlp_acc_seed_sd=float(np.std([m["acc"] for m in per_seed])),
                             mlp_bal=float(np.mean([m["bal"] for m in per_seed])),
                             mlp_f1=float(np.mean([m["f1"] for m in per_seed])),
                             linear_acc=lin["acc"], linear_bal=lin["bal"], linear_f1=lin["f1"],
                             floor_acc=float(floor.mean())))
        # SSI of each estimator against its own native subspace; RAND draws against each
        # estimator's native subspace, as in the published null design.
        comparisons = [(m, 0, m) for m in METHODS + ["PCA"]]
        comparisons += [("RAND", d, m) for d in range(1, rand_draws + 1) for m in METHODS]
        for method, draw, reference in comparisons:
            mlp_c, lin_c = results[(method, "cross", draw)]
            mlp_n, lin_n = results[(reference, "native", 0)]
            c_m = (mlp_c["predictions"] == test_y).mean(0)
            n_m = (mlp_n["predictions"] == test_y).mean(0)
            c_l = (np.asarray(lin_c["predicted"]) == test_y).astype(float)
            n_l = (np.asarray(lin_n["predicted"]) == test_y).astype(float)
            lo, hi = _ssi_difference(c_m, n_m, c_l, n_l, floor, groups, n_boot, int(study.meta["seed"]),
                                     study.cfg.min_den)
            rows.append(dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                             concept=concept, method=method, direction="SSI", draw=draw, M=rank, n=len(test_y),
                             native_reference=reference,
                             SSI_mlp=float(ssi(c_m.mean(), n_m.mean(), floor.mean(), study.cfg.min_den)),
                             SSI_linear=float(ssi(c_l.mean(), n_l.mean(), floor.mean(), study.cfg.min_den)),
                             SSI_mlp_minus_linear_ci_lo=lo, SSI_mlp_minus_linear_ci_hi=hi))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- online-code MDL
def component_order(groups, seed):
    units = np.unique(groups)
    rank = {u: i for i, u in enumerate(np.random.default_rng(seed).permutation(units))}
    return np.argsort(np.asarray([rank[g] for g in groups]), kind="stable")


def block_ends(ordered_groups, fractions=VOITA):
    """Block end positions at the given fractions, moved forward to component boundaries."""
    n = len(ordered_groups)
    starts = np.flatnonzero(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]])
    ends = []
    for f in fractions:
        j = int(np.searchsorted(starts, int(np.ceil(f * n))))
        end = int(starts[j]) if j < len(starts) else n
        if end == 0:
            end = int(starts[1]) if len(starts) > 1 else n
        if not ends or end > ends[-1]:
            ends.append(end)
    if ends[-1] != n:
        ends.append(n)
    return ends


def kt_bits(codes, K):
    """Sequential Krichevsky-Trofimov (add-1/2) label code, in bits."""
    counts = np.zeros(K)
    bits = 0.
    for i, c in enumerate(codes):
        bits -= np.log2((counts[c] + .5) / (i + .5 * K))
        counts[c] += 1
    return float(bits)


def _block_probabilities(Z_train, y_train, Z_block, K, device, dtype):
    import torch
    from idiom_subspace.fastprobe import EraseFeatures, _predict, _to_tensor, fit_batched
    Xt, Tt = _to_tensor(Z_train, device, dtype), _to_tensor(Z_block, device, dtype)
    empty = torch.zeros(len(MDL_CS), Z_train.shape[1], 0, device=device, dtype=dtype)
    feats = EraseFeatures(Xt, Xt[:1], Tt, empty, empty)
    W, b, _ = fit_batched(feats, y_train, K, MDL_CS, max_iter=1000)
    with torch.no_grad():
        return _predict(feats, "test", W, b, np.arange(K)).double().cpu().numpy()


def online_code(Z, codes, K, groups, seed, device, dtype):
    order = component_order(groups, seed)
    Z, codes, groups = Z[order], codes[order], groups[order]
    ends = block_ends(groups)
    bits = ends[0] * np.log2(K)
    charge = np.log2(len(MDL_CS) * len(EPSILONS))
    chosen = []
    for start, stop in zip(ends[:-1], ends[1:]):
        probs = _block_probabilities(Z[:start], codes[:start], Z[start:stop], K, device, dtype)
        prior = (np.bincount(codes[:start], minlength=K) + .5) / (start + .5 * K)
        target = codes[start:stop]
        best = None
        for ci, C in enumerate(MDL_CS):
            for eps in EPSILONS:
                p = (1 - eps) * probs[ci][np.arange(stop - start), target] + eps * prior[target]
                value = float(-np.log2(np.maximum(p, 1e-300)).sum())
                if best is None or value < best[0]:
                    best = (value, C, eps)
        bits += best[0] + charge
        chosen.append(dict(start=int(start), stop=int(stop), C=best[1], eps=best[2], bits=best[0]))
    return float(bits), chosen, float(kt_bits(codes, K))


def mdl_part(ctx, study, log):
    from idiom_subspace.fastprobe import default_dtype
    device, dtype = study._device, default_dtype(study._device)
    orders = int(ctx.option("mdl_orders", 3))
    rand_draws = int(ctx.option("mdl_rand_draws", 3))
    rows = []
    for concept in ["F", "S"]:
        rank = rank_for(study, concept)
        y = labels(study, concept)
        universe = np.unique(y[study.idx("bn-fig", "space", concept)])
        K = len(universe)
        reps = []
        for condition in ["bn-fig", "bn-context", "bn-idiom", "en-idiom"]:
            if condition in set(study._condition):
                reps.append((condition, "AMBIENT", "native", 0, None))
        for method in METHODS:
            reps.append(("bn-fig", method, "cross", 0, study.sub("en-idiom", concept, method, rank)))
            reps.append(("bn-fig", method, "native", 0, study.sub("bn-fig", concept, method, rank)))
        for d in range(1, rand_draws + 1):
            reps.append(("bn-fig", "RAND", "cross", d, study.sub("en-idiom", concept, "RAND", rank, draw=d)))
        for condition, method, direction, draw, sub in reps:
            ix = np.concatenate([study.idx(condition, "probe", concept), study.idx(condition, "validation", concept)])
            keep = np.isin(y[ix], universe)
            ix = ix[keep]
            codes = np.searchsorted(universe, y[ix])
            Z = study.X[ix] if sub is None else sub.project(study.X[ix])
            groups = study.items.iloc[ix].component_id.astype(str).to_numpy()
            log(f"{study.meta['model']} {concept}: MDL {condition} {method} {direction} draw {draw}")
            for order in range(orders):
                bits, blocks, prior = online_code(Z, codes, K, groups, 1000 * int(study.meta["seed"]) + order,
                                                  device, dtype)
                uniform = len(codes) * np.log2(K)
                rows.append(dict(model=study.meta["model"], split=int(study.meta["seed"]), layer=study.meta["layer"],
                                 concept=concept, condition=condition, method=method, direction=direction, draw=draw,
                                 M=rank if sub is not None else study.X.shape[1], order=order, n=len(codes), K=K,
                                 online_bits=bits, uniform_bits=float(uniform), prior_bits=prior,
                                 compression_vs_uniform=float(uniform / bits),
                                 compression_vs_prior=float(prior / bits), gain_over_prior_bits=float(prior - bits),
                                 blocks=len(blocks) + 1,
                                 first_trained_block=int(blocks[0]["start"]) if blocks else None,
                                 eps_zero_blocks=int(sum(b["eps"] == 0 for b in blocks))))
    return pd.DataFrame(rows)


def run(ctx):
    log = Log(ctx.work, ctx.experiment)
    parts = ctx.option("parts", "mlp,mdl").split(",")
    mlp, mdl = [], []
    for split in ctx.splits:
        for number, name in enumerate(ctx.models, 1):
            key = f"{model_slug(name)}_split{split}"
            saved = cell_tables(ctx, key)
            if saved is None:
                log(f"loading {name} split {split}", model=name, split=split, completed=number - 1,
                    total=len(ctx.models))
                cell = load_cell(ctx, name, split, conditions=["en-idiom", "bn-fig", "bn-context", "bn-idiom"])
                study = make_study(ctx, cell, "nonlinear")
                tables = {}
                if "mlp" in parts:
                    tables["mlp.csv"] = mlp_part(ctx, study, log)
                if "mdl" in parts:
                    tables["mdl.csv"] = mdl_part(ctx, study, log)
                saved = cell_tables(ctx, key, tables)
                del study, cell
            else:
                log(f"restored {key}")
            mlp.append(saved.get("mlp.csv"))
            mdl.append(saved.get("mdl.csv"))
    tables = {"mlp.csv": concat(mlp), "mdl.csv": concat(mdl)}
    mdl_all = tables["mdl.csv"]
    if not mdl_all.empty:
        keys = ["model", "split", "concept", "condition", "method", "direction"]
        summary = mdl_all.groupby(keys, dropna=False).agg(
            online_bits=("online_bits", "mean"), online_bits_sd=("online_bits", "std"),
            prior_bits=("prior_bits", "mean"), uniform_bits=("uniform_bits", "mean"),
            compression_vs_uniform=("compression_vs_uniform", "mean"),
            compression_vs_prior=("compression_vs_prior", "mean"), n=("n", "first"), K=("K", "first")).reset_index()
        tables["mdl_summary.csv"] = summary
    finish(ctx, tables, dict(parts=parts, voita_fractions=VOITA, mdl_Cs=MDL_CS, epsilons=EPSILONS,
                             mlp_hidden=HIDDEN, mlp_alphas=ALPHAS, mlp_seeds=SEEDS))
    log("finished")
