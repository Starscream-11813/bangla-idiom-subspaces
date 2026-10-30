"""S1--S4, calibrated inference, and probe robustness on prepared features.

Every transform/subspace is fitted on SPACE; probes use PROBE and tune on
VALIDATION. Test rows only enter final scores and explicitly paired inference.
"""
from pathlib import Path
import hashlib
import json
import os
import uuid
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.metrics import accuracy_score
from .geometry import (Subspace, fit_estimator, fit_probe, tune_probe, shared_subspace_index, mlr_coefficients,
                       null_adjusted_index, paired_cluster_bootstrap, monte_carlo_pvalue, holm,
                       resolve_probe_backend)


_SOURCE_DIGEST = None


def _source_digest():
    """Digest of the analysis source as first seen by this process.

    Computed once, so editing or pulling code during a long run cannot give
    later layers a different checkpoint identity than the code actually running.
    """
    global _SOURCE_DIGEST
    if _SOURCE_DIGEST is None:
        digest = hashlib.sha256()
        for source in (Path(__file__), Path(__file__).with_name("geometry.py"), Path(__file__).with_name("fastprobe.py")):
            digest.update(source.read_bytes())
        _SOURCE_DIGEST = digest.hexdigest()
    return _SOURCE_DIGEST


_source_digest()


def lexical_baselines(items, cfg):
    """Word/character TF-IDF baseline; vocabulary and probe fit on PROBE only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    rows = []
    for condition in items.condition.unique():
        f = items[items.condition.eq(condition)]
        for field in ["span", "text"]:
            if field == "span" and condition == "bn-context": continue
            use = f[f.span_found] if field == "span" else f
            for concept in cfg.concepts:
                if concept not in use: continue
                data = {s: use[use.split.eq(s) & use[concept].ge(0)] for s in ["probe", "validation", "test"]}
                if min(map(len, data.values())) == 0: continue
                tr, va, te = [data[s] for s in ["probe", "validation", "test"]]
                if tr[concept].nunique() > max(100, len(tr)//3): continue
                vectorizer = FeatureUnion([
                    ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, token_pattern=r"(?u)\b\w+\b")),
                    ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=20000))])
                try: xtr = vectorizer.fit_transform(tr[field])
                except ValueError: continue
                xva, xte = vectorizer.transform(va[field]), vectorizer.transform(te[field])
                best, best_c, best_val = None, None, -1.
                for c in sorted(cfg.probe_Cs):
                    probe = (LogisticRegression(C=c, max_iter=1000, random_state=cfg.seeds[0]) if tr[concept].nunique() > 1
                             else DummyClassifier(strategy="most_frequent"))
                    probe.fit(xtr, tr[concept])
                    score = accuracy_score(va[concept], probe.predict(xva))
                    if score > best_val: best, best_c, best_val = probe, c, score
                rows.append(dict(condition=condition, field=field, concept=concept, C=best_c,
                                 R=accuracy_score(te[concept], best.predict(xte)), validation_R=best_val,
                                 R0=float(te[concept].eq(tr[concept].mode().iloc[0]).mean()), n=len(te),
                                 vocabulary_fit_split="probe", vectorizer="word_1_2_plus_char_2_5"))
    return pd.DataFrame(rows)


def layer_ssi_table(grid, min_den=.02):
    """Layer-wise point indices with absolute scores and explicit single-draw RAND."""
    if grid.empty: return pd.DataFrame()
    keys = [c for c in ["model", "seed", "label_seed", "layer", "split_mode", "concept", "M", "condition", "group"] if c in grid]
    estimated = grid[grid.method.ne("AMBIENT")]
    own = estimated[estimated.source_condition.eq(estimated.condition)]
    refs = own[keys+["method", "R"]].rename(columns={"R": "R_pp"})
    cross = estimated[estimated.source_condition.ne(estimated.condition)].copy()
    result = cross.merge(refs, on=keys+["method"], validate="many_to_one")
    for baseline in ["RAND", "PCA"]:
        columns = keys+["source_condition"]
        null = cross[cross.method.eq(baseline)][columns+["R"]].rename(columns={"R": baseline+"_R"})
        result = result.merge(null, on=columns, how="left", validate="many_to_one")
    result["denominator"] = result.R_pp - result.R0
    result["SSI"] = [shared_subspace_index(a,b,c,min_den) for a,b,c in zip(result.R,result.R_pp,result.R0)]
    result["RAND_SSI_single_draw"] = [shared_subspace_index(a,b,c,min_den) for a,b,c in zip(result.RAND_R,result.R_pp,result.R0)]
    result["PCA_SSI"] = [shared_subspace_index(a,b,c,min_den) for a,b,c in zip(result.PCA_R,result.R_pp,result.R0)]
    result["uncertainty"] = "point_estimates_only_no_layer_selection_on_test"
    return result


def groups_for(frame):
    out = {"all": np.ones(len(frame), dtype=bool)}
    for col in ["anchored", "cultural", "religious", "historical", "rural", "rare"]:
        if col in frame:
            a = frame[col].fillna(False).to_numpy(bool)
            out[col], out["non" + col] = a, ~a
    # Counterpart-transparency tertiles (SPACE-fitted cut-offs; middle tertile excluded).
    for col in ["transparent", "opaque"]:
        if col in frame:
            out[col] = frame[col].fillna(False).to_numpy(bool)
    if {"rare", "anchored"} <= set(frame):
        a, r = frame.anchored.to_numpy(bool), frame.rare.to_numpy(bool)
        for name, mask in [("anchored_common", a & ~r), ("unanchored_common", ~a & ~r),
                           ("anchored_rare", a & r), ("unanchored_rare", ~a & r)]: out[name] = mask
    return out


def matched_anchor_masks(space, test, seed=0):
    """Frequency/length/token-count matched test cohorts, with SPACE-only bins.

    Matching is a sensitivity analysis of observed covariates, not randomization
    of counterpart availability. No outcomes enter binning or sampling.
    """
    required = {"anchored", "rare", "idiom_len", "token_count"}
    if not required <= set(space) or not required <= set(test): return {}
    buckets = []
    for col in ["idiom_len", "token_count"]:
        edges = np.unique(np.nanquantile(space[col].to_numpy(float), [.25, .5, .75]))
        buckets.append(np.searchsorted(edges, test[col].to_numpy(float), side="right"))
    frequency = (test.freq_raw.astype(str).str.strip().str.casefold().to_numpy()
                 if "freq_raw" in test else test.rare.to_numpy(bool))
    strata = pd.DataFrame(dict(length=buckets[0], tokens=buckets[1], frequency=frequency))
    a = test.anchored.to_numpy(bool)
    selected = np.zeros(len(test), dtype=bool)
    rng = np.random.default_rng(seed)
    for ix in strata.groupby(["length", "tokens", "frequency"], sort=True).indices.values():
        yes, no = ix[a[ix]], ix[~a[ix]]
        n = min(len(yes), len(no))
        if n:
            selected[rng.choice(yes, n, replace=False)] = True
            selected[rng.choice(no, n, replace=False)] = True
    return {"anchored_matched": selected & a, "unanchored_matched": selected & ~a}


def _margin(model, X, y):
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.where(np.asarray(y) == model.classes_[0], 1., -1.)
    result = -proba.max(axis=1)
    lookup = {c: j for j, c in enumerate(model.classes_)}
    columns = np.asarray([lookup.get(label, -1) for label in y])
    known = columns >= 0
    rows = np.flatnonzero(known)
    competitors = proba[known].copy()
    competitors[np.arange(len(rows)), columns[known]] = -np.inf
    result[known] = proba[rows, columns[known]] - competitors.max(axis=1)
    return result


class LayerStudy:
    """One layer's tasks, with train-only estimator reuse and resumable readouts.

    Checkpoints contain JSON and numeric arrays, never executable pickles. A
    completion marker is published only after its evidence is written. The
    fingerprint includes the data, labels, source and readout mathematics, while
    excluding null-draw/bootstrapping budgets so these can be extended safely.
    """
    CHECKPOINT_VERSION = "layer-study-v3"

    def __init__(self, X, items, cfg, meta, checkpoint_dir=None, progress=None):
        self.X, self.items, self.cfg, self.meta = np.asarray(X), items.reset_index(drop=True), cfg, dict(meta)
        self.rows, self.skips, self.evidence, self.subspaces = [], [], {}, {}
        self._estimators, self._indices, self._evaluations, self._groups = {}, {}, {}, {}
        self._permutation_diagnostics, self._permutation_units = {}, {}
        self._languages = self.items.groupby("condition", sort=False).lang.first().to_dict()
        self._condition = self.items.condition.to_numpy()
        self._split = self.items.split.to_numpy()
        self.progress = progress
        self.backend = resolve_probe_backend(cfg)
        self.probe_max_iter = int(getattr(cfg, "probe_max_iter", 5000))
        self.mlr_variant = getattr(cfg, "mlr_variant", "centered")
        self.grid_rand_draws = max(1, int(getattr(cfg, "grid_rand_draws", 1)))
        self._device = None
        if self.backend == "torch":
            from .fastprobe import resolve_device
            self._device = resolve_device(getattr(cfg, "probe_device", "auto"))
        self.checkpoint_dir = None
        if checkpoint_dir is not None:
            digest = hashlib.sha256(self.CHECKPOINT_VERSION.encode())
            digest.update(_source_digest().encode())
            settings = {name: getattr(cfg, name, None) for name in (
                "concepts", "methods", "ranks", "pairs", "probe_Cs", "min_group",
                "compute_leakage", "estimator_C", "estimator_ridge", "probe_max_iter",
                "mlr_variant", "grid_rand_draws", "open_class_ridge")}
            settings["probe_backend_resolved"] = self.backend
            settings["meta"] = {k: v for k, v in self.meta.items() if k not in {"layer_role", "role"}}
            digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
            columns = [c for c in ["condition", "lang", "split", "idiom_id", "item_id", "component_id",
                       "idiom_group", "anchored", "rare", "cultural", "religious", "historical", "rural",
                       "freq_raw", "idiom_len", "token_count", "F", "Ft", "D", "S", "L",
                       "cp_transparency", "transparent", "opaque"] if c in self.items]
            digest.update(self.items[columns].to_json(orient="split", force_ascii=False).encode())
            digest.update(str((self.X.shape, self.X.dtype.str)).encode())
            for start in range(0, len(self.X), 4096):
                digest.update(np.ascontiguousarray(self.X[start:start+4096]).tobytes())
            self._checkpoint_stamp = digest.hexdigest()
            self.checkpoint_dir = Path(checkpoint_dir) / self._checkpoint_stamp
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_json(self.checkpoint_dir / "manifest.json", dict(
                version=self.CHECKPOINT_VERSION, stamp=self._checkpoint_stamp,
                settings=settings, evidence_format="numeric_npz_no_pickle"))

    @staticmethod
    def _atomic_json(path, value):
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        def convert(v):
            if isinstance(v, np.generic): return v.item()
            if isinstance(v, np.ndarray): return v.tolist()
            raise TypeError(type(v).__name__)
        temporary.write_text(json.dumps(value, ensure_ascii=False, default=convert), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _key(key):
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()

    def _path(self, kind, key, extension):
        return self.checkpoint_dir / f"{kind}_{self._key(key)}.{extension}"

    def _load_evaluation(self, key):
        if key in self._evaluations: return True, self._evaluations[key]
        if self.checkpoint_dir is None or not getattr(self.cfg, "resume", True): return False, None
        marker = self._path("eval", key, "json")
        if not marker.exists(): return False, None
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state["status"] == "unavailable": return True, None
        evidence = self._path("eval", key, "npz")
        if not evidence.exists(): return False, None
        with np.load(evidence, allow_pickle=False) as values:
            result = {name: values[name].item() if values[name].ndim == 0 else values[name] for name in values.files}
        return True, result

    def _save_evaluation(self, key, value, retain=True):
        if retain: self._evaluations[key] = value
        if self.checkpoint_dir is None: return
        if value is not None:
            path = self._path("eval", key, "npz")
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            with temporary.open("wb") as destination:
                np.savez(destination, **{k: v for k, v in value.items() if k != "model"})
            os.replace(temporary, path)
        self._atomic_json(self._path("eval", key, "json"), dict(status="completed" if value is not None else "unavailable"))

    def _report(self, stage, completed, total, message, **context):
        if self.progress is not None:
            self.progress.update(stage=stage, completed=completed, total=total, message=message,
                                 **dict(self.meta, **context))

    def idx(self, condition, split, concept):
        key = (condition, split, concept)
        if key not in self._indices:
            labels = pd.to_numeric(self.items[concept], errors="coerce").to_numpy()
            self._indices[key] = np.flatnonzero((self._condition == condition) & (self._split == split) & (labels >= 0))
        return self._indices[key]

    def _sub_key(self, condition, concept, method, rank, permutation, draw, subset):
        selection = None if subset is None else hashlib.sha256(np.asarray(subset, dtype=np.int64).tobytes()).hexdigest()
        unit = getattr(self.cfg, "permutation_unit", "component") if permutation else None
        return (condition, concept, method, rank, bool(permutation), draw, selection, unit)

    def _permuted_labels(self, condition, concept, ix, y, draw):
        requested = getattr(self.cfg, "permutation_unit", "component")
        if requested not in {"component", "idiom", "idiom_type"}:
            raise ValueError("permutation_unit must be component or an explicitly diagnostic idiom type unit.")
        column = "component_id" if requested == "component" else "idiom_group"
        if column not in self.items:
            raise ValueError(f"Permutation unit {requested!r} requires {column} metadata.")
        key = (condition, concept, column, hashlib.sha256(np.asarray(ix, dtype=np.int64).tobytes()).hexdigest())
        if key not in self._permutation_units:
            frame = self.items.iloc[ix]
            if frame[column].isna().any(): raise ValueError("Permutation units cannot contain missing values.")
            units = list(frame.groupby(column, sort=True).indices.values())
            by_size = {}
            for positions in units: by_size.setdefault(len(positions), []).append(positions)
            movable = sum(size * len(bundles) for size, bundles in by_size.items() if len(bundles) > 1)
            diagnostic = dict(permutation_unit=column, permutation_units=len(units),
                              permutation_frozen_units=sum(len(b) for b in by_size.values() if len(b) == 1),
                              permutation_movable_fraction=movable / len(ix))
            self._permutation_units[key] = by_size, diagnostic
        by_size, diagnostic = self._permutation_units[key]
        rng = np.random.default_rng(self.meta["seed"] + 1009 * (draw + 1))
        shuffled = y.copy()
        for size in sorted(by_size):
            bundles = by_size[size]
            for destination, source in zip(bundles, rng.permutation(len(bundles))):
                shuffled[destination] = y[bundles[source]]
        return shuffled, diagnostic

    def sub(self, condition, concept, method, rank, permutation=None, draw=0, subset=None):
        key = self._sub_key(condition, concept, method, rank, permutation, draw, subset)
        if key in self.subspaces: return self.subspaces[key]
        ix = self.idx(condition, "space", concept) if subset is None else subset
        if len(ix) < 3: return None
        y = self.items[concept].to_numpy()[ix]
        if len(np.unique(y)) < 2 and method not in {"RAND", "PCA"}: return None
        estimator_key = key[:3] + key[4:]
        if estimator_key not in self._estimators:
            if permutation:
                y, diagnostic = self._permuted_labels(condition, concept, ix, y, draw)
                self._permutation_diagnostics[estimator_key] = diagnostic
            self._estimators[estimator_key] = fit_estimator(
                self.X[ix], y, method, max([rank] + list(self.cfg.ranks)),
                seed=self.meta["seed"] + 1009 * draw, **self._estimator_options())
        sub = self._estimators[estimator_key].at_rank(rank)
        # Requested/effective rank mismatch is never silently called a matched-M result.
        if sub.effective_rank != rank: return None
        self.subspaces[key] = sub
        return sub

    def _estimator_options(self):
        ridge = getattr(self, "_ridge_override", None)
        if ridge is None:
            ridge = getattr(self.cfg, "estimator_ridge", 0.)
        return dict(ridge=ridge, C=getattr(self.cfg, "estimator_C", 1.),
                    mlr_variant=self.mlr_variant, backend=self.backend, max_iter=self.probe_max_iter,
                    device=self._device)

    def _forget_draw(self, condition, concept, method, rank, permutation, draw, subset):
        """Release a null draw's estimator after its readouts are checkpointed."""
        key = self._sub_key(condition, concept, method, rank, permutation, draw, subset)
        self.subspaces.pop(key, None)
        self._estimators.pop(key[:3] + key[4:], None)

    def _prefetch_permutation_estimators(self, requests):
        """Fit MLR estimators for many permutation draws in one GPU batch.

        ``requests`` holds (condition, concept, method, rank, draw) tuples. Other
        estimators are closed-form and are fitted lazily by ``sub``.
        """
        if self.backend != "torch":
            return
        grouped = {}
        for condition, concept, method, rank, draw in requests:
            if method != "MLR":
                continue
            key = self._sub_key(condition, concept, method, rank, True, draw, None)
            if key[:3] + key[4:] in self._estimators:
                continue
            grouped.setdefault((condition, concept), {})[draw] = key[:3] + key[4:]
        for (condition, concept), draws in grouped.items():
            ix = self.idx(condition, "space", concept)
            if len(ix) < 3:
                continue
            original = self.items[concept].to_numpy()[ix]
            if len(np.unique(original)) < 2:
                continue
            labels, keys = [], []
            for draw, estimator_key in sorted(draws.items()):
                y, diagnostic = self._permuted_labels(condition, concept, ix, original, draw)
                self._permutation_diagnostics[estimator_key] = diagnostic
                labels.append((draw, estimator_key, y))
            batch = int(getattr(self.cfg, "estimator_batch", 32))
            for start in range(0, len(labels), batch):
                chunk = labels[start:start + batch]
                coefs = mlr_coefficients(self.X[ix], [np.unique(y, return_inverse=True)[1] for _, _, y in chunk],
                                         getattr(self.cfg, "estimator_C", 1.), variant=self.mlr_variant,
                                         backend="torch", max_iter=self.probe_max_iter, device=self._device)
                for (draw, estimator_key, y), coef in zip(chunk, coefs):
                    self._estimators[estimator_key] = fit_estimator(
                        self.X[ix], y, "MLR", max(self.cfg.ranks), seed=self.meta["seed"] + 1009 * draw,
                        mlr_coef=coef, **self._estimator_options())

    def _prefetch(self, specs, label="readouts"):
        """Compute missing readouts in GPU batches and checkpoint each one.

        Each spec is (condition, concept, source, source_concept, method, rank,
        representation, permutation, draw, subset, need_margin) with the exact
        semantics of ``_evaluate_sub``; results are stored under the same keys,
        so the sequential task loops afterwards only restore them. A no-op for
        the scikit-learn backend. Without resumable checkpoints, use the scalar
        Torch path: nonzero-draw results would otherwise be fitted twice.
        """
        if (self.backend != "torch" or not specs or self.checkpoint_dir is None
                or not getattr(self.cfg, "resume", True)):
            return
        from .fastprobe import compact_features, erase_features, tune_readouts, default_dtype
        pending, seen = [], set()
        for spec in specs:
            (condition, concept, source, source_concept, method, rank, representation,
             permutation, draw, subset, need_margin) = spec
            sub_key = self._sub_key(source, source_concept or concept, method, rank, permutation, draw, subset)
            key = ("readout", sub_key, condition, concept, representation, "linear", need_margin)
            if key in seen:
                continue
            seen.add(key)
            found, _ = self._load_evaluation(key)
            if not found:
                pending.append((key, spec))
        if not pending:
            return
        dtype = default_dtype(self._device)
        groups = {}
        for key, spec in pending:
            (condition, concept, source, source_concept, method, rank, representation,
             permutation, draw, subset, need_margin) = spec
            kind = "compact" if method != "AMBIENT" and representation == "project" else "erase"
            # Keep only task metadata here. Fitting estimators and retaining
            # their factors for all null draws defeats readout batch bounds.
            groups.setdefault((condition, concept, kind, bool(need_margin)), []).append((key, spec))
        completed = 0
        planned = sum(len(members) for members in groups.values())
        self._report("batched_" + label, 0, planned, f"GPU-batched {label}: {planned} probe fits in {len(groups)} groups")
        for (condition, concept, kind, need_margin), members in groups.items():
            indices = [self.idx(condition, split, concept) for split in ["probe", "validation", "test"]]
            tr, va, te = indices
            y = self.items[concept].to_numpy()
            if kind == "compact":
                width = max(1, max(spec[5] for _, spec in members))
                rows = len(tr) + len(va) + len(te)
                size = max(1, min(int(getattr(self.cfg, "probe_batch_compact", 512)), int(4e7 // max(rows * width, 1))))
            else:
                size = max(1, int(getattr(self.cfg, "probe_batch_ambient", 32)))
            if any(spec[7] and spec[4] == "MLR" for _, spec in members):
                size = min(size, max(1, int(getattr(self.cfg, "estimator_batch", 32))))
            for start in range(0, len(members), size):
                chunk = members[start:start + size]
                payloads = []
                try:
                    if min(map(len, indices)):
                        perm = [(s[2], s[3] or s[1], s[4], s[5], s[8]) for _, s in chunk
                                if s[7] and s[9] is None]
                        self._prefetch_permutation_estimators(perm)
                    for key, spec in chunk:
                        (condition, concept, source, source_concept, method, rank, representation,
                         permutation, draw, subset, need_margin) = spec
                        if min(map(len, indices)) == 0:
                            self._save_evaluation(key, None, retain=not draw)
                            continue
                        if method == "AMBIENT":
                            payload = ("erase", np.zeros((self.X.shape[1], 0)), np.zeros((self.X.shape[1], 0)))
                        else:
                            sub = self.sub(source, source_concept or concept, method, rank, permutation, draw, subset)
                            if sub is None:
                                self._save_evaluation(key, None, retain=not draw)
                                continue
                            payload = (("compact", sub.input_factor) if representation == "project"
                                       else ("erase", sub.input_factor, sub.basis))
                        payloads.append((key, spec, payload))
                    if payloads:
                        if kind == "compact":
                            feats = compact_features([self.X[tr] @ p[1] for _, _, p in payloads],
                                                     [self.X[va] @ p[1] for _, _, p in payloads],
                                                     [self.X[te] @ p[1] for _, _, p in payloads], self._device, dtype)
                        else:
                            feats = erase_features(self.X[tr], self.X[va], self.X[te], [p[1] for _, _, p in payloads],
                                                   [p[2] for _, _, p in payloads], self._device, dtype)
                        results = tune_readouts(feats, y, indices, self.cfg.probe_Cs, need_margin=need_margin,
                                                max_iter=max(self.probe_max_iter, 1000))
                        for (key, spec, _), result in zip(payloads, results):
                            permutation, draw = spec[7], spec[8]
                            if permutation:
                                result.update(self._permutation_diagnostics[key[1][:3] + key[1][4:]])
                            self._save_evaluation(key, result, retain=not draw)
                        del feats, results
                finally:
                    for _, spec in chunk:
                        if spec[8]:
                            self._forget_draw(spec[2], spec[3] or spec[1], spec[4], spec[5], spec[7], spec[8], spec[9])
                completed += len(chunk)
                self._report("batched_" + label, completed, planned, f"GPU-batched {label}", batch=len(chunk))

    def _fit_evaluation(self, parts, condition, concept, kind="linear", need_margin=True):
        tr, va, te = [self.idx(condition, s, concept) for s in ["probe", "validation", "test"]]
        if min(map(len, [tr, va, te])) == 0: return None
        y = self.items[concept].to_numpy()
        return self._fit_parts(parts, (tr, va, te), y, kind, need_margin)

    def _fit_parts(self, parts, indices, y, kind="linear", need_margin=True):
        tr, va, te = indices
        xtr, xva, xte = parts
        if self.backend == "torch" and kind == "linear":
            from .fastprobe import compact_features, tune_readouts, default_dtype
            feats = compact_features([xtr], [xva], [xte], self._device, default_dtype(self._device))
            return tune_readouts(feats, y, indices, self.cfg.probe_Cs, need_margin=need_margin,
                                 max_iter=max(self.probe_max_iter, 1000))[0]
        probe, C = tune_probe(xtr, y[tr], xva, y[va], self.cfg.probe_Cs,
                              seed=self.meta["seed"], kind=kind, max_iter=self.probe_max_iter)
        classes, counts = np.unique(y[tr], return_counts=True)
        majority = classes[np.argmax(counts)]
        return dict(probe_converged=bool(getattr(probe, "converged_", True)),
                    correct=probe.predict(xte) == y[te],
                    validation_correct=probe.predict(xva) == y[va],
                    floor=y[te] == majority, margin=_margin(probe, xte, y[te]) if need_margin else np.empty(0),
                    test_idx=te, val_idx=va, C=C,
                    unseen_test_classes=int((~np.isin(y[te], classes)).sum()))

    def evaluate(self, features, condition, concept, kind="linear"):
        """Public uncached evaluation; internal tasks use explicit content keys."""
        if features is self.X:
            return self._evaluate_sub(condition, concept, condition, concept, "AMBIENT", self.X.shape[1], kind=kind)
        indices = [self.idx(condition, split, concept) for split in ["probe", "validation", "test"]]
        return self._fit_evaluation([features[ix] for ix in indices], condition, concept, kind)

    def _evaluate_sub(self, condition, concept, source=None, source_concept=None, method="AMBIENT", rank=0,
                      representation="project", kind="linear", permutation=False, draw=0, subset=None,
                      need_margin=True):
        sub_key = self._sub_key(source, source_concept or concept, method, rank, permutation, draw, subset)
        key = ("readout", sub_key, condition, concept, representation, kind, need_margin)
        found, cached = self._load_evaluation(key)
        if found:
            if not draw: self._evaluations[key] = cached
            return cached
        indices = [self.idx(condition, split, concept) for split in ["probe", "validation", "test"]]
        if min(map(len, indices)) == 0:
            self._save_evaluation(key, None, retain=not draw); return None
        if method == "AMBIENT":
            parts = [self.X[ix] for ix in indices]
        else:
            sub = self.sub(source, source_concept or concept, method, rank, permutation, draw, subset)
            if sub is None:
                self._save_evaluation(key, None, retain=not draw); return None
            # A complete complement QR costs O(D^3). Compact residual coordinates
            # help when much of the space is erased; small-rank erasures retain
            # ambient coordinates, with the identical Euclidean/L2 metric.
            transform = sub.project if representation == "project" else (
                sub.erase_coordinates if method != "LEACE" and rank * 4 >= self.X.shape[1] else sub.erase)
            parts = [transform(self.X[ix]) for ix in indices]
        result = self._fit_evaluation(parts, condition, concept, kind, need_margin)
        if permutation and result is not None:
            result.update(self._permutation_diagnostics[sub_key[:3] + sub_key[4:]])
        self._save_evaluation(key, result, retain=not draw)
        return result

    def _custom_readout(self, key, sub, indices, labels, representation="project"):
        key = ("custom", key, representation)
        found, result = self._load_evaluation(key)
        if found: return result
        transform = (lambda x: x) if sub is None else sub.project if representation == "project" else sub.erase
        result = self._fit_parts([transform(self.X[ix]) for ix in indices], indices, labels, need_margin=False)
        self._save_evaluation(key, result)
        return result

    def _group_info(self, condition, concept, matched=False):
        key = (condition, concept, matched)
        if key not in self._groups:
            frame = self.items.iloc[self.idx(condition, "test", concept)]
            masks = groups_for(frame)
            if matched:
                masks.update(matched_anchor_masks(self.items.iloc[self.idx(condition, "space", concept)], frame, self.meta["seed"]))
            components = frame.component_id.to_numpy()
            self._groups[key] = (masks, {name: (int(mask.sum()), int(np.unique(components[mask]).size)) for name, mask in masks.items()})
        return self._groups[key]

    def _grid_task(self, concept, method, rank, source, target):
        key = ("grid", concept, method, rank, source, target)
        marker = self._path("grid", key, "json") if self.checkpoint_dir is not None else None
        if marker is not None and getattr(self.cfg, "resume", True) and marker.exists():
            stored = json.loads(marker.read_text(encoding="utf-8"))
            self.rows.extend(dict(row, **self.meta) for row in stored["rows"])
            if stored.get("skip"):
                self.skips.append(stored["skip"]); return
            ev = self._evaluate_sub(target, concept, source, concept, method, rank)
            if ev is not None and method != "AMBIENT": self.evidence[(concept, method, rank, source, target)] = ev
            return
        start = len(self.rows)
        ev = self._evaluate_sub(target, concept, source, concept, method, rank)
        skip = None
        if ev is None:
            skip = dict(concept=concept, source=source, target=target, method=method, M=rank,
                        reason="insufficient_data_or_rank")
            self.skips.append(skip)
        else:
            leak = None
            leakage = method != "AMBIENT" and getattr(self.cfg, "compute_leakage", True)
            if leakage:
                leak = self._evaluate_sub(target, concept, source, concept, method, rank, representation="erase")
            shown, shown_leak = ev, leak
            if method == "RAND" and self.grid_rand_draws > 1:
                # Average several random subspaces (the reference uses five).
                draws = [ev] + [self._evaluate_sub(target, concept, source, concept, method, rank, draw=d)
                                for d in range(1, self.grid_rand_draws)]
                draws = [d for d in draws if d is not None]
                shown = dict(ev, correct=np.mean([d["correct"] for d in draws], axis=0),
                             validation_correct=np.mean([d["validation_correct"] for d in draws], axis=0))
                if leakage:
                    leaks = [leak] + [self._evaluate_sub(target, concept, source, concept, method, rank,
                                                         representation="erase", draw=d)
                                      for d in range(1, self.grid_rand_draws)]
                    leaks = [d for d in leaks if d is not None]
                    shown_leak = dict(leak, correct=np.mean([d["correct"] for d in leaks], axis=0))
            gap = np.nan
            if method not in {"AMBIENT", "RAND"}:
                sub = self.sub(source, concept, method, rank)
                gap = sub.spectral_gap if sub is not None else np.nan
            self.add_rows(shown, shown_leak, concept, method, rank, source, target,
                          formal=shown_leak if method == "LEACE" else None,
                          draws=self.grid_rand_draws if method == "RAND" else 1, gap=gap)
            if method != "AMBIENT": self.evidence[(concept, method, rank, source, target)] = ev
        if marker is not None:
            self._atomic_json(marker, dict(rows=self.rows[start:], skip=skip))

    def _grid_specs(self, tasks):
        specs = []
        leakage = getattr(self.cfg, "compute_leakage", True)
        for concept, method, rank, source, target in tasks:
            draws = range(self.grid_rand_draws) if method == "RAND" else [0]
            for draw in draws:
                specs.append((target, concept, source, concept, method, rank, "project", False, draw, None, True))
                if method != "AMBIENT" and leakage:
                    specs.append((target, concept, source, concept, method, rank, "erase", False, draw, None, True))
        return specs

    def run_grid(self):
        self.rows, self.skips = [], []
        conditions = list(self.items.condition.unique())
        pairs = sorted(set(map(tuple, self.cfg.pairs)) | {(c, c) for c in conditions})
        tasks = []
        for concept in self.cfg.concepts:
            if concept not in self.items: continue
            training_labels = self.items.loc[self.items.split.eq("probe") & self.items[concept].ge(0)].drop_duplicates("idiom_id")[concept]
            if (len(training_labels) < 3 or training_labels.nunique() > max(100, len(training_labels) // 3)
                    or training_labels.value_counts().ge(3).mean() < .5):
                self.skips.append(dict(concept=concept, reason="target_too_sparse_for_a_reliable_probe"))
                continue
            for target in conditions: tasks.append((concept, "AMBIENT", self.X.shape[1], target, target))
            for source, target in pairs:
                if source not in conditions or target not in conditions:
                    self.skips.append(dict(concept=concept, source=source, target=target, reason="condition_unavailable"))
                    continue
                for method in self.cfg.methods:
                    for rank in self.cfg.ranks:
                        tasks.append((concept, method, rank, source, target))
        self._report("grid", 0, len(tasks), "planning readout tasks", grid_tasks=len(tasks),
                     backend=self.backend)
        self._prefetch(self._grid_specs(tasks), label="grid readouts")
        for completed, task in enumerate(tasks, 1):
            concept, method, rank, source, target = task
            self._report("grid", completed-1, len(tasks), "readout task", concept=concept, method=method,
                         M=rank, source=source, target=target)
            self._grid_task(*task)
            self._report("grid", completed, len(tasks), "readout task completed", concept=concept, method=method,
                         M=rank, source=source, target=target)
        return pd.DataFrame(self.rows)

    def add_rows(self, ev, leak, concept, method, rank, source, target, formal=None, draws=1, gap=np.nan):
        masks, counts = self._group_info(target, concept)
        for group, mask in masks.items():
            n, n_components = counts[group]
            if n < (1 if group == "all" else self.cfg.min_group): continue
            self.rows.append(dict(self.meta, concept=concept, method=method, M=rank,
                                  effective_rank=rank, source_condition=source, condition=target,
                                  e=self._languages[source], p=self._languages[target], group=group, n=n,
                                  n_components=n_components,
                                  R=float(ev["correct"][mask].mean()),
                                  L=float(leak["correct"][mask].mean()) if leak else np.nan,
                                  L_formal=float(formal["correct"][mask].mean()) if formal else np.nan,
                                  R0=float(ev["floor"][mask].mean()), C=ev["C"],
                                  validation_R=float(ev["validation_correct"].mean()),
                                  leakage_convention="exact_oblique" if method == "LEACE" else "orthogonal_complement",
                                  leakage_computed=leak is not None,
                                  unseen_test_classes=ev["unseen_test_classes"],
                                  subspace_draws=draws, probe_backend=self.backend, spectral_gap=gap,
                                  probe_converged=bool(ev.get("probe_converged", True)),
                                  leak_probe_converged=bool(leak.get("probe_converged", True)) if leak else None))

    def inference(self):
        rows, predictions = [], []
        method = self.cfg.inference_method
        primary_pairs = set(map(tuple, getattr(self.cfg, "primary_test_pairs", [("en-idiom", "bn-fig")])))
        primary_concepts = set(getattr(self.cfg, "primary_test_concepts", ["F", "S"]))
        primary_groups = set(getattr(self.cfg, "primary_test_groups", ["all"]))
        inference_groups = set(getattr(self.cfg, "inference_groups", ["all"]))
        minimum_movable = getattr(self.cfg, "min_permutation_movable_fraction", .5)
        tasks = [(concept, source, target) for concept in self.cfg.concepts for source, target in self.cfg.pairs]
        plans = {}
        completed_draws, total_draws = 0, 0
        for concept, source, target in tasks:
            ranks = [m for m in self.cfg.ranks if (concept, method, m, source, target) in self.evidence
                     and (concept, method, m, target, target) in self.evidence]
            if not ranks: continue
            primary = (source, target) in primary_pairs and concept in primary_concepts
            budget = self.cfg.null_draws if primary else getattr(self.cfg, "diagnostic_null_draws", 19)
            space_idx = self.idx(source, "space", concept)
            _, info = self._permuted_labels(source, concept, space_idx, self.items[concept].to_numpy()[space_idx], 0)
            movable = info["permutation_movable_fraction"] >= minimum_movable
            component = info["permutation_unit"] == "component_id"
            perm_budget = (getattr(self.cfg, "perm_null_draws", None) or budget) if primary else budget
            permutation_draws = (perm_budget if component else getattr(self.cfg, "diagnostic_null_draws", 19)) if movable else 0
            plans[(concept, source, target)] = (max(ranks), budget, info, movable, component, permutation_draws)
            total_draws += 1 + budget + permutation_draws
        specs = []
        for (concept, source, target), (rank, budget, _, _, _, permutation_draws) in plans.items():
            specs += [(target, concept, source, concept, "RAND", rank, "project", False, d, None, False)
                      for d in range(1, budget + 1)]
            specs.append((target, concept, source, concept, "PCA", rank, "project", False, 1, None, False))
            specs += [(target, concept, source, concept, method, rank, "project", True, d, None, False)
                      for d in range(1, permutation_draws + 1)]
        self._prefetch(specs, label="null draws")
        for concept in self.cfg.concepts:
            for source, target in self.cfg.pairs:
                if (concept, source, target) not in plans: continue
                rank, budget, permutation_info, enough_movable, component_permutation, permutation_draws = plans[(concept, source, target)]
                # Prespecified highest numerically supported matched rank; never test-selected.
                ev = self.evidence[(concept, method, rank, source, target)]
                own = self.evidence[(concept, method, rank, target, target)]
                assert np.array_equal(ev["test_idx"], own["test_idx"])
                frame = self.items.iloc[ev["test_idx"]].copy()
                base = dict(self.meta, concept=concept, method=method, M=rank,
                            source_condition=source, condition=target,
                            e=self._languages[source], p=self._languages[target])
                primary_task = (source, target) in primary_pairs and concept in primary_concepts
                permutation_inferential = enough_movable and component_permutation and primary_task
                null = {"RAND": [], "PCA": [], "PERM": []}
                for null_kind in null:
                    ndraw = 1 if null_kind == "PCA" else budget
                    if null_kind == "PERM": ndraw = permutation_draws
                    for draw in range(ndraw):
                        self._report("nulls", completed_draws, total_draws, "null readout", concept=concept,
                                     source=source, target=target, null=null_kind, draw=draw+1)
                        cev = self._evaluate_sub(target, concept, source, concept,
                                                 method if null_kind == "PERM" else null_kind, rank,
                                                 permutation=null_kind == "PERM", draw=draw+1, need_margin=False)
                        if cev is not None: null[null_kind].append(cev["correct"])
                        completed_draws += 1
                        self._report("nulls", completed_draws, total_draws, "null readout completed", concept=concept,
                                     source=source, target=target, null=null_kind, draw=draw+1)
                null_mean_vec = np.mean(null["RAND"], axis=0) if null["RAND"] else np.full(len(frame), np.nan)
                group_masks, group_counts = self._group_info(target, concept,
                                                             matched=any("matched" in group for group in inference_groups))
                for group, mask in group_masks.items():
                    if group not in inference_groups: continue
                    n, n_components = group_counts[group]
                    if n < (2 if group == "all" else self.cfg.min_group): continue
                    if n_components < 2: continue
                    values = dict(cross=ev["correct"][mask], within=own["correct"][mask],
                                  floor=ev["floor"][mask], null=null_mean_vec[mask])
                    stat = lambda a: shared_subspace_index(a["cross"].mean(), a["within"].mean(),
                                                          a["floor"].mean(), self.cfg.min_den)
                    boot = paired_cluster_bootstrap(values, frame.component_id.to_numpy()[mask], stat,
                                                     self.cfg.bootstrap, self.meta["seed"])
                    adjusted = lambda a: null_adjusted_index(a["cross"].mean(), a["within"].mean(),
                                                             a["null"].mean(), self.cfg.min_den)
                    adj_boot = paired_cluster_bootstrap(values, frame.component_id.to_numpy()[mask], adjusted,
                                                         self.cfg.bootstrap, self.meta["seed"])
                    nulls = {k: [stat(dict(values, cross=vector[mask])) for vector in v] for k, v in null.items()}
                    formal_family = primary_task and group in primary_groups
                    row = dict(base, group=group, n=n, n_components=n_components,
                               R_ep=values["cross"].mean(), R_pp=values["within"].mean(), R0=values["floor"].mean(),
                               SSI=boot["estimate"], ci_lo=boot["ci_lo"], ci_hi=boot["ci_hi"], n_valid=boot["n_valid"],
                               SSI_adjusted=adj_boot["estimate"], adjusted_ci_lo=adj_boot["ci_lo"],
                               adjusted_ci_hi=adj_boot["ci_hi"],
                               primary_test=formal_family, null_requested_draws=budget,
                               **permutation_info,
                               PERM_inferential=permutation_inferential and formal_family,
                               PERM_status=("insufficient_movable_component_mass" if not enough_movable else
                                            "diagnostic_noncomponent_unit" if not component_permutation else
                                            "primary_component_bundle_null" if formal_family else "diagnostic_nonprimary_family"),
                               uncertainty={"strict": "test_counterpart_component_bootstrap_conditional_on_fits",
                                            "counterpart_closure": "test_counterpart_component_bootstrap_conditional_on_fits",
                                            "input_disjoint": "test_input_component_bootstrap_conditional_on_fits"}.get(
                                   self.cfg.split_mode, "test_idiom_type_bootstrap_counterpart_overlap_remains"))
                    for kind, vals in nulls.items():
                        finite = np.asarray(vals)[np.isfinite(vals)]
                        row[kind + "_SSI_mean"] = float(finite.mean()) if len(finite) else np.nan
                        row[kind + "_R_mean"] = float(np.mean([v[mask].mean() for v in null[kind]])) if null[kind] else np.nan
                        row[kind + "_draws"] = len(finite)
                        # PCA is one deterministic baseline, not a randomization distribution.
                        diagnostic_p = monte_carlo_pvalue(boot["estimate"], vals) if kind != "PCA" else np.nan
                        row[kind + "_p_diagnostic"] = diagnostic_p
                        allowed = formal_family and (kind == "RAND" or kind == "PERM" and permutation_inferential)
                        row[kind + "_p"] = diagnostic_p if allowed else np.nan
                        row[kind + "_minimum_p"] = 1 / (len(finite) + 1) if len(finite) and kind != "PCA" else np.nan
                        row["SSI_adjusted_" + kind] = null_adjusted_index(
                            values["cross"].mean(), values["within"].mean(), row[kind + "_R_mean"], self.cfg.min_den)
                    rows.append(row)
                keep = [c for c in ["idiom_id", "source_id", "item_id", "component_id", "rare", "freq_raw", "idiom_len",
                                    "token_count", "anchored", "cultural", "religious", "historical", "rural",
                                    "cp_transparency", "transparent", "opaque"] if c in frame]
                pred = frame[keep].copy()
                for k, v in base.items(): pred[k] = v
                pred["cross_correct"], pred["within_correct"], pred["floor_correct"] = ev["correct"], own["correct"], ev["floor"]
                pred["cross_margin"], pred["within_margin"] = ev["margin"], own["margin"]
                if "freq_raw" in pred:
                    freq = pred.freq_raw.astype(str).str.casefold().str.strip()
                    pred["very_common"], pred["very_rare"] = freq.eq("very common"), freq.eq("very rare")
                predictions.append(pred)
        result = pd.DataFrame(rows)
        if not result.empty:
            planned_family_size = len(primary_pairs) * len(primary_concepts) * len(primary_groups)
            perm_draws = getattr(self.cfg, "perm_null_draws", None) or self.cfg.null_draws
            for null in ["RAND", "PERM"]:
                planned_minimum_p = 1 / ((self.cfg.null_draws if null == "RAND" else perm_draws) + 1)
                observed = int(result[null + "_p"].notna().sum())
                result[null + "_formal_family_size_this_refit"] = planned_family_size
                result[null + "_observed_formal_tests_this_refit"] = observed
                result[null + "_planned_minimum_p_primary"] = planned_minimum_p
                result[null + "_best_possible_holm_p_this_refit"] = (
                    min(1., planned_family_size * planned_minimum_p) if planned_family_size else np.nan)
                result[null + "_resolution_supports_0_05_this_refit"] = (
                    result[null + "_best_possible_holm_p_this_refit"].le(.05) & result[null + "_p"].notna())
        self._report("nulls", completed_draws, completed_draws, "null tasks completed")
        return result, pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    def contrasts(self):
        """Paired component-bootstrap contrasts; CIs, without bootstrap-sign p-values."""
        rows = []
        for (concept, method, rank, source, target), ev in self.evidence.items():
            if method != self.cfg.inference_method or source == target: continue
            own = self.evidence.get((concept, method, rank, target, target))
            if own is None: continue
            # Report only the same prespecified maximal rank used by inference().
            higher = any((concept, method, m, source, target) in self.evidence
                         and (concept, method, m, target, target) in self.evidence for m in self.cfg.ranks if m > rank)
            if higher: continue
            f = self.items.iloc[ev["test_idx"]]
            masks = groups_for(f)
            masks.update(matched_anchor_masks(self.items.iloc[self.idx(target, "space", concept)], f, self.meta["seed"]))
            for a, b in [("transparent", "opaque"), ("anchored", "nonanchored"), ("religious", "nonreligious"),
                         ("historical", "nonhistorical"), ("cultural", "noncultural"),
                         ("anchored_common", "unanchored_common"), ("anchored_rare", "unanchored_rare"),
                         ("anchored_matched", "unanchored_matched")]:
                if a not in masks or b not in masks: continue
                if min(masks[a].sum(), masks[b].sum()) < self.cfg.min_group: continue
                if min(f.loc[masks[a], "component_id"].nunique(), f.loc[masks[b], "component_id"].nunique()) < 2: continue
                values = dict(cross=ev["correct"], within=own["correct"], floor=ev["floor"], a=masks[a], b=masks[b])
                def stat(v):
                    def s(m):
                        if not m.any(): return np.nan
                        return shared_subspace_index(v["cross"][m].mean(), v["within"][m].mean(), v["floor"][m].mean(), self.cfg.min_den)
                    return s(v["a"].astype(bool)) - s(v["b"].astype(bool))
                boot = paired_cluster_bootstrap(values, f.component_id.to_numpy(), stat, self.cfg.bootstrap, self.meta["seed"])
                rows.append(dict(self.meta, concept=concept, method=method, M=rank, source_condition=source,
                                 condition=target, contrast=a+" - "+b, n_a=int(masks[a].sum()), n_b=int(masks[b].sum()),
                                 **boot, uncertainty="paired_component_bootstrap_CI_only"))
        return pd.DataFrame(rows)

    def asymmetry(self):
        rows, done = [], set()
        for (concept, method, rank, source, target), ev in self.evidence.items():
            pair = tuple(sorted([source, target]))
            if source == target or method != self.cfg.inference_method or (concept, rank, pair) in done: continue
            reverse = self.evidence.get((concept, method, rank, target, source))
            own = self.evidence.get((concept, method, rank, target, target))
            rown = self.evidence.get((concept, method, rank, source, source))
            if reverse is None or own is None or rown is None: continue
            done.add((concept, rank, pair))
            f, r = self.items.iloc[ev["test_idx"]], self.items.iloc[reverse["test_idx"]]
            common = f[["idiom_id", "component_id"]].reset_index(drop=True).reset_index(names="a").merge(
                r[["idiom_id"]].reset_index(drop=True).reset_index(names="b"), on="idiom_id")
            if len(common) < 3 or common.component_id.nunique() < 2: continue
            ai, bi = common.a.to_numpy(), common.b.to_numpy()
            values = dict(ep=ev["correct"][ai], pp=own["correct"][ai], floor=ev["floor"][ai],
                          pe=reverse["correct"][bi], ee=rown["correct"][bi], rfloor=reverse["floor"][bi])
            stat = lambda v: (shared_subspace_index(v["ep"].mean(), v["pp"].mean(), v["floor"].mean(), self.cfg.min_den)
                              - shared_subspace_index(v["pe"].mean(), v["ee"].mean(), v["rfloor"].mean(), self.cfg.min_den))
            boot = paired_cluster_bootstrap(values, common.component_id.to_numpy(), stat, self.cfg.bootstrap, self.meta["seed"])
            rows.append(dict(self.meta, concept=concept, method=method, M=rank, first_direction=source+"->"+target,
                             reverse_direction=target+"->"+source, n=len(common), cohort="matched_sense_ids_only", **boot))
        return pd.DataFrame(rows)

    def figurative_literal_geometry(self):
        """Item-wise cosine margins; requires actual reviewed literal/control contexts."""
        base = self.items[(self.items.condition == "bn-fig") & (self.items.split == "test")]
        rows = []
        for control in ["bn-lit", "bn-ctrl"]:
            target = self.items[(self.items.condition == control) & (self.items.split == "test")]
            para = self.items[(self.items.condition == "bn-para") & (self.items.split == "test")]
            if min(len(base), len(target), len(para)) == 0: continue
            paired = base.reset_index(names="x").merge(target[["idiom_id"]].reset_index(names="l"), on="idiom_id")
            paired = paired.merge(para[["idiom_id"]].reset_index(names="f"), on="idiom_id")
            for row in paired.to_dict("records"):
                x, f, l = self.X[row["x"]], self.X[row["f"]], self.X[row["l"]]
                cosine = lambda a, b: float(a @ b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
                sf, sl = cosine(x, f), cosine(x, l)
                covars = {k: row[k] for k in ["idiom_len", "token_count", "rare", "anchored", "cultural", "religious", "historical", "rural", "freq_raw"] if k in row}
                if "freq_raw" in covars:
                    freq = str(covars["freq_raw"]).strip().casefold()
                    covars.update(very_common=freq == "very common", very_rare=freq == "very rare")
                rows.append(dict(self.meta, **covars, idiom_id=row["idiom_id"], component_id=row["component_id"], control=control,
                                 cosine_figurative=sf, cosine_literal_control=sl, figurative_margin=sf-sl,
                                 interpretation="geometric_association_not_causal_literal_collapse"))
        return pd.DataFrame(rows)

    def language(self):
        """Pooled two-condition concept estimates, both directions and principal angles."""
        rows = []
        for a, b in [("bn-idiom", "en-idiom"), ("bn-fig", "en-trans")]:
            subset = self.items.condition.isin([a, b]).to_numpy()
            if not all(c in set(self.items.condition) for c in [a, b]): continue
            lang = self.items.lang.eq("en").to_numpy(int)
            for concept in self.cfg.concepts:
                if concept not in self.items: continue
                known = subset & self.items[concept].ge(0).to_numpy()
                ix = {s: np.flatnonzero(known & self.items.split.eq(s).to_numpy())
                      for s in ["space", "probe", "validation", "test"]}
                if min(map(len, ix.values())) < 2: continue
                x, y = self.X[ix["space"]], self.items[concept].to_numpy()[ix["space"]]
                readout_indices = [ix[s] for s in ["probe", "validation", "test"]]
                for method in self.cfg.methods:
                    gkey, fkey = ("language", a, b, concept, method, "G"), ("language", a, b, concept, method, "F")
                    if gkey not in self._estimators:
                        self._estimators[gkey] = fit_estimator(x, lang[ix["space"]], method, 1, self.meta["seed"],
                            ridge=getattr(self.cfg, "estimator_ridge", 0.), C=getattr(self.cfg, "estimator_C", 1.))
                    if fkey not in self._estimators:
                        self._estimators[fkey] = fit_estimator(x, y, method, max(self.cfg.ranks), self.meta["seed"],
                            ridge=getattr(self.cfg, "estimator_ridge", 0.), C=getattr(self.cfg, "estimator_C", 1.))
                    gsub = self._estimators[gkey].at_rank(1)
                    for rank in self.cfg.ranks:
                        fsub = self._estimators[fkey].at_rank(rank)
                        if fsub.effective_rank != rank or gsub.effective_rank != 1: continue
                        row = dict(self.meta, concept=concept, method=method, M=rank, pair=a + "/" + b,
                                   min_principal_angle_deg=float(np.rad2deg(subspace_angles(fsub.basis, gsub.basis)).min()))
                        for name, sub, labels in [("F_to_G", fsub, lang), ("G_to_F", gsub, self.items[concept].to_numpy())]:
                            for side, representation in [("inside", "project"), ("outside", "erase")]:
                                if representation == "erase" and not getattr(self.cfg, "compute_leakage", True):
                                    row[name + "_" + side + "_error"] = np.nan; continue
                                # G_to_F uses the same one-dimensional G subspace
                                # for every F rank and therefore shares one fit.
                                readout_rank = rank if name == "F_to_G" else 1
                                ev = self._custom_readout(("S2", a, b, concept, method, name, readout_rank),
                                                          sub, readout_indices, labels, representation)
                                row[name + "_" + side + "_error"] = 1 - ev["correct"].mean()
                        row["projection_convention"] = "exact_oblique" if method == "LEACE" else "orthogonal"
                        rows.append(row)
        return pd.DataFrame(rows)

    def identity_open_class(self):
        """Idiom identity as an open-class concept (the reference paper's unseen-speaker analogue).

        Subspaces for identity are estimated on N SPACE idioms, each seen in two
        different exact-span contexts (``bn-fig`` and ``bn-fig2``), and evaluated on
        *unseen* TEST idioms: each held-out idiom's second context must retrieve its
        first context by cosine 1-nearest neighbour among all held-out idioms (and
        vice versa). Retention/leakage are matching accuracies inside/outside the
        subspace; the ambient space is the best case and random subspaces the
        reference. Purity reports the F-probe error inside the identity subspace.
        LDA/LEACE use Ledoit-Wolf shrinkage (few rows per class).
        """
        rows = []
        pairs = getattr(self.cfg, "identity_pairs", [["bn-fig", "bn-fig2"], ["bn-context", "bn-context2"]])
        for first, second in pairs:
            if {first, second} <= set(self._condition):
                rows.extend(self._identity_rows(first, second))
        return pd.DataFrame(rows)

    def _identity_rows(self, first, second):
        """Identity study for one pair of conditions (two contexts of each idiom)."""
        def paired(split):
            a = self.items[(self.items.condition == first) & (self.items.split == split)]
            b = self.items[(self.items.condition == second) & (self.items.split == split)]
            return a.reset_index(names="i1")[["i1", "idiom_id"]].merge(
                b.reset_index(names="i2")[["i2", "idiom_id"]], on="idiom_id")
        estimation, evaluation = paired("space"), paired("test")
        if len(estimation) < 10 or len(evaluation) < 10:
            return []
        A, B = self.X[evaluation.i1.to_numpy()], self.X[evaluation.i2.to_numpy()]

        def matching(P, Q):
            P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
            Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
            sims = Q @ P.T
            ranks = []
            for s in (sims, sims.T):
                true = np.diag(s)
                ranks.append(1 + (s > true[:, None] + 1e-12).sum(1))
            ranks = np.concatenate(ranks)
            return float(np.mean(ranks == 1)), float(np.mean(1 / ranks))

        center = self.X[np.r_[estimation.i1.to_numpy(), estimation.i2.to_numpy()]].mean(0)
        ambient_top1, ambient_mrr = matching(A - center, B - center)
        methods = [m for m in getattr(self.cfg, "identity_methods", ["LDA", "LEACE", "CPCA", "COV", "PCA", "RAND"])
                   if m in {"LDA", "LEACE", "CPCA", "COV", "PCA", "RAND", "MLR"}]
        ranks = sorted(getattr(self.cfg, "identity_ranks", [8, 16, 32, 64, 128, 256]))
        sizes = sorted({min(int(n), len(estimation)) for n in getattr(self.cfg, "identity_N", [50, 100, 200, 400, 800, 1600])}
                       | {len(estimation)})
        reps = int(getattr(self.cfg, "identity_reps", 3))
        f_labels = pd.to_numeric(self.items["F"], errors="coerce").to_numpy() if "F" in self.items else None
        f_indices = [self.idx(first, s, "F") for s in ["probe", "validation", "test"]] if f_labels is not None else None
        rows = []
        total = sum(len(methods) for n in sizes for _ in range(reps if n < len(estimation) else 1))
        done = 0
        for n in sizes:
            for rep in range(reps if n < len(estimation) else 1):
                order = np.random.default_rng(self.meta["seed"] * 7919 + rep).permutation(len(estimation))[:n]
                chosen = estimation.iloc[order]
                X_est = self.X[np.r_[chosen.i1.to_numpy(), chosen.i2.to_numpy()]]
                y_est = np.r_[np.arange(n), np.arange(n)]
                for method in methods:
                    done += 1
                    self._report("S4_identity", done, total, "identity open-class sweep", N=n, method=method,
                                 condition=first)
                    ridge = "ledoit_wolf" if method in {"LDA", "LEACE"} else 0.
                    estimator = fit_estimator(X_est, y_est, method, max(ranks), seed=self.meta["seed"] + rep,
                                              **dict(self._estimator_options(), ridge=ridge))
                    for rank in ranks:
                        if rank > estimator.effective_rank:
                            continue
                        sub = estimator.at_rank(rank)
                        top1, mrr = matching(sub.project(A), sub.project(B))
                        leak_top1, leak_mrr = matching(sub.erase(A), sub.erase(B))
                        row = dict(self.meta, concept="I", condition=first, paired_condition=second,
                                   representation="context_only_idiom_masked" if "context" in first else "idiom_span",
                                   method=method, M=rank, N=n, rep=rep,
                                   n_eval_idioms=len(evaluation), R=top1, MRR=mrr, L=leak_top1, L_MRR=leak_mrr,
                                   ambient_R=ambient_top1, ambient_MRR=ambient_mrr, chance=1 / len(evaluation),
                                   spectral_gap=sub.spectral_gap, estimator_ridge=str(ridge),
                                   task="unseen_idiom_identities_two_contexts")
                        if f_labels is not None and n == len(estimation) and min(map(len, f_indices)) > 0:
                            ev = self._custom_readout(("S4_identity_purity", first, method, rank, n), sub, f_indices, f_labels)
                            row.update(purity_concept="F", purity_err=1 - ev["correct"].mean(),
                                       purity_floor_err=1 - ev["floor"].mean())
                        rows.append(row)
        return rows

    def subspace_overlap(self):
        """Alignment-free geometric sharing: principal angles between subspaces
        estimated separately on each condition (same estimator and rank).

        ``mean_cos2`` = tr(P_a P_b)/M is 1 for identical and M/D in expectation
        for independent random M-dimensional subspaces. For LEACE the compared
        spaces are the images of the oblique operators.
        """
        rows = []
        dims = self.X.shape[1]
        conditions = set(self._condition)
        for concept in self.cfg.concepts:
            if concept not in self.items: continue
            for source, target in self.cfg.pairs:
                if source == target or source not in conditions or target not in conditions: continue
                for method in self.cfg.methods:
                    if method == "RAND": continue
                    for rank in self.cfg.ranks:
                        a = self.sub(source, concept, method, rank)
                        b = self.sub(target, concept, method, rank)
                        if a is None or b is None: continue
                        cos = np.clip(np.linalg.svd(a.basis.T @ b.basis, compute_uv=False), 0, 1)
                        rows.append(dict(self.meta, concept=concept, method=method, M=rank,
                                         source_condition=source, condition=target,
                                         mean_cos2=float(np.mean(cos ** 2)), max_cos=float(cos.max()),
                                         min_cos=float(cos.min()), random_expected_mean_cos2=rank / dims,
                                         spectral_gap_source=a.spectral_gap, spectral_gap_target=b.spectral_gap,
                                         note="principal angles between separately estimated image subspaces"))
        return pd.DataFrame(rows)

    def language_subspace(self):
        """Multi-dimensional language subspace from paired translation differences.

        For matched SPACE records, the top-M right singular vectors of
        X_bn - X_en span the directions along which the two languages differ
        (the mean offset first). We report F/S probe errors inside and outside
        that subspace and principal angles to the concept subspace estimated on
        the pooled pair, replacing the one-dimensional binary language test.
        """
        rows = []
        for a, b in [("bn-idiom", "en-idiom"), ("bn-fig", "en-trans"), ("bn-para", "en-para"), ("bn-sentence", "en-trans")]:
            if not all(c in set(self._condition) for c in [a, b]): continue
            space_a = self.items[(self.items.condition == a) & (self.items.split == "space")]
            space_b = self.items[(self.items.condition == b) & (self.items.split == "space")]
            paired = space_a.reset_index(names="ia")[["ia", "idiom_id"]].merge(
                space_b.reset_index(names="ib")[["ib", "idiom_id"]], on="idiom_id")
            if len(paired) < 10: continue
            differences = self.X[paired.ia.to_numpy()] - self.X[paired.ib.to_numpy()]
            _, singular, vt = np.linalg.svd(differences, full_matrices=False)
            subset = self.items.condition.isin([a, b]).to_numpy()
            for concept in self.cfg.concepts:
                if concept not in self.items: continue
                known = subset & pd.to_numeric(self.items[concept], errors="coerce").ge(0).to_numpy()
                ix = {s: np.flatnonzero(known & self.items.split.eq(s).to_numpy()) for s in ["space", "probe", "validation", "test"]}
                if min(map(len, ix.values())) < 2: continue
                labels = self.items[concept].to_numpy()
                readout = [ix[s] for s in ["probe", "validation", "test"]]
                for rank in self.cfg.ranks:
                    if rank > len(singular): continue
                    basis = vt[:rank].T
                    lang = Subspace(np.zeros(self.X.shape[1]), basis, rank, "LANG_DIFF", rank)
                    row = dict(self.meta, concept=concept, M=rank, pair=a + "/" + b, n_pairs=len(paired),
                               language_variance_share=float((singular[:rank] ** 2).sum() / (singular ** 2).sum()))
                    for side, representation in [("inside", "project"), ("outside", "erase")]:
                        if representation == "erase" and not getattr(self.cfg, "compute_leakage", True):
                            row[f"concept_error_{side}_language"] = np.nan; continue
                        ev = self._custom_readout(("S2_langdiff", a, b, concept, rank), lang, readout, labels, representation)
                        row[f"concept_error_{side}_language"] = 1 - ev["correct"].mean() if ev else np.nan
                        row["floor_error"] = 1 - ev["floor"].mean() if ev else np.nan
                    for method in self.cfg.methods:
                        if method == "RAND": continue
                        key = ("language", a, b, concept, method, "F")
                        if key not in self._estimators:
                            self._estimators[key] = fit_estimator(self.X[ix["space"]], labels[ix["space"]], method,
                                                                  max(self.cfg.ranks), self.meta["seed"], **self._estimator_options())
                        concept_sub = self._estimators[key].at_rank(rank)
                        if concept_sub.effective_rank != rank: continue
                        cos = np.clip(np.linalg.svd(concept_sub.basis.T @ basis, compute_uv=False), 0, 1)
                        row[f"{method}_mean_cos2_with_language"] = float(np.mean(cos ** 2))
                    row["random_expected_mean_cos2"] = rank / self.X.shape[1]
                    rows.append(row)
        return pd.DataFrame(rows)

    def fls(self):
        rows = []
        if "L" not in self.items: return pd.DataFrame()
        for condition in self.items.condition.unique():
            # Same taxonomy, English literal labels only; exclude equal F/L assignments.
            valid = self.items.F.ge(0) & self.items.L.ge(0) & self.items.F.ne(self.items.L)
            if not (valid & self.items.condition.eq(condition)).any(): continue
            for method in self.cfg.methods:
                for rank in self.cfg.ranks:
                    for first, second in [("F", "L"), ("L", "F")]:
                        ix = {s: np.flatnonzero(valid & self.items.condition.eq(condition) & self.items.split.eq(s))
                              for s in ["space", "probe", "validation", "test"]}
                        if min(map(len, ix.values())) < 2: continue
                        sub = self.sub(condition, first, method, rank, subset=ix["space"])
                        if sub is None: continue
                        labels = self.items[second].to_numpy()
                        row = dict(self.meta, condition=condition, method=method, M=rank, subspace=first, other=second,
                                   n=len(ix["test"]), meaning="cross_concept_probe_error_not_per_item_geometry")
                        for name, representation in [("purity", "project"), ("interference", "erase")]:
                            if representation == "erase" and not getattr(self.cfg, "compute_leakage", True):
                                row[name + "_err"] = np.nan; continue
                            ev = self._custom_readout(("S4_FLS", condition, first, second, method, rank), sub,
                                                      [ix[s] for s in ["probe", "validation", "test"]], labels, representation)
                            row[name + "_err"] = 1 - ev["correct"].mean()
                        row["projection_convention"] = "exact_oblique" if method == "LEACE" else "orthogonal"
                        rows.append(row)
        return pd.DataFrame(rows)

    def open_class(self):
        rows, tasks = [], []
        leakage = getattr(self.cfg, "compute_leakage", True)
        methods = [m for m in getattr(self.cfg, "open_class_methods", ["MLR", "LEACE", "RAND", "PCA"])
                   if m in self.cfg.methods]
        for concept in [c for c in ["F", "S"] if c in self.cfg.concepts]:
            for condition in [c for c in getattr(self.cfg, "open_class_conditions", ["bn-fig", "bn-idiom", "en-idiom"])
                              if c in set(self._condition)]:
                ix = self.idx(condition, "space", concept)
                if len(ix) < 4: continue
                units = self.items.iloc[ix].groupby("idiom_group" if "idiom_group" in self.items else "idiom_id", sort=True).indices
                keys = np.array(list(units))
                for rep in range(self.cfg.open_class_reps):
                    rng = np.random.default_rng(self.meta["seed"] * 1009 + rep)
                    order = rng.permutation(keys)
                    for n in sorted(set(min(n, len(keys)) for n in self.cfg.open_class_N) | {len(keys)}):
                        small = ix[np.concatenate([units[k] for k in order[:n]])]
                        for method in methods:
                            for rank in self.cfg.ranks:
                                tasks.append((concept, condition, rep, n, small, method, rank))
        specs = []
        for concept, condition, rep, n, small, method, rank in tasks:
            specs.append((condition, concept, condition, concept, method, rank, "project", False, 0, small, True))
            if leakage:
                specs.append((condition, concept, condition, concept, method, rank, "erase", False, 0, small, True))
            other = self._other_concept(concept)
            if other:   # purity: the other concept read inside this concept's subspace
                specs.append((condition, other, condition, concept, method, rank, "project", False, 0, small, True))
        # Small estimation sets have fewer rows than dimensions; LEACE/LDA whitening then
        # needs shrinkage (Ledoit-Wolf by default). Subset estimators are keyed by their rows.
        ridge = getattr(self.cfg, "open_class_ridge", "ledoit_wolf")
        self._ridge_override = ridge
        try:
            return self._open_class_rows(tasks, specs, leakage, rows, ridge)
        finally:
            self._ridge_override = None

    def _other_concept(self, concept):
        other = {"F": "S", "S": "F"}.get(concept)
        return other if other in self.items and other in self.cfg.concepts else None

    def _open_class_rows(self, tasks, specs, leakage, rows, ridge):
        self._prefetch(specs, label="open-class readouts")
        for done, (concept, condition, rep, n, small, method, rank) in enumerate(tasks, 1):
            ev = self._evaluate_sub(condition, concept, condition, concept, method, rank, subset=small)
            self._report("S4_open_class", done, len(tasks), "open-class readout", concept=concept,
                         condition=condition, N=n, method=method, M=rank)
            if ev is None: continue
            leak = (self._evaluate_sub(condition, concept, condition, concept, method, rank,
                                      representation="erase", subset=small) if leakage else None)
            other = self._other_concept(concept)
            pure = self._evaluate_sub(condition, other, condition, concept, method, rank, subset=small) if other else None
            rows.append(dict(self.meta, concept=concept, condition=condition, method=method, M=rank,
                             N=n, n_records=len(small), rep=rep, R=ev["correct"].mean(),
                             L=leak["correct"].mean() if leak is not None else np.nan, R0=ev["floor"].mean(),
                             purity_concept=other, purity_err=1 - pure["correct"].mean() if pure else np.nan,
                             purity_floor_err=1 - pure["floor"].mean() if pure else np.nan,
                             estimator_ridge=str(ridge) if method in {"LEACE", "LDA"} else "none",
                             task="unseen_idiom_types_fixed_semantic_inventory"))
        return pd.DataFrame(rows)

    def baselines(self):
        rows = []
        for source, target in self.cfg.pairs:
            a = self.items[(self.items.condition == source) & (self.items.split == "test")]
            b = self.items[(self.items.condition == target) & (self.items.split == "test")]
            if a.empty or b.empty or source == target: continue
            # source_id distinguishes senses of a repeated idiom.
            key = "idiom_id"
            paired = a.reset_index(names="a_idx").merge(b.reset_index(names="b_idx"), on=key, suffixes=("_a", "_b"))
            if len(paired) < 3: continue
            if paired[key].duplicated().any():
                self.skips.append(dict(source=source, target=target, reason="retrieval_pair_not_one_to_one")); continue
            A, B = self.X[paired.a_idx], self.X[paired.b_idx]
            for centering in ["raw", "per_condition_space"]:
                if centering == "per_condition_space":
                    ai = np.flatnonzero(self.items.condition.eq(source) & self.items.split.eq("space"))
                    bi = np.flatnonzero(self.items.condition.eq(target) & self.items.split.eq("space"))
                    if not len(ai) or not len(bi): continue
                    A_, B_ = A - self.X[ai].mean(0), B - self.X[bi].mean(0)
                else: A_, B_ = A, B
                aa, bb = A_ - A_.mean(0), B_ - B_.mean(0)
                den = np.linalg.norm(aa.T @ aa) * np.linalg.norm(bb.T @ bb)
                cka = np.linalg.norm(aa.T @ bb) ** 2 / den if den > 0 else np.nan
                an = A_ / np.maximum(np.linalg.norm(A_, axis=1, keepdims=True), 1e-12)
                bn = B_ / np.maximum(np.linalg.norm(B_, axis=1, keepdims=True), 1e-12)
                # Chunk queries to avoid the n^2 similarity matrix in host memory.
                ranks = []
                for start in range(0, len(an), 256):
                    sim = an[start:start+256] @ bn.T
                    right = sim[np.arange(len(sim)), np.arange(start, start + len(sim))]
                    # Midrank ties; tied vectors must not produce perfect retrieval.
                    ranks.extend(1 + (sim > right[:, None] + 1e-12).sum(1) + .5 * ((np.abs(sim - right[:, None]) <= 1e-12).sum(1) - 1))
                ranks = np.asarray(ranks)
                rows.append(dict(self.meta, source_condition=source, condition=target, n=len(ranks),
                                 centering=centering, CKA=cka, retr_top1=np.mean(ranks == 1),
                                 retr_MRR=np.mean(1 / ranks), chance_top1=1 / len(ranks),
                                 space=self.meta.get("ambient_space", "original_hidden" if getattr(self.cfg, "pca_dim", None) is None
                                                     else f"PCA_{self.cfg.pca_dim}")))
        return pd.DataFrame(rows)

    def robustness(self):
        rows = []
        for condition in ["bn-fig", "bn-idiom", "bn-context", "en-idiom"]:
            for concept in [c for c in ["F", "S"] if c in self.cfg.concepts]:
                ix = self.idx(condition, "probe", concept)
                if len(ix) < 10: continue
                # MDL coding order is fixed by counterpart component; never split a unit across blocks.
                components = self.items.iloc[ix].component_id.to_numpy()
                rng = np.random.default_rng(self.meta["seed"])
                units = rng.permutation(np.unique(components))
                blocks = np.array_split(units, min(5, len(units)))
                labels = self.items[concept].to_numpy()
                universe = np.unique(labels[self.idx(condition, "space", concept)])
                nclass = len(universe)
                if nclass < 2: continue
                train, bits = [], 0.
                for block in blocks:
                    test = ix[np.isin(components, block)]
                    if not len(test): continue
                    if not train:
                        bits += len(test) * np.log2(nclass)
                    else:
                        fitted = fit_probe(self.X[train], labels[train], seed=self.meta["seed"])
                        proba = fitted.predict_proba(self.X[test]); lookup = {c: j for j, c in enumerate(fitted.classes_)}
                        q = np.array([proba[j, lookup[y]] if y in lookup else 1e-12 for j, y in enumerate(labels[test])])
                        bits -= np.log2(np.maximum(q, 1e-12)).sum()
                    train.extend(test.tolist())
                rows.append(dict(self.meta, condition=condition, concept=concept, analysis="ambient_online_MDL",
                                 bits=bits, uniform_bits=len(ix)*np.log2(nclass), n=len(ix),
                                 compression=len(ix)*np.log2(nclass)/bits if bits else np.nan))
                for kind in ["linear", "mlp"]:
                    ev = self.evaluate(self.X, condition, concept, kind)
                    if ev is not None:
                        rows.append(dict(self.meta, condition=condition, concept=concept, analysis="ambient_probe",
                                         probe=kind, R=ev["correct"].mean(), R0=ev["floor"].mean(), n=len(ev["correct"])))
        return pd.DataFrame(rows)


def moderator_analysis(predictions, out_dir):
    """Associational transfer logit; failed diagnostics never yield coefficients.

    Perfect separation, nonidentifiable designs and invalid or unconverged
    inference are recorded explicitly rather than reported as associations.
    """
    import statsmodels.api as sm
    from statsmodels.tools import sm_exceptions
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if predictions.empty: return pd.DataFrame()
    rows = []
    strata = [c for c in ["model", "seed", "label_seed", "layer", "concept", "method", "M", "source_condition", "condition"] if c in predictions]
    for key, frame in predictions.groupby(strata, dropna=False):
        base = dict(zip(strata, key if isinstance(key, tuple) else (key,)))
        candidates = [c for c in ["cp_transparency", "anchored", "rare", "very_common", "very_rare", "idiom_len", "token_count", "cultural", "religious", "historical", "rural", "within_correct"] if c in frame]
        varying = [c for c in candidates if frame[c].nunique() > 1]
        x = frame[varying].apply(pd.to_numeric, errors="coerce").astype(float)
        scales = x.std().replace(0, 1)
        x = sm.add_constant((x - x.mean()) / scales, has_constant="add")
        diagnostic = dict(base, n=len(frame), n_components=frame.component_id.nunique(),
                          n_parameters=x.shape[1], interpretation="association_only")
        if len(frame) <= 5 * x.shape[1] or frame.component_id.nunique() < 5 or frame.cross_correct.nunique() < 2:
            rows.append(dict(diagnostic, status="insufficient_outcomes_or_components",
                             reason="Too few observations/components or no outcome variation.")); continue
        if not np.isfinite(x.to_numpy()).all() or not frame.cross_correct.isin([0, 1, False, True]).all():
            rows.append(dict(diagnostic, status="invalid_design", reason="Nonfinite predictors or nonbinary/missing outcomes.")); continue
        if frame.component_id.isna().any() or frame.component_id.astype(str).str.strip().eq("").any():
            rows.append(dict(diagnostic, status="invalid_design", reason="Missing component identifiers.")); continue
        design_rank = int(np.linalg.matrix_rank(x.to_numpy()))
        diagnostic["design_rank"] = design_rank
        if design_rank < x.shape[1]:
            rows.append(dict(diagnostic, status="rank_deficient_design",
                             reason=f"Design rank {design_rank} is below {x.shape[1]} parameters; coefficients are not identifiable.")); continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = sm.GLM(frame.cross_correct.astype(int), x, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": frame.component_id.to_numpy()})
                warning_messages = [str(w.message) for w in caught]
                diagnostic["warnings"] = " | ".join(warning_messages)
                separation = any(isinstance(w.message, getattr(sm_exceptions, "PerfectSeparationWarning", ()))
                                 or "perfect separation" in str(w.message).lower() for w in caught)
                convergence = any(isinstance(w.message, sm_exceptions.ConvergenceWarning) for w in caught)
                if separation or convergence:
                    rows.append(dict(diagnostic, status="fit_failed", reason="Perfect separation warning." if separation
                                     else "Convergence warning; association estimates unavailable.")); continue
                if not getattr(fit, "converged", False):
                    rows.append(dict(diagnostic, status="not_converged", reason="GLM optimizer did not converge.")); continue
                inferential_values = np.concatenate([np.asarray(fit.params), np.asarray(fit.bse),
                                                    np.asarray(fit.pvalues), np.asarray(fit.tvalues)])
                if not np.isfinite(inferential_values).all():
                    rows.append(dict(diagnostic, status="fit_failed", reason="Nonfinite coefficient, standard error, p value or test statistic.")); continue
                for name in fit.params.index:
                    rows.append(dict(diagnostic, status="ok", term=name, beta_standardized=fit.params[name],
                                     se=fit.bse[name], p=fit.pvalues[name]))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError,
                    sm_exceptions.PerfectSeparationError) as exc:
                rows.append(dict(diagnostic, status="fit_failed", reason=f"{type(exc).__name__}: {exc}",
                                 warnings=" | ".join(str(w.message) for w in caught)))
    result = pd.DataFrame(rows)
    if "p" in result: result["p_holm_all_terms"] = holm(result.p.to_numpy())
    result.to_csv(out / "moderators.csv", index=False)
    return result
