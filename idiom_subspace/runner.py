"""Command-line orchestration. Importing this package performs no experiment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from .config import Config, model_slug


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def convert(v):
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    from .progress import atomic_json
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    serial=json.loads(json.dumps(value,ensure_ascii=False,default=convert))
    atomic_json(path,clean(serial))


def find_data(path):
    if path != "auto":
        if not Path(path).is_file(): raise FileNotFoundError(path)
        return Path(path)
    hits = []
    for root in [Path("/kaggle/input"), Path(".local/data"), Path("../.local/data"), Path.home()/".cache/kagglehub/datasets/sakhadib/bangla-bagdhara"]:
        if root.exists(): hits.extend(p for p in root.rglob("*.json") if "bagdhara" in str(p).lower())
    if len(hits) != 1:
        raise ValueError(f"Found {len(hits)} corpus JSONs. Pass --data explicitly (or download-data first).")
    return hits[0]


def environment():
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "sentence-transformers", "statsmodels", "accelerate", "bitsandbytes"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (FileNotFoundError, subprocess.CalledProcessError): git, dirty = None, None
    source_root = Path(__file__).resolve().parents[1]
    files = sorted((source_root / "idiom_subspace").glob("*.py")) + [source_root/"run_research.py", source_root/"launch_gpu_jobs.py"]
    hashes = {str(f.relative_to(source_root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}
    return dict(python=sys.version, platform=platform.platform(), packages=packages, git_commit=git,
                dirty_worktree=dirty, source_files_sha256=hashes)


def prepare(cfg, out, seed, label_seed=None):
    from .data import load_corpus, split_corpus, build_items
    from .labels import build_labels
    out.mkdir(parents=True, exist_ok=True)
    data_path = find_data(cfg.data)
    raw, audit = load_corpus(data_path)
    audit["sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    split = split_corpus(raw, seed=seed, fractions=cfg.split_fractions, mode=cfg.split_mode)
    write_json(out / "data_audit.json", audit)
    write_json(out / "split_audit.json", split.attrs["split_report"])
    if label_seed is not None:
        split, report, artifacts = build_labels(split, seed=label_seed, k=cfg.classes,
                                                backend=cfg.label_backend, encoder=cfg.label_encoder, revision=cfg.label_revision)
        write_json(out / "label_audit.json", report)
        centers = artifacts.get("kmeans")
        if centers is not None: np.save(out / "taxonomy_centers.npy", centers.cluster_centers_)
    items = build_items(split, controls_path=cfg.controls)
    write_json(out / "item_audit.json", items.attrs["item_audit"])
    # Explicit external controls are included in the run when they are supplied.
    selected = set(cfg.conditions) | {"bn-lit", "bn-ctrl", "bn-rom", "natural-en", "natural-hi", "natural-ur", "natural-kok"}
    items = items[items.condition.isin(selected)].reset_index(drop=True)
    split.to_json(out / "corpus_prepared.json", orient="records", force_ascii=False, indent=1)
    items.to_json(out / "items.json", orient="records", force_ascii=False, indent=1)
    meta = [c for c in ["idiom_id", "source_id", "source_record_id", "idiom_group", "component_id", "split", "anchored",
                        "rare", "cultural", "religious", "historical", "rural", "F", "L", "S", "D", "Ft",
                        "cp_transparency", "transparent", "opaque"] if c in split]
    split[meta].to_csv(out / "corpus_index.csv", index=False)
    return split, items


def annotation_templates(df, cfg, out, n=200):
    from .labels import export_annotation_sample, annotation_sample
    from .behavior import write_score_template
    out.mkdir(parents=True, exist_ok=True)
    sample = annotation_sample(df, n=n, seed=cfg.seeds[0])
    export_annotation_sample(sample, out / "taxonomy_annotations.csv", n=n, seed=cfg.seeds[0])
    # Include references, but leave model output and human score blank.
    score_rows = []
    # Spend comprehension annotation on the held-out, exact-span population.
    behavior_population = df[df.split.eq("test")]
    exact = [bool(r.idiom and r.idiom in r.bn_ex) for r in behavior_population.itertuples()]
    behavior_population = behavior_population.loc[exact]
    behavior_sample = annotation_sample(behavior_population, n=n, seed=cfg.seeds[0])
    for row in behavior_sample.to_dict("records"):
        for spec in cfg.models:
            score_rows.append(dict(idiom_id=row["idiom_id"], model=spec["name"], source_id=row["source_id"], idiom=row["idiom"],
                                   prompt=f'Explain the figurative meaning of this Bangla expression: {row["idiom"]}',
                                   generation="", reference_meaning=row["fig_bn"]))
    write_score_template(pd.DataFrame(score_rows, columns=["idiom_id", "model", "source_id", "idiom", "prompt", "generation", "reference_meaning"]),
                         out / "behavior_scores.csv")
    codebook = []
    if "F" in df:
        for label, rows in df[df.split.eq("space") & df.F.ge(0)].groupby("F"):
            examples = rows.fig_en.sample(n=min(3, len(rows)), random_state=cfg.seeds[0]).tolist()
            codebook.append(dict(F=label, human_class_name="", human_definition="", uncertain_rule="",
                                 space_example_1=examples[0] if examples else "",
                                 space_example_2=examples[1] if len(examples)>1 else "",
                                 space_example_3=examples[2] if len(examples)>2 else ""))
        pd.DataFrame(codebook).to_csv(out / "taxonomy_codebook.csv", index=False, encoding="utf-8-sig")
    controls = []
    literal = []
    prompts = []
    for row in sample.to_dict("records"):
        literal.append(dict(idiom_id=row["idiom_id"], source_id=row["source_id"], idiom=row["idiom"],
                            literal_bn=row["lit_bn"], literal_en="", approved=False))
        for condition, lang in [("bn-lit", "bn"), ("bn-ctrl", "bn"), ("bn-rom", "bn-Latn"),
                                ("natural-en", "en"), ("natural-hi", "hi"), ("natural-ur", "ur"), ("natural-kok", "kok")]:
            controls.append(dict(idiom_id=row["idiom_id"], source_id=row["source_id"], idiom=row["idiom"],
                                 figurative_meaning=row["fig_bn"], condition=condition, lang=lang,
                                 text="", span="", approved=False, source="", reviewer="", notes=""))
        prompt = f'এই অভিব্যক্তিটির রূপক অর্থ ব্যাখ্যা করুন: {row["idiom"]}'
        start = prompt.find(row["idiom"])
        donor = row["sims"][0] if row["sims"] else ""
        donor_prompt = f'Explain the figurative meaning of this expression: {donor}' if donor else ""
        ds = donor_prompt.find(donor) if donor else -1
        prompts.append(dict(idiom_id=row["idiom_id"], source_id=row["source_id"], prompt=prompt,
                            span=[start, start+len(row["idiom"])], donor_prompt=donor_prompt,
                            donor_span=[ds, ds+len(donor)] if donor else None))
    pd.DataFrame(controls).to_csv(out / "controls.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(literal).to_csv(out / "literal_glosses.csv", index=False, encoding="utf-8-sig")
    write_json(out / "intervention_prompts.json", prompts)
    write_json(out / "annotation_manifest.json", dict(
        requested_records=n, taxonomy_control_records=len(sample), behavior_records=len(behavior_sample),
        behavior_population="test_records_with_exact_bn_span", sampling="round_robin_F_anchored_rare_strata",
        note="Rare strata are oversampled; not a population prevalence sample. Scores and generations are blank."))


def apply_literal_reviews(df, path):
    """English literal glosses for L: human-approved rows and, explicitly marked, machine rows.

    ``approved`` true/yes/1 marks a human-reviewed gloss; ``approved=machine``
    marks an unreviewed machine translation (``translate-literals``). Both are
    recorded per record in ``lit_en_source`` so results can be stratified.
    """
    from .data import _TRUE
    reviews = pd.read_csv(path, keep_default_na=False)
    if not {"idiom_id", "literal_en", "approved"} <= set(reviews):
        raise ValueError("Literal CSV requires idiom_id,literal_en,approved")
    flag = reviews.approved.astype(str).str.strip().str.casefold()
    approved = reviews[(flag.isin(_TRUE) | flag.eq("machine")) & reviews.literal_en.astype(str).str.strip().ne("")].copy()
    approved["origin"] = np.where(flag.loc[approved.index].isin(_TRUE), "human_review", "machine_translation")
    if approved.idiom_id.duplicated().any(): raise ValueError("Duplicate approved literal gloss IDs")
    if not approved.idiom_id.isin(df.idiom_id).all(): raise ValueError("Unknown idiom IDs in literal reviews")
    mapping = approved.set_index("idiom_id")
    df = df.copy()
    use = df.idiom_id.isin(mapping.index)
    df.loc[use, "lit_en"] = df.loc[use, "idiom_id"].map(mapping.literal_en)
    df.loc[use, "lit_en_source"] = df.loc[use, "idiom_id"].map(mapping.origin) + ":" + str(Path(path).resolve())
    return df


def save_table(path, frames):
    nonempty = [f for f in frames if not f.empty]
    result = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    if not result.empty: atomic_table(path, result)
    return result


PIPELINE_VERSION = "4.0"
SCIENTIFIC_REVISION = ("v4: reference oblique LEACE, equal-centroid CPCA, reference MLR (no intercept, unit-norm rows), "
                       "input-disjoint split with form-variant links, multi-draw RAND grid baseline")


def atomic_table(path, frame):
    import os
    from .progress import atomic_replace
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False)
    atomic_replace(temp, path)


def atomic_npz(path, **arrays):
    import os
    from .progress import atomic_replace
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream: np.savez(stream, **arrays)
    atomic_replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


CORE_FIELDS = ["concepts", "methods", "ranks", "pairs", "pca_dim", "probe_Cs", "inference_method",
               "estimator_ridge", "estimator_C", "compute_leakage", "split_mode", "split_fractions",
               "classes", "label_backend", "label_encoder", "label_revision", "mlr_variant",
               "probe_max_iter", "grid_rand_draws"]


_ANALYSIS_CODE = None


def analysis_code():
    """Source hashes of the analysis modules, fixed at first use in this process."""
    global _ANALYSIS_CODE
    if _ANALYSIS_CODE is None:
        source = Path(__file__).parent
        _ANALYSIS_CODE = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                          for name in ["geometry.py", "studies.py", "data.py", "labels.py", "runner.py", "fastprobe.py"]}
    return dict(_ANALYSIS_CODE)


def analysis_signature(cfg, frame, manifest):
    from .geometry import resolve_probe_backend
    columns = [c for c in ["item_id", "idiom_id", "idiom_group", "component_id", "split", "condition", "lang",
                           "F", "L", "S", "D", "Ft", "token_count", "anchored", "rare", "freq_raw",
                           "cultural", "religious", "historical", "rural", "idiom_len", "cp_transparency"] if c in frame]
    return digest(dict(version=PIPELINE_VERSION, code=analysis_code(), features=manifest.get("fingerprint"),
                       revision=manifest.get("resolved_revision"), config={k:getattr(cfg,k) for k in CORE_FIELDS},
                       probe_backend=resolve_probe_backend(cfg),
                       rows=json.loads(frame[columns].to_json(orient="records"))))


def inference_signature(cfg, core):
    names = ["null_draws", "diagnostic_null_draws", "perm_null_draws", "bootstrap", "min_den", "min_group",
             "inference_groups", "primary_test_groups", "primary_test_pairs", "primary_test_concepts",
             "permutation_unit", "min_permutation_movable_fraction", "open_class_N", "open_class_reps",
             "open_class_methods", "open_class_conditions", "open_class_ridge", "identity_N", "identity_ranks",
             "identity_reps", "identity_methods", "identity_pairs", "robustness"]
    return digest(dict(core=core, settings={k:getattr(cfg,k) for k in names}))


def planned_primary_tests(cfg, *, per_refit=False):
    concepts=set(cfg.primary_test_concepts)
    pairs=set(map(tuple,cfg.primary_test_pairs))
    groups=set(cfg.primary_test_groups)
    count=len(concepts)*len(pairs)*len(groups)
    return count if per_refit else count*len(cfg.models)*len(cfg.seeds)*len(cfg.label_seeds)


def _run_impl(cfg, only_model=None, literal_reviews=None):
    from types import SimpleNamespace
    from .extraction import extract_features
    from .geometry import fitted_preprocessor, holm, resolve_probe_backend
    from .studies import LayerStudy, moderator_analysis, lexical_baselines, layer_ssi_table
    from .data import build_items
    from .labels import build_labels
    from .progress import Progress, atomic_json
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    previous = read_json_if_present(out / "pipeline_manifest.json")
    if not previous and (out / "run_config.json").exists():
        raise ValueError("This output contains an earlier pipeline. Choose a fresh --out and reuse its cache with --cache-dir OLD/cache.")
    if previous and previous.get("version") != PIPELINE_VERSION:
        raise ValueError("Output pipeline version differs. Choose a fresh --out; existing feature caches can be reused.")
    inputs={"corpus":hashlib.sha256(find_data(cfg.data).read_bytes()).hexdigest()}
    for field,path in [("controls",cfg.controls),("literal_reviews",literal_reviews)]:
        inputs[field]=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    current_core=dict(code=analysis_code(),settings={k:getattr(cfg,k) for k in CORE_FIELDS},inputs=inputs)
    if previous and previous.get("analysis_core") and previous["analysis_core"]!=current_core:
        raise ValueError("Analysis inputs/code changed. Choose a fresh --out and reuse its feature cache with --cache-dir.")
    # Extraction-only folders may later gain a taxonomy/analysis configuration.
    core=previous.get("analysis_core") if previous else None
    if not cfg.extract_only: core=current_core
    atomic_json(out / "pipeline_manifest.json", dict(version=PIPELINE_VERSION,analysis_core=core,
        scientific_revision=SCIENTIFIC_REVISION))
    write_json(out / "run_config.json", cfg.as_dict())
    write_json(out / "environment.json", environment())
    models = [s for s in cfg.models if only_model is None or s["name"] == only_model]
    if not models: raise ValueError(f"Model {only_model} not in config")
    seeds = cfg.seeds[:1] if cfg.extract_only else cfg.seeds
    label_seeds = cfg.label_seeds[:1] if cfg.extract_only else cfg.label_seeds
    planned = len(seeds)*len(label_seeds)*len(models)
    family=planned_primary_tests(cfg)
    design=dict(primary_family_size=family,min_p=1/(cfg.null_draws+1),
        minimum_attainable_holm_p=min(1.,family/(cfg.null_draws+1)),alpha=.05,
        can_reject_at_alpha=family>0 and family/(cfg.null_draws+1)<=.05,
        interpretation="Separate RAND/PERM families; configured models/refits remain in the family even if missing or undefined. Nonprimary comparisons are descriptive.")
    write_json(out/"inference_design.json",dict(design,status="planned",observed_primary_pvalues=0))
    global_inference, global_predictions, coverage = [], [], []
    completed = 0
    with Progress(out / "progress.json", kind="orchestrator", plan_jobs=planned,
                  mode="extraction" if cfg.extract_only else "analysis") as overall:
        for seed in seeds:
            for ls in label_seeds:
                prepared = out / f"split_{seed}" / f"labels_{ls}"; prepared.mkdir(parents=True, exist_ok=True)
                overall.update(stage="prepare", message="Preparing corpus, spelling-disjoint split and SPACE taxonomy", seed=seed,label_seed=ls)
                df, items = prepare(cfg, prepared, seed, None if cfg.extract_only else ls)
                if literal_reviews and not cfg.extract_only:
                    df = apply_literal_reviews(df, literal_reviews)
                    df, report, _ = build_labels(df, seed=ls, k=cfg.classes, backend=cfg.label_backend,
                                                encoder=cfg.label_encoder, revision=cfg.label_revision)
                    report["literal_gloss_sources"] = df.lit_en_source.astype(str).str.split(":").str[0].value_counts().to_dict()
                    items = build_items(df, cfg.controls)
                    approved = items.get("control_approved", pd.Series(False,index=items.index)).fillna(False).astype(bool)
                    items = items[items.condition.isin(set(cfg.conditions)|set(items.loc[approved,"condition"]))].reset_index(drop=True)
                    write_json(prepared / "label_audit.json", report)
                    df.to_json(prepared / "corpus_prepared.json", orient="records", force_ascii=False)
                    items.to_json(prepared / "items.json", orient="records", force_ascii=False)
                    cols = [c for c in pd.read_csv(prepared / "corpus_index.csv", nrows=0).columns if c in df]
                    atomic_table(prepared / "corpus_index.csv", df[cols])
                if not cfg.extract_only:
                    lexical_path = prepared / "bow_baselines.csv"
                    if not (cfg.resume and lexical_path.exists()):
                        overall.update(stage="lexical_baselines",message="Word/character baselines (CPU)")
                        lexical = lexical_baselines(items,cfg)
                        if not lexical.empty: atomic_table(lexical_path,lexical)
                for spec in models:
                    name = spec["name"]
                    dest = prepared / model_slug(name); dest.mkdir(parents=True, exist_ok=True)
                    status = dest / "status.json"
                    atomic_json(status,dict(status="running",model=name))
                    with Progress(dest / "progress.json",model=name,seed=seed,label_seed=ls,kind="model_refit",
                                  cpu_threads=cfg.cpu_threads,mode="extraction" if cfg.extract_only else "analysis",
                                  echo=cfg.progress) as reporter:
                        try:
                            reporter.update(stage="extract",completed=0,total=len(items),unit="items",message="Finding or extracting original hidden states")
                            features,pooling,manifest = extract_features(items,spec,cfg,out/"cache",progress=reporter)
                            write_json(dest/"model_revision.json",manifest)
                            atomic_table(dest/"pooling.csv",pooling)
                            if cfg.extract_only:
                                reporter.update(stage="extracted",completed=len(items),total=len(items),message="Embedding cache complete; analysis pending")
                                atomic_json(status,dict(status="completed",stage="extracted",analysis_pending=True,model=name))
                                completed+=1
                                overall.update(stage="jobs",completed=completed,total=planned,unit="model extractions",message="Original hidden states cached")
                                continue
                            valid=pooling.valid.to_numpy(bool)
                            frame=items.loc[valid].reset_index(drop=True)
                            frame["token_count"]=pooling.loc[valid,"token_count"].to_numpy()
                            for split in ["space","probe","validation","test"]:
                                if not frame.split.eq(split).any(): raise ValueError(f"No valid features in {split}")
                            core=analysis_signature(cfg,frame,manifest)
                            old_analysis=read_json_if_present(dest/"analysis_manifest.json")
                            if old_analysis and old_analysis.get("signature")!=core:
                                raise ValueError("Analysis inputs/code changed in this output. Use a fresh --out and reuse the existing cache.")
                            write_json(dest/"analysis_manifest.json",dict(version=PIPELINE_VERSION,signature=core,
                                ambient_space="original_hidden" if cfg.pca_dim is None else f"PCA_{cfg.pca_dim}",
                                feature_cache_fingerprint=manifest.get("fingerprint"),resolved_revision=manifest.get("resolved_revision"),
                                probe_backend=resolve_probe_backend(cfg),mlr_variant=cfg.mlr_variant,
                                grid_rand_draws=cfg.grid_rand_draws,split_mode=cfg.split_mode,
                                description="Primary LEACE uses the exact oblique readout; PCA sensitivity results do not establish full-space containment."))
                            checkpoints=dest/"checkpoints"/core
                            infer_key=inference_signature(cfg,core)
                            stage_records=read_json_if_present(dest/"stage_checkpoints.json") or {}
                            grids,candidates=[],[]
                            def make_study(layer):
                                li=manifest["layers"].index(layer)
                                raw=np.asarray(features[valid,li,:],dtype=float)
                                pre_path=dest/f"preprocess_layer_{layer}.npz"
                                reporter.update(stage="preprocess",layer=layer,message="SPACE-only centering and optional PCA")
                                if cfg.resume and pre_path.exists():
                                    with np.load(pre_path,allow_pickle=False) as saved:
                                        mean,scale,components=saved["mean"],float(saved["scale"]),saved["components"]
                                    z=(raw-mean)@components.T/scale if cfg.pca_dim is not None else (raw-mean)/scale
                                    pre=SimpleNamespace(mean_=mean,scale_=scale)
                                else:
                                    pre=fitted_preprocessor(raw[frame.split.eq("space")],cfg.pca_dim,seed=seed)
                                    components=pre.pca.components_ if pre.pca is not None else np.eye(raw.shape[1])
                                    atomic_npz(pre_path,mean=pre.mean_,scale=pre.scale_,components=components)
                                    z=pre.transform(raw)
                                meta=dict(model=name,seed=seed,label_seed=ls,layer=layer,split_mode=cfg.split_mode,
                                          ambient_space="original_hidden" if cfg.pca_dim is None else f"PCA_{cfg.pca_dim}")
                                study=LayerStudy(z,frame,cfg,meta,checkpoint_dir=checkpoints/f"layer_{layer}",progress=reporter)
                                return study,pre,components
                            reporter.start_milestone("layer",total=len(manifest["layers"]),label="layers analysed")
                            for layer in manifest["layers"]:
                                study,pre,components=make_study(layer)
                                grid=study.run_grid()
                                if grid.empty: raise ValueError("No estimable tasks; inspect split and label coverage")
                                grids.append(grid)
                                selection=grid[(grid.concept=="F")&(grid.method==cfg.inference_method)
                                               &(grid.source_condition=="bn-fig")&(grid.condition=="bn-fig")&(grid.group=="all")]
                                if not selection.empty and not(spec.get("kind")=="decoder" and layer==0):
                                    candidates.append((float(selection.validation_R.mean()),layer))
                                atomic_table(dest/f"grid_layer_{layer}.csv",grid)
                                write_json(dest/f"skipped_layer_{layer}.json",study.skips)
                                reporter.update(stage="layer_complete",completed=len(grids),total=len(manifest["layers"]),unit="layers",
                                                layer=layer,message=f"Layer {layer} checkpoint complete")
                                reporter.milestone("layer",len(grids),len(manifest["layers"]))
                                del study
                            if not candidates: raise ValueError("No F/bn-fig validation task for layer selection")
                            best_score,best=sorted(candidates,key=lambda pair:(-pair[0],pair[1]))[0]
                            write_json(dest/"layer_selection.json",dict(layer=best,validation_score=best_score,
                                criterion="mean across supported configured ranks, F within bn-fig",selected_using="validation_only",
                                candidates=candidates,decoder_embedding_layer_excluded=spec.get("kind")=="decoder"))
                            reporter.update(stage="selected_layer",layer=best,
                                            message=f"Restoring validation-selected layer {best} for inference and studies")
                            study,pre,components=make_study(best)
                            study.run_grid()  # restores completed task checkpoints, not a second set of fits
                            study.meta["layer_role"]="validation_selected"
                            grid_all=save_table(dest/"S1_S3_grid.csv",grids)
                            layer_indices=layer_ssi_table(grid_all,cfg.min_den)
                            if not layer_indices.empty:atomic_table(dest/"ssi_all_layers.csv",layer_indices)
                            if "S3" in cfg.studies:
                                if cfg.resume and stage_records.get("inference")==infer_key and (dest/"S3_inference.csv").exists():
                                    inference=pd.read_csv(dest/"S3_inference.csv")
                                    predictions=pd.read_csv(dest/"predictions.csv") if (dest/"predictions.csv").exists() else pd.DataFrame()
                                else:
                                    inference,predictions=study.inference()
                                    if not inference.empty:
                                        for null in ["RAND","PERM"]:
                                            inference[null+"_p_holm"]=holm(inference[null+"_p"].to_numpy(),family_size=planned_primary_tests(cfg,per_refit=True))
                                        atomic_table(dest/"S3_inference.csv",inference)
                                    if not predictions.empty:
                                        atomic_table(dest/"predictions.csv",predictions)
                                        moderator_analysis(predictions,dest)
                                    stage_records["inference"]=infer_key
                                    write_json(dest/"stage_checkpoints.json",stage_records)
                                if not inference.empty:global_inference.append(inference)
                                if not predictions.empty:global_predictions.append(predictions)
                            job_coverage=[]
                            for stage,filename,function in [("S2","S2_language.csv",study.language),
                                ("S2","S2_language_subspace.csv",study.language_subspace),
                                ("S3","S3_subspace_overlap.csv",study.subspace_overlap),
                                ("S3","S3_baselines.csv",study.baselines),("S3","S3_contrasts.csv",study.contrasts),
                                ("S3","S3_asymmetry.csv",study.asymmetry),("S4","S4_FLS.csv",study.fls),
                                ("S4","S4_geometry.csv",study.figurative_literal_geometry),("S4","S4_openclass.csv",study.open_class),
                                ("S4","S4_identity.csv",study.identity_open_class),
                                ("robustness","probe_robustness.csv",study.robustness)]:
                                if stage not in cfg.studies or(stage=="robustness" and not cfg.robustness):continue
                                record_key=digest(dict(stage=filename,settings=infer_key))
                                if cfg.resume and stage_records.get(filename,{}).get("signature")==record_key:
                                    count=stage_records[filename]["rows"]
                                else:
                                    reporter.update(stage=stage,completed=0,total=1,message=filename)
                                    table=function();count=len(table)
                                    if count:atomic_table(dest/filename,table)
                                    stage_records[filename]=dict(signature=record_key,rows=count)
                                    write_json(dest/"stage_checkpoints.json",stage_records)
                                job_coverage.append(dict(model=name,seed=seed,label_seed=ls,output=filename,
                                    status="completed" if count else "unavailable_data_or_rank",rows=count))
                            # A generic orthogonal activation hook must not silently substitute for oblique LEACE.
                            if spec.get("kind")=="decoder" and cfg.inference_method!="LEACE":
                                # Span edits at the final block have no later attention block
                                # through which to influence subsequent prompt positions.
                                final_hidden=manifest.get("model_layer_count",max(manifest["layers"]))
                                eligible=[(score,layer) for score,layer in candidates if 0 < layer < final_hidden]
                                intervention_best=sorted(eligible,key=lambda pair:(-pair[0],pair[1]))[0][1] if eligible else None
                                write_json(dest/"intervention_layer_selection.json",dict(layer=intervention_best,
                                    selected_using="validation_only",criterion="same F/bn-fig criterion, excluding embedding and final block",
                                    final_hidden_state_index=final_hidden,status="eligible" if eligible else "no_eligible_sampled_layer"))
                                if intervention_best is not None:
                                    study,pre,components=make_study(intervention_best)
                                else:
                                    study=None
                                for condition in ["bn-fig","en-idiom"]:
                                    if study is None:continue
                                    supported={concept:[m for m in cfg.ranks if study.sub(condition,concept,cfg.inference_method,m) is not None]
                                               if concept in frame else [] for concept in ["F","L"]}
                                    common=sorted(set(supported["F"])&set(supported["L"]))
                                    for concept in ["F","L"]:
                                        if not supported[concept]:continue
                                        rank=max(common) if common else max(supported[concept])
                                        sub=study.sub(condition,concept,cfg.inference_method,rank)
                                        if sub is None:continue
                                        ambient=components.T@sub.basis
                                        center=pre.mean_+(sub.center*pre.scale_)@components
                                        atomic_npz(dest/f"basis_{concept}_{condition}.npz",basis=ambient,center=center,
                                            model_id=name,layer_index=intervention_best-1,hidden_state_index=intervention_best,space="decoder_block_output",
                                            resolved_revision=manifest["resolved_revision"],quantization=json.dumps(spec.get("quantization")),
                                            precision=manifest.get("dtype",cfg.precision))
                            coverage.extend(job_coverage);write_json(dest/"coverage.json",job_coverage)
                            atomic_json(status,dict(status="completed",model=name,selected_layer=best))
                            reporter.update(stage="completed",completed=1,total=1,message="All requested analyses checkpointed")
                        except BaseException as exc:
                            atomic_json(status,dict(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                                                    model=name,error=type(exc).__name__,message=str(exc)))
                            raise
                    completed+=1
                    overall.update(stage="jobs",completed=completed,total=planned,unit="model refits",message="Model refit complete")
        inf=save_table(out/"inference_all.csv",global_inference)
        if not inf.empty:
            for null in ["RAND","PERM"]:inf[null+"_p_holm_global"]=holm(inf[null+"_p"].to_numpy(),family_size=family)
            observed=int(inf.get("RAND_p",pd.Series(dtype=float)).notna().sum())
            inf["family_size_global"]=family
            inf["minimum_p_resolution"]=1/(cfg.null_draws+1)
            atomic_table(out/"inference_all.csv",inf)
            keys=["model","concept","method","M","source_condition","condition","group"]
            summary=inf.groupby(keys,dropna=False).agg(SSI_mean=("SSI","mean"),SSI_sd=("SSI","std"),n_refits=("SSI","count")).reset_index()
            atomic_table(out/"refit_summary.csv",summary)
            write_json(out/"inference_design.json",dict(design,status="completed_requested_jobs",
                observed_primary_pvalues=observed,completed_model_refits=completed,
                configured_model_refits=len(cfg.models)*len(cfg.seeds)*len(cfg.label_seeds)))
        save_table(out/"predictions_all.csv",global_predictions);write_json(out/"coverage.json",coverage)


def read_json_if_present(path):
    try:return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:return None


def run(cfg,only_model=None,literal_reviews=None):
    # Small/medium probe matrices become dramatically slower with excessive BLAS workers.
    from threadpoolctl import threadpool_limits
    literal_reviews = literal_reviews or getattr(cfg, "literal_reviews", None)
    with threadpool_limits(limits=cfg.cpu_threads):
        return _run_impl(cfg,only_model,literal_reviews)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Idiom-subspace research pipeline v4")
    parser.add_argument("command", choices=["audit", "templates", "run", "behavior", "intervene", "intervention-effects",
                                            "validate-annotations", "validate-scorer", "download-data", "translate-literals"])
    parser.add_argument("--config"); parser.add_argument("--data"); parser.add_argument("--out")
    parser.add_argument("--model"); parser.add_argument("--scores"); parser.add_argument("--geometry")
    parser.add_argument("--controls"); parser.add_argument("--literal-reviews"); parser.add_argument("--annotations")
    parser.add_argument("--automatic"); parser.add_argument("--human")
    parser.add_argument("--basis"); parser.add_argument("--literal-basis"); parser.add_argument("--records")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--cache-dir",action="append",default=[],help="Existing feature-cache root; repeat for several old jobs")
    parser.add_argument("--cache-only","--analysis-only",dest="cache_only",action="store_true",help="Require completed embeddings; never load model weights")
    parser.add_argument("--extract-only",action="store_true",help="Cache embeddings and stop before probes")
    parser.add_argument("--no-resume",action="store_true",help="Recompute analysis tasks instead of reading task checkpoints")
    parser.add_argument("--no-progress",action="store_true",help="Disable terminal bars; dashboard telemetry is retained")
    parser.add_argument("--threads",type=int,help="CPU/BLAS thread limit for analysis (default 2)")
    parser.add_argument("--seeds",help="Comma-separated split seeds; overrides the config (replicate refits are opt-in)")
    parser.add_argument("--label-seeds",help="Comma-separated taxonomy seeds; overrides the config")
    parser.add_argument("--probe-backend",choices=["auto","torch","sklearn"],help="Probe engine (default from config: auto)")
    parser.add_argument("--split-mode",choices=["input_disjoint","counterpart_closure","strict","idiom_only"],
                        help="Split policy override")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    for key in ["data", "out", "controls"]:
        value = getattr(args, key)
        if value is not None: setattr(cfg, key, value)
    if args.cache_dir:cfg.feature_cache_dirs += args.cache_dir
    if args.cache_only:cfg.cache_only=True
    if args.extract_only:cfg.extract_only=True
    if args.no_resume:cfg.resume=False
    if args.no_progress:cfg.progress=False
    if args.threads is not None:cfg.cpu_threads=args.threads
    if args.seeds:cfg.seeds=[int(v) for v in args.seeds.split(",") if v.strip()]
    if args.label_seeds:cfg.label_seeds=[int(v) for v in args.label_seeds.split(",") if v.strip()]
    if args.probe_backend:cfg.probe_backend=args.probe_backend
    if args.split_mode:cfg.split_mode=args.split_mode
    cfg.validate()
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    if args.command == "download-data":
        import kagglehub
        print(kagglehub.dataset_download("sakhadib/bangla-bagdhara")); return
    if args.command == "run":
        run(cfg, args.model, args.literal_reviews); return
    if args.command == "translate-literals":
        from .data import load_corpus
        from .literal import translate_literals
        from .progress import Progress
        df, _ = load_corpus(find_data(cfg.data))
        with Progress(out / "progress.json", model=args.model or cfg.literal_mt_model, kind="translation",
                      echo=cfg.progress) as reporter:
            table = translate_literals(df, model=args.model or cfg.literal_mt_model, progress=reporter)
        path = out / "literal_glosses_mt.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{int(table.approved.eq('machine').sum())} machine-translated literal glosses written to {path}. "
              f"Review/edit them, then pass --literal-reviews {path} (or set literal_reviews in the config).")
        return
    if args.command in {"audit", "templates"}:
        # Audit is deliberately offline: TF-IDF does not download a label encoder.
        if args.command == "audit":
            df, items = prepare(cfg, out, cfg.seeds[0])
        else:
            df, items = prepare(cfg, out, cfg.seeds[0], cfg.label_seeds[0])
            annotation_templates(df, cfg, out / "annotations", args.n)
        print(json.dumps(df.attrs.get("split_report", {}), indent=2)); return
    if args.command == "behavior":
        from .behavior import geometry_behavior_analysis
        if not args.geometry or not args.scores: parser.error("behavior requires --geometry and --scores")
        geometry_behavior_analysis(pd.read_csv(args.geometry), pd.read_csv(args.scores), out,
                                   bootstrap_iterations=cfg.bootstrap, seed=cfg.seeds[0]); return
    if args.command == "intervention-effects":
        from .behavior import paired_intervention_effects
        if not args.scores: parser.error("intervention-effects requires --scores")
        metadata = pd.read_csv(args.geometry) if args.geometry else None
        paired_intervention_effects(pd.read_csv(args.scores), out, metadata=metadata,
                                    bootstrap_iterations=cfg.bootstrap, seed=cfg.seeds[0]); return
    if args.command == "validate-annotations":
        from .labels import validate_annotations
        if not args.annotations: parser.error("--annotations required")
        write_json(out / "annotation_agreement.json", validate_annotations(args.annotations)); return
    if args.command == "validate-scorer":
        from .behavior import validate_rubric_concordance
        if not args.automatic or not args.human: parser.error("--automatic and --human required")
        write_json(out / "scorer_validation.json", validate_rubric_concordance(args.automatic, args.human)); return
    if args.command == "intervene":
        if not args.basis or not args.records or not args.model: parser.error("intervene requires --basis, --records, --model")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from .interventions import InterventionBasis, run_interventions
        def load_basis(path):
            with np.load(path, allow_pickle=False) as v:
                return InterventionBasis(v["basis"], v["center"], str(v["model_id"]), int(v["layer_index"]), str(v["space"]))
        basis = load_basis(args.basis)
        if basis.model_id != args.model: raise ValueError("Basis model must exactly match generation model")
        spec = next((s for s in cfg.models if s["name"] == args.model), {})
        with np.load(args.basis, allow_pickle=False) as values:
            basis_values={key:str(values[key]) for key in ["resolved_revision","quantization","precision"] if key in values}
        revision = basis_values.get("resolved_revision", spec.get("revision", "main"))
        if "quantization" in basis_values and str(basis_values["quantization"]) != json.dumps(spec.get("quantization")):
            raise ValueError("Basis quantization must match the generation model configuration")
        if args.literal_basis:
            with np.load(args.literal_basis, allow_pickle=False) as lv:
                for key in ["resolved_revision","quantization","precision"]:
                    if key in lv and str(lv[key]) != basis_values.get(key):
                        raise ValueError(f"Figurative and literal bases must use the same {key}")
        precision=basis_values.get("precision",cfg.precision).removeprefix("torch.")
        if precision=="auto":precision="float16" if torch.cuda.is_available() else "float32"
        if precision not in {"float16","float32","float64","bfloat16"}:
            raise ValueError(f"Unsupported recorded intervention precision: {precision}")
        kwargs = dict(revision=revision, trust_remote_code=False,
                      torch_dtype=getattr(torch,precision))
        if spec.get("quantization"):
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        if not spec.get("quantization"): model.to("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(args.model, revision=revision, use_fast=True)
        records = json.loads(Path(args.records).read_text(encoding="utf-8"))
        result = run_interventions(model, tok, records, basis.layer_index, basis,
                                   load_basis(args.literal_basis) if args.literal_basis else None, seed=cfg.seeds[0])
        pd.DataFrame(result).to_csv(out / "intervention_generations.csv", index=False)


if __name__ == "__main__": main()
