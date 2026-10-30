"""Run the follow-up experiments X1--X19 against a finished discovery campaign.

    python run_experiments.py list
    python run_experiments.py preflight --campaign CAMPAIGN --out OUT
    python run_experiments.py X5 --campaign CAMPAIGN --out OUT
    python run_experiments.py verify --check X5
    python run_experiments.py verify --group X1-X8

CAMPAIGN is the discovery campaign written by run_research.py (opened read-only); split 1 is read from its
prospective follow-up directory, CAMPAIGN/paper_followup_seed1_20261021 (run_replication.py), unless --followup
says otherwise. OUT receives checkpoints and item-level evidence. Small aggregates go to results/experiments/.
Every experiment resumes from its checkpoints when re-run with the same OUT.

The experiments form three groups, each with a reproduction gate that is checked before its other results are
trusted: X5 must reproduce the planned discovery tests (X1-X8), X10 must reproduce X7 (X9-X14), and X15 must
reproduce the planned test and the estimator table (X15-X19). X16 refuses to run if its frozen plan,
configs/plans/PLAN_X16.json, has changed. scripts/run_experiments.sh runs everything in order with the gates;
docs/RUNNING.md is the runbook.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# code: (module, results folder, default splits, title, group)
EXPERIMENTS = {
    "X1": ("x1_zero_shot", "X1_zero_shot", [0], "strict zero-shot readout (English-trained probe on Bangla)", "X1-X8"),
    "X2": ("x2_concepts", "X2_concepts", [0], "planned test for FBN (Bangla-gloss taxonomy) and DOM (usage domain)",
           "X1-X8"),
    "X3": ("x3_leace", "X3_leace_ablation", [0, 1], "LEACE whitening ridge / covariance sample size / probe C",
           "X1-X8"),
    "X4": ("x4_pooling", "X4_pooling_matched", [0], "pooling-matched 2x2 (new forward passes)", "X1-X8"),
    "X5": ("x5_balanced", "X5_balanced_metrics", [0, 1], "balanced accuracy and macro-F1 with every null draw",
           "X1-X8"),
    "X6": ("x6_nonlinear", "X6_nonlinear_mdl", [0], "MLP readouts and corrected online-code MDL", "X1-X8"),
    "X7": ("x7_alignment", "X7_alignment", [0], "CSLS retrieval and Procrustes alignment", "X1-X8"),
    "X8": ("x8_layerwise", "X8_layerwise", [0], "COV/MLR/LEACE at every cached layer with nulls", "X1-X8"),
    "X9": ("x9_erasers", "X9_erasers_nonlinear", [0],
           "INLP/R-LACE and rotated erasers; linear and MLP readouts after erasure", "X9-X14"),
    "X10": ("x10_heldout_alignment", "X10_heldout_alignment", [0],
            "Procrustes maps fitted on held-out PROBE/VALIDATION pairs (SPACE refit = X7 gate)", "X9-X14"),
    "X11": ("x11_taxonomy", "X11_taxonomy_robustness", [0],
            "planned test, input ladder and gloss contrast for re-seeded F and a tag taxonomy (TAG)", "X9-X14"),
    "X12": ("x12_penalty_path", "X12_penalty_path", [0, 1],
            "zero-shot and supervised readouts along MLR's penalty path and LEACE's ridge path", "X9-X14"),
    "X13": ("x13_idiom_isolated", "X13_idiom_isolated", [0],
            "the idiom without context (bn-idiom) as a second planned family, with absolute effects", "X9-X14"),
    "X14": ("x14_decoder", "X14_decoder_powered", [0],
            "optional: the decoder (Qwen3-8B) with the encoders' planned design and full nulls", "X9-X14"),
    "X15": ("x15_matched_nulls", "X15_matched_nulls", [0],
            "covariance-matched random null and label-free PCA reference for MLR and COV (bn-fig, bn-idiom)",
            "X15-X19"),
    "X16": ("x16_idiom_prospective", "X16_idiom_prospective", [1],
            "frozen-plan prospective test of the idiom without context (bn-idiom) on the follow-up split", "X15-X19"),
    "X17": ("x17_human_items", "X17_human_items", [0],
            "item-level link between the volunteers' judgments and cross-lingual readouts", "X15-X19"),
    "X18": ("x18_refit_bootstrap", "X18_refit_bootstrap", [0], "bootstrap intervals that refit the subspaces and probes",
            "X15-X19"),
    "X19": ("x19_counterpart_type", "X19_counterpart_type", [0],
            "English subspaces from idiom-like vs paraphrase-like counterparts vs size-matched random subsets",
            "X15-X19"),
}

# Per group: run order (the gate first), the gate, the --smoke budgets and the options its experiments read.
GROUPS = {
    "X1-X8": dict(
        order=["X5", "X1", "X3", "X7", "X8", "X2", "X6", "X4"], gate="X5", preflight="preflight", verify="verify",
        smoke=dict(null_draws=9, perm_draws=9, bootstrap=50, zs_rand_draws=9, zs_perm_draws=9, layer_rand_draws=9,
                   layer_perm_draws=9, mlp_rand_draws=1, mdl_orders=1, mdl_rand_draws=1, n_jobs=1, mlp_small=True),
        options=["null_draws", "perm_draws", "bootstrap", "zs_rand_draws", "zs_perm_draws", "layer_rand_draws",
                 "layer_perm_draws", "mlp_rand_draws", "mdl_orders", "mdl_rand_draws", "n_jobs", "dom_k", "fbn_k",
                 "concepts", "fbn_backend", "fbn_model", "methods", "phase", "device", "parts", "layers"]),
    "X9-X14": dict(
        order=["X10", "X12", "X13", "X9", "X11", "X14"], gate="X10", preflight="preflight2", verify="verify2",
        smoke=dict(bootstrap=50, sign_draws=199, rand_draws=1, shuffle_draws=1, n_jobs=1, mlp_small=True,
                   inlp_max_rank=12, rlace_steps=300, rlace_pilot_steps=100, rlace_eval_every=50,
                   rlace_lrs="0.01,0.03", null_draws=9, perm_draws=9, zs_rand_draws=9, zs_perm_draws=9),
        options=["bootstrap", "sign_draws", "rand_draws", "shuffle_draws", "n_jobs", "inlp_max_rank", "rlace_steps",
                 "rlace_pilot_steps", "rlace_batch", "rlace_eval_every", "null_draws", "perm_draws", "zs_rand_draws",
                 "zs_perm_draws", "tag_k", "guard_eps", "rlace_weight_decay", "rlace_init_scale", "parts", "fit_sets",
                 "rlace_lrs", "reseeds", "concepts", "methods"]),
    "X15-X19": dict(
        order=["X15", "X16", "X17", "X19", "X18"], gate="X15", preflight="preflight3", verify="verify3",
        smoke=dict(bootstrap=50, null_draws=9, perm_draws=9, cov_draws=9, zs_rand_draws=9, zs_perm_draws=9,
                   refit_draws=3, subset_draws=3, human_perm=49, human_boot=49),
        options=["bootstrap", "null_draws", "perm_draws", "cov_draws", "zs_rand_draws", "zs_perm_draws", "refit_draws",
                 "subset_draws", "human_perm", "human_boot"]),
}
ORDER = [code for group in GROUPS.values() for code in group["order"]]
INT_OPTIONS = ["null_draws", "perm_draws", "bootstrap", "zs_rand_draws", "zs_perm_draws", "layer_rand_draws",
               "layer_perm_draws", "mlp_rand_draws", "mdl_orders", "mdl_rand_draws", "n_jobs", "dom_k", "fbn_k",
               "sign_draws", "rand_draws", "shuffle_draws", "inlp_max_rank", "rlace_steps", "rlace_pilot_steps",
               "rlace_batch", "rlace_eval_every", "tag_k", "cov_draws", "refit_draws", "subset_draws", "human_perm",
               "human_boot"]
FLOAT_OPTIONS = ["guard_eps", "rlace_weight_decay", "rlace_init_scale"]
TEXT_OPTIONS = {
    "concepts": "concepts: X2 (default FBN,DOM) or X11 (default F1,F2,F3,TAG)",
    "methods": "estimators: X2 or X11 (default MLR,COV)",
    "parts": "parts: X6 (default mlp,mdl) or X9 (default linear,mlp)",
    "fbn_model": "X2 NLLB checkpoint (default facebook/nllb-200-distilled-600M)",
    "device": "X4 extraction device (default auto)",
    "layers": "X8 layer subset, comma-separated",
    "fit_sets": "X10 pair sets (default space,probe,validation,probe_validation)",
    "rlace_lrs": "X9 R-LACE learning rates for the pilot (default 0.003,0.01,0.03,0.1)",
    "reseeds": "X11 k-means seeds for the re-seeded F taxonomies (default 1,2,3)",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS, key=lambda c: int(c[1:])) +
                   ["all", "preflight", "verify", "list"])
    p.add_argument("--campaign", type=Path, help="discovery campaign directory (read-only)")
    p.add_argument("--out", type=Path, help="work directory for checkpoints and item-level evidence")
    p.add_argument("--followup", type=Path, help="follow-up directory (default CAMPAIGN/paper_followup_seed1_20261021)")
    p.add_argument("--results", type=Path, help="aggregate destination (default results/experiments)")
    p.add_argument("--group", choices=sorted(GROUPS, key=lambda g: int(g.split("-")[0][1:])),
                   help="preflight, verify or all: restrict to one group of experiments")
    p.add_argument("--x7", type=Path, help="X7 aggregates for the X9-X14 reproduction gates "
                                          "(default RESULTS/X7_alignment)")
    p.add_argument("--human", type=Path, help="volunteers' judgments for X17 "
                                             "(default results/human_validation/judgments.csv)")
    p.add_argument("--models", help="comma-separated subset of the configured model names")
    p.add_argument("--splits", help="comma-separated splits: 0 (discovery) and/or 1 (follow-up)")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for torch/BLAS (default 8)")
    p.add_argument("--allow-cpu", action="store_true", help="run without CUDA (smoke tests only; very slow)")
    p.add_argument("--smoke", action="store_true", help="tiny null/bootstrap budgets for a pipeline check")
    for name in INT_OPTIONS:
        p.add_argument("--" + name.replace("_", "-"), type=int)
    for name in FLOAT_OPTIONS:
        p.add_argument("--" + name.replace("_", "-"), type=float)
    for name, text in TEXT_OPTIONS.items():
        p.add_argument("--" + name.replace("_", "-"), help=text)
    p.add_argument("--fbn-backend", choices=["nllb", "tfidf"], help="X2 FBN embedder (default nllb)")
    p.add_argument("--phase", choices=["all", "extract", "analyse"], help="X4 phase (default all)")
    p.add_argument("--perm-mlr", action="store_true", help="also run (slow) permuted-label MLR nulls in X1/X8")
    p.add_argument("--check", help="verify only these experiments, e.g. X5 or X10,X12")
    return p


def groups_for(args):
    return [args.group] if args.group else list(GROUPS)


def make_context(args, code):
    from experiments.common import Context, FOLLOWUP_NAME, RESULTS, base_config, configured_models
    _, folder, default_splits, _, group = EXPERIMENTS[code]
    spec = GROUPS[group]
    campaign = args.campaign.resolve()
    cfg = base_config(campaign)
    options = dict(spec["smoke"]) if args.smoke else {}
    for key in spec["options"]:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    if group == "X1-X8" and args.perm_mlr:
        options["perm_mlr"] = True
    if group == "X15-X19" and args.human is not None:
        options["human"] = str(args.human.resolve())
    splits = [int(s) for s in args.splits.split(",")] if args.splits else default_splits
    return Context(experiment=folder, campaign=campaign, out=args.out.resolve(),
                   followup=(args.followup or campaign / FOLLOWUP_NAME).resolve(),
                   results=(args.results or RESULTS).resolve(), models=configured_models(cfg, args.models),
                   splits=splits, cfg=cfg, options=options)


def run_one(args, code):
    module, folder, _, title, _ = EXPERIMENTS[code]
    ctx = make_context(args, code)
    print(f"=== {code} {folder}: {title}\n    models={ctx.models}\n    splits={ctx.splits} options={ctx.options}",
          flush=True)
    started = time.time()
    importlib.import_module(f"experiments.{module}").run(ctx)
    print(f"=== {code} finished in {(time.time() - started) / 3600:.2f} h; aggregates in {ctx.portable}", flush=True)


def preflight(args):
    status = 0
    for group in groups_for(args):
        print(f"--- preflight {group}", flush=True)
        check = importlib.import_module(f"experiments.{GROUPS[group]['preflight']}").preflight
        out = args.out.resolve()
        if group == "X1-X8":
            code = check(args.campaign, out, args.followup, args.models)
        elif group == "X9-X14":
            splits = [int(s) for s in args.splits.split(",")] if args.splits else [0, 1]   # X12 also reads split 1
            code = check(args.campaign, out, args.followup, args.models, splits)
        else:
            code = check(args.campaign, out, args.followup, args.models, args.human)
        status = max(status, int(code or 0))
    return status


def verify(args):
    only = args.check.split(",") if args.check else None
    status = 0
    for group in groups_for(args):
        codes = [c for c in only if EXPERIMENTS.get(c, (None,) * 5)[4] == group] if only else None
        if only and not codes:
            continue
        check = importlib.import_module(f"experiments.{GROUPS[group]['verify']}").verify
        if group == "X9-X14":
            x7 = args.x7 or ((args.results / "X7_alignment") if args.results else None)
            code = check(args.results, args.out, codes, x7)
        else:
            code = check(args.results, args.out, codes)
        status = max(status, int(code or 0))
    if only:
        unknown = [c for c in only if c not in EXPERIMENTS]
        if unknown:
            print("unknown experiments:", ", ".join(unknown), flush=True)
            status = max(status, 1)
    return status


def main(argv=None):
    args = parser().parse_args(argv)
    if args.experiment == "list":
        for group, spec in GROUPS.items():
            print(f"{group} (gate {spec['gate']})")
            for code in spec["order"]:
                module, folder, splits, title, _ = EXPERIMENTS[code]
                print(f"  {code:4s} {folder:24s} {title}; default splits {splits}")
        return
    if args.experiment == "verify":
        sys.exit(verify(args))
    if args.campaign is None or args.out is None:
        sys.exit("--campaign and --out are required")
    if args.experiment == "preflight":
        sys.exit(preflight(args))
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("CUDA is not available. These experiments are meant for the GPU machine; "
                 "pass --allow-cpu only for smoke tests.")
    codes = [c for g in groups_for(args) for c in GROUPS[g]["order"]] if args.experiment == "all" \
        else [args.experiment]
    failures = []
    with threadpool_limits(limits=args.threads):
        for code in codes:
            try:
                run_one(args, code)
            except Exception:
                traceback.print_exc()
                failures.append(code)
                if args.experiment != "all":
                    raise
    if failures:
        print("FAILED:", ", ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
