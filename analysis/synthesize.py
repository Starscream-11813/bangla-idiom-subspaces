"""Second-pass synthesis of the frozen discovery campaign.

Reads only the derived campaign tables in results/discovery (and the class aggregates
written by rebuild_taxonomy.py). Nothing is refitted. Every quantity here is a
descriptive re-analysis of saved outputs; the only confirmatory tests remain
the twelve planned MLR tests in results/discovery/primary_confirmatory.csv.

    python analysis/synthesize.py   # writes results/discovery/synthesis/*.csv and analysis/figures/
"""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("IDIO_RESULTS", HERE.parent / "results"))
SRC = RESULTS / "discovery"
OUT = SRC / "synthesis"
FIG = Path(os.environ.get("IDIO_FIGURES", HERE / "figures"))
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Models ordered by the amount of cross-lingual (parallel) training signal.
MODELS = ["csebuetnlp/banglabert", "csebuetnlp/banglishbert", "bert-base-multilingual-cased",
          "xlm-roberta-large", "google/muril-large-cased", "setu4993/LaBSE"]
NAME = {"csebuetnlp/banglabert": "BanglaBERT", "csebuetnlp/banglishbert": "BanglishBERT",
        "bert-base-multilingual-cased": "mBERT", "xlm-roberta-large": "XLM-R",
        "google/muril-large-cased": "MuRIL", "setu4993/LaBSE": "LaBSE", "Qwen/Qwen3-8B": "Qwen3-8B"}
GROUP = {"csebuetnlp/banglabert": "Bangla-centric", "csebuetnlp/banglishbert": "Bangla-centric",
         "bert-base-multilingual-cased": "multilingual MLM", "xlm-roberta-large": "multilingual MLM",
         "google/muril-large-cased": "parallel-supervised", "setu4993/LaBSE": "parallel-supervised"}
COLOR = {"csebuetnlp/banglabert": "#E69F00", "csebuetnlp/banglishbert": "#D55E00",
         "bert-base-multilingual-cased": "#56B4E9", "xlm-roberta-large": "#0072B2",
         "google/muril-large-cased": "#009E73", "setu4993/LaBSE": "#CC79A7", "Qwen/Qwen3-8B": "#222222"}
MARK = {"Bangla-centric": "o", "multilingual MLM": "s", "parallel-supervised": "^"}
LAYERS = {"csebuetnlp/banglabert": 12, "csebuetnlp/banglishbert": 12, "bert-base-multilingual-cased": 12,
          "xlm-roberta-large": 24, "google/muril-large-cased": 24, "setu4993/LaBSE": 12}
SELECTED = {"xlm-roberta-large": 24, "bert-base-multilingual-cased": 12, "setu4993/LaBSE": 6,
            "google/muril-large-cased": 12, "csebuetnlp/banglabert": 9, "csebuetnlp/banglishbert": 12}
TOP_RANK = {"F": 11, "S": 3}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .5, "figure.dpi": 150, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})
TABLES = []

# Role colours, with the default (Okabe-Ito) values. HOUSE_STYLE=1 restyled the figures for the paper with a
# plotting module that is not part of this release; the default style is the one supported here.
import os
import sys
HOUSE = os.environ.get("HOUSE_STYLE") == "1"
METHOD_COLOR = {"RAND": "#999999", "PCA": "#56B4E9", "MLR": "#D55E00", "LEACE": "#CC79A7", "LDA": "#E69F00",
                "COV": "#009E73"}
INK, QWEN, GREY, SCATTER = "k", "#222222", "#555555", "#0072B2"
ERASE_COLOR = {"ambient": "#bbbbbb", "en_LEACE": "#0072B2", "en_COV": "#56B4E9", "bn_LEACE": "#D55E00"}
TRANSPARENCY_COLOR = {"opaque": "#0072B2", "transparent": "#E69F00"}
SHADE = {"random": "#eeeeee", "unsupervised": "#f4f4f4", "filter": "#fde3df", "pattern": "#dff2e7"}
METHOD_LS = {"RAND": ":", "PCA": ":", "LDA": "-.", "MLR": "-", "LEACE": "--", "COV": "-"}
VS, POSTHOC = (r"\textit{vs.}", r"\textit{post hoc}") if HOUSE else ("vs", "post-hoc")   # Latin, italic in print
FTAG = (lambda c: rf"F\textsubscript{{{c}}}") if HOUSE else (lambda c: f"F{c}")       # class tags, as in the paper
if HOUSE:
    raise SystemExit("HOUSE_STYLE=1 needs the paper's figure-style module, which is not part of this release.")


def load(name):
    return pd.read_csv(SRC / name, low_memory=False)


def save(name, frame):
    if not HOUSE:                       # house-style runs only redraw figures; the derived tables stay frozen
        frame.to_csv(OUT / name, index=False)
    return frame


def house_legends(fig):
    """The paper's legend policy. Captions key the encoders, so a legend of encoders only is dropped;
    any other legend drawn beside a panel moves into one row above the panels, so that every panel keeps its width.
    Legends inside or above a panel stay where they are."""
    encoders = set(NAME.values())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    moved = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        labels = [t.get_text() for t in leg.get_texts()]
        if all(label in encoders for label in labels):
            leg.remove()
            continue
        if leg.get_window_extent(renderer).x0 >= ax.get_window_extent(renderer).x1 - 2:
            moved += [(h, label) for h, label in zip(leg.legend_handles, labels) if label not in encoders]
            leg.remove()
    if moved:
        handles, labels = zip(*moved)
        if fig.get_layout_engine() is not None:
            fig.legend(handles, labels, loc="outside upper center", ncol=len(handles), frameon=False)
        else:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                       frameon=False)


def savefig(fig, name):
    if HOUSE:
        house_legends(fig)
        # The paper sets multi-panel figures as subfigures: lettered titles ("a  ...") become subcaptions, and
        # panels titled without letters (one per encoder) are lettered in reading order.
        import re
        titled = [(ax, ax.get_title(loc="left") or ax.get_title()) for ax in fig.axes]
        titled = [(ax, t) for ax, t in titled if t]
        lettered = [(ax, re.match(r"^([a-z])\s\s+(.*)$", t, flags=re.S)) for ax, t in titled]
        if len(titled) > 1:
            if all(m for _, m in lettered):
                panels = [([ax], m.group(1), " ".join(m.group(2).split())) for ax, m in lettered]
            else:
                order = sorted(titled, key=lambda at: (-round(at[0].get_position().y1, 2), at[0].get_position().x0))
                panels = [([ax], "abcdefgh"[k], " ".join(t.split())) for k, (ax, t) in enumerate(order)]
            saved = {ax: (ax.get_title(loc="left"), ax.get_title()) for axes, _, _ in panels for ax in axes}
            for axes, _, _ in panels:
                axes[0].set_title("", loc="left")
                axes[0].set_title("")
            try:
                hs.save_panels(fig, name, folder=FIG, panels=panels)
                return
            except ValueError as error:         # letters not in reading order: keep the drawn titles
                print(f"savefig: {error}; keeping the drawn titles")
                for ax, (left, centre) in saved.items():
                    ax.set_title(left, loc="left")
                    ax.set_title(centre)
                (FIG / f"{name}.tex").unlink(missing_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    if not HOUSE:
        fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)


def md(frame, title, floatfmt=3):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:.{floatfmt}f}"
        return str(v)
    lines = [f"#### {title}", "", "| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    TABLES.append("\n".join(lines) + "\n")


def legend_handles(models=MODELS, extra=()):
    hs = [Line2D([], [], color=COLOR[m], marker=MARK[GROUP[m]], ls="", ms=5, label=NAME[m]) for m in models]
    return hs + list(extra)


# ---------------------------------------------------------------------------
# 1. The sharing ladder: function, retrieval and pattern geometry
# ---------------------------------------------------------------------------
inf = load("all_selected_layer_inference.csv")
inf = inf[inf.group.eq("all") & inf.method.eq("MLR")].copy()
inf["pair"] = inf.source_condition + "→" + inf.condition
dec = load("decoder_inference.csv")
dec["pair"] = dec.source_condition + "→" + dec.condition
base = load("S3_baselines.csv")
base = base[base.centering.eq("per_condition_space")].copy()
base["pair"] = base.source_condition + "→" + base.condition
ov = load("S3_subspace_overlap.csv")
ov["pair"] = ov.source_condition + "→" + ov.condition
ov["shared_dims"] = ov.mean_cos2 * ov.M
ov["random_dims"] = ov.random_expected_mean_cos2 * ov.M

RUNGS = [("en-para→bn-para", "gloss → gloss"),
         ("en-trans→bn-fig", "translation → idiom in context"),
         ("en-idiom→bn-idiom", "counterpart → idiom (no context)"),
         ("en-idiom→bn-fig", "counterpart → idiom in context"),
         ("en-idiom→bn-context", "counterpart → masked context")]
ladder = []
for concept in ["F", "S"]:
    for model in MODELS:
        for pair, label in RUNGS:
            r = inf[(inf.model == model) & (inf.concept == concept) & (inf.pair == pair)]
            b = base[(base.model == model) & (base.pair == pair)]
            o = ov[(ov.model == model) & (ov.concept == concept) & (ov.pair == pair) & ov.method.eq("COV")
                   & ov.M.eq(TOP_RANK[concept])]
            m = ov[(ov.model == model) & (ov.concept == concept) & (ov.pair == pair) & ov.method.eq("MLR")
                   & ov.M.eq(TOP_RANK[concept])]
            ladder.append(dict(model=NAME[model], concept=concept, pair=pair, rung=label,
                               SSI=r.SSI.iloc[0], ci_lo=r.ci_lo.iloc[0], ci_hi=r.ci_hi.iloc[0],
                               cross_pp_above_floor=100 * (r.R_ep.iloc[0] - r.R0.iloc[0]),
                               within_pp_above_floor=100 * (r.R_pp.iloc[0] - r.R0.iloc[0]),
                               retrieval_top1=b.retr_top1.iloc[0], retrieval_MRR=b.retr_MRR.iloc[0],
                               chance_top1=b.chance_top1.iloc[0], CKA=b.CKA.iloc[0],
                               COV_shared_dims=o.shared_dims.iloc[0], MLR_shared_dims=m.shared_dims.iloc[0],
                               random_dims=o.random_dims.iloc[0]))
ladder = save("ladder.csv", pd.DataFrame(ladder))

# Same-target comparisons (identical TEST rows, floor and native reference).
same_target = []
for concept in ["F", "S"]:
    for model in MODELS:
        g = inf[(inf.model == model) & (inf.concept == concept)].set_index("pair")
        same_target.append(dict(model=NAME[model], concept=concept,
                                gloss_to_gloss=g.loc["en-para→bn-para", "SSI"],
                                counterpart_to_gloss=g.loc["en-idiom→bn-para", "SSI"],
                                translation_to_span=g.loc["en-trans→bn-fig", "SSI"],
                                counterpart_to_span=g.loc["en-idiom→bn-fig", "SSI"]))
same_target = pd.DataFrame(same_target)
same_target["gloss_over_counterpart_ratio"] = same_target.gloss_to_gloss / same_target.counterpart_to_gloss
same_target["translation_minus_counterpart"] = same_target.translation_to_span - same_target.counterpart_to_span
save("same_target_comparisons.csv", same_target)
md(same_target[same_target.concept.eq("F")].drop(columns="concept"),
   "Same-target source comparisons, F (MLR SSI; identical TEST rows within each pair of columns)")
md(same_target[same_target.concept.eq("S")].drop(columns="concept"),
   "Same-target source comparisons, S (MLR SSI)")

ladder_F = ladder[ladder.concept.eq("F")]
for value, title in [("SSI", "Sharing ladder, F: MLR SSI by rung"),
                     ("retrieval_top1", "Sharing ladder: translation-pair retrieval top-1 (per-condition centred)"),
                     ("COV_shared_dims", "Sharing ladder, F: shared pattern dimensions (sum of cos² of COV rank-11 principal angles)")]:
    t = ladder_F.pivot_table(index="rung", columns="model", values=value, sort=False)[[NAME[m] for m in MODELS]]
    t.insert(0, "rung", t.index)
    t["median"] = t.drop(columns="rung").median(axis=1)
    md(t.reset_index(drop=True), title, 3 if value != "retrieval_top1" else 4)

fig = plt.figure(figsize=(7.1, 4.6), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
ax_a1, ax_a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
# a: same Bangla target, literal vs idiomatic English source (identical TEST rows per target).
for ax, (lit, idi, target, lit_label) in zip([ax_a1, ax_a2], [
        ("gloss_to_gloss", "counterpart_to_gloss", "Bangla gloss", "English gloss"),
        ("translation_to_span", "counterpart_to_span", "Bangla idiom in context", "English translation")]):
    d = same_target[same_target.concept.eq("F")].set_index("model").loc[[NAME[m] for m in MODELS]]
    rows = list(d.index)
    yy = np.arange(len(rows))[::-1]
    for yi, (name, r) in zip(yy, d.iterrows()):
        model = next(m for m in MODELS if NAME[m] == name)
        ax.plot([r[idi], r[lit]], [yi, yi], color=COLOR[model], lw=1.2)
        ax.plot(r[lit], yi, marker=MARK[GROUP[model]], color=COLOR[model], ms=5)
        ax.plot(r[idi], yi, marker=MARK[GROUP[model]], mfc="white", mec=COLOR[model], ms=5)
    if target == "Bangla idiom in context":
        q = dec[dec.concept.eq("F")].set_index("pair")
        ax.plot([q.loc["en-idiom→bn-fig", "SSI"], q.loc["en-trans→bn-fig", "SSI"]], [-1, -1], color=QWEN, lw=1.2, ls="--")
        ax.plot(q.loc["en-trans→bn-fig", "SSI"], -1, marker="D", color=QWEN, ms=4)
        ax.plot(q.loc["en-idiom→bn-fig", "SSI"], -1, marker="D", mfc="white", mec=QWEN, ms=4)
        ax.set_yticks(list(yy) + [-1], rows + ["Qwen3-8B*"])
    else:
        ax.set_yticks(yy, rows)
    ax.axvline(0, color=INK, lw=.6)
    ax.set_xlim(-.05, 1.12)
    ax.set_xlabel("SSI (MLR), F")
    short = lit_label.replace("English ", "EN ")
    ax.set_title(f"{'a' if ax is ax_a1 else 'b'}  Same target: {target}\n"
                 f"    ● {short} source  ○ EN counterpart source", loc="left", fontsize=7.5)
y = np.arange(len(RUNGS))[::-1]
for model in MODELS:
    d = ladder_F[ladder_F.model.eq(NAME[model])].set_index("pair").loc[[p for p, _ in RUNGS]]
    jitter = (MODELS.index(model) - 2.5) * .07
    for ax, col in [(ax_b, "retrieval_top1"), (ax_c, "COV_shared_dims")]:
        ax.plot(d[col].to_numpy(), y + jitter, color=COLOR[model], marker=MARK[GROUP[model]], ms=4, lw=.8, alpha=.9)
ax_b.set_xscale("log")
ax_b.axvline(ladder_F.chance_top1.min(), color=INK, lw=.6, ls=":")
ax_b.annotate("chance", (ladder_F.chance_top1.min(), y[0] + .3), fontsize=6.5, xytext=(3, 0), textcoords="offset points")
ax_b.set_yticks(y, [label for _, label in RUNGS])
ax_b.set_xlabel("pair retrieval top-1 (log scale)")
ax_b.set_title("c  Representational alignment", loc="left")
ax_c.axvline(ladder_F.random_dims.max(), color=INK, lw=.6, ls=":")
ax_c.annotate("random", (ladder_F.random_dims.max(), y[0] + .3), fontsize=6.5, xytext=(3, 0), textcoords="offset points")
ax_c.set_yticks(y, [""] * len(y))
ax_c.set_xlabel("shared F-pattern dimensions (COV rank 11, of 11)")
ax_c.set_title("d  Concept-pattern geometry", loc="left")
ax_c.legend(handles=legend_handles(extra=[Line2D([], [], color=QWEN, marker="D", ls="--", ms=4,
                                                  label="Qwen3-8B*")]),
            loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
savefig(fig, "fig1_sharing_ladder")

# ---------------------------------------------------------------------------
# 2. Patterns versus filters: which estimator's English directions read Bangla
# ---------------------------------------------------------------------------
cross = load("cross_language_estimators.csv")
cross = cross[cross.group.eq("all")]
cont = load("selected_layer_containment.csv")
cont = cont[cont.group.eq("all") & cont.condition.eq("bn-fig")]
FAMILY = {"RAND": "random", "PCA": "unsupervised", "MLR": "filter", "LDA": "filter", "LEACE": "filter",
          "CPCA": "pattern", "COV": "pattern"}
METHODS = ["RAND", "PCA", "MLR", "LDA", "LEACE", "CPCA", "COV"]
est = []
for concept in ["F", "S"]:
    for model in MODELS:
        amb = cont[(cont.model == model) & (cont.concept == concept) & cont.method.eq("AMBIENT")].iloc[0]
        for method in METHODS:
            for M in sorted(cross[(cross.concept == concept)].M.unique()):
                c = cross[(cross.model == model) & (cross.concept == concept) & (cross.method == method) & (cross.M == M)]
                n = cont[(cont.model == model) & (cont.concept == concept) & (cont.method == method) & (cont.M == M)]
                if c.empty or n.empty:
                    continue
                c, n = c.iloc[0], n.iloc[0]
                r0 = c.R0
                den = n.R - r0
                o = ov[(ov.model == model) & (ov.concept == concept) & ov.pair.eq("en-idiom→bn-fig") & (ov.method == method) & (ov.M == M)]
                est.append(dict(model=NAME[model], concept=concept, method=method, family=FAMILY[method], M=M,
                                cross_R=c.R, native_R=n.R, floor=r0, ambient_R=amb.R,
                                SSI=(c.R - r0) / den if den > .02 else np.nan,
                                cross_minus_ambient_pp=100 * (c.R - amb.R),
                                cross_L=c.L, native_L=n.L,
                                shared_dims=o.shared_dims.iloc[0] if len(o) else np.nan,
                                max_cos=o.max_cos.iloc[0] if len(o) else np.nan,
                                random_dims=o.random_dims.iloc[0] if len(o) else np.nan,
                                spectral_gap_source=o.spectral_gap_source.iloc[0] if len(o) else np.nan))
est = save("estimator_families.csv", pd.DataFrame(est))
for concept in ["F", "S"]:
    top = est[est.concept.eq(concept) & est.M.eq(TOP_RANK[concept])]
    for value, title, scale in [("cross_R", f"English directions → Bangla idiom spans, {concept}, rank {TOP_RANK[concept]}: accuracy (%)", 100),
                                ("native_R", f"Native Bangla directions → Bangla idiom spans, {concept}, rank {TOP_RANK[concept]}: accuracy (%)", 100),
                                ("SSI", f"SSI by estimator, {concept}, rank {TOP_RANK[concept]} (each estimator's own native reference)", 1)]:
        t = (scale * top.pivot_table(index="method", columns="model", values=value)).reindex(METHODS)[[NAME[m] for m in MODELS]]
        t.insert(0, "estimator", t.index)
        if value == "cross_R":
            amb = (100 * top.groupby("model").ambient_R.first())[[NAME[m] for m in MODELS]]
            t.loc["ambient"] = ["native ambient probe (reference)"] + amb.tolist()
        md(t.reset_index(drop=True), title, 2 if scale == 100 else 3)

# Difference-in-differences: oblique (LEACE) vs orthogonal (COV) readout of the same rank-11 image.
did = []
for concept in ["F", "S"]:
    M = TOP_RANK[concept]
    for model in MODELS:
        e = est[(est.model == NAME[model]) & (est.concept == concept) & est.M.eq(M)].set_index("method")
        did.append(dict(model=NAME[model], concept=concept, M=M,
                        native_LEACE_minus_COV_pp=100 * (e.loc["LEACE", "native_R"] - e.loc["COV", "native_R"]),
                        cross_LEACE_minus_COV_pp=100 * (e.loc["LEACE", "cross_R"] - e.loc["COV", "cross_R"]),
                        native_MLR_minus_COV_pp=100 * (e.loc["MLR", "native_R"] - e.loc["COV", "native_R"]),
                        cross_MLR_minus_COV_pp=100 * (e.loc["MLR", "cross_R"] - e.loc["COV", "cross_R"])))
did = pd.DataFrame(did)
did["LEACE_extra_cross_penalty_pp"] = did.cross_LEACE_minus_COV_pp - did.native_LEACE_minus_COV_pp
did["MLR_extra_cross_penalty_pp"] = did.cross_MLR_minus_COV_pp - did.native_MLR_minus_COV_pp
save("pattern_filter_did.csv", did)
md(did.drop(columns="M"), "Filter penalty: oblique/discriminative readout minus pattern readout (pp), native vs cross-lingual", 2)

fig = plt.figure(figsize=(7.1, 3.7), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
ax_a, ax_b, ax_c = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
F11 = est[est.concept.eq("F") & est.M.eq(11)]
ROWS = ["RAND", "PCA", "MLR", "LDA", "LEACE", "COV"]
yy = np.arange(len(ROWS) + 1)[::-1]
for i, m in enumerate(ROWS):
    ax_a.axhspan(yy[i] - .45, yy[i] + .45, color=SHADE[FAMILY[m]], zorder=0, lw=0)
for model in MODELS:
    d = F11[F11.model.eq(NAME[model])].set_index("method").reindex(ROWS)
    jitter = (MODELS.index(model) - 2.5) * .11
    floor = d.floor.iloc[0]
    ax_a.plot(100 * (d.cross_R - floor), yy[:-1] + jitter, color=COLOR[model], marker=MARK[GROUP[model]], ms=4.5, lw=0)
    ax_a.plot(100 * (d.ambient_R.iloc[0] - floor), yy[-1] + jitter, color=COLOR[model], marker=MARK[GROUP[model]],
              mfc="white", ms=4.5, lw=0)
ax_a.axvline(0, color=INK, lw=.6)
ax_a.set_yticks(yy, [f"{m} ({FAMILY[m]})" for m in ROWS] + ["Bangla ambient probe\n(full space, reference)"])
ax_a.set_xlabel("F accuracy above floor on Bangla idiom spans (pp)")
ax_a.set_title("a  English rank-11 directions → Bangla", loc="left")
ax_a.set_xlim(-2, 19)
for model in MODELS:
    d = did[(did.model == NAME[model]) & did.concept.eq("F")].iloc[0]
    ax_b.plot([0, 1], [d.native_LEACE_minus_COV_pp, d.cross_LEACE_minus_COV_pp], color=COLOR[model],
              marker=MARK[GROUP[model]], ms=4, lw=1)
ax_b.axhline(0, color=INK, lw=.6)
ax_b.set_xticks([0, 1], ["native\n(BN→BN)", "cross-lingual\n(EN→BN)"])
ax_b.set_xlim(-.25, 1.25)
ax_b.set_ylabel("LEACE − COV (pp)")
ax_b.set_title("b  Same rank-11 image: filter readout − pattern readout", loc="left")
ax_b.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6.5)
fam_order = ["MLR", "LDA", "PCA", "COV"]
yo = np.arange(len(fam_order))[::-1]
for model in MODELS:
    d = F11[F11.model.eq(NAME[model])].set_index("method").reindex(fam_order)
    jitter = (MODELS.index(model) - 2.5) * .1
    ax_c.plot(d.shared_dims, yo + jitter, color=COLOR[model], marker=MARK[GROUP[model]], ms=4, lw=0)
ax_c.axvline(F11.random_dims.max(), color=INK, lw=.6, ls=":")
ax_c.set_yticks(yo, ["MLR (filter)", "LDA (filter)†", "PCA (variance)", "COV = CPCA = LEACE image"])
ax_c.set_xlabel(f"shared dimensions, English {VS} Bangla F subspaces (of 11)")
ax_c.set_title("c  Subspace overlap (dotted: random)", loc="left")
savefig(fig, "fig2_patterns_vs_filters")

# Rank curves of cross-lingual readout per estimator (appendix).
if HOUSE:   # two rows of three, so that each subfigure is wide enough for its encoder's name
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.5), sharey=True, layout="constrained")
    axes = axes.ravel()
    axes[3].set_ylabel("EN→BN F accuracy (%)")
else:
    fig, axes = plt.subplots(1, 6, figsize=(7.1, 1.9), sharey=True)
for ax, model in zip(axes, MODELS):
    d = est[est.model.eq(NAME[model]) & est.concept.eq("F")]
    for method, color in [("RAND", METHOD_COLOR["RAND"]), ("PCA", METHOD_COLOR["PCA"]), ("MLR", METHOD_COLOR["MLR"]), ("LEACE", METHOD_COLOR["LEACE"]),
                          ("LDA", METHOD_COLOR["LDA"]), ("COV", METHOD_COLOR["COV"])]:
        s = d[d.method.eq(method)].sort_values("M")
        ax.plot(s.M, 100 * s.cross_R, color=color, marker="o", ms=2.5, lw=1.6 if (HOUSE and method == "COV") else 1,
                ls=METHOD_LS[method] if HOUSE else "-", label=method)
    ax.axhline(100 * d.ambient_R.iloc[0], color=INK, lw=.6, ls="--")
    ax.set_title(NAME[model])
    ax.set_xticks([1, 4, 8, 11])
    ax.set_xlabel("rank")
axes[0].set_ylabel("EN→BN F accuracy (%)")
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
savefig(fig, "figA1_rank_curves_by_estimator")

# ---------------------------------------------------------------------------
# 3. Reading transfers, erasing does not
# ---------------------------------------------------------------------------
eras = []
for concept in ["F", "S"]:
    M = TOP_RANK[concept]
    for model in MODELS:
        e = est[(est.model == NAME[model]) & (est.concept == concept) & est.M.eq(M)].set_index("method")
        amb, r0 = e.ambient_R.iloc[0], e.floor.iloc[0]
        row = dict(model=NAME[model], concept=concept, M=M, ambient=amb, floor=r0)
        for method in ["LEACE", "COV", "MLR", "RAND"]:
            row[f"native_{method}_leak"] = e.loc[method, "native_L"]
            row[f"english_{method}_leak"] = e.loc[method, "cross_L"]
        row["english_LEACE_share_of_native_erasure"] = ((amb - row["english_LEACE_leak"]) /
                                                        (amb - row["native_LEACE_leak"]))
        eras.append(row)
eras = save("erasure_transfer.csv", pd.DataFrame(eras))
t = eras[["model", "concept", "ambient", "native_LEACE_leak", "english_LEACE_leak", "native_COV_leak",
          "english_COV_leak", "english_RAND_leak", "floor", "english_LEACE_share_of_native_erasure"]].copy()
for c in t.columns[2:-1]:
    t[c] = 100 * t[c]
md(t, "Erasure: Bangla accuracy after removing a rank-11 (F) / rank-3 (S) subspace (%)", 2)

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), layout="constrained")
for ax, concept in zip(axes, ["F", "S"]):
    d = eras[eras.concept.eq(concept)].set_index("model").loc[[NAME[m] for m in MODELS]]
    xx = np.arange(len(d))
    w = .2
    ax.bar(xx - 1.5 * w, 100 * d.ambient, w, color=ERASE_COLOR["ambient"], label="no erasure (ambient)")
    ax.bar(xx - .5 * w, 100 * d.english_LEACE_leak, w, color=ERASE_COLOR["en_LEACE"], label="English-fitted LEACE")
    ax.bar(xx + .5 * w, 100 * d.english_COV_leak, w, color=ERASE_COLOR["en_COV"], label="English-fitted COV")
    ax.bar(xx + 1.5 * w, 100 * d.native_LEACE_leak, w, color=ERASE_COLOR["bn_LEACE"], label="Bangla-fitted LEACE")
    ax.axhline(100 * d.floor.iloc[0], color=INK, lw=.7, ls=":")
    ax.set_xticks(xx, d.index, rotation=25, ha="right")
    ax.set_ylim(100 * d.floor.iloc[0] - (6 if concept == "F" else 4), None)
    ax.set_ylabel(f"Bangla {concept} accuracy (%)")
    ax.set_title(f"{'a' if concept == 'F' else 'b'}  {concept}, rank-{TOP_RANK[concept]} erasure (dotted: floor)", loc="left")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
savefig(fig, "fig3_erasure_does_not_transfer")

# ---------------------------------------------------------------------------
# 4. Context carries the shared gist
# ---------------------------------------------------------------------------
cond = load("selected_layer_conditions.csv")
cond = cond[cond.group.eq("all")]
COND = [("en-para", "English gloss*"), ("bn-para", "Bangla gloss"), ("en-trans", "English translation"),
        ("bn-sentence", "Bangla sentence"), ("bn-fig", "Bangla idiom in context"),
        ("bn-context", "Bangla context, idiom masked"), ("en-idiom", "English counterpart"),
        ("bn-idiom", "Bangla idiom, no context")]
read = []
for concept in ["F", "S"]:
    M = TOP_RANK[concept]
    for model in MODELS:
        for c, label in COND:
            r = cond[(cond.model == model) & (cond.concept == concept) & (cond.condition == c) & cond.method.eq("MLR") & cond.M.eq(M)]
            a = cond[(cond.model == model) & (cond.concept == concept) & (cond.condition == c) & cond.method.eq("AMBIENT")]
            read.append(dict(model=NAME[model], concept=concept, condition=c, label=label, n=r.n.iloc[0],
                             floor=r.R0.iloc[0], MLR_R=r.R.iloc[0], ambient_R=a.R.iloc[0],
                             MLR_pp_above_floor=100 * (r.R.iloc[0] - r.R0.iloc[0]),
                             ambient_pp_above_floor=100 * (a.R.iloc[0] - a.R0.iloc[0])))
read = save("readability_ladder.csv", pd.DataFrame(read))
bow = load("bow_baselines.csv")
BOW = {"en-para": ("en-para", "text"), "bn-para": ("bn-para", "text"), "en-trans": ("en-trans", "text"),
       "bn-sentence": ("bn-sentence", "text"), "bn-fig": ("bn-fig", "span"), "bn-context": ("bn-context", "text"),
       "en-idiom": ("en-idiom", "span"), "bn-idiom": ("bn-idiom", "span")}
for concept in ["F", "S"]:
    t = read[read.concept.eq(concept)].pivot_table(index="label", columns="model", values="MLR_pp_above_floor", sort=False)
    t = t[[NAME[m] for m in MODELS]]
    t.insert(0, "condition", t.index)
    t["median"] = t.drop(columns="condition").median(axis=1)
    lex = []
    for c, label in COND:
        cc, field = BOW[c]
        b = bow[(bow.condition == cc) & (bow.field == field) & (bow.concept == concept)].iloc[0]
        lex.append(100 * (b.R - b.R0))
    t["TF-IDF lexical baseline"] = lex
    md(t.reset_index(drop=True), f"Native readability ladder, {concept}: accuracy above the majority floor (pp), MLR rank {TOP_RANK[concept]}", 2)

ctrl = load("matched_input_controls.csv")
CTRL = [("bn-context", "masked context"), ("bn-idiom", "idiom without context"),
        ("bn-sentence", "whole sentence"), ("bn-para", "Bangla gloss")]
t = ctrl.copy()
t["model"] = t.model.map(NAME)
for c in ["cross_span", "cross_control", "span_minus_control", "ci_lo", "ci_hi"]:
    t[c] = 100 * t[c]
t["control"] = t.control.map(dict(CTRL))
save("matched_controls_pp.csv", t)
md(t[t.concept.eq("F")][["model", "control", "n", "cross_span", "cross_control", "span_minus_control", "ci_lo", "ci_hi"]],
   "Matched input controls, F: English-direction accuracy on the idiom span minus on the control (pp; same sense IDs)", 2)

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1]))
dF = read[read.concept.eq("F")]
yy = np.arange(len(COND))[::-1]
for model in MODELS:
    d = dF[dF.model.eq(NAME[model])].set_index("condition").loc[[c for c, _ in COND]]
    jitter = (MODELS.index(model) - 2.5) * .07
    axes[0].plot(d.MLR_pp_above_floor, yy + jitter, color=COLOR[model], marker=MARK[GROUP[model]], ms=4, lw=0)
lex = []
for c, _ in COND:
    cc, field = BOW[c]
    b = bow[(bow.condition == cc) & (bow.field == field) & bow.concept.eq("F")].iloc[0]
    lex.append(100 * (b.R - b.R0))
axes[0].plot(lex, yy, color=INK, marker="|", ms=10, lw=0, mew=1.5)
axes[0].set_yticks(yy, [label for _, label in COND])
axes[0].axvline(0, color=INK, lw=.6)
axes[0].set_xlabel("native F accuracy above floor (pp)")
axes[0].set_title("a  Native F readability (| = TF-IDF)", loc="left")
cF = ctrl[ctrl.concept.eq("F")]
yy2 = np.arange(len(CTRL))[::-1]
for model in MODELS:
    d = cF[cF.model.eq(model)].set_index("control").loc[[c for c, _ in CTRL]]
    jitter = (MODELS.index(model) - 2.5) * .1
    axes[1].errorbar(100 * d.span_minus_control, yy2 + jitter,
                     xerr=[100 * (d.span_minus_control - d.ci_lo), 100 * (d.ci_hi - d.span_minus_control)],
                     color=COLOR[model], marker=MARK[GROUP[model]], ms=4, lw=0, elinewidth=.8, capsize=0)
axes[1].axvline(0, color=INK, lw=.6)
axes[1].set_yticks(yy2, [f"{VS} masked\ncontext", f"{VS} idiom\nw/o context", f"{VS} whole\nsentence",
                         f"{VS} Bangla\ngloss"])
axes[1].set_xlabel("span − control, F accuracy (pp)")
axes[1].set_title("b  Matched controls (95% component CI)", loc="left")
axes[1].legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
savefig(fig, "fig4_context_carries_gist")

# ---------------------------------------------------------------------------
# 5. Gist, not identity
# ---------------------------------------------------------------------------
ident = load("identity_rank64_full_space.csv")
ident["model_name"] = ident.model.map(NAME)
t = ident.pivot_table(index=["representation", "method"], columns="model_name", values="R")[[NAME[m] for m in MODELS]] * 100
t = t.reset_index()
amb = ident.groupby(["representation", "model_name"]).ambient_R.first().unstack()[[NAME[m] for m in MODELS]] * 100
for rep in amb.index:
    t.loc[len(t)] = [rep, "AMBIENT"] + amb.loc[rep].tolist()
save("identity_summary.csv", t)
md(t.sort_values(["representation", "method"]), "Unseen-idiom identity, top-1 (%) among 1,889 TEST pairs; rank 64, N = 3,125 (chance 0.053%)", 2)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), layout="constrained")
for ax, rep, title in [(axes[0], "idiom_span", "a  Idiom spans (shared surface form)"),
                       (axes[1], "context_only_idiom_masked", "b  Idiom-masked contexts")]:
    d = ident[ident.representation.eq(rep)]
    xx = np.arange(len(MODELS))
    for j, (method, color) in enumerate([("RAND", METHOD_COLOR["RAND"]), ("PCA", METHOD_COLOR["PCA"]), ("COV", METHOD_COLOR["COV"]),
                                         ("LEACE", METHOD_COLOR["LEACE"]), ("LDA", METHOD_COLOR["LDA"])]):
        v = [100 * d[(d.model == m) & (d.method == method)].R.iloc[0] for m in MODELS]
        ax.bar(xx + (j - 2) * .15, v, .15, color=color, label=method)
    ambv = [100 * d[d.model == m].ambient_R.iloc[0] for m in MODELS]
    ax.plot(xx, ambv, "k_", ms=12, mew=1.6, label="ambient")
    ax.axhline(100 / 1889, color=INK, lw=.6, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(xx, [NAME[m] for m in MODELS], rotation=25, ha="right")
    ax.set_ylabel("top-1 (%), log scale")
    ax.set_title(title, loc="left")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1), frameon=False)
savefig(fig, "fig5_identity_form_vs_context")

# ---------------------------------------------------------------------------
# 6. Where sharing emerges (layers)
# ---------------------------------------------------------------------------
lay = load("primary_layer_profiles.csv")
lay["depth"] = lay.apply(lambda r: r.layer / LAYERS[r.model], axis=1)
save("layer_profiles_relative_depth.csv", lay[["model", "concept", "layer", "depth", "R", "R_pp", "R0", "SSI",
                                               "RAND_R", "PCA_R", "PCA_SSI"]].assign(model=lambda d: d.model.map(NAME)))
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), layout="constrained")
for model in MODELS:
    d = lay[lay.model.eq(model) & lay.concept.eq("F")].sort_values("layer")
    axes[0].plot(d.depth, 100 * (d.R - d.R0), color=COLOR[model], marker=MARK[GROUP[model]], ms=3, lw=1)
    axes[1].plot(d.depth, 100 * (d.R_pp - d.R0), color=COLOR[model], marker=MARK[GROUP[model]], ms=3, lw=1)
    axes[2].plot(d.depth, 100 * (d.PCA_R - d.R0), color=COLOR[model], marker=MARK[GROUP[model]], ms=3, lw=1)
    sel = d[d.layer.eq(SELECTED[model])]
    axes[0].plot(sel.depth, 100 * (sel.R - sel.R0), marker="o", mfc="none", mec=INK, ms=7, lw=0)
for ax, title in zip(axes, ["a  EN MLR → BN", "b  BN MLR → BN (native)",
                            "c  EN PCA → BN"]):
    ax.axhline(0, color=INK, lw=.6)
    ax.set_xlabel("relative depth (layer / #layers)")
    ax.set_title(title, loc="left")
axes[0].set_ylabel("F accuracy above floor (pp)")
axes[2].legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
savefig(fig, "fig6_layer_emergence")

# ---------------------------------------------------------------------------
# 7. Direction asymmetry
# ---------------------------------------------------------------------------
PAIRS = [("en-idiom→bn-fig", "bn-fig→en-idiom", "counterpart ↔ idiom in context"),
         ("en-trans→bn-fig", "bn-fig→en-trans", "translation ↔ idiom in context"),
         ("en-idiom→bn-idiom", "bn-idiom→en-idiom", "counterpart ↔ idiom (no context)"),
         ("en-para→bn-para", "bn-para→en-para", "gloss ↔ gloss")]
asym = []
for concept in ["F", "S"]:
    for model in MODELS:
        g = inf[(inf.model == model) & (inf.concept == concept)].set_index("pair")
        for fwd, rev, label in PAIRS:
            a, b = g.loc[fwd, "SSI"], g.loc[rev, "SSI"]
            asym.append(dict(model=NAME[model], concept=concept, pair=label, en_to_bn=a, bn_to_en=b,
                             bn_to_en_minus_en_to_bn=b - a,
                             bn_private_over_shared=1 / a - 1 if a > 0 else np.nan,
                             en_private_over_shared=1 / b - 1 if b > 0 else np.nan))
    gq = dec[dec.concept.eq(concept)].set_index("pair")
    for fwd, rev, label in PAIRS:
        if fwd in gq.index and rev in gq.index:
            a, b = gq.loc[fwd, "SSI"], gq.loc[rev, "SSI"]
            asym.append(dict(model="Qwen3-8B", concept=concept, pair=label, en_to_bn=a, bn_to_en=b,
                             bn_to_en_minus_en_to_bn=b - a, bn_private_over_shared=1 / a - 1 if a > 0 else np.nan,
                             en_private_over_shared=1 / b - 1 if b > 0 else np.nan))
asym = save("asymmetry.csv", pd.DataFrame(asym))
t = asym[asym.concept.eq("F")].pivot_table(index="model", columns="pair", values="bn_to_en_minus_en_to_bn", sort=False)
t.insert(0, "model", t.index)
md(t.reset_index(drop=True), "Direction asymmetry, F: SSI(Bangla→English) − SSI(English→Bangla)", 3)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), layout="constrained")
PMARK = {"counterpart ↔ idiom in context": "o", "translation ↔ idiom in context": "s",
         "counterpart ↔ idiom (no context)": "v", "gloss ↔ gloss": "*"}
for ax, concept in zip(axes, ["F", "S"]):
    d = asym[asym.concept.eq(concept)]
    for r in d.itertuples():
        model = next((m for m in list(NAME) if NAME[m] == r.model), None)
        ax.plot(r.en_to_bn, r.bn_to_en, marker=PMARK[r.pair], color=COLOR[model], ms=6 if r.pair == "gloss ↔ gloss" else 4.5, lw=0)
    lim = [min(-.05, d[["en_to_bn", "bn_to_en"]].min().min() - .05), max(1.1, d[["en_to_bn", "bn_to_en"]].max().max() + .05)]
    ax.plot(lim, lim, color=INK, lw=.6, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SSI English → Bangla")
    ax.set_ylabel("SSI Bangla → English")
    ax.set_title(f"{'a' if concept == 'F' else 'b'}  {concept}", loc="left")
if HOUSE:
    # Marker shape encodes the pair here, so encoders are keyed by colour; one legend row above the panels.
    handles = [Line2D([], [], color=COLOR[m], lw=2.4, label=NAME[m]) for m in MODELS]
    handles += [Line2D([], [], color=QWEN, lw=2.4, label="Qwen3-8B")]
    handles += [Line2D([], [], color=GREY, marker=mk, ls="", ms=5, label=p) for p, mk in PMARK.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=6, frameon=False, handlelength=1.4)
else:
    handles = legend_handles(MODELS) + [Line2D([], [], color=QWEN, marker="o", ls="", ms=4, label="Qwen3-8B")]
    handles += [Line2D([], [], color=GREY, marker=mk, ls="", ms=5, label=p) for p, mk in PMARK.items()]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
savefig(fig, "fig7_direction_asymmetry")

# ---------------------------------------------------------------------------
# 8. Which meanings travel: per-class transfer
# ---------------------------------------------------------------------------
cls = load("class_recall_diagnostic.csv")
tax = pd.read_csv(OUT / "F_taxonomy.csv")
cF = cls[cls.concept.eq("F")].copy()
cF["model_name"] = cF.model.map(NAME)
per = cF.groupby("label").agg(n=("n", "first"), cross_recall=("cross_recall", "mean"),
                              within_recall=("within_recall", "mean")).reset_index()
per["transfer_ratio"] = per.cross_recall / per.within_recall
per = per.merge(tax, left_on="label", right_on="F")
per = save("class_transfer.csv", per)
rho_neutral = spearmanr(per[per.label.ne(10)].transfer_ratio, per[per.label.ne(10)].neutral_share)
rho_transp = spearmanr(per[per.label.ne(10)].transfer_ratio, per[per.label.ne(10)].cp_transparency_mean)
rho_size = spearmanr(per[per.label.ne(10)].transfer_ratio, per[per.label.ne(10)].n)
t = per.sort_values("transfer_ratio", ascending=False)[["F", "name", "n", "within_recall", "cross_recall", "transfer_ratio",
                                                         "neutral_share", "cp_transparency_mean", "keywords"]].copy()
t["within_recall"] *= 100; t["cross_recall"] *= 100
md(t, "Which F classes transfer (MLR primary readouts, mean over six encoders; recall in %)", 2)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), layout="constrained", gridspec_kw=dict(width_ratios=[1.7, 1]))
order = per.sort_values("transfer_ratio", ascending=False).label.tolist()
mat = cF.pivot_table(index="label", columns="model_name", values="cross_recall").loc[order][[NAME[m] for m in MODELS]] * 100
wit = cF.pivot_table(index="label", columns="model_name", values="within_recall").loc[order][[NAME[m] for m in MODELS]] * 100
im = axes[0].imshow(mat.to_numpy(), cmap=hs.CMAP_LAV if HOUSE else "viridis", aspect="auto", vmin=0, vmax=56)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        axes[0].text(j, i, f"{mat.iloc[i, j]:.0f}", ha="center", va="center", fontsize=6,
                     color=(("white" if mat.iloc[i, j] >= 30 else INK) if HOUSE else
                            ("white" if mat.iloc[i, j] < 30 else "black")))
axes[0].set_xticks(range(len(MODELS)), [NAME[m] for m in MODELS], rotation=30, ha="right")
names = dict(zip(per.label, per.name))
axes[0].set_yticks(range(len(order)), [f"{FTAG(c)}: {names[c]}" + (" (floor class)" if c == 10 else "") for c in order])
axes[0].grid(False)
axes[0].set_title("a  English-direction recall per F class (%)", loc="left")
fig.colorbar(im, ax=axes[0], fraction=.04, pad=.01, label="recall (%)")
p = per[per.label.ne(10)]
axes[1].scatter(100 * p.within_recall, 100 * p.cross_recall, s=12 + p.n / 6, color=SCATTER, alpha=.75)
for r in p.itertuples():
    axes[1].annotate(FTAG(r.label), (100 * r.within_recall, 100 * r.cross_recall), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
lim = [0, 52]
axes[1].plot(lim, lim, "k--", lw=.6)
axes[1].set_xlabel("native recall (%)")
axes[1].set_ylabel("English-direction recall (%)")
axes[1].set_title(f"b  English {VS} native recall" if HOUSE else f"b  Per class: EN-direction {VS} native",
                  loc="left")
savefig(fig, "fig8_which_meanings_travel")

# ---------------------------------------------------------------------------
# 9. Pattern geometry predicts sharing across 66 model × pair cells
# ---------------------------------------------------------------------------
cells = []
for model in MODELS:
    for pair in inf.pair.unique():
        r = inf[(inf.model == model) & inf.concept.eq("F") & (inf.pair == pair)]
        o = ov[(ov.model == model) & ov.concept.eq("F") & (ov.pair == pair) & ov.method.eq("COV") & ov.M.eq(11)]
        m = ov[(ov.model == model) & ov.concept.eq("F") & (ov.pair == pair) & ov.method.eq("MLR") & ov.M.eq(11)]
        b = base[(base.model == model) & (base.pair == pair)]
        if r.empty or o.empty or b.empty:
            continue
        cells.append(dict(model=NAME[model], pair=pair, SSI=r.SSI.iloc[0], COV_shared_dims=o.shared_dims.iloc[0],
                          MLR_shared_dims=m.shared_dims.iloc[0], retrieval_MRR=b.retr_MRR.iloc[0], CKA=b.CKA.iloc[0]))
cells = save("geometry_function_cells.csv", pd.DataFrame(cells))
corr = {}
for x_name in ["COV_shared_dims", "MLR_shared_dims", "CKA"]:
    for y_name in ["SSI", "retrieval_MRR"]:
        ok = cells[[x_name, y_name]].dropna()
        rho = spearmanr(ok[x_name], ok[y_name])
        # Within-model correlation (pairs vary, model fixed), averaged.
        within = np.nanmean([spearmanr(g[x_name], g[y_name]).statistic for _, g in cells.groupby("model")])
        corr[f"{x_name}~{y_name}"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(ok),
                                          mean_within_model_rho=float(within))
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), layout="constrained")
for ax, (y_name, ylab) in zip(axes, [("SSI", "functional SSI (MLR), F"), ("retrieval_MRR", "pair retrieval MRR")]):
    for model in MODELS:
        d = cells[cells.model.eq(NAME[model])]
        ax.scatter(d.COV_shared_dims, d[y_name], color=COLOR[model], marker=MARK[GROUP[model]], s=16)
    k = corr[f"COV_shared_dims~{y_name}"]
    ax.set_xlabel("shared F-pattern dimensions (COV, of 11)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{'a' if y_name == 'SSI' else 'b'}  {ylab.split(',')[0]}", loc="left")
    ax.text(.03, .95, f"Spearman ρ = {k['rho']:.2f} (66 cells)\nmean within-model ρ = {k['mean_within_model_rho']:.2f}",
            transform=ax.transAxes, va="top", fontsize=6.8)
axes[1].legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
savefig(fig, "fig9_geometry_predicts_sharing")

# ---------------------------------------------------------------------------
# 10. Floors, denominators and item-level predictors
# ---------------------------------------------------------------------------
tr = load("transparency_group_decomposition.csv")
tr["model"] = tr.model.map(NAME)
save("transparency_decomposition.csv", tr)
mods = load("posthoc_complete_case_moderators.csv")
mods["model"] = mods.model.map(NAME)
keep_terms = ["within_correct", "cp_transparency", "idiom_len", "token_count", "rare", "religious", "cultural"]
mt = mods[mods.term.isin(keep_terms)][["model", "concept", "term", "odds_ratio_per_sd", "odds_ratio_ci_lo",
                                        "odds_ratio_ci_hi", "p", "p_holm_all_terms"]]
save("moderator_odds_ratios.csv", mt)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.25]))
d = tr[tr.concept.eq("F") & tr.group.isin(["transparent", "opaque"])]
xx = np.arange(len(MODELS))
for j, (group, color) in enumerate([("opaque", TRANSPARENCY_COLOR["opaque"]), ("transparent", TRANSPARENCY_COLOR["transparent"])]):
    g = d[d.group.eq(group)].set_index("model").loc[[NAME[m] for m in MODELS]]
    axes[0].bar(xx + (j - .5) * .38, 100 * (g.R_ep), .36, color=color, alpha=.85, label=f"{group} counterparts")
    axes[0].plot(xx + (j - .5) * .38, 100 * g.R0, "k_", ms=9, mew=1.5)
    for xi, s in zip(xx + (j - .5) * .38, g.SSI):
        axes[0].text(xi, 1, "—" if not np.isfinite(s) else f"{s:.2f}", ha="center", fontsize=5.8, color="white", rotation=90)
axes[0].set_xticks(xx, [NAME[m] for m in MODELS], rotation=30, ha="right")
axes[0].set_ylabel("F accuracy (%); bar text = SSI")
axes[0].set_title("a  Transparency: accuracy ↑, SSI ↓ (—: floor)", loc="left", pad=16)
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=6, ncol=2, borderaxespad=0.1)
terms = ["within_correct", "cp_transparency", "idiom_len", "rare"]
labels = {"within_correct": "native readout correct", "cp_transparency": "counterpart transparency (SD)",
          "idiom_len": "idiom length (SD)", "rare": "rare idiom"}
for i, term in enumerate(terms):
    for concept, off, mk in [("F", -.12, "o"), ("S", .12, "s")]:
        g = mt[(mt.term == term) & (mt.concept == concept)].set_index("model").loc[[NAME[m] for m in MODELS]]
        for j, model in enumerate(MODELS):
            r = g.loc[NAME[model]]
            yv = i + off + (j - 2.5) * .03
            axes[1].plot([r.odds_ratio_ci_lo, r.odds_ratio_ci_hi], [yv, yv], color=COLOR[model], lw=.7)
            axes[1].plot(r.odds_ratio_per_sd, yv, marker=mk, color=COLOR[model], ms=3.5, mfc="white" if concept == "S" else COLOR[model])
axes[1].axvline(1, color=INK, lw=.6)
axes[1].set_xscale("log")
axes[1].set_yticks(range(len(terms)), [labels[t] for t in terms])
axes[1].invert_yaxis()
axes[1].set_xlabel("odds ratio, EN-direction success (log; ● F, □ S)")
axes[1].set_title(f"b  Item-level predictors ({POSTHOC} GLM)", loc="left", pad=22)
savefig(fig, "fig10_floors_and_item_predictors")

# ---------------------------------------------------------------------------
# 11. Decoder (exploratory)
# ---------------------------------------------------------------------------
g = load("decoder_S1_S3_grid.csv")
g = g[g.group.eq("all")] if "group" in g else g
qrows = []
for concept in ["F", "S"]:
    for pair, label in [("en-para→bn-para", "gloss → gloss"), ("en-trans→bn-fig", "translation → idiom in context"),
                        ("en-idiom→bn-fig", "counterpart → idiom in context"), ("bn-fig→en-idiom", "idiom in context → counterpart")]:
        r = dec[(dec.concept == concept) & (dec.pair == pair)]
        if r.empty:
            continue
        r = r.iloc[0]
        qrows.append(dict(concept=concept, pair=label, M=r.M, cross_R=100 * r.R_ep, native_R=100 * r.R_pp,
                          floor=100 * r.R0, SSI=r.SSI, ci_lo=r.ci_lo, ci_hi=r.ci_hi, null_draws=r.RAND_draws))
qrows = save("decoder_ladder.csv", pd.DataFrame(qrows))
md(qrows, "Qwen3-8B (exploratory; layer 16, MLR rank 3; 9 primary null draws, 3 diagnostic)", 3)
fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.1))
for meth, color in [("MLR", METHOD_COLOR["MLR"]), ("PCA", METHOD_COLOR["PCA"]), ("RAND", METHOD_COLOR["RAND"])]:
    cr = g[g.source_condition.eq("en-idiom") & g.condition.eq("bn-fig") & g.method.eq(meth) & g.M.eq(3) & g.concept.eq("F")].sort_values("layer")
    na = g[g.source_condition.eq("bn-fig") & g.condition.eq("bn-fig") & g.method.eq(meth) & g.M.eq(3) & g.concept.eq("F")].sort_values("layer")
    ax.plot(cr.layer, 100 * (cr.R - cr.R0), color=color, marker="o", ms=2.5, lw=1, label=f"{meth} EN→BN")
    if meth == "MLR":
        ax.plot(na.layer, 100 * (na.R - na.R0), color=color, marker="o", ms=2.5, lw=1, ls="--", label="MLR BN→BN")
ax.axhline(0, color=INK, lw=.6)
ax.set_xlabel("Qwen3-8B layer")
ax.set_ylabel("F accuracy above floor (pp)")
# Above the axes: inside, the native curve runs through the legend.
ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2,
          handlelength=2.6)                     # long enough to show the native curve's dashes
savefig(fig, "figA2_decoder_layers")

# ---------------------------------------------------------------------------
# Primary table (unchanged confirmatory family) for convenience
# ---------------------------------------------------------------------------
prim = load("primary_confirmatory.csv")
p = prim[["model", "concept", "layer", "M", "R_ep", "R_pp", "R0", "SSI", "ci_lo", "ci_hi", "RAND_R_mean", "PCA_R_mean",
          "PERM_R_mean", "RAND_p_holm_all_jobs", "PERM_p_holm_all_jobs", "both_nulls_reject"]].copy()
p["model"] = p.model.map(NAME)
for c in ["R_ep", "R_pp", "R0", "RAND_R_mean", "PCA_R_mean", "PERM_R_mean"]:
    p[c] = 100 * p[c]
p = p.set_index("model").loc[[NAME[m] for m in MODELS]].reset_index().sort_values(["concept"], kind="stable")
md(p, "Planned confirmatory family (unchanged): MLR, English counterpart → Bangla idiom in context", 3)

summary = dict(spearman=corr,
               class_transfer_vs_neutral_share=dict(rho=float(rho_neutral.statistic), p=float(rho_neutral.pvalue)),
               class_transfer_vs_transparency=dict(rho=float(rho_transp.statistic), p=float(rho_transp.pvalue)),
               class_transfer_vs_size=dict(rho=float(rho_size.statistic), p=float(rho_size.pvalue)))
if not HOUSE:
    (OUT / "synthesis_summary.json").write_text(json.dumps(summary, indent=2))
    (FIG / "tables.md").write_text("# Generated tables\n\n" + "\n".join(TABLES), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
