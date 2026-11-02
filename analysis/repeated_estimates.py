"""Compare every repeated estimate in the released results with the one the paper reports.

    python analysis/repeated_estimates.py [--out results/repeated_estimates.csv]

Several experiments re-evaluate a baseline that an earlier analysis already reported, and solver and batching
tolerances make the two differ slightly. The paper reports the first analysis's estimate of each quantity, so a
repeated quantity has one value throughout. This script reads only the released tables and writes one row per
comparison: the reported value and its file, the re-evaluated value and its file, and their gap. It also prints the
largest gap per kind of readout and the erasure medians the paper reports.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SHORT = {"csebuetnlp/banglabert": "BanglaBERT", "csebuetnlp/banglishbert": "BanglishBERT",
         "bert-base-multilingual-cased": "mBERT", "xlm-roberta-large": "XLM-R",
         "google/muril-large-cased": "MuRIL", "setu4993/LaBSE": "LaBSE"}
MODELS = list(SHORT.values())
RANK = {"F": 11, "S": 3}
KEY = ["model", "concept", "method"]
ESTIMATORS = "discovery/synthesis/estimator_families.csv"
ERASURE = "discovery/synthesis/erasure_transfer.csv"
INFERENCE = "discovery/all_selected_layer_inference.csv"
X1 = "experiments/X1_zero_shot/summary.csv"
X4 = "experiments/X4_pooling_matched/cells.csv"
X5 = "experiments/X5_balanced_metrics/tests.csv"
X7 = "experiments/X7_alignment/aligned.csv"
X8 = "experiments/X8_layerwise/layers.csv"
X9 = "experiments/X9_erasers_nonlinear/summary.csv"
X10 = "experiments/X10_heldout_alignment/aligned.csv"
X12 = "experiments/X12_penalty_path/path.csv"
X13 = "experiments/X13_idiom_isolated/tests.csv"
X13_ABSOLUTE = "experiments/X13_idiom_isolated/absolute.csv"
X15 = "experiments/X15_matched_nulls/tests.csv"
X19 = "experiments/X19_counterpart_type/subsets.csv"
E1 = "replication/scores.csv"
E2_E3 = "replication/mechanisms.csv"
E4 = "replication/isotropy.csv"
READOUT, FULL_SPACE, ERASE, RATIO = "subspace readout", "full-space probe", "erasure drop", "SSI or share"


def read(name):
    """A released table, with models under the short names of the synthesis tables."""
    table = pd.read_csv(RESULTS / name)
    if "model" in table:
        table["model"] = table.model.map(lambda m: SHORT.get(m, m))
    return table


class Comparisons:
    def __init__(self):
        self.rows = []

    def add(self, kind, quantity, split, target, key, source, value, other_source, other, scale=100.0):
        """One repeated quantity; `scale` puts accuracy gaps in points. Undefined values have nothing to compare."""
        if np.isfinite(value) and np.isfinite(other):
            self.rows.append(dict(kind=kind, quantity=quantity, split=split, target=target, model=key[0],
                                  concept=key[1], method=key[2], reported_from=f"results/{source}", reported=value,
                                  rerun_from=f"results/{other_source}", rerun=other, gap=scale * abs(value - other)))


def in_context(out):
    """Idioms in context, split seed 0: accuracies against the discovery tables, balanced accuracy against X5."""
    est = read(ESTIMATORS)
    est = est[est.M.eq(est.concept.map(RANK))].set_index(KEY)
    bal = read(X5)
    bal = bal[bal.split.eq(0)].set_index(KEY)
    accuracy = {"accuracy through English directions": ("cross_R", READOUT),
                "accuracy through Bangla directions": ("native_R", READOUT),
                "full-space accuracy": ("ambient_R", FULL_SPACE)}
    balanced = {"balanced accuracy through English directions": "bal",
                "balanced accuracy through Bangla directions": "bal_native"}
    x1 = read(X1)
    x1 = x1[x1.source.eq("en-idiom") & x1.target.eq("bn-fig") & x1.method.isin(["MLR", "COV", "LEACE"])]
    x4 = read(X4)
    x8 = read(X8)
    x13 = read(X13_ABSOLUTE)
    x15 = read(X15)
    x19 = read(X19)
    x19 = x19[x19["subset"].eq("all")]
    x5 = bal.reset_index()
    reruns = [
        (X1, x1.drop_duplicates(KEY), dict(zip([*accuracy, *balanced], ["supervised_cross_acc", "native_acc", None,
                                                                        "supervised_cross_bal", "native_bal"]))),
        (X4, x4[x4.condition.eq("bn-fig")], dict(zip([*accuracy, *balanced], ["acc", "acc_native", "ambient_native_acc",
                                                                              "bal", "bal_native"]))),
        (X5, x5, dict(zip(accuracy, ["acc", "acc_native", None]))),
        (X7, read(X7).drop_duplicates(KEY), dict(zip([*accuracy, *balanced], ["published_cross_acc", None, "ambient_acc",
                                                                              "published_cross_bal", None]))),
        (X8, x8[x8.split.eq(0) & x8.is_selected.astype(bool)], dict(zip([*accuracy, *balanced],
                                                                        ["acc", "acc_native", None, "bal", "bal_native"]))),
        (X10, read(X10).query("split == 0").drop_duplicates(KEY),
         dict(zip([*accuracy, *balanced], ["published_cross_acc", None, "ambient_acc", "published_cross_bal", None]))),
        (X13_ABSOLUTE, x13[x13.target.eq("bn-fig")],
         dict(zip([*accuracy, *balanced], ["cross_acc", "native_acc", "ambient_acc", "cross_bal", "native_bal"]))),
        (X15, x15[x15.target.eq("bn-fig")], dict(zip([*accuracy, *balanced], ["acc", "acc_native", None, "bal", "bal_native"]))),
        (X19, x19[x19.target.eq("bn-fig")],
         dict(zip([*accuracy, *balanced], ["cross_acc", "native_acc", None, "cross_bal", "native_bal"]))),
    ]
    for source, table, columns in reruns:
        for row in table.to_dict("records"):
            key = tuple(row[k] for k in KEY)
            for quantity, column in columns.items():
                if column is None:
                    continue
                if quantity in accuracy and key in est.index:
                    reported, kind = accuracy[quantity]
                    out.add(kind, quantity, 0, "bn-fig", key, ESTIMATORS, est.loc[key, reported], source, row[column])
                elif quantity in balanced and key in bal.index:
                    out.add(READOUT, quantity, 0, "bn-fig", key, X5, bal.loc[key, balanced[quantity]], source, row[column])


def idiom_alone(out):
    """The idiom alone, split seed 0: X13 first reported it; the accuracy SSI of MLR comes from the discovery tables."""
    alone = read(X13).set_index(KEY)
    x15 = read(X15)
    x19 = read(X19)
    x19 = x19[x19["subset"].eq("all")]
    columns = {"accuracy through English directions": ("acc", "cross_acc"),
               "accuracy through Bangla directions": ("acc_native", "native_acc"),
               "balanced accuracy through English directions": ("bal", "cross_bal"),
               "balanced accuracy through Bangla directions": ("bal_native", "native_bal")}
    for quantity, (reported, in_x19) in columns.items():
        for source, table, column in [(X15, x15[x15.target.eq("bn-idiom")], reported),
                                      (X19, x19[x19.target.eq("bn-idiom")], in_x19)]:
            for row in table.to_dict("records"):
                key = tuple(row[k] for k in KEY)
                out.add(READOUT, quantity, 0, "bn-idiom", key, X13, alone.loc[key, reported], source, row[column])
    pairs = read(INFERENCE)
    pairs = pairs[pairs.group.eq("all") & pairs.method.eq("MLR") & pairs.source_condition.eq("en-idiom")
                  & pairs.condition.eq("bn-idiom")].set_index(KEY)
    for key in pairs.index:
        out.add(RATIO, "accuracy SSI", 0, "bn-idiom", key, INFERENCE, pairs.loc[key, "SSI"], X13, alone.loc[key, "SSI_acc"], 1.0)


def erasure(out):
    """Erasers without a rotation against the discovery tables; the held-out rotation against X10, its first report."""
    er = read(ERASURE).set_index(["model", "concept"])
    x9 = read(X9)
    raw = x9[x9.condition.eq("raw") & x9.eraser.isin(["COV", "LEACE", "MLR"])]
    for row in raw.to_dict("records"):
        key = (row["model"], row["concept"], row["eraser"])
        c = er.loc[(row["model"], row["concept"])]
        out.add(ERASE, "accuracy removed by the English-fitted eraser", 0, "bn-fig", key, ERASURE,
                100 * (c.ambient - c[f"english_{row['eraser']}_leak"]), X9, row["drop_pp"], 1.0)
        out.add(ERASE, "accuracy removed by the Bangla-fitted eraser", 0, "bn-fig", key, ERASURE,
                100 * (c.ambient - c[f"native_{row['eraser']}_leak"]), X9, row["native_drop_pp"], 1.0)
    for row in raw.drop_duplicates(["model", "concept"]).to_dict("records"):
        out.add(FULL_SPACE, "full-space accuracy", 0, "bn-fig", (row["model"], row["concept"], "AMBIENT"), ERASURE,
                er.loc[(row["model"], row["concept"]), "ambient"], X9, row["ambient_acc"])
    rotated = read(X10)
    rotated = rotated[rotated.split.eq(0) & rotated.fit_on.eq("probe_validation") & rotated.alignment.eq("procrustes")
                      & rotated.shuffle_draw.eq(0)].set_index(KEY)
    for row in x9[x9.condition.eq("procrustes") & x9.eraser.isin(["COV", "LEACE"])].to_dict("records"):
        key = (row["model"], row["concept"], row["eraser"])
        out.add(RATIO, "erase share after the held-out rotation", 0, "bn-fig", key, X10,
                rotated.loc[key, "erase_share_aligned"], X9, row["erase_share"], 1.0)


def zero_shot(out):
    """The unpenalized settings of the penalty path repeat X1's zero-shot SSIs."""
    x1 = read(X1)
    x1 = x1[x1.source.eq("en-idiom") & x1.target.eq("bn-fig") & x1.centering.eq("per_condition")].set_index(KEY)
    x12 = read(X12)
    x12 = x12[x12.split.eq(0) & x12.setting.isin(["C=1", "ridge=0", "COV"])]
    for row in x12.to_dict("records"):
        key = tuple(row[k] for k in KEY)
        out.add(RATIO, "zero-shot SSI", 0, "bn-fig", key, X1, x1.loc[key, "SSI_zs_acc"], X12, row["SSI_zs_acc"], 1.0)


def seed_one(out):
    """Split seed 1: E2's default MLR, E3's English covariance and E4's baseline repeat E1's readouts, as X5 does."""
    e1 = read(E1)
    e1 = e1[e1.source.eq("en-idiom") & e1.target.eq("bn-fig")].set_index([*KEY, "representation"]).accuracy
    mechanisms = read(E2_E3)
    for row in mechanisms[mechanisms.setting.eq("C=1")].to_dict("records"):
        key = (row["model"], row["concept"], "MLR")
        out.add(READOUT, "accuracy through English directions", 1, "bn-fig", key, E1, e1[(*key, "project")], E2_E3, row["accuracy"])
    for row in mechanisms[mechanisms.setting.eq("english")].to_dict("records"):
        key = (row["model"], row["concept"], "LEACE")
        kind, quantity = (READOUT, "accuracy through English directions") if row["representation"] == "project" else (
            ERASE, "accuracy left by the English-fitted eraser")
        out.add(kind, quantity, 1, "bn-fig", key, E1, e1[(*key, row["representation"])], E2_E3, row["accuracy"])
    isotropy = read(E4)
    for row in isotropy[isotropy.setting.eq("baseline") & isotropy.source.eq("en-idiom")].to_dict("records"):
        key = (row["model"], row["concept"], row["method"])
        out.add(READOUT, "accuracy through English directions", 1, "bn-fig", key, E1, e1[(*key, "project")], E4, row["accuracy"])
    x5 = read(X5)
    for row in x5[x5.split.eq(1)].to_dict("records"):
        key = tuple(row[k] for k in KEY)
        out.add(READOUT, "accuracy through English directions", 1, "bn-fig", key, E1, e1[(*key, "project")], X5, row["acc"])


def erasure_medians():
    """Medians over the six encoders of the accuracy points each eraser removes, as the paper reports them."""
    er = read(ERASURE).set_index("model")
    lines = []
    for concept in ["F", "S"]:
        d = er[er.concept.eq(concept)].loc[MODELS]
        for method in ["COV", "LEACE", "MLR"]:
            english = (100 * (d.ambient - d[f"english_{method}_leak"])).median()
            bangla = (100 * (d.ambient - d[f"native_{method}_leak"])).median()
            lines.append(f"  {concept} {method:<5} English-fitted {english:5.2f}, Bangla-fitted {bangla:5.2f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS / "repeated_estimates.csv")
    args = parser.parse_args()
    out = Comparisons()
    for section in [in_context, idiom_alone, erasure, zero_shot, seed_one]:
        section(out)
    table = pd.DataFrame(out.rows).sort_values(["kind", "quantity", "split", "target", "model", "concept", "method",
                                                "rerun_from"])
    table.to_csv(args.out, index=False, float_format="%.6f", lineterminator="\n")
    print(f"{len(table)} comparisons in {args.out}")
    print("Largest gap from the reported estimate (points; SSI and share units for the last group):")
    print(table.groupby("kind").gap.agg(["count", "max", "median"]).round(3).to_string())
    print("Accuracy points removed by each eraser, median over six encoders, discovery split:")
    print("\n".join(erasure_medians()))


if __name__ == "__main__":
    main()
