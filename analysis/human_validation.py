"""Agreement and class-level summaries of the two volunteers' judgments (300 TEST items, 25 per F class).

    python analysis/human_validation.py [--out results/human_validation]

Reads data/human_validation/annotator_{A,B}.csv (answers entered in spreadsheets; no session times exist) and
writes judgments.csv (the answers without corpus text), agreement.csv and classes.csv. Agreement follows the plan:
unweighted Cohen's kappa with a 2,000-draw item bootstrap (seed 20261024), `uncertain` kept as a category, plus
Gwet's AC1. Everything class-level (shares per F class beside the class's English-direction transfer from
results/discovery/synthesis/class_transfer.csv, and the Spearman correlations printed at the end) is exploratory.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[1]
SHEETS = REPO / "data" / "human_validation"
SYNTHESIS = REPO / "results" / "discovery" / "synthesis"
FIELDS = {"f_plausibility": ["good", "partial", "poor", "uncertain"],
          "function": ["referential", "evaluative", "both", "uncertain"],
          "counterpart_type": ["idiom", "proverb", "literal_paraphrase", "other", "uncertain"],
          "counterpart_equivalence": ["equivalent", "partly_equivalent", "not_equivalent", "uncertain"]}
SOURCE = ["idiom", "bangla_example", "english_example", "bangla_figurative_gloss", "english_figurative_gloss",
          "english_counterpart", "assigned_F", "assigned_F_name"]
DRAWS, SEED = 2000, 20261024


def load_sheets():
    sheets = {}
    for code in "AB":
        d = pd.read_csv(SHEETS / f"annotator_{code}.csv", keep_default_na=False, encoding="utf-8-sig")
        assert d.collection_method.eq("spreadsheet").all() and d.annotation_timing_status.eq("unavailable").all()
        sheets[code] = d.set_index("sample_id").sort_index()
    for column in SOURCE:
        assert sheets["A"][column].astype(str).equals(sheets["B"][column].astype(str)), f"source differs: {column}"
    for code, d in sheets.items():
        assert len(d) == 300 and d.index.is_unique
        for field, allowed in FIELDS.items():
            assert d[field].isin(allowed).all(), (code, field)
    a, b = sheets["A"], sheets["B"]
    assert a.index.equals(b.index), "the two sheets must hold the same items"
    assert a.assigned_F.value_counts().eq(25).all() and a.assigned_F.nunique() == 12      # 25 items per F class
    return sheets


def agreement(a, b):
    rng = np.random.default_rng(SEED)
    n = len(a)
    rows = []
    for field in FIELDS:
        x, y = a[field].to_numpy(), b[field].to_numpy()
        boot = []
        for _ in range(DRAWS):
            ix = rng.integers(0, n, n)
            value = cohen_kappa_score(x[ix], y[ix])
            if np.isfinite(value):
                boot.append(value)
        lo, hi = np.quantile(boot, [.025, .975])
        cats = sorted(set(x) | set(y))
        pa = float(np.mean(x == y))
        shares = np.array([(np.mean(x == c) + np.mean(y == c)) / 2 for c in cats])
        pe = float((shares * (1 - shares)).sum() / (len(cats) - 1)) if len(cats) > 1 else 0.0
        rows.append(dict(field=field, agree=pa, kappa=float(cohen_kappa_score(x, y)), lo=float(lo), hi=float(hi),
                         ac1=(pa - pe) / (1 - pe)))
    return pd.DataFrame(rows)


def classes(a, b):
    item = pd.DataFrame(index=a.index)
    item["F"] = a.assigned_F
    item["name"] = a.assigned_F_name
    mean2 = lambda mask_a, mask_b: (mask_a.astype(int) + mask_b.astype(int)) / 2  # noqa: E731
    item["good"] = mean2(a.f_plausibility.eq("good"), b.f_plausibility.eq("good"))
    item["poor"] = mean2(a.f_plausibility.eq("poor"), b.f_plausibility.eq("poor"))
    item["evaluative"] = mean2(a.function.isin(["evaluative", "both"]), b.function.isin(["evaluative", "both"]))
    item["idiom_cp"] = mean2(a.counterpart_type.eq("idiom"), b.counterpart_type.eq("idiom"))
    item["equivalent"] = mean2(a.counterpart_equivalence.eq("equivalent"), b.counterpart_equivalence.eq("equivalent"))
    table = item.groupby(["F", "name"]).mean().reset_index()
    transfer = pd.read_csv(SYNTHESIS / "class_transfer.csv")[["label", "within_recall", "cross_recall", "transfer_ratio"]]
    transfer = transfer.rename(columns={"label": "F"})
    return table.merge(transfer, on="F", validate="one_to_one").sort_values("transfer_ratio", ascending=False)


def partial_spearman(x, y, given):
    """Spearman correlation of x and y with the ranks of `given` regressed out of both."""
    rx, ry, rg = (pd.Series(np.asarray(v, float)).rank().to_numpy() for v in (x, y, given))
    resid = lambda r: r - np.polyval(np.polyfit(rg, r, 1), rg)  # noqa: E731
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "human_validation")
    args = parser.parse_args(argv)
    sheets = load_sheets()
    a, b = sheets["A"], sheets["B"]
    agree = agreement(a, b)
    per_class = classes(a, b)
    judgments = pd.DataFrame({"sample_id": a.index, "assigned_F": a.assigned_F.to_numpy()})
    for code, d in sheets.items():
        for field in FIELDS:
            judgments[f"{field}_{code}"] = d.loc[a.index, field].to_numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    judgments.to_csv(args.out / "judgments.csv", index=False)
    agree.to_csv(args.out / "agreement.csv", index=False)
    per_class.to_csv(args.out / "classes.csv", index=False)

    print(agree.round(3).to_string(index=False))
    both = lambda field, value: float(np.mean(a[field].eq(value) & b[field].eq(value)))  # noqa: E731
    print(f"\nBoth volunteers (share of {len(a)} items): class good {both('f_plausibility', 'good'):.2f}, "
          f"class poor {both('f_plausibility', 'poor'):.2f}, referential {both('function', 'referential'):.2f}, "
          f"evaluative {both('function', 'evaluative'):.2f}, counterpart an idiom {both('counterpart_type', 'idiom'):.2f}, "
          f"counterpart equivalent {both('counterpart_equivalence', 'equivalent'):.2f}")
    print("\nExploratory, over the 12 F classes (Spearman):")
    for column in ["evaluative", "good", "idiom_cp"]:
        r = spearmanr(per_class[column], per_class.transfer_ratio)
        print(f"  transfer ratio vs {column:10s} rho {r.statistic:+.2f}  p {r.pvalue:.2f}")
    print(f"  transfer ratio vs evaluative, net of native recall: partial rho "
          f"{partial_spearman(per_class.evaluative, per_class.transfer_ratio, per_class.within_recall):+.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
