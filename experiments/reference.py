"""Published numbers that the new runs must reproduce before their new outputs are used.

Split 0: the planned discovery family (results/discovery).
Split 1: the follow-up H1 tests and readout scores (results/replication).
"""
from __future__ import annotations

import pandas as pd

from .common import REPO

DISCOVERY = REPO / "results" / "discovery" / "primary_confirmatory.csv"
FOLLOWUP_TESTS = REPO / "results" / "replication" / "hypotheses.csv"
FOLLOWUP_SCORES = REPO / "results" / "replication" / "scores.csv"


def planned(split):
    """One row per (model, concept, method) with published accuracies, SSI and p values."""
    if split == 0 and DISCOVERY.exists():
        d = pd.read_csv(DISCOVERY)
        return pd.DataFrame(dict(model=d.model, concept=d.concept, method=d.method, ref_cross=d.R_ep,
                                 ref_native=d.R_pp, ref_floor=d.R0, ref_SSI=d.SSI, ref_RAND_p=d.RAND_p,
                                 ref_PERM_p=d.PERM_p))
    if split == 1 and FOLLOWUP_TESTS.exists():
        h = pd.read_csv(FOLLOWUP_TESTS)
        h = h[h.hypothesis.str.startswith("H1_")]
        rows = []
        for (model, concept), g in h.groupby(["model", "concept"]):
            r = g.set_index("hypothesis")
            rows.append(dict(model=model, concept=concept, method="COV",
                             ref_cross=r.cross_accuracy.iloc[0], ref_native=r.native_accuracy.iloc[0],
                             ref_floor=r["floor"].iloc[0], ref_SSI=r.estimate.iloc[0],
                             ref_RAND_p=r.p.get("H1_RAND"), ref_PERM_p=r.p.get("H1_PERM")))
        return pd.DataFrame(rows)
    return pd.DataFrame()


def scores(split):
    """Follow-up readout accuracies (split 1 only): method/source/representation -> accuracy."""
    if split == 1 and FOLLOWUP_SCORES.exists():
        return pd.read_csv(FOLLOWUP_SCORES)
    return pd.DataFrame()


def compare(table, split, on=("model", "concept", "method")):
    """Attach published values and absolute differences (accuracy, SSI, p)."""
    ref = planned(split)
    if ref.empty or table.empty:
        return table
    merged = table.merge(ref, on=list(on), how="left")
    if "acc" in merged:
        merged["diff_cross_acc"] = merged.acc - merged.ref_cross
        merged["diff_native_acc"] = merged.acc_native - merged.ref_native
        merged["diff_SSI_acc"] = merged.SSI_acc - merged.ref_SSI
    if "RAND_p_acc" in merged:
        merged["diff_RAND_p"] = merged.RAND_p_acc - merged.ref_RAND_p
        merged["diff_PERM_p"] = merged.PERM_p_acc - merged.ref_PERM_p
    return merged
