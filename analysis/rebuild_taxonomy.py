"""Rebuild the campaign's F taxonomy and export class-level aggregates only.

Uses the campaign code (commit 8d98424), split seed 0, label seed 0 and the
SPACE-fitted all-MiniLM-L6-v2 constructor. The rebuilt class counts are
checked against the campaign's SPACE/PROBE/TEST counts before anything is
written. Only aggregates and cluster keywords are saved (no corpus text).

    python analysis/rebuild_taxonomy.py /path/to/bagdhara.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from idiom_subspace.data import load_corpus, split_corpus  # noqa: E402
from idiom_subspace.labels import build_labels  # noqa: E402

# Campaign label_balance.csv (identical for every model).
EXPECTED = {
    "space": {10: 528, 3: 455, 7: 397, 11: 393, 0: 358, 1: 357, 5: 354, 6: 302, 2: 295, 4: 278, 8: 235, 9: 191},
    "probe": {10: 394, 3: 315, 11: 315, 0: 222, 7: 205, 6: 201, 1: 195, 5: 182, 2: 169, 4: 168, 8: 146, 9: 76},
    "test": {3: 401, 10: 360, 4: 226, 5: 223, 7: 219, 1: 217, 2: 217, 11: 206, 6: 200, 0: 148, 8: 127, 9: 46},
}
# Short names assigned from the SPACE keywords below (author-facing labels, not annotations).
NAMES = {0: "people & character", 1: "fear & agitation", 2: "deception & secrecy", 3: "work & tasks",
         4: "money & livelihood", 5: "beating & punishment", 6: "worthlessness & triviality",
         7: "insult, sarcasm & flattery", 8: "family & kinship", 9: "folly & cunning",
         10: "residual: time, place, objects", 11: "danger, death & fate"}
SENTIMENT = {0: "mixed", 1: "negative", 2: "neutral", 3: "positive"}


def main(data_path):
    raw, _ = load_corpus(data_path)
    split = split_corpus(raw, seed=0, fractions=(.40, .25, .10, .25), mode="input_disjoint")
    df, report, _ = build_labels(split, seed=0, k=12, backend="sentence-transformers",
                                 encoder="sentence-transformers/all-MiniLM-L6-v2", revision="main")
    for part, expected in EXPECTED.items():
        got = df[df.split.eq(part) & df.F.ge(0)].F.value_counts().to_dict()
        assert got == expected, f"rebuilt {part} counts differ from the campaign: {got}"
    keywords = {k["F"]: k["keywords"] for k in report["cluster_keywords"]}
    rows = []
    for c in range(12):
        members = df[df.F.eq(c)]
        test = members[members.split.eq("test")]
        s = members.S[members.S.ge(0)].map(SENTIMENT).value_counts(normalize=True)
        rows.append(dict(F=c, name=NAMES[c], space_n=int(members.split.eq("space").sum()), test_n=len(test),
                         keywords=", ".join(w for w in keywords[c] if len(w) > 1),
                         negative_share=float(s.get("negative", 0)), neutral_share=float(s.get("neutral", 0)),
                         positive_share=float(s.get("positive", 0)), mixed_share=float(s.get("mixed", 0)),
                         cp_transparency_mean=float(members.cp_transparency.mean()),
                         idiom_len_mean=float(members.idiom_len.mean()),
                         cultural_share=float(members.cultural.astype(float).mean()),
                         religious_share=float(members.religious.astype(float).mean())))
    out = REPO / "results" / "discovery" / "synthesis"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "F_taxonomy.csv", index=False)
    (out / "F_taxonomy_provenance.json").write_text(json.dumps(dict(
        code_commit="8d984246eb96bfeb8b061c7b4bbf90065cebb381", split_seed=0, label_seed=0,
        encoder="sentence-transformers/all-MiniLM-L6-v2",
        encoder_revision=report.get("label_encoder_resolved_revision"),
        counts_match_campaign=True, reseed_ARI=report.get("F_SPACE_reseed_ARI")), indent=2))
    print(pd.DataFrame(rows)[["F", "name", "space_n", "test_n", "neutral_share", "keywords"]].to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
