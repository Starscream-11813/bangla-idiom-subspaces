"""Covariate-only sampling checks for matching and annotation coverage."""
import numpy as np
import pandas as pd
from idiom_subspace.labels import annotation_sample
from idiom_subspace.studies import matched_anchor_masks, layer_ssi_table


def test_matching_balances_full_frequency_categories_without_outcomes():
    frame = pd.DataFrame(dict(anchored=[True]*6+[False]*4,
                              rare=[False]*6+[False]*4, idiom_len=[2]*10, token_count=[4]*10,
                              freq_raw=["common"]*4+["very common"]*2+["common"]*2+["very common"]*2))
    masks = matched_anchor_masks(frame, frame, seed=2)
    yes = frame.loc[masks["anchored_matched"], "freq_raw"].value_counts().to_dict()
    no = frame.loc[masks["unanchored_matched"], "freq_raw"].value_counts().to_dict()
    assert yes == no == {"common": 2, "very common": 2}
    changed = frame.assign(cross_correct=np.arange(10) % 2, F=np.arange(10))
    for key, mask in matched_anchor_masks(changed, changed, seed=2).items():
        np.testing.assert_array_equal(mask, masks[key])


def test_annotation_sample_keeps_rare_strata_and_never_duplicates():
    frame = pd.DataFrame(dict(F=[0]*50+[1]*2, anchored=[True]*50+[False]*2, rare=False,
                              idiom_id=np.arange(52)))
    sample = annotation_sample(frame, n=8, seed=2)
    assert len(sample) == 8 and sample.idiom_id.is_unique
    assert (~sample.anchored).sum() == 2
    assert sample.equals(annotation_sample(frame, n=8, seed=2))


def test_layer_indices_keep_absolute_scores_and_invalid_denominators():
    rows=[]
    for layer, within in [(0,.2),(1,.6)]:
        for method, score in [("MLR",.4),("RAND",.25),("PCA",.3)]:
            base=dict(model="m",layer=layer,concept="F",M=2,condition="bn-fig",group="all",method=method,R0=.2)
            rows.append(dict(base,source_condition="bn-fig",R=within))
            rows.append(dict(base,source_condition="en-idiom",R=score))
    result=layer_ssi_table(pd.DataFrame(rows))
    first=result[(result.layer==0)&(result.method=="MLR")].iloc[0]
    second=result[(result.layer==1)&(result.method=="MLR")].iloc[0]
    assert np.isnan(first.SSI) and first.R==.4 and first.R_pp==.2
    assert np.isclose(second.SSI,.5) and np.isclose(second.RAND_SSI_single_draw,.125)
