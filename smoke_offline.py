"""Exercise real extraction, analysis and zero-fit resumption with random weights.

This creates synthetic implementation evidence only. It downloads no models and
does not use, modify or score the research corpus.
"""
import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="out_offline_smoke",help="Dedicated synthetic validation directory")
    parser.add_argument("--probe-backend",default="auto",choices=["auto","torch","sklearn"],
                        help="Probe engine to exercise (torch also runs on CPU)")
    args=parser.parse_args(argv)
    root=Path(args.out).resolve()
    marker=root/"synthetic_validation.json"
    if root.exists() and any(root.iterdir()) and not marker.exists():
        parser.error("Use an empty output directory; this folder contains other files.")
    root.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({"kind":"synthetic random-weight validation; not research evidence"}),encoding="utf8")
    from transformers import BertConfig,BertModel,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    import torch
    from idiom_subspace.config import Config
    from idiom_subspace.runner import analysis_code,digest,run

    torch.manual_seed(42)
    model_dir=root/"synthetic-tiny-bert"
    if not model_dir.exists():
        model_dir.mkdir()
        vocab=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","expr","anger","wealth",
               "means","wrath","savings","fury","income","context","meaning"]+[str(i) for i in range(100)]
        backend=Tokenizer(WordLevel({w:i for i,w in enumerate(vocab)},unk_token="[UNK]"))
        backend.pre_tokenizer=Whitespace()
        backend.post_processor=TemplateProcessing(single="[CLS] $A [SEP]",special_tokens=[("[CLS]",2),("[SEP]",3)])
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token="[UNK]",pad_token="[PAD]",
            cls_token="[CLS]",sep_token="[SEP]",mask_token="[MASK]")
        tokenizer.save_pretrained(model_dir)
        BertModel(BertConfig(vocab_size=len(vocab),hidden_size=12,num_hidden_layers=2,
            num_attention_heads=3,intermediate_size=24,max_position_embeddings=64)).save_pretrained(model_dir)
    rows=[]
    for j in range(80):
        topic="wealth savings income" if j%2 else "anger fury wrath"
        rows.append(dict(id=str(j),idiom=f"expr {j}",figurative_meaning_en=f"{topic} meaning {j}",
            figurative_meaning_bn=f"{topic} meaning {j}",literal_meaning=f"expr {j}",
            example_sentences_in_bangla=[f"{topic} expr {j} context"],
            example_sentences_in_english=[f"{topic} expr {j} context"],similar_in_english=[f"expr {j}"],
            sentiment="positive" if j%2 else "negative",usage_domain=["daily life"],tags=["emotion"],
            frequency="common",cultural_significance=False,religious_significance=False,historical_significance=False))
    corpus=root/"synthetic_corpus.json";corpus.write_text(json.dumps(rows),encoding="utf8")
    cfg=Config(data=str(corpus),out=str(root/("analysis_"+digest(analysis_code())[:8])),
        models=[dict(name="synthetic-tiny-bert",revision="local-random")],
        split_fractions=[.4,.2,.2,.2],seeds=[3],label_seeds=[0],label_backend="tfidf",classes=2,
        conditions=["bn-fig","en-idiom"],pairs=[["en-idiom","bn-fig"],["bn-fig","en-idiom"]],
        concepts=["F","S"],methods=["RAND","PCA","MLR","CPCA","LEACE"],ranks=[1,2,3],
        pca_dim=None,layers=[0,1,2],device="cpu",precision="float32",max_length=64,batch_size=16,
        bootstrap=10,null_draws=3,diagnostic_null_draws=2,probe_Cs=[1.],min_group=100,
        open_class_N=[10,20],open_class_reps=1,studies=["S1","S2","S3","S4"],robustness=False,
        probe_backend=args.probe_backend,grid_rand_draws=2)
    cfg.validate()
    previous_cwd=Path.cwd()
    try:
        os.chdir(root)
        start=time.perf_counter();run(cfg);first_seconds=time.perf_counter()-start
        cfg.cache_only=True;start=time.perf_counter()
        with patch("transformers.AutoModel.from_pretrained",side_effect=AssertionError("resume loaded weights")), \
             patch("idiom_subspace.studies.fit_estimator",side_effect=AssertionError("resume refitted estimator")), \
             patch("idiom_subspace.studies.tune_probe",side_effect=AssertionError("resume refitted probe")),              patch("idiom_subspace.fastprobe.tune_readouts",side_effect=AssertionError("resume refitted batched probe")):
            run(cfg)
        report=dict(validation="synthetic random-weight model; not research findings",status="passed",
            probe_backend=args.probe_backend,
            first_seconds=first_seconds,resumed_seconds=time.perf_counter()-start,output=cfg.out,
            real_extraction=True,resume_loaded_weights=False,resume_refitted=False)
        (root/"validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
        print(json.dumps(report,indent=2))
    finally:os.chdir(previous_cwd)


if __name__=="__main__":main()
