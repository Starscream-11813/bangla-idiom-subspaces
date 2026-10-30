"""Explicit, serializable experiment configuration. No installation on import."""
from dataclasses import asdict, dataclass, field
import json
import math
import re
from pathlib import Path


# input_disjoint (default): no idiom type, form variant or identical model input
# crosses splits. counterpart_closure (alias "strict"): additionally links every
# listed English counterpart transitively. idiom_only: Bangla types only.
SPLIT_MODES = {"input_disjoint", "strict", "counterpart_closure", "idiom_only"}


def model_slug(name):
    """Keep model output directories portable and inside their parent folder."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(". ") or "model"


@dataclass
class Config:
    data: str = "auto"
    out: str = "out_v3"
    controls: str | None = None
    # English literal glosses for L (FLS). CSV from `translate-literals` (approved=machine)
    # and/or human review (approved=true); None leaves L unavailable.
    literal_reviews: str | None = None
    literal_mt_model: str = "facebook/nllb-200-distilled-600M"
    feature_cache_dirs: list = field(default_factory=list)
    cache_only: bool = False
    extract_only: bool = False
    resume: bool = True
    progress: bool = True
    cpu_threads: int = 2
    split_mode: str = "input_disjoint"
    split_fractions: list = field(default_factory=lambda: [.40, .25, .10, .25])
    # One split and one taxonomy by default. Replicate refits are opt-in:
    # list more seeds here (or pass --seeds/--label-seeds) after a first full run.
    seeds: list = field(default_factory=lambda: [0])
    label_seeds: list = field(default_factory=lambda: [0])
    label_backend: str = "sentence-transformers"
    label_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    label_revision: str = "main"
    classes: int = 12
    models: list = field(default_factory=lambda: [
        {"name": "xlm-roberta-large", "revision": "main"},
        {"name": "bert-base-multilingual-cased", "revision": "main"},
        {"name": "setu4993/LaBSE", "revision": "main"},
        {"name": "google/muril-large-cased", "revision": "main"},
        {"name": "csebuetnlp/banglabert", "revision": "main", "normalization": "csebuetnlp"},
        {"name": "csebuetnlp/banglishbert", "revision": "main", "normalization": "csebuetnlp"}])
    conditions: list = field(default_factory=lambda: ["bn-fig", "bn-sentence", "bn-idiom", "bn-context",
                                                    "bn-para", "en-idiom", "en-trans", "en-para"])
    pairs: list = field(default_factory=lambda: [
        ["en-idiom", "bn-fig"], ["bn-fig", "en-idiom"],
        ["en-idiom", "bn-idiom"], ["bn-idiom", "en-idiom"],
        ["en-trans", "bn-fig"], ["bn-fig", "en-trans"],
        ["en-para", "bn-para"], ["bn-para", "en-para"],
        ["en-idiom", "bn-context"], ["en-idiom", "bn-para"], ["en-idiom", "bn-sentence"],
        ["en-idiom", "bn-lit"], ["en-idiom", "bn-ctrl"],
        ["en-idiom", "bn-rom"], ["natural-en", "bn-fig"],
        ["natural-hi", "bn-fig"], ["natural-ur", "bn-fig"], ["natural-kok", "bn-fig"]])
    concepts: list = field(default_factory=lambda: ["F", "S"])
    methods: list = field(default_factory=lambda: ["RAND", "PCA", "MLR", "LDA", "CPCA", "COV", "LEACE"])
    ranks: list = field(default_factory=lambda: [1, 2, 3, 4, 8, 11])
    pca_dim: int | None = None
    layer_step: int = 3
    layers: list | None = None
    max_length: int = 128
    batch_size: int = 8
    device: str = "auto"
    precision: str = "auto"
    bootstrap: int = 1000
    null_draws: int = 1999
    diagnostic_null_draws: int = 19
    inference_groups: list = field(default_factory=lambda: ["all"])
    primary_test_groups: list = field(default_factory=lambda: ["all"])
    primary_test_pairs: list = field(default_factory=lambda: [["en-idiom", "bn-fig"]])
    primary_test_concepts: list = field(default_factory=lambda: ["F", "S"])
    permutation_unit: str = "component"
    min_permutation_movable_fraction: float = .5
    compute_leakage: bool = True
    estimator_ridge: float = 0.0
    estimator_C: float = 1.0
    # "reference" reproduces concept_space MLR (no intercept, unit-norm class
    # weight rows before the SVD); "centered" is the earlier v3 variant.
    mlr_variant: str = "reference"
    min_den: float = .02
    min_group: int = 20
    probe_Cs: list = field(default_factory=lambda: [1.])
    # Probe engine: "auto" uses batched PyTorch L-BFGS on CUDA when available,
    # otherwise scikit-learn. Both minimise scikit-learn's L2 logistic loss.
    probe_backend: str = "auto"
    probe_device: str = "auto"
    probe_max_iter: int = 3000
    probe_batch_compact: int = 512
    probe_batch_ambient: int = 32
    estimator_batch: int = 32
    # Random-subspace grid baseline averaged over several draws (reference: 5).
    grid_rand_draws: int = 5
    # Optional separate budget for label-permutation draws (default: null_draws).
    perm_null_draws: int | None = None
    open_class_N: list = field(default_factory=lambda: [50, 100, 200, 400, 800, 1600, 3200])
    open_class_reps: int = 3
    open_class_methods: list = field(default_factory=lambda: ["MLR", "LEACE", "RAND", "PCA"])
    open_class_conditions: list = field(default_factory=lambda: ["bn-fig", "bn-idiom", "en-idiom"])
    # LEACE/LDA covariance shrinkage for the small estimation sets of the open-class sweep.
    open_class_ridge: str | float = "ledoit_wolf"
    # Identity open-class study (needs the bn-fig2 condition; see configs/identity_openclass.json).
    identity_N: list = field(default_factory=lambda: [50, 100, 200, 400, 800, 1600])
    identity_ranks: list = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    identity_reps: int = 3
    identity_methods: list = field(default_factory=lambda: ["LDA", "LEACE", "CPCA", "COV", "PCA", "RAND"])
    # Two contexts per idiom: spans (largely lexical identity) and idiom-masked contexts (semantic identity).
    identity_pairs: list = field(default_factory=lambda: [["bn-fig", "bn-fig2"], ["bn-context", "bn-context2"]])
    inference_method: str = "MLR"
    robustness: bool = True
    studies: list = field(default_factory=lambda: ["S1", "S2", "S3", "S4", "robustness"])

    @classmethod
    def load(cls, path=None):
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self):
        numeric = list(self.split_fractions) + list(self.probe_Cs) + [self.estimator_ridge, self.estimator_C,
            self.min_den, self.min_permutation_movable_fraction]
        if any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) for v in numeric):
            raise ValueError("Configuration numbers must be finite")
        for key in ["classes", "batch_size", "max_length", "layer_step", "bootstrap", "null_draws",
                    "diagnostic_null_draws", "cpu_threads", "min_group", "open_class_reps", "probe_max_iter",
                    "probe_batch_compact", "probe_batch_ambient", "estimator_batch", "grid_rand_draws", "identity_reps"]:
            value=getattr(self,key)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(f"{key} must be a positive integer")
        if self.perm_null_draws is not None and (not isinstance(self.perm_null_draws, int)
                                                 or isinstance(self.perm_null_draws, bool) or self.perm_null_draws < 1):
            raise ValueError("perm_null_draws must be null or a positive integer")
        if self.probe_backend not in {"auto", "torch", "sklearn"}:
            raise ValueError("probe_backend must be auto, torch or sklearn")
        if self.mlr_variant not in {"reference", "centered"}:
            raise ValueError("mlr_variant must be reference or centered")
        if not (self.open_class_ridge == "ledoit_wolf" or (isinstance(self.open_class_ridge, (int, float))
                and not isinstance(self.open_class_ridge, bool) and self.open_class_ridge >= 0)):
            raise ValueError("open_class_ridge must be 'ledoit_wolf' or a nonnegative number")
        for key in ["seeds", "label_seeds"]:
            values=getattr(self,key)
            if len(set(values))!=len(values) or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in values):
                raise ValueError(f"{key} must contain distinct nonnegative integers")
        if self.min_den < 0:
            raise ValueError("min_den must be nonnegative")
        if len(self.split_fractions) != 4 or abs(sum(self.split_fractions) - 1) > 1e-8:
            raise ValueError("split_fractions must sum to one: space/probe/validation/test")
        if min(self.split_fractions) <= 0 or self.split_mode not in SPLIT_MODES:
            raise ValueError(f"Invalid split configuration; split_mode must be one of {sorted(SPLIT_MODES)}")
        if not self.seeds or not self.label_seeds or self.classes < 2:
            raise ValueError("At least one split seed, label seed and two classes required")
        if self.batch_size < 1 or self.max_length < 4 or self.layer_step < 1:
            raise ValueError("Invalid extraction settings")
        if self.bootstrap < 1 or self.null_draws < 1:
            raise ValueError("bootstrap and null_draws must be positive")
        if self.cpu_threads < 1 or not self.inference_groups:
            raise ValueError("cpu_threads must be positive and inference_groups nonempty")
        if self.permutation_unit not in {"component", "idiom"}:
            raise ValueError("permutation_unit must be component or idiom")
        if not 0 <= self.min_permutation_movable_fraction <= 1:
            raise ValueError("min_permutation_movable_fraction must be between zero and one")
        if self.estimator_ridge < 0 or self.estimator_C<=0 or not self.probe_Cs or min(self.probe_Cs) <= 0:
            raise ValueError("Invalid estimator ridge or probe regularization")
        if self.diagnostic_null_draws < 1 or not set(self.primary_test_groups)<=set(self.inference_groups):
            raise ValueError("Diagnostic null draws must be positive; primary groups must be included in inference_groups")
        if not self.ranks or any(not isinstance(r, int) or isinstance(r, bool) or r < 1 for r in self.ranks):
            raise ValueError("ranks must contain positive integers")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("ranks must be distinct")
        supported = {"RAND", "PCA", "MLR", "LDA", "CPCA", "COV", "LEACE"}
        if not self.methods or not set(self.methods) <= supported or self.inference_method not in self.methods:
            raise ValueError("Unknown method or inference_method missing from methods")
        if self.pca_dim is not None and (not isinstance(self.pca_dim, int) or self.pca_dim < 1):
            raise ValueError("pca_dim must be null or a positive integer")
        if self.layers is not None and (not self.layers or any(not isinstance(v, int) or v < 0 for v in self.layers)):
            raise ValueError("layers must be null or nonnegative integer indices")
        names = [m['name'] for m in self.models]
        if not names or any(not isinstance(n,str) or not n.strip() for n in names) or len(names) != len(set(names)):
            raise ValueError("Model names must be nonempty and distinct")
        if len({model_slug(n).casefold() for n in names})!=len(names):
            raise ValueError("Model output directory names collide")
        if any(len(pair) != 2 for pair in self.pairs + self.primary_test_pairs):
            raise ValueError("Each condition pair requires source and target")
        if any(m['name'] == self.label_encoder for m in self.models):
            raise ValueError("The taxonomy encoder must never be one of the probed models")

    def as_dict(self):
        return asdict(self)
