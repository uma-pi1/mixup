from dataclasses import dataclass
from enum import StrEnum


class ModelName(StrEnum):
    GCN = "GCN"
    GIN = "GIN"


class DatasetName(StrEnum):
    REDDIT_BINARY = "REDDIT-BINARY"
    REDDIT_MULTI_5K = "REDDIT-MULTI-5K"
    IMDB_BINARY = "IMDB-BINARY"
    IMDB_MULTI = "IMDB-MULTI"
    PROTEINS = "PROTEINS"
    COLLAB = "COLLAB"
    MUTAG = "MUTAG"
    ENZYMES = "ENZYMES"
    NCI1 = "NCI1"


class PreLossMixupName(StrEnum):
    EMB_MIXUP = "emb_mixup"


class PreBatchMixupName(StrEnum):
    IF_MIXUP = "if_mixup"
    IF_MIXUP_SE = "if_mixup_se"
    G_MIXUP = "g_mixup"
    FGW_MIXUP = "fgw_mixup"
    S_MIXUP = "s_mixup"
    S_MIXUP_SE = "s_mixup_se"
    SUBMIX = "submix"
    GEOMIX = "geomix"
    GEOMIX_SE = "geomix_se"
    GED_MIXUP = "ged_mixup"


class AugmentationName(StrEnum):
    DROP_EDGE = "drop_edge"
    DROP_NODE = "drop_node"
    DROP_PATH = "drop_path"
    PERTURB_NODE_ATTR = "perturb_node_attr"


@dataclass
class CLConfig:
    num_trials: int
    study_timeout: None | int
    train_timeout: None | int
    device: int
    seed: int
    cv_seed: int
    num_workers: int
    log_dir: str
    data_dir: str
    num_test_rounds: int
    max_epochs: int
    patience: int
    use_baseline: bool
    use_params_from: None | str
    model_name: ModelName
    dataset_name: DatasetName
    method_name: PreLossMixupName | PreBatchMixupName | AugmentationName | None
    num_outer_folds: int
    num_inner_folds: int
    use_inner_holdout: bool
    fold: int
    reload_dataloaders_every_n_epochs: int
    skip_model_selection: bool
    verbose: bool
    test_every_trial: bool
    label_corruption_prob: float
