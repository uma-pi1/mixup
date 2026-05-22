import enum

from sqlalchemy import (
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    Column,
    DateTime,
    Boolean,
    String,
    Integer,
    JSON,
    Double,
    ForeignKey,
    Index,
    Enum,
)


metadata_obj = MetaData()

experiments = Table(
    "experiments",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("start_date", DateTime, nullable=False),
    Column("model_name", String(255), nullable=False),
    Column("dataset_name", String(255), nullable=False),
    Column("method_name", String(255)),
    Column("num_folds", Integer, nullable=False),
    Column("num_test_rounds", Integer, nullable=False),
    Column("seed", Integer, nullable=False),
    Column("cv_seed", Integer, nullable=False),
    Column("config", JSON, nullable=False),
    Column("commit_hash", String(255), nullable=False),
    Column("use_params_from", String(510)),
)

studies = Table(
    "studies",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("optuna_study_id", Integer, nullable=False),
    Column(
        "experiment_id",
        Integer,
        ForeignKey(
            "experiments.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    ),
    Column("start_date", DateTime, nullable=False),
    Column("end_date", DateTime),
    Column("is_complete", Boolean),
    Column("best_trial_number", Integer),
    Column("fold", Integer, nullable=False),
    Column("err_msg", Text()),
    Column("best_avg_train_loss", Double()),
    Column("best_avg_train_loss_epoch", Integer()),
    Column("best_avg_val_acc", Double()),
    Column("best_avg_val_acc_epoch", Integer()),
    Column("best_avg_val_loss", Double()),
    Column("best_avg_val_loss_epoch", Integer()),
    UniqueConstraint("experiment_id", "fold"),
)

trials = Table(
    "trials",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column(
        "study_id",
        Integer,
        ForeignKey("studies.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("optuna_trial_id", Integer(), nullable=False),
    Column("trial_number", Integer(), nullable=False),
    Column("duration", Double()),
    Column("result", Double()),
)


class TrialMetric(enum.Enum):
    TRAIN_LOSS_EPOCH = "train_loss_epoch"
    VAL_LOSS = "val_loss"
    VAL_ACC = "val_acc"
    TEST_ACC = "test_acc"
    TEST_LOSS = "test_loss"
    TEST_TRAIN_LOSS_EPOCH = "test_train_loss_epoch"


trial_metrics = Table(
    "trial_metrics",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column(
        "trial_id",
        Integer,
        ForeignKey("trials.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("inner_fold", Integer),
    Column("epoch", Integer, nullable=False),
    Column("key", Enum(TrialMetric), nullable=False),
    Column("value", Double()),
    UniqueConstraint("trial_id", "inner_fold", "epoch", "key"),
    Index("ix_trial_metrics_key", "key"),
)

trial_hyperparams = Table(
    "trial_hyperparams",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column(
        "trial_id",
        Integer,
        ForeignKey("trials.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("key", String(255), nullable=False),
    Column("value", String(255)),
    Index("ix_trial_hyperparams_key", "key"),
)

results = Table(
    "results",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column(
        "study_id",
        Integer,
        ForeignKey(
            "studies.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    ),
    Column("final_train_loss", Double()),
    Column("test_acc", Double()),
    Column("test_loss", Double()),
    Column("created_at", DateTime, nullable=False),
)

best_study_params = Table(
    "best_study_params",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column(
        "study_id",
        Integer,
        ForeignKey(
            "studies.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    ),
    Column("key", String(255), nullable=False),
    Column("value", String(255)),
    UniqueConstraint("study_id", "key"),
)

test_time_graphs = Table(
    "test_time_graphs",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column(
        "study_id",
        Integer,
        ForeignKey(
            "studies.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    ),
    Column("graph_id", Integer, nullable=False),
)
