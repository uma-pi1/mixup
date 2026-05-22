from argparse import ArgumentParser, ArgumentTypeError
from enum import Enum
from typing import Union, cast

from graph_mixup.config.typing import (
    CLConfig,
    ModelName,
    DatasetName,
    PreLossMixupName,
    PreBatchMixupName,
    AugmentationName,
)


class Parser:
    def __init__(self) -> None:
        self.parser = ArgumentParser()

    def parse_args(self) -> CLConfig:
        self.parser.add_argument(
            "--num_trials",
            type=int,
            required=True,
            help="Total number of completed trials (using MaxTrialsCallback).",
        )
        self.parser.add_argument(
            "--study_timeout",
            type=int,
            help="Timeout in seconds for the HP optimization of a single fold. Aborts the study even if num_trials is not reached.",
            required=True,
        )
        self.parser.add_argument(
            "--train_timeout",
            type=int,
            help="Timeout in seconds for the training of a single HP configuration.",
        )
        self.parser.add_argument(
            "--device", type=int, default=0, help="Device ID"
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="""Seed that is set in the beginning of each fold's HPO. This 
            affects torch, numpy, optuna, and Python's random module. Note that due 
            to scatter ops on the GPU, results might still differ from one another even 
            though a seed is specified.""",
        )
        self.parser.add_argument(
            "--cv_seed",
            type=int,
            default=0,
            help="""Seed that is used to determine the dataset folds during cross-validation. 
            If a different value is used, then the dataset is split differently (with high probability).""",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="number of workers in data loaders",
        )
        self.parser.add_argument(
            "--log_dir",
            type=str,
            default="logs",
        )
        self.parser.add_argument(
            "--data_dir", type=str, default="data", help="data storage location"
        )
        self.parser.add_argument(
            "--num_test_rounds",
            type=int,
            help="the number of times a model is initialized, trained, and tested during model assessment",
        )
        self.parser.add_argument(
            "--max_epochs",
            type=int,
            default=1000,
            help="number of epochs to train",
        )
        self.parser.add_argument(
            "--patience",
            type=int,
            help="Parameter for early stopping. If the validation accuracy has "
            "not improved for the last `patience` validation rounds, "
            "training will stop. ",
            required=True,
        )
        self.parser.add_argument(
            "--use_baseline",
            action="store_true",
            help="Uses model HPs from the corresponding baseline experiment.",
        )
        self.parser.add_argument(
            "--use_params_from",
            type=str,
            help="Use the best parameters from a prior study of the given name. "
            "Will directly proceed to model assessment (i.e., no model "
            "selection).",
        )
        self.parser.add_argument(
            "--model_name",
            type=ModelName,
            choices=[cast(Enum, model).value for model in ModelName],
            required=True,
        )
        self.parser.add_argument(
            "--dataset_name",
            type=DatasetName,
            choices=[cast(Enum, dataset).value for dataset in DatasetName],
            required=True,
        )
        self.parser.add_argument(
            "--method_name",
            type=self._method_name_type,
            choices=[
                cast(Enum, method).value
                for method in list(PreLossMixupName)
                + list(PreBatchMixupName)
                + list(AugmentationName)
            ],
            help="Choose mixup or augmentation method (can be None).",
        )
        self.parser.add_argument(
            "--num_outer_folds",
            type=int,
            help="number of folds in cross-validation",
            required=True,
        )
        self.parser.add_argument(
            "--num_inner_folds",
            type=int,
            help="number of inner folds in cross-validation",
            required=True,
        )
        self.parser.add_argument(
            "--use_inner_holdout",
            action="store_true",
            help="If set, uses inner holdout validation instead of inner cross "
            "validation. The validation set size is then given as 1 / num_inner_folds.",
        )
        self.parser.add_argument(
            "--fold",
            type=int,
            help="outer fold index for current study from [0, num_folds - 1]",
            required=True,
        )
        self.parser.add_argument(
            "--reload_dataloaders_every_n_epochs",
            type=int,
            default=0,
        )
        self.parser.add_argument(
            "--skip_model_selection",
            action="store_true",
        )
        self.parser.add_argument(
            "--verbose", "-v", action="store_true", default=False
        )
        self.parser.add_argument(
            "--test_every_trial", action="store_true", default=False
        )

        self.parser.add_argument(
            "--label_corruption_prob",
            type=float,
            default=0.0,
            help="Probability of corrupting the label of a training graph (between 0 and 1).",
        )

        config = CLConfig(**vars(self.parser.parse_args()))

        self._validate_fold_index(config.fold, config.num_outer_folds)
        self._validate_baseline_method(config.use_baseline, config.method_name)

        return config

    @staticmethod
    def _method_name_type(
        method_name: str,
    ) -> Union[PreLossMixupName, PreBatchMixupName, AugmentationName]:
        if method_name in set(PreLossMixupName):
            return PreLossMixupName(method_name)

        if method_name in set(PreBatchMixupName):
            return PreBatchMixupName(method_name)

        if method_name in set(AugmentationName):
            return AugmentationName(method_name)

        raise ArgumentTypeError(f"method {method_name} not supported")

    @staticmethod
    def _validate_fold_index(fold: int, num_folds: int) -> None:
        if not 0 <= fold < num_folds:
            raise ArgumentTypeError(
                "fold index must be in range [0, 1, ..., num_folds - 1]"
            )

    @staticmethod
    def _validate_baseline_method(
        use_baseline: bool, method_name: None | str
    ) -> None:
        if use_baseline and method_name is None:
            raise ArgumentTypeError(
                "Baseline can only be used if a method for the current "
                "study is specified. Either omit use_baseline or specify a"
                "method_name."
            )
