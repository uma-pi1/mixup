import logging

import torch

from graph_mixup.config.parser import Parser
from graph_mixup.logging.fs_study_logger import FileSystemStudyLogger
from graph_mixup.logging.logger_container import LoggerContainer
from graph_mixup.logging.rdb.rdb_logger import RDBLogger
from graph_mixup.studies.objectives import TrainObjective, TestObjective
from graph_mixup.studies.study_manager import StudyManager


def main():
    torch.set_float32_matmul_precision("high")

    parser = Parser()
    config = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if config.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open("commit_hash.txt", "r") as f:
        commit_hash = f.read()

    study_manager = StudyManager(config)

    loggers = LoggerContainer(
        config,
        commit_hash,
        study_manager.optuna_study_id,
        FileSystemStudyLogger,
        RDBLogger,
    )

    try:
        if config.use_params_from is None:
            # ===
            # Model Selection: Find best hyperparameters
            # ===

            train_objective = TrainObjective(config)
            best_trial_results, all_trials = study_manager.model_selection(
                train_objective
            )
            loggers.log_model_selection_result(best_trial_results, all_trials)

        # ===
        # Model Assessment: Elicit Model Performance
        # ===

        for test_round in range(config.num_test_rounds):
            test_objective = TestObjective(config, test_round)
            test_result = study_manager.model_assessment(test_objective)
            loggers.log_model_assessment_result(test_result)

        loggers.log_study_completed()

    except Exception as e:
        loggers.log_study_incomplete(e)
        raise

    finally:
        loggers.dispose()


if __name__ == "__main__":
    main()
