import os

import optuna
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from sqlalchemy import URL

from graph_mixup.config.typing import CLConfig
from graph_mixup.resource_locators import ResourceLocator


def get_optuna_db_url():
    load_dotenv(override=True)
    url_object = URL.create(
        drivername=os.getenv("OPTUNA_DB_CONNECTION"),
        username=os.getenv("OPTUNA_DB_USER"),
        password=os.getenv("OPTUNA_DB_PASSWORD"),
        host=os.getenv("OPTUNA_DB_HOST"),
        database=os.getenv("OPTUNA_DB_DATABASE"),
    )
    db_url = url_object.render_as_string(False)
    return db_url


def create_optuna_study(
    config: CLConfig,
) -> optuna.Study:
    locator = ResourceLocator(config)

    return optuna.create_study(
        storage=get_optuna_db_url(),
        sampler=TPESampler(seed=config.seed),
        study_name=locator.get_optuna_study_name(),
        direction="maximize",
    )


def load_optuna_study(config: CLConfig) -> optuna.Study:
    return optuna.load_study(
        study_name=config.use_params_from,
        storage=get_optuna_db_url(),
    )
