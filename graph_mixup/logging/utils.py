import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import URL


def value_mean_std_dict(values: list[float | None]) -> dict[str, Any]:
    cleaned_values = [v for v in values if v is not None]

    mean = np.mean(cleaned_values).item() if len(cleaned_values) > 0 else None
    std = (
        np.std(cleaned_values, ddof=1).item()
        if len(cleaned_values) > 0
        else None
    )

    return dict(
        values=values,
        mean=mean,
        std=std,
    )


def get_experiment_db_url():
    load_dotenv(override=True)
    url_object = URL.create(
        drivername=os.getenv("EXPERIMENT_DB_CONNECTION"),
        username=os.getenv("EXPERIMENT_DB_USER"),
        password=os.getenv("EXPERIMENT_DB_PASSWORD"),
        host=os.getenv("EXPERIMENT_DB_HOST"),
        database=os.getenv("EXPERIMENT_DB_DATABASE"),
    )
    db_url = url_object.render_as_string(False)
    return db_url
