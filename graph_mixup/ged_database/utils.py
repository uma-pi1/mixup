import os

from dotenv import load_dotenv
from sqlalchemy import Engine, URL, create_engine


def get_engine() -> Engine:
    load_dotenv(override=True)
    url_object = URL.create(
        drivername=os.getenv("GED_DB_CONNECTION"),
        username=os.getenv("GED_DB_USER"),
        password=os.getenv("GED_DB_PASSWORD"),
        host=os.getenv("GED_DB_HOST"),
        database=os.getenv("GED_DB_DATABASE"),
    )
    db_url = url_object.render_as_string(False)

    return create_engine(db_url)
