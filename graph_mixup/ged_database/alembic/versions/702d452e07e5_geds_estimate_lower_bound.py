"""geds_estimate->lower_bound

Revision ID: 702d452e07e5
Revises: d521d698f845
Create Date: 2025-01-27 13:11:20.908776

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "702d452e07e5"
down_revision: Union[str, None] = "d521d698f845"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "geds",
        "estimate",
        new_column_name="lower_bound",
        existing_type=sa.Integer,
    )


def downgrade() -> None:
    op.alter_column(
        "geds",
        "lower_bound",
        new_column_name="estimate",
        existing_type=sa.Integer,
    )
