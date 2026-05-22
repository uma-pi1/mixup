"""mixup parents nullable

Revision ID: bab74799014a
Revises: 702d452e07e5
Create Date: 2025-02-14 12:54:11.026398

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "bab74799014a"
down_revision: Union[str, None] = "702d452e07e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "mixup_attrs",
        "parent_0_id",
        existing_type=sa.INTEGER(),
        nullable=True,
    )
    op.alter_column(
        "mixup_attrs",
        "parent_1_id",
        existing_type=sa.INTEGER(),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "mixup_attrs",
        "parent_0_id",
        existing_type=sa.INTEGER(),
        nullable=False,
    )
    op.alter_column(
        "mixup_attrs",
        "parent_1_id",
        existing_type=sa.INTEGER(),
        nullable=False,
    )
