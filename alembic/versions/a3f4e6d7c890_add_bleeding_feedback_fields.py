"""add_bleeding_feedback_fields

Revision ID: a3f4e6d7c890
Revises: 56d95a4717f5
Create Date: 2026-03-25 17:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f4e6d7c890'
down_revision: Union[str, Sequence[str], None] = '56d95a4717f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    if not _has_table(inspector, table_name):
        return False
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "analyses") and not _has_column(inspector, "analyses", "bleeding_type"):
        op.add_column("analyses", sa.Column("bleeding_type", sa.String(), nullable=True))

    if _has_table(inspector, "analysis_reviews"):
        if not _has_column(inspector, "analysis_reviews", "is_bleeding"):
            op.add_column(
                "analysis_reviews",
                sa.Column("is_bleeding", sa.Boolean(), nullable=False, server_default=sa.false()),
            )
            op.alter_column("analysis_reviews", "is_bleeding", server_default=None)

        if not _has_column(inspector, "analysis_reviews", "bleeding_type"):
            op.add_column("analysis_reviews", sa.Column("bleeding_type", sa.String(), nullable=True))

        if not _has_column(inspector, "analysis_reviews", "model_trained"):
            op.add_column(
                "analysis_reviews",
                sa.Column("model_trained", sa.Boolean(), nullable=False, server_default=sa.false()),
            )
            op.alter_column("analysis_reviews", "model_trained", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_column(inspector, "analysis_reviews", "model_trained"):
        op.drop_column("analysis_reviews", "model_trained")

    if _has_column(inspector, "analysis_reviews", "bleeding_type"):
        op.drop_column("analysis_reviews", "bleeding_type")

    if _has_column(inspector, "analysis_reviews", "is_bleeding"):
        op.drop_column("analysis_reviews", "is_bleeding")

    if _has_column(inspector, "analyses", "bleeding_type"):
        op.drop_column("analyses", "bleeding_type")
