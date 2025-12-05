"""Initial tables for AION analysis storage."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_init"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )

    op.create_table(
        "aion_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String()),
        sa.Column("is_public", sa.Boolean(), server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )

    op.create_table(
        "system_factors",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("factor_code", sa.String(length=10), nullable=False, unique=True),
        sa.Column("factor_name", sa.String(length=50), nullable=False),
        sa.Column("description", sa.String()),
    )

    op.create_table(
        "model_factor_weights",
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("aion_models.id"), nullable=False),
        sa.Column("factor_id", sa.Integer(), sa.ForeignKey("system_factors.id"), nullable=False),
        sa.Column("weight", sa.Numeric(5, 2), nullable=False),
        sa.PrimaryKeyConstraint("model_id", "factor_id"),
    )

    op.create_table(
        "market_data_daily",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("date", sa.DateTime(), nullable=False),
        sa.Column("open", sa.Float()),
        sa.Column("high", sa.Float()),
        sa.Column("low", sa.Float()),
        sa.Column("close", sa.Float()),
        sa.Column("volume", sa.Float()),
    )
    op.create_index("ix_market_data_daily_ticker", "market_data_daily", ["ticker"])
    op.create_index("ix_market_data_daily_date", "market_data_daily", ["date"])

    op.create_table(
        "analysis_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False, unique=True),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("total_score", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("action_card", sa.String(length=50)),
        sa.Column("factors", postgresql.JSONB(), nullable=False),
        sa.Column("weight_denominator", sa.Float()),
        sa.Column("calculated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.create_index("ix_analysis_scores_task_id", "analysis_scores", ["task_id"], unique=True)
    op.create_index("ix_analysis_scores_ticker", "analysis_scores", ["ticker"])


def downgrade() -> None:
    op.drop_index("ix_analysis_scores_ticker", table_name="analysis_scores")
    op.drop_index("ix_analysis_scores_task_id", table_name="analysis_scores")
    op.drop_table("analysis_scores")

    op.drop_index("ix_market_data_daily_date", table_name="market_data_daily")
    op.drop_index("ix_market_data_daily_ticker", table_name="market_data_daily")
    op.drop_table("market_data_daily")

    op.drop_table("model_factor_weights")
    op.drop_table("system_factors")
    op.drop_table("aion_models")
    op.drop_table("users")
