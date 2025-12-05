"""Add narrative layer persistence tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "0003_narrative_layer"
down_revision: Union[str, None] = "0002_rebuild_weight_system"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "soft_factor_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("factor_code", sa.String(length=20), nullable=False),
        sa.Column("asof_date", sa.Date(), nullable=False),
        sa.Column("score", sa.Float()),
        sa.Column("confidence", sa.Float()),
        sa.Column("reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("ticker", "factor_code", "asof_date", name="uq_soft_factor_scores_identity"),
    )
    op.create_index("ix_soft_factor_scores_ticker", "soft_factor_scores", ["ticker"])
    op.create_index("ix_soft_factor_scores_asof_date", "soft_factor_scores", ["asof_date"])

    op.create_table(
        "context_store",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("asof_date", sa.Date(), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_refs", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("ticker", "asof_date", "type", name="uq_context_store_identity"),
    )
    op.create_index("ix_context_store_ticker", "context_store", ["ticker"])
    op.create_index("ix_context_store_asof_date", "context_store", ["asof_date"])

    op.create_table(
        "narrative_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.String()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_narrative_jobs_ticker", "narrative_jobs", ["ticker"])

    op.create_table(
        "narrative_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("narrative_jobs.id"), nullable=False, unique=True),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("output_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("tokens", sa.Integer()),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_narrative_reports_ticker", "narrative_reports", ["ticker"])


def downgrade() -> None:
    op.drop_index("ix_narrative_reports_ticker", table_name="narrative_reports")
    op.drop_table("narrative_reports")

    op.drop_index("ix_narrative_jobs_ticker", table_name="narrative_jobs")
    op.drop_table("narrative_jobs")

    op.drop_index("ix_context_store_asof_date", table_name="context_store")
    op.drop_index("ix_context_store_ticker", table_name="context_store")
    op.drop_table("context_store")

    op.drop_index("ix_soft_factor_scores_asof_date", table_name="soft_factor_scores")
    op.drop_index("ix_soft_factor_scores_ticker", table_name="soft_factor_scores")
    op.drop_table("soft_factor_scores")
