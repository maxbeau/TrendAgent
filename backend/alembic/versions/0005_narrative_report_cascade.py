"""Add cascade delete from narrative_reports to narrative_jobs."""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0005_narrative_report_cascade"
down_revision: Union[str, None] = "0004_drop_narrative_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("narrative_reports") as batch_op:
        batch_op.drop_constraint("narrative_reports_job_id_fkey", type_="foreignkey")
        batch_op.create_foreign_key(
            "narrative_reports_job_id_fkey",
            "narrative_jobs",
            ["job_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table("narrative_reports") as batch_op:
        batch_op.drop_constraint("narrative_reports_job_id_fkey", type_="foreignkey")
        batch_op.create_foreign_key(
            "narrative_reports_job_id_fkey",
            "narrative_jobs",
            ["job_id"],
            ["id"],
        )
