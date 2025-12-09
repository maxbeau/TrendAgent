"""Add advanced insight fields to analysis_scores"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0011_add_analysis_insights"
down_revision = "0010_soft_factor_code_len"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analysis_scores", sa.Column("scenarios", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("analysis_scores", sa.Column("key_variables", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("analysis_scores", sa.Column("stock_strategy", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("analysis_scores", sa.Column("option_strategies", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("analysis_scores", sa.Column("risk_management", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("analysis_scores", sa.Column("execution_notes", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("analysis_scores", "execution_notes")
    op.drop_column("analysis_scores", "risk_management")
    op.drop_column("analysis_scores", "option_strategies")
    op.drop_column("analysis_scores", "stock_strategy")
    op.drop_column("analysis_scores", "key_variables")
    op.drop_column("analysis_scores", "scenarios")
