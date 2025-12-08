"""Allow longer soft factor codes for policy tailwind data."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0010_soft_factor_code_len"
down_revision = "0009_remove_fundamental_roic"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "soft_factor_scores",
        "factor_code",
        existing_type=sa.String(length=20),
        type_=sa.String(length=64),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "soft_factor_scores",
        "factor_code",
        existing_type=sa.String(length=64),
        type_=sa.String(length=20),
        existing_nullable=False,
    )
