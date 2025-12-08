"""Drop formula_text from factor_categories."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0007_drop_formula_from_cat"
down_revision = "0006_formula_in_categories"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("factor_categories", "formula_text")


def downgrade() -> None:
    op.add_column("factor_categories", sa.Column("formula_text", sa.String(), nullable=True))
