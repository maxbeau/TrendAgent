"""Drop tokens column from narrative_reports."""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0004_drop_narrative_tokens"
down_revision: Union[str, None] = "0003_narrative_layer"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("narrative_reports") as batch_op:
        batch_op.drop_column("tokens")


def downgrade() -> None:
    with op.batch_alter_table("narrative_reports") as batch_op:
        batch_op.add_column(sa.Column("tokens", sa.Integer))
