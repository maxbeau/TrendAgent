"""Remove fundamental.roic factor and weights to match doc formula."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0009_remove_fundamental_roic"
down_revision = "0008_add_tam_expansion_factor"
branch_labels = None
depends_on = None

FACTOR_CODE = "fundamental.roic"
FACTOR_NAME = "ROIC"
FACTOR_DESC = "Return on invested capital"
FACTOR_WEIGHT = 15.0


def _get_factor_id(bind) -> int:
    row = bind.execute(
        sa.text("SELECT id FROM system_factors WHERE factor_code=:code"),
        {"code": FACTOR_CODE},
    ).first()
    return row[0] if row else None


def upgrade() -> None:
    bind = op.get_bind()
    factor_id = _get_factor_id(bind)
    if factor_id is None:
        return
    bind.execute(
        sa.text("DELETE FROM model_factor_weights WHERE factor_id=:factor_id"),
        {"factor_id": factor_id},
    )
    bind.execute(
        sa.text("DELETE FROM system_factors WHERE id=:factor_id"),
        {"factor_id": factor_id},
    )


def downgrade() -> None:
    bind = op.get_bind()
    factor_id = _get_factor_id(bind)
    if factor_id is None:
        category_row = bind.execute(
            sa.text("SELECT id FROM factor_categories WHERE category_code=:code"),
            {"code": "F3"},
        ).first()
        if not category_row:
            raise RuntimeError("Cannot recreate fundamental.roic without F3 category")
        category_id = category_row[0]
        result = bind.execute(
            sa.text(
                "INSERT INTO system_factors (category_id, factor_code, name, description) "
                "VALUES (:category_id, :factor_code, :name, :description) RETURNING id"
            ),
            {
                "category_id": category_id,
                "factor_code": FACTOR_CODE,
                "name": FACTOR_NAME,
                "description": FACTOR_DESC,
            },
        )
        factor_id = result.scalar_one()

    model_rows = bind.execute(sa.text("SELECT id FROM aion_models")).fetchall()
    for (model_id,) in model_rows:
        bind.execute(
            sa.text(
                "INSERT INTO model_factor_weights (model_id, factor_id, weight) "
                "VALUES (:model_id, :factor_id, :weight) "
                "ON CONFLICT (model_id, factor_id) DO NOTHING"
            ),
            {"model_id": model_id, "factor_id": factor_id, "weight": FACTOR_WEIGHT},
        )
