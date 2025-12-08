"""Add TAM expansion factor to F3 weights."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0008_add_tam_expansion_factor"
down_revision = "0007_drop_formula_from_cat"
branch_labels = None
depends_on = None

FACTOR_CODE = "fundamental.tam_expansion"
FACTOR_NAME = "TAM 扩张"
FACTOR_DESC = "TAM 扩张软因子（Massive 新闻 + LLM）"
FACTOR_WEIGHT = 15.0


def _get_factor_id(bind) -> int:
    row = bind.execute(
        sa.text("SELECT id FROM system_factors WHERE factor_code=:code"), {"code": FACTOR_CODE}
    ).first()
    return row[0] if row else None


def upgrade() -> None:
    bind = op.get_bind()
    category_row = bind.execute(
        sa.text("SELECT id FROM factor_categories WHERE category_code=:code"), {"code": "F3"}
    ).first()
    if not category_row:
        raise RuntimeError("Factor category F3 not found; cannot insert TAM factor.")
    category_id = category_row[0]

    factor_id = _get_factor_id(bind)
    if factor_id is None:
        result = bind.execute(
            sa.text(
                "INSERT INTO system_factors (category_id, factor_code, name, description) "
                "VALUES (:category_id, :factor_code, :name, :description) "
                "RETURNING id"
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


def downgrade() -> None:
    bind = op.get_bind()
    factor_id = _get_factor_id(bind)
    if factor_id is None:
        return
    bind.execute(
        sa.text("DELETE FROM model_factor_weights WHERE factor_id=:factor_id"),
        {"factor_id": factor_id},
    )
    bind.execute(sa.text("DELETE FROM system_factors WHERE id=:factor_id"), {"factor_id": factor_id})
