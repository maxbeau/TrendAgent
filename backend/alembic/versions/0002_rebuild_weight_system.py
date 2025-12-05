"""Rebuild weight system with category and factor hierarchy."""

from __future__ import annotations

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_rebuild_weight_system"
down_revision: Union[str, None] = "0001_init"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _seed_model_and_weights() -> None:
    bind = op.get_bind()

    # reset categories and factors to the new canonical definitions
    bind.execute(sa.text("DELETE FROM factor_categories"))

    category_table = sa.table(
        "factor_categories",
        sa.column("id", sa.Integer),
        sa.column("category_code", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.String),
    )
    op.bulk_insert(
        category_table,
        [
            {"id": 1, "category_code": "F1", "name": "宏观驱动", "description": "Macro"},
            {"id": 2, "category_code": "F2", "name": "行业景气度", "description": "Industry"},
            {"id": 3, "category_code": "F3", "name": "公司基本面", "description": "Fundamental"},
            {"id": 4, "category_code": "F4", "name": "技术结构", "description": "Technical"},
            {"id": 5, "category_code": "F5", "name": "资金流向", "description": "Capital Flow"},
            {"id": 6, "category_code": "F6", "name": "市场情绪", "description": "Sentiment"},
            {"id": 7, "category_code": "F7", "name": "催化事件", "description": "Catalyst"},
            {"id": 8, "category_code": "F8", "name": "波动赔率", "description": "Volatility"},
        ],
    )
    bind.execute(
        sa.text(
            "SELECT setval(pg_get_serial_sequence('factor_categories','id'), "
            "(SELECT MAX(id) FROM factor_categories))"
        )
    )

    factor_table = sa.table(
        "system_factors",
        sa.column("id", sa.Integer),
        sa.column("category_id", sa.Integer),
        sa.column("factor_code", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.String),
    )

    factors = [
        # F1 Macro
        {"id": 1, "category_id": 1, "factor_code": "macro.liquidity_direction", "name": "流动性方向"},
        {"id": 2, "category_id": 1, "factor_code": "macro.rate_trend", "name": "利率趋势"},
        {"id": 3, "category_id": 1, "factor_code": "macro.credit_spread", "name": "信用利差"},
        {"id": 4, "category_id": 1, "factor_code": "macro.global_demand", "name": "全球需求"},
        # F2 Industry (placeholders for future implementation)
        {"id": 5, "category_id": 2, "factor_code": "industry.growth", "name": "行业增速"},
        {"id": 6, "category_id": 2, "factor_code": "industry.capex_cycle", "name": "CapEx 周期"},
        {"id": 7, "category_id": 2, "factor_code": "industry.margin_trend", "name": "利润率趋势"},
        {"id": 8, "category_id": 2, "factor_code": "industry.policy_tailwind", "name": "政策顺风"},
        # F3 Fundamental
        {"id": 9, "category_id": 3, "factor_code": "fundamental.growth", "name": "成长性"},
        {"id": 10, "category_id": 3, "factor_code": "fundamental.profitability", "name": "盈利能力"},
        {"id": 11, "category_id": 3, "factor_code": "fundamental.cash_flow", "name": "现金流质量"},
        {"id": 12, "category_id": 3, "factor_code": "fundamental.leverage", "name": "财务稳健"},
        {"id": 13, "category_id": 3, "factor_code": "fundamental.roic", "name": "ROIC"},
        # F4 Technical
        {"id": 14, "category_id": 4, "factor_code": "technical.trend_structure", "name": "趋势结构"},
        {"id": 15, "category_id": 4, "factor_code": "technical.relative_strength", "name": "相对强弱"},
        {"id": 16, "category_id": 4, "factor_code": "technical.volume_profile", "name": "量价行为"},
        {"id": 17, "category_id": 4, "factor_code": "technical.volatility_structure", "name": "波动结构"},
        # F5 Flow
        {"id": 18, "category_id": 5, "factor_code": "flow.institutional", "name": "机构资金"},
        {"id": 19, "category_id": 5, "factor_code": "flow.options_activity", "name": "期权异动"},
        {"id": 20, "category_id": 5, "factor_code": "flow.gamma_exposure", "name": "Gamma"},
        {"id": 21, "category_id": 5, "factor_code": "flow.etf_behavior", "name": "ETF 行为"},
        # F6 Sentiment
        {"id": 22, "category_id": 6, "factor_code": "sentiment.vix", "name": "VIX"},
        {"id": 23, "category_id": 6, "factor_code": "sentiment.put_call", "name": "Put/Call"},
        {"id": 24, "category_id": 6, "factor_code": "sentiment.skew", "name": "Skew"},
        {"id": 25, "category_id": 6, "factor_code": "sentiment.fear_greed", "name": "情绪指数"},
        # F7 Catalyst
        {"id": 26, "category_id": 7, "factor_code": "catalyst.event_strength", "name": "事件强度"},
        {"id": 27, "category_id": 7, "factor_code": "catalyst.timing", "name": "时间接近度"},
        {"id": 28, "category_id": 7, "factor_code": "catalyst.probability", "name": "发生概率"},
        # F8 Volatility
        {"id": 29, "category_id": 8, "factor_code": "volatility.iv_vs_hv", "name": "IV vs HV"},
        {"id": 30, "category_id": 8, "factor_code": "volatility.skew", "name": "波动偏斜"},
        {"id": 31, "category_id": 8, "factor_code": "volatility.risk_reward", "name": "赔率结构"},
    ]
    op.bulk_insert(factor_table, factors)
    bind.execute(
        sa.text(
            "SELECT setval(pg_get_serial_sequence('system_factors','id'), "
            "(SELECT MAX(id) FROM system_factors))"
        )
    )

    model_row = bind.execute(sa.text("SELECT id FROM aion_models WHERE model_name=:name"), {"name": "v8.0"}).first()
    if model_row:
        model_id = model_row[0]
    else:
        model_id = uuid.uuid4()
        bind.execute(
            sa.text(
                "INSERT INTO aion_models (id, model_name, description, is_public, created_at) "
                "VALUES (:id, :name, :description, true, now())"
            ),
            {"id": model_id, "name": "v8.0", "description": "AION v8.0 baseline model"},
        )

    category_weight_table = sa.table(
        "model_category_weights",
        sa.column("model_id", postgresql.UUID),
        sa.column("category_id", sa.Integer),
        sa.column("weight", sa.Numeric),
    )
    op.bulk_insert(
        category_weight_table,
        [
            {"model_id": model_id, "category_id": 1, "weight": 10.0},
            {"model_id": model_id, "category_id": 2, "weight": 15.0},
            {"model_id": model_id, "category_id": 3, "weight": 20.0},
            {"model_id": model_id, "category_id": 4, "weight": 15.0},
            {"model_id": model_id, "category_id": 5, "weight": 10.0},
            {"model_id": model_id, "category_id": 6, "weight": 10.0},
            {"model_id": model_id, "category_id": 7, "weight": 10.0},
            {"model_id": model_id, "category_id": 8, "weight": 10.0},
        ],
    )

    factor_weight_table = sa.table(
        "model_factor_weights",
        sa.column("model_id", postgresql.UUID),
        sa.column("factor_id", sa.Integer),
        sa.column("weight", sa.Numeric),
    )
    op.bulk_insert(
        factor_weight_table,
        [
            # F1 Macro
            {"model_id": model_id, "factor_id": 1, "weight": 40.0},
            {"model_id": model_id, "factor_id": 2, "weight": 20.0},
            {"model_id": model_id, "factor_id": 3, "weight": 20.0},
            {"model_id": model_id, "factor_id": 4, "weight": 20.0},
            # F2 Industry
            {"model_id": model_id, "factor_id": 5, "weight": 30.0},
            {"model_id": model_id, "factor_id": 6, "weight": 30.0},
            {"model_id": model_id, "factor_id": 7, "weight": 20.0},
            {"model_id": model_id, "factor_id": 8, "weight": 20.0},
            # F3 Fundamental
            {"model_id": model_id, "factor_id": 9, "weight": 25.0},
            {"model_id": model_id, "factor_id": 10, "weight": 25.0},
            {"model_id": model_id, "factor_id": 11, "weight": 20.0},
            {"model_id": model_id, "factor_id": 12, "weight": 15.0},
            {"model_id": model_id, "factor_id": 13, "weight": 15.0},
            # F4 Technical
            {"model_id": model_id, "factor_id": 14, "weight": 40.0},
            {"model_id": model_id, "factor_id": 15, "weight": 30.0},
            {"model_id": model_id, "factor_id": 16, "weight": 20.0},
            {"model_id": model_id, "factor_id": 17, "weight": 10.0},
            # F5 Flow
            {"model_id": model_id, "factor_id": 18, "weight": 40.0},
            {"model_id": model_id, "factor_id": 19, "weight": 30.0},
            {"model_id": model_id, "factor_id": 20, "weight": 20.0},
            {"model_id": model_id, "factor_id": 21, "weight": 10.0},
            # F6 Sentiment
            {"model_id": model_id, "factor_id": 22, "weight": 40.0},
            {"model_id": model_id, "factor_id": 23, "weight": 30.0},
            {"model_id": model_id, "factor_id": 24, "weight": 20.0},
            {"model_id": model_id, "factor_id": 25, "weight": 10.0},
            # F7 Catalyst
            {"model_id": model_id, "factor_id": 26, "weight": 50.0},
            {"model_id": model_id, "factor_id": 27, "weight": 30.0},
            {"model_id": model_id, "factor_id": 28, "weight": 20.0},
            # F8 Volatility
            {"model_id": model_id, "factor_id": 29, "weight": 40.0},
            {"model_id": model_id, "factor_id": 30, "weight": 30.0},
            {"model_id": model_id, "factor_id": 31, "weight": 30.0},
        ],
    )


def upgrade() -> None:
    op.drop_table("model_factor_weights")

    op.rename_table("system_factors", "factor_categories")
    op.alter_column("factor_categories", "factor_code", new_column_name="category_code")
    op.alter_column("factor_categories", "factor_name", new_column_name="name")

    op.create_table(
        "system_factors",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("category_id", sa.Integer(), sa.ForeignKey("factor_categories.id"), nullable=False),
        sa.Column("factor_code", sa.String(length=64), unique=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String()),
    )

    op.create_table(
        "model_category_weights",
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("aion_models.id"), nullable=False),
        sa.Column("category_id", sa.Integer(), sa.ForeignKey("factor_categories.id"), nullable=False),
        sa.Column("weight", sa.Numeric(5, 2), nullable=False),
        sa.PrimaryKeyConstraint("model_id", "category_id"),
    )

    op.create_table(
        "model_factor_weights",
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("aion_models.id"), nullable=False),
        sa.Column("factor_id", sa.Integer(), sa.ForeignKey("system_factors.id"), nullable=False),
        sa.Column("weight", sa.Numeric(5, 2), nullable=False),
        sa.PrimaryKeyConstraint("model_id", "factor_id"),
    )

    _seed_model_and_weights()


def downgrade() -> None:
    op.drop_table("model_factor_weights")
    op.drop_table("model_category_weights")
    op.drop_table("system_factors")

    op.alter_column("factor_categories", "category_code", new_column_name="factor_code")
    op.alter_column("factor_categories", "name", new_column_name="factor_name")
    op.rename_table("factor_categories", "system_factors")

    op.create_table(
        "model_factor_weights",
        sa.Column("model_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("aion_models.id"), nullable=False),
        sa.Column("factor_id", sa.Integer(), sa.ForeignKey("system_factors.id"), nullable=False),
        sa.Column("weight", sa.Numeric(5, 2), nullable=False),
        sa.PrimaryKeyConstraint("model_id", "factor_id"),
    )
