"""Add formula_text to factor_categories for per-factor formulas."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0006_formula_in_categories"
down_revision = "0005_narrative_report_cascade"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("factor_categories", sa.Column("formula_text", sa.String(), nullable=True))

    default_formulas = {
        "F1": "Score = 0.4×流动性方向 + 0.2×利率趋势 + 0.2×信用利差 + 0.2×全球需求",
        "F2": "Score = 0.3×行业增速 + 0.3×CapEx方向 + 0.2×利润率 + 0.2×政策顺风",
        "F3": "Score = 0.25×成长性 + 0.25×盈利能力 + 0.2×现金流 + 0.15×财务稳健 + 0.15×TAM",
        "F4": "Score = 0.4×趋势结构 + 0.3×RS强度 + 0.2×量价行为 + 0.1×波动结构",
        "F5": "Score = 0.4×机构资金 + 0.3×ETF行为 + 0.2×期权异动 + 0.1×被动流向",
        "F6": "Score = 0.4×VIX + 0.3×P/C Ratio + 0.2×Skew + 0.1×情绪指数",
        "F7": "Score = 0.5×事件强度 + 0.3×时间接近度 + 0.2×发生概率",
        "F8": "Score = 0.4×波动位置 + 0.3×波动方向 + 0.3×赔率结构（IV vs HV、Skew、R/R）",
    }
    bind = op.get_bind()
    for code, formula in default_formulas.items():
        bind.execute(
            sa.text("UPDATE factor_categories SET formula_text=:formula WHERE category_code=:code").bindparams(
                formula=formula, code=code
            )
        )


def downgrade() -> None:
    op.drop_column("factor_categories", "formula_text")
