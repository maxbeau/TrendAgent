from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.risk_reward import summarize_risk_reward

logger = logging.getLogger(__name__)


FACTOR_LABELS: Dict[str, str] = {
    "macro": "宏观周期",
    "industry": "产业趋势",
    "fundamental": "基本面强度",
    "technical": "技术形态",
    "flow": "资金流向",
    "sentiment": "情绪定位",
    "catalyst": "催化密度",
    "volatility": "波动与赔率",
}

FACTOR_ALIASES: Dict[str, List[str]] = {
    "macro": ["macro", "macro_economy", "F1"],
    "industry": ["industry", "sector", "F2"],
    "fundamental": ["fundamental", "valuation", "F3"],
    "technical": ["technical", "price_action", "F4"],
    "flow": ["flow", "positioning", "F5"],
    "sentiment": ["sentiment", "emotion", "F6"],
    "catalyst": ["catalyst", "trigger", "F7"],
    "volatility": ["volatility", "risk", "F8"],
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_price(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"${value:,.0f}"


def _probability_from_score(score: Optional[float]) -> float:
    if score is None:
        return 0.5
    return max(0.2, min(0.8, score / 5.0))


def _direction_from_score(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if score >= 4.0:
        return "bullish"
    if score <= 2.5:
        return "bearish"
    return "neutral"


def _impact_from_score(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if score >= 3.5:
        return "bullish"
    if score <= 2.5:
        return "bearish"
    return "neutral"


def _suggestion_from_summary(summary: Optional[str]) -> str:
    if not summary:
        return "重新运行 AION 引擎以刷新要点"
    snippet = summary.split("·")[0].strip()
    return snippet or summary[:80]


def _pick_factor_payload(factors: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    aliases = FACTOR_ALIASES.get(key, [key])
    for alias in aliases:
        payload = factors.get(alias)
        if isinstance(payload, dict):
            return payload
    return None


def _build_key_variables(factors: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, label in FACTOR_LABELS.items():
        payload = _pick_factor_payload(factors, key)
        if not payload:
            continue
        score = _safe_float(payload.get("score"))
        impact = _impact_from_score(score)
        threshold = "数据不足" if score is None else f"{score:.1f} 分"
        suggestion = _suggestion_from_summary(payload.get("summary"))
        priority = abs((score or 3.0) - 3.0)
        rows.append(
            {
                "name": f"{label}",
                "threshold": threshold,
                "impact": impact,
                "suggestion": suggestion,
                "_priority": priority,
            }
        )
    rows.sort(key=lambda item: item.get("_priority", 0), reverse=True)
    return [
        {k: v for k, v in item.items() if k != "_priority"}
        for item in rows[:4]
    ]


def _build_stock_strategy(rr_payload: Dict[str, Any], direction: str) -> Optional[Dict[str, Any]]:
    close = _safe_float(rr_payload.get("close"))
    high = _safe_float(rr_payload.get("recent_high"))
    low = _safe_float(rr_payload.get("recent_low"))
    if close is None:
        return None

    entry_lower = close * (0.95 if direction == "bullish" else 0.9)
    entry_upper = close * (1.02 if direction == "bullish" else 0.97)
    profit = high if high and high > close else close * (1.08 if direction == "bullish" else 0.94)

    add_conditions = [f"突破 {_format_price(high)} 后关注" if high else "关注放量突破"]
    reduce_conditions = [f"跌破 {_format_price(low)}" if low else "跌破关键均线"]

    if direction == "bearish":
        add_conditions = [f"反弹至 {_format_price(entry_upper)} 附近" if entry_upper else "反弹遇阻时加仓"]
        reduce_conditions = [f"跌破 {_format_price(low)} 后分批止盈" if low else "跌破目标价分批止盈"]

    return {
        "entry_zone": f"{_format_price(entry_lower)}–{_format_price(entry_upper)}",
        "add_conditions": add_conditions,
        "reduce_conditions": reduce_conditions,
        "profit_target": _format_price(profit),
    }


def _build_option_strategies(rr_payload: Dict[str, Any], direction: str) -> List[Dict[str, Any]]:
    close = _safe_float(rr_payload.get("close"))
    high = _safe_float(rr_payload.get("recent_high"))
    low = _safe_float(rr_payload.get("recent_low"))
    if close is None:
        return []

    upper_strike = high or close * 1.1
    lower_strike = low or close * 0.9

    if direction == "bearish":
        return [
            {
                "name": "Put Protection",
                "legs": [
                    {"action": "buy", "type": "put", "strike": _format_price(close * 0.95), "expiration": "3M"},
                    {"action": "sell", "type": "put", "strike": _format_price(lower_strike), "expiration": "3M"},
                ],
                "expiration_notes": "3M",
                "rationale": "应对基本面下行与波动扩张",
            }
        ]

    return [
        {
            "name": "Bull Call Spread",
            "legs": [
                {"action": "buy", "type": "call", "strike": _format_price(close), "expiration": "3-6M"},
                {"action": "sell", "type": "call", "strike": _format_price(upper_strike), "expiration": "3-6M"},
            ],
            "expiration_notes": "3-6M",
            "rationale": "顺势看多同时控制权利金",
        }
    ]


def _build_risk_management(score: Optional[float], rr_payload: Dict[str, Any]) -> Dict[str, Any]:
    base = "20–25%" if (score or 0) >= 3.5 else "15–20%"
    max_exposure = "35%" if (score or 0) >= 3.5 else "25%"
    low = _format_price(_safe_float(rr_payload.get("recent_low")))
    add = f"回踩 {low} 加仓" if low != "—" else "回踩 5% 内加仓"
    stop = f"跌破 {low} 止损" if low != "—" else "跌破关键支撑止损"
    odds = "极佳" if (score or 0) >= 4.0 else "中性" if (score or 0) >= 3.0 else "防守"
    rr = rr_payload.get("ratio")
    rr_text = f"胜率 × 盈亏比 ≈ {rr:.1f}" if isinstance(rr, (int, float)) and rr else "胜率 × 盈亏比 待更新"
    return {
        "initial_position": f"{base}（核心仓位）",
        "max_exposure": max_exposure,
        "add_rule": add,
        "stop_loss_rule": stop,
        "odds_rating": odds,
        "win_rate_rr": rr_text,
    }


def _build_execution_notes(action_card: str) -> Dict[str, List[str]]:
    action = action_card.lower() if action_card else ""
    if "short" in action or "reduce" in action:
        cycles = ["T+1", "T+5"]
    else:
        cycles = ["T+3", "T+7"]
    return {
        "observation_cycle": cycles,
        "signals_to_watch": [
            "机构持仓变化",
            "期权成交偏度",
            "财报/指引更新",
        ],
    }


def _build_scenarios(
    rr_payload: Dict[str, Any],
    score: Optional[float],
    action_card: str,
) -> List[Dict[str, Any]]:
    direction = _direction_from_score(score)
    base_prob = _probability_from_score(score)
    alt_prob = round(max(0.15, 1 - base_prob), 2)
    high = _safe_float(rr_payload.get("recent_high"))
    low = _safe_float(rr_payload.get("recent_low"))

    support = [low] if low else []
    resistance = [high] if high else []

    base_description = "综合八维因子后，主路径维持当前趋势。"
    alt_description = "若关键因子走弱，则进入震荡/回调通道。"

    alt_type = "bear_case" if direction == "bullish" else "bull_case"
    return [
        {
            "type": "base_case",
            "label": "主路径 (Base Case)",
            "probability": round(base_prob, 2),
            "direction": direction,
            "description": base_description,
            "support": support,
            "resistance": resistance,
            "timeframe_notes": "1-3 个月滚动观察",
            "catalysts": [action_card] if action_card else None,
        },
        {
            "type": alt_type,
            "label": "次要路径 (Alt Case)",
            "probability": alt_prob,
            "direction": "bearish" if direction == "bullish" else "bullish",
            "description": alt_description,
            "support": support,
            "resistance": resistance,
            "timeframe_notes": "关注关键宏观/业绩事件",
            "catalysts": ["权重因子突变", "监管/供给冲击"],
        },
    ]


async def build_advanced_insights(
    ticker: str,
    *,
    total_score: Optional[float],
    action_card: str,
    factors: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        rr_payload = await summarize_risk_reward(ticker)
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("advanced_insights_risk_reward_failed", extra={"ticker": ticker, "error": str(exc)})
        rr_payload = {}

    direction = _direction_from_score(total_score)
    return {
        "scenarios": _build_scenarios(rr_payload, total_score, action_card),
        "key_variables": _build_key_variables(factors),
        "stock_strategy": _build_stock_strategy(rr_payload, direction),
        "option_strategies": _build_option_strategies(rr_payload, direction),
        "risk_management": _build_risk_management(total_score, rr_payload),
        "execution_notes": _build_execution_notes(action_card),
    }
