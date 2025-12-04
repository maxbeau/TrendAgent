from datetime import datetime


def calculate(ticker: str, model_version: str) -> dict:
    """Mock calculation until the actual factor pipeline is in place."""
    return {
        "ticker": ticker,
        "model_version": model_version,
        "total_score": 3.8,
        "action_card": "Waiting for confirmation",
        "calculated_at": datetime.utcnow(),
    }


def build_narrative_context(ticker: str) -> dict:
    """Return a minimal skeleton for the narrative builder."""
    return {
        "ticker": ticker,
        "primary_insight": "Placeholder narrative context",
        "qualitative_signals": [],
    }
