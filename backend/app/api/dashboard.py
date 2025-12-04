from fastapi import APIRouter

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_dashboard_summary() -> dict:
    """Return a static placeholder summary until the engine is ready."""
    return {
        "model": "AION v4.0",
        "tickers": ["NVDA", "MSFT", "AAPL"],
        "highlights": [
            "Macro momentum improving",
            "F3: Fundamental strength in NVDA",
            "F8: IV compression in tech"],
    }
