from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}
