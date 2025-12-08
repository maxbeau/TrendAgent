from fastapi import APIRouter

from . import dashboard, engine, health, market, narrative, meta


router = APIRouter(prefix="/api")
router.include_router(dashboard.router)
router.include_router(engine.router)
router.include_router(health.router)
router.include_router(market.router)
router.include_router(narrative.router)
router.include_router(meta.router)
