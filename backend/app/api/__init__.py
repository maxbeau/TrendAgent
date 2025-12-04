from fastapi import APIRouter

from . import dashboard, engine, health


router = APIRouter(prefix="/api")
router.include_router(dashboard.router)
router.include_router(engine.router)
router.include_router(health.router)
