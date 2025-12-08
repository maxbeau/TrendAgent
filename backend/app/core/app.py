from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import router as api_router


def create_app() -> FastAPI:
    """Build the FastAPI application with all routers and middleware."""
    settings = get_settings()
    app = FastAPI(title="TrendAgent API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app
