from fastapi import FastAPI

from app.api import router as api_router


def create_app() -> FastAPI:
    """Build the FastAPI application with all routers and middleware."""
    app = FastAPI(title="TrendAgent API", version="0.1.0")
    app.include_router(api_router)
    return app
