from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings


settings = get_settings()

def _normalized_db_url(raw: str) -> str:
    """
    Ensure async driver (asyncpg) is used in all environments.
    """
    if raw.startswith("postgres://"):
        return raw.replace("postgres://", "postgresql+asyncpg://", 1)
    if raw.startswith("postgresql://") and "+asyncpg" not in raw:
        return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    return raw


engine = create_async_engine(
    _normalized_db_url(settings.database_url),
    future=True,
    echo=False,
    connect_args={"statement_cache_size": 0},
)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
