from typing import Optional, List

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MacroThresholds(BaseModel):
    net_liquidity_change: list[float] = Field(default_factory=lambda: [-50000, -20000, 0.0, 20000])
    yield_curve: list[float] = Field(default_factory=lambda: [-1.0, 0.0, 0.5, 1.0])
    credit_spread: list[float] = Field(default_factory=lambda: [2.0, 3.0, 4.0, 6.0])
    rate_expectations: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])
    global_demand: list[float] = Field(default_factory=lambda: [-0.03, -0.01, 0.0, 0.02])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(..., env="DATABASE_URL")
    supabase_url: str = ""
    supabase_key: str = ""
    fmp_api_key: str = ""
    massive_api_key: str = ""
    fred_api_key: str = ""
    openai_api_key: str = ""
    openai_model_name: str = "gpt-5.1"
    openai_small_model_name: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    http_proxy: str = Field(default="", env="HTTP_PROXY")
    https_proxy: str = Field(default="", env="HTTPS_PROXY")

    # Frontend dev server origins allowed to call the API.
    # In production, set ALLOWED_ORIGINS as a comma-separated string of domains.
    # Example: "https://your-app.vercel.app,https://another-domain.com"
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        env="ALLOWED_ORIGINS",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v

    fmp_base_url: str = "https://financialmodelingprep.com/api"
    massive_base_url: str = "https://api.massive.com"
    fred_base_url: str = "https://api.stlouisfed.org"
    macro_thresholds: MacroThresholds = MacroThresholds()


def get_settings() -> Settings:
    return Settings()
