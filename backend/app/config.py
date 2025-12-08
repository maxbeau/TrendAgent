from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MacroThresholds(BaseModel):
    net_liquidity_change: list[float] = Field(default_factory=lambda: [-50000, -20000, 0.0, 20000])
    yield_curve: list[float] = Field(default_factory=lambda: [-1.0, 0.0, 0.5, 1.0])
    credit_spread: list[float] = Field(default_factory=lambda: [2.0, 3.0, 4.0, 6.0])
    rate_expectations: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])
    global_demand: list[float] = Field(default_factory=lambda: [-0.03, -0.01, 0.0, 0.02])


class Settings(BaseSettings):
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
    yfinance_proxy: Optional[str] = "http://127.0.0.1:7890"

    # Frontend dev server origins allowed to call the API.
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    fmp_base_url: str = "https://financialmodelingprep.com/api"
    massive_base_url: str = "https://api.massive.com"
    fred_base_url: str = "https://api.stlouisfed.org"
    macro_thresholds: MacroThresholds = MacroThresholds()

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
