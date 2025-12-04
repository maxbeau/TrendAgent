from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://localhost/trendagent"
    supabase_url: str = ""
    supabase_key: str = ""
    fmp_api_key: str = ""
    massive_api_key: str = ""
    fred_api_key: str = ""
    openai_api_key: str = ""
    openai_model_name: str = "gpt-5.1"
    openai_base_url: str = "https://api.openai.com/v1"
    yfinance_proxy: Optional[str] = "http://127.0.0.1:7890"

    fmp_base_url: str = "https://financialmodelingprep.com/api"
    massive_base_url: str = "https://api.massive.com"
    fred_base_url: str = "https://api.stlouisfed.org"

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
