from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from app.config import get_settings

settings = get_settings()


def _require_api_key() -> None:
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is missing; set OPENAI_API_KEY in the environment.")


def build_llm(*, model: Optional[str] = None, temperature: float = 0.2) -> ChatOpenAI:
    """
    Standard ChatOpenAI client for primary models (叙事/策略等主链路).
    """
    _require_api_key()
    return ChatOpenAI(
        model=model or settings.openai_model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
    )


def build_small_llm(*, temperature: float = 0.2) -> ChatOpenAI:
    """
    Smaller/cheaper model for压缩/评分类任务，若未配置则回退主模型。
    """
    _require_api_key()
    model_name = settings.openai_small_model_name or settings.openai_model_name
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
    )
