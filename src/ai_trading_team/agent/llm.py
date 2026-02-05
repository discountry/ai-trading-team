"""LLM client configuration for langchain-anthropic."""

from typing import Any

from ai_trading_team.config import Config


def create_llm(config: Config) -> Any:
    """Create LangChain LLM client.

    Args:
        config: Application configuration

    Returns:
        Configured ChatAnthropic instance

    Note:
        Uses third-party API endpoint from config (e.g., api.minimax.io)
    """
    # Lazy import to avoid loading langchain when not needed
    from langchain_anthropic import ChatAnthropic
    from pydantic import SecretStr

    return ChatAnthropic(
        model_name="MiniMax-M2.1",
        api_key=SecretStr(config.api.anthropic_api_key),
        base_url=config.api.anthropic_base_url,
        max_tokens_to_sample=4096,
        timeout=60.0,
        stop=None,
    )
