"""Configuration management for AI Trading Team."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class APIConfig:
    """API configuration for exchanges and LLM."""

    anthropic_base_url: str = ""
    anthropic_api_key: str = ""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    weex_api_key: str = ""
    weex_api_secret: str = ""
    weex_passphrase: str = ""


@dataclass
class TradingConfig:
    """Trading parameters configuration."""

    symbol: str = "BTCUSDT"
    exchange: str = "weex"
    leverage: int = 20  # Default 20x leverage (competition limit)
    max_position_percent: float = 75.0  # Max 75% of available balance for positions
    dry_run: bool = False  # Simulate trades without real execution
    prompt_style: str = "neutral"  # Prompt style: "neutral", "long" (bullish), "short" (bearish)


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""

    bot_token: str = ""
    chat_id: str = ""
    account_label: str = ""


@dataclass
class Config:
    """Main configuration container."""

    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Config":
        """Load configuration from environment variables.

        Args:
            env_path: Path to .env file (optional)

        Returns:
            Config instance populated from environment
        """
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        api = APIConfig(
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
            weex_api_key=os.getenv("WEEX_API_KEY", ""),
            weex_api_secret=os.getenv("WEEX_API_SECRET", ""),
            weex_passphrase=os.getenv("WEEX_PASSPHRASE", ""),
        )

        leverage_raw = int(os.getenv("TRADING_LEVERAGE", "20"))
        leverage = min(max(leverage_raw, 1), 20)

        # Validate prompt_style
        prompt_style_raw = os.getenv("PROMPT_STYLE", "neutral").strip().lower()
        if prompt_style_raw not in ("neutral", "long", "short"):
            prompt_style_raw = "neutral"

        trading = TradingConfig(
            symbol=os.getenv("TRADING_SYMBOL", "BTCUSDT"),
            exchange=os.getenv("TRADING_EXCHANGE", "weex").strip().lower(),
            leverage=leverage,
            max_position_percent=float(os.getenv("TRADING_MAX_POSITION_PERCENT", "75.0")),
            dry_run=os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes"),
            prompt_style=prompt_style_raw,
        )

        telegram = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            account_label=os.getenv("TELEGRAM_ACCOUNT_LABEL", ""),
        )

        return cls(api=api, trading=trading, telegram=telegram)
