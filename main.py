"""AI Trading Team - Entry point with event-driven signal system.

This trading bot:
1. Fetches market data from Binance (configured via TRADING_SYMBOL)
2. Uses EVENT-DRIVEN signals (only on state changes, not periodic evaluation)
3. Uses LangChain agent to make trading decisions when signals trigger
4. Executes trades on the configured exchange (WEEX/Binance, or MockExecutor in DRY_RUN)
5. Implements comprehensive risk control (25% force stop, 10% profit signals)

Signal System:
- MA Crossover: Price crosses above/below MA (triggers on crossover only)
- RSI Extremes: RSI enters/exits overbought/oversold zones
- Funding Rate: Significant funding rate shifts
- Long/Short Ratio: Significant ratio changes

Signals are NOT periodic - they only fire when state CHANGES.
"""

import asyncio
import contextlib
import html
import logging
import signal
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ai_trading_team.agent.commands import AgentAction
from ai_trading_team.agent.schemas import AgentDecision
from ai_trading_team.agent.trader import LangChainTradingAgent
from ai_trading_team.config import Config
from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.core.health import HealthMonitor
from ai_trading_team.core.session import SessionManager
from ai_trading_team.core.types import EventType, OrderType, Side
from ai_trading_team.data.manager import BinanceDataManager
from ai_trading_team.execution.binance.executor import BinanceExecutor
from ai_trading_team.execution.models import Position
from ai_trading_team.execution.mock_executor import MockExecutor
from ai_trading_team.execution.weex.executor import WEEXExecutor
from ai_trading_team.execution.weex.stream import WEEXPrivateStream
from ai_trading_team.logging import setup_logging
from ai_trading_team.notifications import TelegramNotifier
from ai_trading_team.risk.monitor import RiskMonitor
from ai_trading_team.risk.rules import DynamicTakeProfitRule, ForceStopLossRule, TrailingStopRule
from ai_trading_team.strategy.signals import (
    Signal,
    SignalAggregator,
    SignalDirection,
    SignalStrength,
    SignalType,
    Timeframe,
)
from ai_trading_team.strategy.state_machine import (
    PositionContext,
    StateTransition,
    StrategyState,
    StrategyStateMachine,
)

# TUI support - imported only when needed
TUI_AVAILABLE = False
TradingApp: type | None = None
try:
    from ai_trading_team.ui.app import TradingApp as _TradingApp

    TradingApp = _TradingApp
    TUI_AVAILABLE = True
except ImportError:
    pass


def normalize_exchange(exchange: str) -> str:
    """Normalize exchange name and validate."""
    normalized = exchange.strip().lower()
    if normalized not in {"weex", "binance"}:
        raise ValueError(f"Unsupported exchange: {exchange}")
    return normalized


def get_symbol_mapping(symbol: str) -> dict[str, str]:
    """Convert trading symbol to exchange-specific formats.

    Args:
        symbol: Base symbol (e.g., "DOGEUSDT", "BTCUSDT")

    Returns:
        Mapping of exchange -> symbol
        - Binance: uppercase (e.g., "DOGEUSDT")
        - WEEX: cmt_ prefix + lowercase (e.g., "cmt_dogeusdt")
    """
    binance_symbol = symbol.upper()
    return {
        "binance": binance_symbol,
        "weex": f"cmt_{symbol.lower()}",
    }


class TradingBot:
    """Main trading bot with event-driven signal system and state recovery."""

    def __init__(self, config: Config) -> None:
        """Initialize trading bot.

        Args:
            config: Application configuration
        """
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._running = False

        # TUI reference for sending updates (set externally)
        self._tui_app: Any = None

        self._exchange = normalize_exchange(config.trading.exchange)
        symbol_map = get_symbol_mapping(config.trading.symbol)
        self._binance_symbol = symbol_map["binance"]
        self._execution_symbol = symbol_map[self._exchange]

        # Core components
        self._data_pool = DataPool()
        self._telegram_notifier = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id,
            account_label=config.telegram.account_label
            or f"{self._exchange}:{self._execution_symbol}",
        )
        self._telegram_lock = asyncio.Lock()
        self._last_position_notice: dict[str, Any] | None = None
        self._position_notify_ready = False
        self._kline_update_counter = 0
        self._kline_update_intervals: set[str] = set()
        self._data_pool.subscribe(self._handle_data_event)

        # Session manager for state persistence
        self._session_manager = SessionManager(
            symbol=self._execution_symbol,
            session_dir="sessions",
            auto_save_interval=30,
        )

        # Data module - Binance for market data
        self._data_manager = BinanceDataManager(
            self._data_pool,
            config.api.binance_api_key,
            config.api.binance_api_secret,
        )

        # Event-driven signal aggregator (replaces old orchestrator)
        self._signal_aggregator = SignalAggregator(
            data_pool=self._data_pool,
            symbol=self._execution_symbol,
        )

        # State machine for trading lifecycle
        self._state_machine = StrategyStateMachine(
            symbol=self._execution_symbol,
            cooldown_seconds=60,
            force_stop_loss_percent=Decimal("25"),
            profit_signal_threshold=Decimal("10"),
        )

        # Agent for trading decisions
        self._agent = LangChainTradingAgent(config, self._execution_symbol)

        # Execution - use MockExecutor in DRY_RUN mode
        self._executor: MockExecutor | WEEXExecutor | BinanceExecutor
        if config.trading.dry_run:
            self._executor = MockExecutor(
                data_pool=self._data_pool,
                initial_balance=Decimal("1000"),
                leverage=config.trading.leverage,
            )
        else:
            if self._exchange == "weex":
                self._executor = WEEXExecutor(
                    config.api.weex_api_key,
                    config.api.weex_api_secret,
                    config.api.weex_passphrase,
                )
            else:
                self._executor = BinanceExecutor(
                    config.api.binance_api_key,
                    config.api.binance_api_secret,
                )

        # WEEX private WebSocket stream for real-time account/position/order updates
        self._weex_stream: WEEXPrivateStream | None = None
        if not config.trading.dry_run and self._exchange == "weex":
            self._weex_stream = WEEXPrivateStream(
                config.api.weex_api_key,
                config.api.weex_api_secret,
                config.api.weex_passphrase,
            )

        # Risk monitor with strategy-specific rules
        self._risk_monitor = RiskMonitor(self._data_pool, self._executor)
        self._setup_risk_rules()

        # Pending signals to process
        self._pending_signals: list[Signal] = []

        # State saving interval
        self._last_state_save = 0.0
        self._state_save_interval = 30.0  # Save every 30 seconds
        self._last_orders_sync = 0.0
        self._orders_sync_interval = 2.0  # Seconds between open order syncs
        self._open_orders_cache: dict[str, dict[str, Any]] = {}
        self._pending_entry_order_id: str | None = None
        self._signal_task: asyncio.Task[None] | None = None
        self._position_sync_interval = 5.0
        self._last_position_sync = 0.0
        self._last_ws_position_update = 0.0
        self._last_ws_account_update = 0.0
        self._last_ws_orders_update = 0.0
        self._ws_position_ttl = 12.0
        self._ws_account_ttl = 12.0
        self._ws_orders_ttl = 8.0
        self._last_backlog_log = 0.0
        self._backlog_log_interval = 30.0
        self._no_position_count = 0
        self._no_position_threshold = 2
        self._pending_close: dict[str, Any] | None = None
        self._pending_close_started = 0.0
        self._profit_signal_task: asyncio.Task[None] | None = None
        self._max_pending_signals = 200
        self._pending_signal_ttl = 300.0
        self._last_queue_drop_log = 0.0
        self._queue_drop_log_interval = 30.0
        self._last_signal_suppress_log = 0.0
        self._signal_suppress_log_interval = 30.0
        self._maker_fee_rate = 0.0004  # 0.04%
        self._taker_fee_rate = 0.0001  # 0.01%
        # 往返手续费：Taker开仓 + Maker平仓 = 0.01% + 0.04% = 0.05%
        self._fee_round_trip_pct = float((self._taker_fee_rate + self._maker_fee_rate) * 100)
        self._min_hold_seconds = 180.0
        # 最低入场波动要求：往返手续费 × 4倍
        self._min_entry_move_pct = self._fee_round_trip_pct * 4.0
        self._min_trade_move_pct = self._fee_round_trip_pct * 2.0
        self._min_close_move_pct = self._min_entry_move_pct
        self._min_flip_move_pct = self._fee_round_trip_pct * 2.0
        self._last_closed_side: Side | None = None
        self._last_closed_price: float | None = None
        self._last_closed_time: datetime | None = None

        # Auto breakeven stop loss tracking
        # When price moves 2% in profit direction, move SL to 0.5% profit (breakeven+)
        self._breakeven_sl_moved = False  # Track if breakeven SL has been set for current position
        self._breakeven_trigger_pct = 2.0  # Trigger at 2% profit
        self._breakeven_sl_lock_pct = 0.5  # Lock in 0.5% profit

        # Health monitor for periodic status logging (every 60 seconds)
        self._health_monitor = HealthMonitor(
            data_manager=self._data_manager,
            executor=self._executor,
            data_pool=self._data_pool,
            state_machine=self._state_machine,
            interval_seconds=60,
        )

    def _setup_risk_rules(self) -> None:
        """Configure risk control rules per STRATEGY.md."""
        # Force stop-loss at 25% margin loss - no agent needed
        self._risk_monitor.add_rule(
            ForceStopLossRule(
                name="force_stop_loss_25",
                force_stop_loss_percent=Decimal("25.0"),
                priority=100,
            )
        )

        # Dynamic take-profit signals at 10% increments - agent decides
        self._risk_monitor.add_rule(
            DynamicTakeProfitRule(
                name="dynamic_take_profit_10",
                profit_threshold_percent=Decimal("10.0"),
                priority=50,
            )
        )

        # Trailing stop for profit protection
        self._risk_monitor.add_rule(
            TrailingStopRule(
                name="trailing_stop",
                activation_profit_percent=Decimal("15.0"),
                trail_distance_percent=Decimal("7.0"),
                priority=80,
            )
        )

        self._logger.info(
            "Risk rules configured: 25% force stop, 10% profit signals, trailing stop"
        )

    def set_tui(self, tui_app: Any) -> None:
        """Set TUI app reference for sending updates.

        Args:
            tui_app: TradingApp instance
        """
        self._tui_app = tui_app

    def _get_current_price(self) -> float:
        """Get current market price from data pool with validation.

        Returns:
            Current price, or 0.0 if unavailable (with warning logged)
        """
        if not self._data_pool.ticker:
            self._logger.warning("Ticker data unavailable when getting current price")
            return 0.0

        price = float(self._data_pool.ticker.get("last_price", 0))
        if price <= 0:
            self._logger.warning(f"Invalid price from ticker: {price}")
            return 0.0

        return price

    def _to_decimal(self, value: Any) -> Decimal:
        try:
            return Decimal(str(value))
        except (TypeError, ValueError, InvalidOperation):
            return Decimal("0")

    def _position_context_from_snapshot(
        self, position: dict[str, Any] | None
    ) -> PositionContext | None:
        if not position:
            return None

        size = self._to_decimal(position.get("size"))
        if size <= 0:
            return None

        side_raw = str(position.get("side") or "").lower()
        if side_raw == "long":
            side = Side.LONG
        elif side_raw == "short":
            side = Side.SHORT
        else:
            return None

        entry_price = self._to_decimal(position.get("entry_price"))
        margin = self._to_decimal(position.get("margin"))
        leverage = int(position.get("leverage") or self._config.trading.leverage)

        return PositionContext(
            symbol=self._execution_symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            margin=margin,
            leverage=leverage,
        )

    def _position_context_from_executor(self, position: Position) -> PositionContext:
        return PositionContext(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            size=position.size,
            margin=position.margin,
            leverage=position.leverage,
        )

    def _sync_state_with_position(self, position: Position | None) -> None:
        if position and not self._state_machine.has_position:
            pos_ctx = self._position_context_from_executor(position)
            self._logger.info("Detected open position; syncing state machine context")
            if self._state_machine.can_transition(StateTransition.AGENT_OPEN):
                self._state_machine.transition(StateTransition.AGENT_OPEN, {"position": pos_ctx})

    def _calculate_lock_in_stop_loss(
        self,
        position: Position,
        threshold_level: float,
    ) -> float | None:
        if threshold_level <= 0:
            return None

        leverage = position.leverage or self._config.trading.leverage or 1
        if leverage <= 0:
            return None

        entry_price = float(position.entry_price)
        if entry_price <= 0:
            return None

        lock_in_percent = max(threshold_level - 10.0, 0.0)
        offset = (lock_in_percent / 100.0) / float(leverage)

        if position.side == Side.LONG:
            target = entry_price * (1 + offset)
            current = self._state_machine._context.position.stop_loss_price
            if current and target < current:
                return current
            return target

        target = entry_price * (1 - offset)
        current = self._state_machine._context.position.stop_loss_price
        if current and target > current:
            return current
        return target

    def _get_signal_category(self, signal_type: SignalType) -> str:
        if signal_type in (SignalType.MA_CROSS_UP, SignalType.MA_CROSS_DOWN):
            return "ma_crossover"
        if signal_type in (
            SignalType.RSI_ENTER_OVERSOLD,
            SignalType.RSI_EXIT_OVERSOLD,
            SignalType.RSI_ENTER_OVERBOUGHT,
            SignalType.RSI_EXIT_OVERBOUGHT,
        ):
            return "rsi_extreme_transition"
        if signal_type in (SignalType.MACD_GOLDEN_CROSS, SignalType.MACD_DEATH_CROSS):
            return "macd_crossover"
        if signal_type in (
            SignalType.BB_BREAK_UPPER,
            SignalType.BB_BREAK_LOWER,
            SignalType.BB_RETURN_UPPER,
            SignalType.BB_RETURN_LOWER,
        ):
            return "bollinger_event"
        if signal_type in (
            SignalType.FUNDING_SPIKE_POSITIVE,
            SignalType.FUNDING_SPIKE_NEGATIVE,
            SignalType.FUNDING_NORMALIZE,
        ):
            return "funding_rate_shift"
        if signal_type in (SignalType.LS_RATIO_SURGE, SignalType.LS_RATIO_DROP):
            return "long_short_ratio_change"
        if signal_type in (SignalType.OI_SURGE, SignalType.OI_DROP):
            return "open_interest_change"
        if signal_type in (SignalType.LIQUIDATION_LONG, SignalType.LIQUIDATION_SHORT):
            return "liquidation_event"
        if signal_type in (
            SignalType.PNL_PROFIT_INCREASE,
            SignalType.PNL_PROFIT_DECREASE,
        ):
            return "pnl_change"
        if signal_type in (
            SignalType.RISK_FORCE_STOP_LOSS,
            SignalType.RISK_TRAILING_STOP,
            SignalType.RISK_TAKE_PROFIT,
        ):
            return "risk_event"
        if signal_type in (
            SignalType.ORDER_FILLED,
            SignalType.ORDER_CANCELLED,
            SignalType.ORDER_PARTIAL_FILL,
        ):
            return "order_event"
        if signal_type in (
            SignalType.BULLISH_CONFLUENCE,
            SignalType.BEARISH_CONFLUENCE,
        ):
            return "multi_signal_confluence"
        return "signal_event"

    def _sanitize_signal_for_agent(self, signal: Signal) -> dict[str, Any]:
        return {
            "category": self._get_signal_category(signal.signal_type),
            "timeframe": signal.timeframe.value,
            "source": signal.source,
            "event": "triggered",
        }

    def _get_volatility_percent(self, snapshot: Any) -> float | None:
        indicators = snapshot.indicators or {}
        composite = indicators.get("ATR_14_COMPOSITE")
        if isinstance(composite, (int, float)):
            return float(composite)

        if not snapshot.klines:
            return None
        for interval in ("15m", "1h", "4h"):
            klines = snapshot.klines.get(interval, [])
            if not klines:
                continue
            last = klines[-1]
            high = self._safe_float(last.get("high")) if isinstance(last, dict) else None
            low = self._safe_float(last.get("low")) if isinstance(last, dict) else None
            close = self._safe_float(last.get("close")) if isinstance(last, dict) else None
            if high is None or low is None or close is None or close <= 0:
                continue
            return (high - low) / close * 100
        return None

    def _should_block_entry(
        self,
        decision: AgentDecision,
        snapshot: Any,
    ) -> tuple[bool, str]:
        if decision.command.action not in (AgentAction.OPEN, AgentAction.ADD):
            return False, ""
        if not decision.command.side:
            return True, "entry blocked: missing side"

        volatility_pct = self._get_volatility_percent(snapshot)
        if volatility_pct is None:
            return True, "entry blocked: volatility unavailable"
        if self._min_entry_move_pct > 0 and volatility_pct < self._min_entry_move_pct:
            return (
                True,
                f"entry blocked: volatility {volatility_pct:.3f}% < "
                f"min {self._min_entry_move_pct:.3f}%",
            )

        last_side = self._last_closed_side
        last_price = self._last_closed_price
        if last_side and last_price and last_price > 0 and decision.command.side != last_side:
            current_price = None
            if decision.command.price is not None:
                current_price = self._safe_float(decision.command.price)
            if current_price is None and snapshot.ticker:
                current_price = self._safe_float(snapshot.ticker.get("last_price"))
            if current_price is None or current_price <= 0:
                current_price = self._get_current_price()
            if current_price > 0:
                move_pct = abs(current_price - last_price) / last_price * 100
                if move_pct < self._min_flip_move_pct:
                    return (
                        True,
                        f"entry blocked: flip move {move_pct:.3f}% < "
                        f"fee {self._min_flip_move_pct:.3f}%",
                    )

        return False, ""

    def _calculate_reward_risk(
        self,
        side: Side,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> float | None:
        if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
            return None

        if side == Side.LONG:
            risk = entry_price - stop_loss_price
            reward = take_profit_price - entry_price
        else:
            risk = stop_loss_price - entry_price
            reward = entry_price - take_profit_price

        if risk <= 0 or reward <= 0:
            return None

        return reward / risk

    def _notify_signal(self, signal_type: str, details: str) -> None:
        """Send signal notification to TUI.

        Args:
            signal_type: Type of signal
            details: Signal details
        """
        if self._tui_app:
            self._tui_app.add_signal(signal_type, details)

    def _notify_agent_log(self, action: str, details: str, result: str = "") -> None:
        """Send agent log to TUI.

        Args:
            action: Action taken
            details: Action details
            result: Result of action
        """
        if self._tui_app:
            self._tui_app.add_agent_log(action, details, result)

    def _notify_risk_event(self, event_type: str, message: str) -> None:
        """Send risk event to TUI.

        Args:
            event_type: Type of risk event
            message: Event message
        """
        if self._tui_app:
            self._tui_app.add_risk_event(event_type, message)

    def _handle_data_event(self, event_type: EventType, data: Any) -> None:
        if event_type == EventType.KLINE_UPDATE:
            if isinstance(data, dict):
                interval = data.get("interval")
                if interval:
                    self._kline_update_intervals.add(interval)
            self._kline_update_counter += 1
            return
        if event_type != EventType.POSITION_UPDATED:
            return
        if not self._telegram_notifier.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._handle_position_notification(data))

    async def _handle_position_notification(self, position: dict[str, Any] | None) -> None:
        async with self._telegram_lock:
            await self._process_position_notification(position)

    async def _process_position_notification(self, position: dict[str, Any] | None) -> None:
        current = self._normalize_position_for_notify(position)
        if not self._position_notify_ready:
            self._position_notify_ready = True
            self._last_position_notice = current
            return

        previous = self._last_position_notice
        self._last_position_notice = current

        if not previous and not current:
            return

        if previous and current and previous["side"] != current["side"]:
            await self._send_position_message("Position Closed", previous)
            await self._send_position_message("Position Opened", current)
            return

        if not previous and current:
            await self._send_position_message("Position Opened", current)
            return

        if previous and not current:
            await self._send_position_message("Position Closed", previous)
            return

        if previous and current:
            delta = current["size"] - previous["size"]
            if abs(delta) < 1e-9:
                return
            title = "Position Increased" if delta > 0 else "Position Reduced"
            await self._send_position_message(title, current, delta=delta)

    def _normalize_position_for_notify(
        self, position: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not position:
            return None

        size = self._safe_float(position.get("size")) or 0.0
        if size <= 0:
            return None

        side_raw = str(position.get("side") or "").lower()
        side = side_raw if side_raw in ("long", "short") else "unknown"
        entry_price = self._safe_float(position.get("entry_price")) or 0.0
        leverage = int(position.get("leverage") or self._config.trading.leverage)
        symbol = str(position.get("symbol") or self._execution_symbol)

        return {
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "leverage": leverage,
        }

    def _format_number(self, value: float) -> str:
        formatted = f"{value:.6f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    async def _send_position_message(
        self,
        title: str,
        position: dict[str, Any],
        delta: float | None = None,
    ) -> None:
        label = self._telegram_notifier.account_label
        lines = [f"<b>{html.escape(title)}</b>"]
        if label:
            lines.append(f"Account: {html.escape(label)}")
        lines.append(f"Symbol: {html.escape(position['symbol'])}")
        lines.append(f"Side: {html.escape(position['side'])}")
        if delta is not None:
            delta_value = self._format_number(delta)
            if delta > 0:
                delta_value = f"+{delta_value}"
            lines.append(f"Delta: {html.escape(delta_value)}")
        lines.append(f"Size: {html.escape(self._format_number(position['size']))}")
        if position["entry_price"] > 0:
            lines.append(
                f"Entry: {html.escape(self._format_number(position['entry_price']))}"
            )
        if position["leverage"] > 0:
            lines.append(f"Leverage: {position['leverage']}x")
        message = "\n".join(lines)
        await self._telegram_notifier.send_message(message)

    def _ensure_snapshot_position(self, snapshot: Any) -> Any:
        if snapshot.position:
            return snapshot
        if not self._state_machine.has_position:
            return snapshot
        pos = self._state_machine.context.position
        if not pos.side or pos.size <= 0:
            return snapshot
        snapshot.position = {
            "symbol": pos.symbol,
            "side": pos.side.value,
            "size": float(pos.size),
            "entry_price": float(pos.entry_price),
            "margin": float(pos.margin),
            "leverage": pos.leverage,
            "unrealized_pnl": float(pos.unrealized_pnl),
            "stop_loss_price": pos.stop_loss_price,
        }
        return snapshot

    def _should_invoke_agent(self, snapshot: Any) -> bool:
        self._agent.update_volatility(snapshot)
        if self._agent.volatility_ready():
            return True
        sample_count, min_samples = self._agent.volatility_sample_status()
        self._logger.info(
            "AI skipped: volatility data incomplete (%d/%d)",
            sample_count,
            min_samples,
        )
        self._notify_agent_log(
            "SKIP",
            "Data incomplete",
            f"volatility {sample_count}/{min_samples}",
        )
        return False

    def _suppress_signals_by_entry_fee(
        self,
        snapshot: Any,
        signals: list[Signal],
    ) -> list[Signal]:
        if not signals:
            return signals
        if not self._state_machine.has_position:
            return signals
        pos = self._state_machine.context.position
        if not pos.side or pos.size <= 0:
            return signals
        try:
            entry_price = float(pos.entry_price)
        except (TypeError, ValueError):
            return signals
        if entry_price <= 0:
            return signals

        current_price = None
        if snapshot.ticker:
            with contextlib.suppress(TypeError, ValueError):
                current_price = float(snapshot.ticker.get("last_price", 0))
        if not current_price or current_price <= 0:
            current_price = self._get_current_price()
        if not current_price or current_price <= 0:
            return signals

        move_pct = abs(current_price - entry_price) / entry_price * 100
        threshold_pct = self._fee_round_trip_pct * 1.5
        if move_pct >= threshold_pct:
            return signals

        critical_types = {
            SignalType.RISK_FORCE_STOP_LOSS,
            SignalType.RISK_TRAILING_STOP,
            SignalType.RISK_TAKE_PROFIT,
            SignalType.ORDER_FILLED,
            SignalType.ORDER_CANCELLED,
            SignalType.ORDER_PARTIAL_FILL,
        }
        filtered = [signal for signal in signals if signal.signal_type in critical_types]

        if len(filtered) != len(signals):
            now_ts = asyncio.get_event_loop().time()
            if now_ts - self._last_signal_suppress_log >= self._signal_suppress_log_interval:
                self._logger.info(
                    "Signal suppression: move %.3f%% < %.3f%% (entry %.6f, current %.6f)",
                    move_pct,
                    threshold_pct,
                    entry_price,
                    current_price,
                )
                self._last_signal_suppress_log = now_ts

        return filtered

    def _queue_signals(self, signals: list[Signal]) -> None:
        if not signals:
            return

        now = datetime.now()
        # Drop stale signals first
        if self._pending_signal_ttl > 0 and self._pending_signals:
            dropped = 0
            while self._pending_signals:
                age = (now - self._pending_signals[0].timestamp).total_seconds()
                if age <= self._pending_signal_ttl:
                    break
                self._pending_signals.pop(0)
                dropped += 1
            if dropped:
                self._logger.warning(f"Dropped {dropped} stale signals from queue")

        self._pending_signals.extend(signals)

        # Cap queue length
        if len(self._pending_signals) > self._max_pending_signals:
            overflow = len(self._pending_signals) - self._max_pending_signals
            for _ in range(overflow):
                self._pending_signals.pop(0)
            now_ts = asyncio.get_event_loop().time()
            if now_ts - self._last_queue_drop_log >= self._queue_drop_log_interval:
                self._logger.warning(
                    f"Signal queue overflow: dropped {overflow} oldest signals"
                )
                self._last_queue_drop_log = now_ts

    def _build_close_info_from_context(self, reason: str) -> dict[str, Any] | None:
        pos = self._state_machine._context.position
        if not pos.side or pos.size <= 0:
            return None
        return {
            "action": "close",
            "side": pos.side.value,
            "size": float(pos.size),
            "entry_price": float(pos.entry_price),
            "exit_price": self._get_current_price(),
            "pnl": float(pos.unrealized_pnl),
            "reason": reason[:100] if reason else None,
        }

    def _set_pending_close(self, info: dict[str, Any] | None) -> None:
        if info:
            self._pending_close = info
        self._pending_close_started = asyncio.get_event_loop().time()
        self._logger.info("Close order placed; awaiting position confirmation")

    def _finalize_close(self, reason: str) -> None:
        info = self._pending_close or self._build_close_info_from_context(reason)
        if info and info.get("size") and info.get("side"):
            pnl = float(info.get("pnl", 0.0))
            entry_price = float(info.get("entry_price", 0.0))
            exit_price = float(info.get("exit_price", 0.0))
            size = float(info.get("size", 0.0))
            side = str(info.get("side") or "")
            self._data_pool.record_trade(
                pnl=pnl,
                entry_price=entry_price,
                exit_price=exit_price,
                side=side,
                size=size,
            )
            operation = {
                "timestamp": datetime.now().isoformat(),
                "action": info.get("action", "close"),
                "side": side,
                "size": size,
                "pnl": pnl,
                "reason": info.get("reason"),
                "result": "success",
                "order_id": info.get("order_id"),
            }
            if info.get("source"):
                operation["source"] = info.get("source")
            if info.get("type"):
                operation["type"] = info.get("type")
            self._data_pool.add_operation(operation)
            self._session_manager.add_operation(operation)
            if exit_price > 0:
                if side == "long":
                    self._last_closed_side = Side.LONG
                elif side == "short":
                    self._last_closed_side = Side.SHORT
                else:
                    self._last_closed_side = None
                self._last_closed_price = exit_price
                self._last_closed_time = datetime.now()

        if self._state_machine.can_transition(StateTransition.POSITION_CLOSED):
            self._state_machine.transition(StateTransition.POSITION_CLOSED)
        self._risk_monitor.reset_rules()
        self._pending_close = None
        self._pending_close_started = 0.0

    async def _upload_execution_log(
        self,
        decision: AgentDecision,
        stage: str,
        order: Any | None,
        extra_output: dict[str, Any] | None = None,
    ) -> None:
        """Upload AI log with order execution details when available."""
        output: dict[str, Any] = {
            "action": decision.command.action.value,
            "symbol": decision.command.symbol or self._execution_symbol,
            "side": decision.command.side.value if decision.command.side else None,
            "size": decision.command.size,
            "order_type": decision.command.order_type.value
            if decision.command.order_type
            else None,
        }
        if order:
            output.update(
                {
                    "order_id": getattr(order, "order_id", None),
                    "executed_size": float(getattr(order, "size", 0) or 0),
                    "executed_price": float(getattr(order, "price", 0) or 0)
                    if getattr(order, "price", None) is not None
                    else None,
                    "status": getattr(order, "status", None).value
                    if getattr(order, "status", None)
                    else None,
                }
            )
        if extra_output:
            output.update(extra_output)

        try:
            await self._executor.upload_ai_log(
                stage=stage,
                model=decision.model,
                input_data={
                    "decision": {
                        "action": decision.command.action.value,
                        "symbol": decision.command.symbol or self._execution_symbol,
                        "side": decision.command.side.value if decision.command.side else None,
                        "size": decision.command.size,
                        "price": decision.command.price,
                        "take_profit_price": decision.command.take_profit_price,
                        "order_type": decision.command.order_type.value
                        if decision.command.order_type
                        else None,
                        "reason": decision.command.reason,
                    }
                },
                output=output,
                explanation=decision.command.reason,
                order_id=getattr(order, "order_id", None),
            )
        except Exception as e:
            self._logger.warning(f"Failed to upload execution AI log: {e}")

    def _safe_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_ws_items(
        self,
        message: dict[str, Any],
        keys: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if not isinstance(message, dict):
            return []

        data = message.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, list):
                    return value

        msg = message.get("msg")
        if isinstance(msg, dict):
            msg_data = msg.get("data")
            if isinstance(msg_data, list):
                return msg_data
            if isinstance(msg_data, dict):
                for key in keys:
                    value = msg_data.get(key)
                    if isinstance(value, list):
                        return value

        return []

    async def _handle_weex_account_ws(self, message: dict[str, Any]) -> None:
        items = self._extract_ws_items(message, ("account", "accounts", "collateral"))
        if not items:
            return

        for item in items:
            coin = str(item.get("coinId") or item.get("coin") or "").upper()
            if coin and coin != "USDT":
                continue
            amount = self._safe_float(item.get("amount"))
            if amount is None:
                continue

            snapshot = self._data_pool.get_snapshot().account or {}
            self._data_pool.update_account(
                {
                    "balance": amount,
                    "available": snapshot.get("available", amount),
                    "margin": snapshot.get("margin", 0.0),
                }
            )
            self._last_ws_account_update = asyncio.get_event_loop().time()
            break

    async def _handle_weex_position_ws(self, message: dict[str, Any]) -> None:
        items = self._extract_ws_items(message, ("position", "positions"))
        if not items:
            return

        snapshot = self._data_pool.get_snapshot().position or {}

        for item in items:
            contract_id = item.get("contractId") or item.get("symbol")
            if contract_id and contract_id != self._execution_symbol:
                continue

            size = self._safe_float(
                item.get("size")
                or item.get("total")
                or item.get("holdSize")
                or item.get("positionSize")
            ) or 0.0
            if size == 0:
                self._data_pool.update_position(None)
                return

            side_raw = str(item.get("side", "")).lower()
            side = side_raw if side_raw in ("long", "short") else snapshot.get("side", "long")

            open_value = self._safe_float(item.get("openValue"))
            entry_price = snapshot.get("entry_price", 0.0)
            if open_value and size:
                entry_price = open_value / size
            else:
                ws_entry = self._safe_float(item.get("openPrice") or item.get("open_price"))
                if ws_entry is not None:
                    entry_price = ws_entry

            margin = self._safe_float(
                item.get("isolatedMargin")
                or item.get("margin")
                or item.get("marginSize")
                or item.get("positionMargin")
            )
            if margin is None:
                margin = snapshot.get("margin", 0.0)

            leverage_raw = self._safe_float(item.get("leverage"))
            leverage = (
                int(leverage_raw)
                if leverage_raw is not None and leverage_raw > 0
                else int(snapshot.get("leverage", self._config.trading.leverage))
            )

            unrealized = self._safe_float(
                item.get("unrealizePnl")
                or item.get("unrealisedPnl")
                or item.get("unrealizedPnl")
                or item.get("unrealizedPnL")
            )
            if unrealized is None:
                unrealized = snapshot.get("unrealized_pnl", 0.0)

            self._data_pool.update_position(
                {
                    "symbol": self._execution_symbol,
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "unrealized_pnl": unrealized,
                    "margin": margin,
                    "leverage": leverage,
                }
            )
            self._last_ws_position_update = asyncio.get_event_loop().time()
            return

    async def _handle_weex_orders_ws(self, message: dict[str, Any]) -> None:
        items = self._extract_ws_items(message, ("order", "orders"))
        if not items:
            return

        for item in items:
            contract_id = item.get("contractId") or item.get("symbol")
            if contract_id and contract_id != self._execution_symbol:
                continue

            order_id = str(item.get("id") or item.get("orderId") or "")
            if not order_id:
                continue

            status_raw = str(item.get("status", "")).upper()
            if status_raw in {"FILLED", "CANCELED", "CANCELLED"}:
                self._open_orders_cache.pop(order_id, None)
                continue

            side_raw = str(item.get("orderSide") or item.get("side") or "").upper()
            side = "long" if side_raw in {"BUY", "LONG"} else "short" if side_raw in {"SELL", "SHORT"} else ""

            self._open_orders_cache[order_id] = {
                "order_id": order_id,
                "orderId": order_id,
                "symbol": self._execution_symbol,
                "side": side,
                "price": self._safe_float(item.get("price")),
                "size": self._safe_float(item.get("size")) or 0.0,
                "quantity": self._safe_float(item.get("size")) or 0.0,
                "filled_size": self._safe_float(item.get("cumFillSize") or item.get("filledSize")) or 0.0,
                "status": str(item.get("status", "OPEN")).lower(),
                "order_type": str(item.get("type", "LIMIT")).lower(),
            }

        self._data_pool.update_orders(list(self._open_orders_cache.values()))
        self._last_ws_orders_update = asyncio.get_event_loop().time()

    async def _start_weex_stream(self) -> None:
        if not self._weex_stream:
            return
        try:
            await self._weex_stream.connect()
            await self._weex_stream.subscribe_account(self._handle_weex_account_ws)
            await self._weex_stream.subscribe_positions(self._handle_weex_position_ws)
            await self._weex_stream.subscribe_orders(self._handle_weex_orders_ws)
            self._logger.info("WEEX private WebSocket subscriptions active")
        except Exception as e:
            self._logger.warning(f"WEEX WebSocket unavailable, fallback to REST: {e}")

    async def _stop_weex_stream(self) -> None:
        if self._weex_stream and self._weex_stream.is_connected:
            await self._weex_stream.disconnect()

    async def start(self) -> None:
        """Start the trading bot with state recovery support."""
        self._logger.info("=" * 60)
        self._logger.info("AI Trading Team - Event-Driven Signal System")
        self._logger.info("=" * 60)
        self._logger.info(f"Binance Symbol: {self._binance_symbol}")
        self._logger.info(f"Trade Exchange: {self._exchange}")
        self._logger.info(f"Trade Symbol: {self._execution_symbol}")
        self._logger.info(f"Leverage: {self._config.trading.leverage}x")
        self._logger.info("Signal System: Event-driven (state changes only)")
        self._logger.info("Signals: MA Crossover, RSI Extremes, Funding Rate, L/S Ratio")

        if self._config.trading.dry_run:
            self._logger.info("Mode: DRY_RUN (simulated trading, $1000 initial balance)")
        else:
            self._logger.info(f"Mode: LIVE (real trading on {self._exchange.upper()})")

        self._logger.info("=" * 60)

        self._running = True

        # Connect to executor
        try:
            await self._executor.connect()
            if self._config.trading.dry_run:
                self._logger.info("Mock executor connected")
            else:
                self._logger.info(f"Connected to {self._executor.name}")

            if self._weex_stream:
                await self._start_weex_stream()

            # Set leverage
            await self._executor.set_leverage(
                self._execution_symbol, self._config.trading.leverage
            )
            self._logger.info(f"Leverage set to {self._config.trading.leverage}x")

            # Initialize trading statistics with initial balance
            account = await self._executor.get_account()
            initial_balance = float(account.total_equity)
            self._data_pool.init_trading_stats(initial_balance)
            self._logger.info(f"Trading stats initialized with balance: ${initial_balance:.2f}")

        except Exception as e:
            if self._config.trading.dry_run:
                self._logger.error(f"Failed to initialize mock executor: {e}")
                return
            else:
                self._logger.error(f"Failed to connect to {self._exchange}: {e}")
                return

        # Attempt state recovery
        await self._recover_state()

        # Start data collection from Binance with multiple timeframes
        # Note: data_manager.start() calls initialize() which fetches all required timeframes
        await self._data_manager.start(
            self._binance_symbol,
            kline_interval="15m",
            kline_intervals=["15m", "1h", "4h"],
        )

        # Wait for data to be ready before proceeding
        if not self._data_manager.is_data_ready:
            self._logger.error("Data initialization failed - cannot start signal processing")
            return

        self._logger.info(f"Data ready: all timeframes loaded for {self._binance_symbol}")

        snapshot = self._data_pool.get_snapshot()
        backfill_samples = self._agent.backfill_volatility(snapshot)
        if backfill_samples:
            self._logger.info(
                "Volatility backfill complete: %d samples loaded",
                backfill_samples,
            )
        oi_samples = await self._backfill_open_interest()
        if oi_samples:
            self._logger.info(
                "Open interest backfill complete: %d samples loaded",
                oi_samples,
            )

        # Start background tasks
        asyncio.create_task(self._market_metrics_loop())
        asyncio.create_task(self._signal_update_loop())
        asyncio.create_task(self._state_save_loop())

        # Start health monitor (logs status every 60 seconds)
        await self._health_monitor.start()

        # Main loop
        await self._run_loop()

    async def _recover_state(self) -> None:
        """Recover state from previous session.

        Reconciles local saved state with actual exchange data.
        """
        # Check for saved session
        if not self._session_manager.has_saved_session:
            self._logger.info("No saved session found, starting fresh")
            self._session_manager.create_session()
            return

        # Load saved session
        saved_state = self._session_manager.load_session()
        if not saved_state:
            self._logger.warning("Failed to load saved session, starting fresh")
            self._session_manager.create_session()
            return

        recovery_info = self._session_manager.get_recovery_info()
        self._logger.info(f"Loaded session: {recovery_info}")

        # Fetch actual position from exchange
        try:
            actual_position = await self._executor.get_position(self._execution_symbol)
        except Exception as e:
            self._logger.error(f"Failed to fetch position for recovery: {e}")
            actual_position = None

        # Reconcile state
        saved_has_position = saved_state.position is not None
        actual_has_position = actual_position is not None and actual_position.size > 0

        if saved_has_position and actual_has_position:
            # Both have positions - verify they match
            self._logger.info("Recovering IN_POSITION state from saved session")

            # Restore state machine to IN_POSITION
            pos_ctx = PositionContext(
                symbol=self._execution_symbol,
                side=actual_position.side,
                entry_price=actual_position.entry_price,
                size=actual_position.size,
                margin=actual_position.margin,
                leverage=actual_position.leverage,
                unrealized_pnl=actual_position.unrealized_pnl,
            )
            self._state_machine._context.current_state = StrategyState.IN_POSITION
            self._state_machine._context.position = pos_ctx

            # Restore P&L tracking from saved state
            if saved_state.highest_pnl_percent:
                self._state_machine._context.position.highest_pnl_percent = Decimal(
                    saved_state.highest_pnl_percent
                )
            if saved_state.last_profit_threshold:
                self._state_machine._context.position.last_profit_signal_threshold = Decimal(
                    saved_state.last_profit_threshold
                )

            self._logger.info(
                f"Recovered position: {actual_position.side.value} "
                f"size={actual_position.size}, entry={actual_position.entry_price}"
            )

        elif not saved_has_position and actual_has_position:
            # Exchange has position but saved state doesn't - adopt it
            self._logger.warning(
                "Exchange has position but saved session doesn't - adopting exchange position"
            )

            pos_ctx = PositionContext(
                symbol=self._execution_symbol,
                side=actual_position.side,
                entry_price=actual_position.entry_price,
                size=actual_position.size,
                margin=actual_position.margin,
                leverage=actual_position.leverage,
                unrealized_pnl=actual_position.unrealized_pnl,
            )
            self._state_machine._context.current_state = StrategyState.IN_POSITION
            self._state_machine._context.position = pos_ctx

        elif saved_has_position and not actual_has_position:
            # Saved state has position but exchange doesn't - position was closed externally
            self._logger.warning(
                "Saved session has position but exchange doesn't - position was closed externally"
            )
            self._session_manager.clear_position()
            self._state_machine._context.current_state = StrategyState.IDLE

        else:
            # Neither has position - restore to IDLE
            self._logger.info("No position on exchange, restoring IDLE state")
            self._state_machine._context.current_state = StrategyState.IDLE

        # Restore trade statistics
        if saved_state.trades_today:
            self._state_machine._context.trades_today = saved_state.trades_today
            self._state_machine._context.wins_today = saved_state.wins_today
            self._state_machine._context.losses_today = saved_state.losses_today

        # Restore pending signals if any
        if saved_state.pending_signals:
            self._logger.info(f"Skipping {len(saved_state.pending_signals)} stale pending signals")
            self._session_manager.clear_pending_signals()

        # Log recovery summary
        self._logger.info(
            f"State recovery complete: state={self._state_machine.state.value}, "
            f"has_position={self._state_machine.has_position}"
        )

    async def _backfill_open_interest(self) -> int:
        rest_client = self._data_manager._rest_client
        history_by_period: dict[str, list[dict[str, Any]]] = {}
        for period in ("15m", "1h", "4h"):
            try:
                history_by_period[period] = await rest_client.get_open_interest_history(
                    self._binance_symbol,
                    period=period,
                    limit=30,
                )
            except Exception as e:
                self._logger.debug(f"Failed to backfill OI ({period}): {e}")
        return self._agent.backfill_open_interest(history_by_period)

    async def _state_save_loop(self) -> None:
        """Periodically save state to disk."""
        while self._running:
            try:
                await asyncio.sleep(self._state_save_interval)
                self._sync_session_state()
                self._session_manager.save_if_dirty()
            except Exception as e:
                self._logger.error(f"Error saving state: {e}")

    def _sync_session_state(self) -> None:
        """Sync current state to session manager."""
        # Update strategy state
        self._session_manager.update_strategy_state(
            state=self._state_machine.state.value,
            previous=self._state_machine._context.previous_state.value
            if self._state_machine._context.previous_state
            else None,
        )

        # Update position if in position
        if self._state_machine.has_position:
            pos = self._state_machine._context.position
            if pos.side:
                self._session_manager.update_position(
                    symbol=self._execution_symbol,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    margin=pos.margin,
                    leverage=pos.leverage,
                )
                self._session_manager.update_pnl_tracking(
                    highest_pnl_percent=pos.highest_pnl_percent,
                    last_profit_threshold=pos.last_profit_signal_threshold,
                )
        else:
            self._session_manager.clear_position()

    async def _market_metrics_loop(self) -> None:
        """Periodically fetch funding rate, long/short ratio, etc."""
        last_oi_hist_refresh = 0.0
        oi_hist_interval = 300.0
        while self._running:
            try:
                rest_client = self._data_manager._rest_client

                # Fetch funding rate
                try:
                    funding = await rest_client.get_funding_rate(self._binance_symbol)
                    self._data_pool.update_funding_rate(
                        {
                            "funding_rate": float(funding.funding_rate),
                            "funding_time": funding.funding_time.isoformat()
                            if funding.funding_time
                            else None,
                        }
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to fetch funding rate: {e}")

                # Fetch long/short ratio
                try:
                    ls_ratio = await rest_client.get_long_short_ratio(self._binance_symbol)
                    self._data_pool.update_long_short_ratio(
                        {
                            "long_ratio": float(ls_ratio.long_ratio),
                            "short_ratio": float(ls_ratio.short_ratio),
                            "long_short_ratio": float(ls_ratio.long_short_ratio),
                            "timestamp": ls_ratio.timestamp.isoformat(),
                        }
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to fetch L/S ratio: {e}")

                # Fetch open interest
                try:
                    oi = await rest_client.get_open_interest(self._binance_symbol)
                    oi_payload = {
                        "open_interest": float(oi.open_interest),
                        "timestamp": oi.timestamp.isoformat(),
                    }
                    self._data_pool.update_open_interest(oi_payload)
                    self._agent.update_open_interest(oi_payload)
                except Exception as e:
                    self._logger.debug(f"Failed to fetch OI: {e}")

                # Fetch mark price
                try:
                    mark = await rest_client.get_mark_price(self._binance_symbol)
                    self._data_pool.update_mark_price(mark)
                except Exception as e:
                    self._logger.debug(f"Failed to fetch mark price: {e}")

                try:
                    now_ts = asyncio.get_event_loop().time()
                    if now_ts - last_oi_hist_refresh >= oi_hist_interval:
                        last_oi_hist_refresh = now_ts
                        history_by_period: dict[str, list[dict[str, Any]]] = {}
                        for period in ("15m", "1h", "4h"):
                            try:
                                history_by_period[period] = (
                                    await rest_client.get_open_interest_history(
                                        self._binance_symbol,
                                        period=period,
                                        limit=30,
                                    )
                                )
                            except Exception as e:
                                self._logger.debug(
                                    f"Failed to refresh OI history ({period}): {e}"
                                )
                        if history_by_period:
                            self._agent.backfill_open_interest(history_by_period)
                except Exception as e:
                    self._logger.debug(f"Failed to refresh OI history: {e}")

            except Exception as e:
                self._logger.error(f"Error in market metrics loop: {e}")

            # Fetch every 30 seconds
            await asyncio.sleep(30)

    async def _signal_update_loop(self) -> None:
        """Update signal sources when new data arrives.

        This is the key difference from the old system:
        - We check for signals when DATA CHANGES, not on a fixed timer
        - Signals are only emitted when state CHANGES
        """
        # Wait for signal aggregator to be ready before processing
        while self._running and not self._signal_aggregator.is_ready:
            # Let aggregator check data readiness
            snapshot = self._data_pool.get_snapshot()
            if snapshot.klines and snapshot.ticker:
                # Trigger readiness check
                self._signal_aggregator.update()
                self._agent.update_volatility(snapshot)
            if self._signal_aggregator.is_ready:
                self._logger.info("Signal aggregator ready - starting signal processing")
                break
            await asyncio.sleep(1)

        if not self._running:
            return

        # Initialize last_refresh with current time to prevent immediate refresh
        current_time = asyncio.get_event_loop().time()

        # Refresh klines periodically for each timeframe
        timeframe_intervals = {
            "15m": 120,  # Check 15m klines every 2min
            "1h": 300,  # Check 1h klines every 5min
            "4h": 600,  # Check 4h klines every 10min
        }
        last_refresh: dict[str, float] = {tf: current_time for tf in timeframe_intervals}
        last_kline_counter = self._kline_update_counter

        while self._running:
            try:
                current_time = asyncio.get_event_loop().time()

                # Refresh klines for each timeframe based on interval
                for interval, refresh_interval in timeframe_intervals.items():
                    if current_time - last_refresh[interval] >= refresh_interval:
                        last_refresh[interval] = current_time
                        await self._refresh_klines(interval)
                if self._kline_update_counter != last_kline_counter:
                    last_kline_counter = self._kline_update_counter
                    intervals = set(self._kline_update_intervals)
                    self._kline_update_intervals.clear()

                    snapshot = self._data_pool.get_snapshot()
                    has_timeframe_updates = False
                    for interval in intervals:
                        timeframe = self._interval_to_timeframe(interval)
                        if not timeframe:
                            continue
                        has_timeframe_updates = True
                        signals = self._signal_aggregator.update(timeframe)
                        if signals:
                            signals = self._suppress_signals_by_entry_fee(snapshot, signals)
                            if signals:
                                for signal in signals:
                                    self._logger.info(
                                        "Signal queued: %s %s %s %s",
                                        signal.signal_type.value,
                                        signal.timeframe.value,
                                        signal.strength.value,
                                        signal.direction.value,
                                    )
                                self._queue_signals(signals)

                    if has_timeframe_updates:
                        self._signal_aggregator.update_indicators()
                        snapshot = self._data_pool.get_snapshot()
                        self._agent.update_volatility(snapshot)

            except Exception as e:
                self._logger.error(f"Error in signal update loop: {e}")

            await asyncio.sleep(1)  # Check every second for near-real-time updates

    async def _refresh_klines(self, interval: str) -> None:
        """Refresh klines for a specific interval.

        Args:
            interval: Kline interval (5m, 15m, 1h, 4h)
        """
        try:
            rest_client = self._data_manager._rest_client
            klines = await rest_client.get_klines(self._binance_symbol, interval, limit=100)
            kline_dicts = [self._data_manager._kline_to_dict(k) for k in klines]
            self._data_pool.update_klines(interval, kline_dicts)
        except Exception as e:
            self._logger.debug(f"Failed to refresh {interval} klines: {e}")

    def _interval_to_timeframe(self, interval: str) -> Timeframe | None:
        """Convert interval string to Timeframe enum.

        Args:
            interval: Interval string (5m, 15m, 1h, 4h)

        Returns:
            Timeframe enum or None
        """
        mapping = {
            "15m": Timeframe.M15,
            "1h": Timeframe.H1,
            "4h": Timeframe.H4,
        }
        return mapping.get(interval)

    async def _run_loop(self) -> None:
        """Main trading loop."""
        risk_check_interval = 5.0  # Check risk every 5 seconds to reduce REST pressure
        last_risk_check = 0.0

        while self._running:
            try:
                current_time = asyncio.get_event_loop().time()

                # Check risk rules (highest priority, most frequent)
                if current_time - last_risk_check >= risk_check_interval:
                    last_risk_check = current_time
                    await self._check_risk()

                # Handle state timeouts (e.g., waiting_entry)
                prev_state = self._state_machine.state
                timed_out = self._state_machine.check_timeout()
                if timed_out and prev_state == StrategyState.WAITING_ENTRY:
                    pending_id = self._pending_entry_order_id
                    if pending_id:
                        try:
                            await self._executor.cancel_order(
                                self._execution_symbol, pending_id
                            )
                            self._logger.warning(
                                f"Canceled stale entry order after timeout: {pending_id}"
                            )
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to cancel stale entry order {pending_id}: {e}"
                            )
                        self._pending_entry_order_id = None

                # Process pending signals (event-driven, not periodic)
                if self._signal_task and self._signal_task.done():
                    with contextlib.suppress(Exception):
                        self._signal_task.result()
                    self._signal_task = None
                if self._profit_signal_task and self._profit_signal_task.done():
                    with contextlib.suppress(Exception):
                        self._profit_signal_task.result()
                    self._profit_signal_task = None

                # Drop stale queued signals before processing
                if self._pending_signals and self._pending_signal_ttl > 0:
                    dropped = 0
                    now_dt = datetime.now()
                    while self._pending_signals:
                        age = (now_dt - self._pending_signals[0].timestamp).total_seconds()
                        if age <= self._pending_signal_ttl:
                            break
                        self._pending_signals.pop(0)
                        dropped += 1
                    if dropped:
                        self._logger.warning(f"Dropped {dropped} stale signals from queue")

                if self._pending_signals and self._signal_task is None:
                    signal = self._pending_signals.pop(0)
                    self._signal_task = asyncio.create_task(self._process_signal(signal))
                elif self._pending_signals and self._signal_task is not None:
                    if current_time - self._last_backlog_log >= self._backlog_log_interval:
                        self._logger.info(
                            "AI busy; %d signals pending", len(self._pending_signals)
                        )
                        self._last_backlog_log = current_time

                # Update position data periodically
                await self._update_position_data()

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    async def _update_position_data(self) -> None:
        """Update position data in data pool."""
        try:
            now = asyncio.get_event_loop().time()
            if now - self._last_position_sync < self._position_sync_interval:
                return
            self._last_position_sync = now

            snapshot = self._data_pool.get_snapshot()
            ws_position_recent = (
                self._weex_stream
                and self._weex_stream.is_connected
                and (now - self._last_ws_position_update) < self._ws_position_ttl
            )
            ws_account_recent = (
                self._weex_stream
                and self._weex_stream.is_connected
                and (now - self._last_ws_account_update) < self._ws_account_ttl
            )
            ws_orders_recent = (
                self._weex_stream
                and self._weex_stream.is_connected
                and (now - self._last_ws_orders_update) < self._ws_orders_ttl
            )

            position = await self._executor.get_position(self._execution_symbol)
            position_size = float(position.size) if position else 0.0
            snapshot_position = snapshot.position if ws_position_recent else None
            snapshot_size = (
                self._safe_float(snapshot_position.get("size")) if snapshot_position else 0.0
            )
            has_position_now = (position and position_size > 0) or (
                snapshot_position and snapshot_size > 0
            )

            if position and position_size > 0:
                self._data_pool.update_position(
                    {
                        "symbol": position.symbol,
                        "side": position.side.value,
                        "size": float(position.size),
                        "entry_price": float(position.entry_price),
                        "unrealized_pnl": float(position.unrealized_pnl),
                        "margin": float(position.margin),
                        "leverage": position.leverage,
                    }
                )
            elif ws_position_recent and snapshot_position and snapshot_size > 0:
                self._logger.debug("Keeping WS position snapshot; REST returned empty")
            elif not ws_position_recent or (snapshot_position and snapshot_size <= 0):
                self._data_pool.update_position(None)

            account = await self._executor.get_account()
            margin_value = float(account.used_margin)
            if margin_value <= 0:
                source_position: Position | None = position
                snapshot_position = snapshot.position if ws_position_recent else None
                if not source_position and snapshot_position:
                    margin_value = self._safe_float(snapshot_position.get("margin")) or 0.0
                    if margin_value <= 0:
                        leverage = (
                            int(snapshot_position.get("leverage") or self._config.trading.leverage)
                        )
                        entry_price = self._safe_float(snapshot_position.get("entry_price")) or 0.0
                        size = self._safe_float(snapshot_position.get("size")) or 0.0
                        if entry_price > 0 and size > 0 and leverage > 0:
                            margin_value = (size * entry_price) / max(leverage, 1)
                elif source_position:
                    margin_value = float(source_position.margin) if source_position.margin else 0.0
                    if margin_value <= 0:
                        leverage = source_position.leverage or self._config.trading.leverage or 1
                        entry_price = (
                            float(source_position.entry_price) if source_position.entry_price else 0.0
                        )
                        if entry_price > 0:
                            margin_value = (
                                float(source_position.size) * entry_price
                            ) / max(leverage, 1)
            self._data_pool.update_account(
                {
                    "balance": float(account.total_equity),
                    "available": float(account.available_balance),
                    "margin": margin_value,
                }
            )
            if not ws_account_recent:
                self._last_ws_account_update = 0.0

            if position:
                self._sync_state_with_position(position)
            elif ws_position_recent and snapshot.position and not self._state_machine.has_position:
                pos_ctx = self._position_context_from_snapshot(snapshot.position)
                if pos_ctx:
                    self._logger.info("Syncing state from WS position snapshot")
                    if self._state_machine.can_transition(StateTransition.AGENT_OPEN):
                        self._state_machine.transition(
                            StateTransition.AGENT_OPEN, {"position": pos_ctx}
                        )

            # Update trading statistics with current equity
            if position and position_size > 0:
                unrealized = float(position.unrealized_pnl)
            elif ws_position_recent and snapshot_position and snapshot_size > 0:
                unrealized = self._safe_float(snapshot_position.get("unrealized_pnl")) or 0.0
            else:
                unrealized = 0.0
            self._data_pool.update_equity(
                current_equity=float(account.total_equity),
                unrealized_pnl=unrealized,
            )

            if has_position_now:
                self._no_position_count = 0
                pnl_unrealized = Decimal(str(unrealized))
                margin_for_percent = Decimal("0")
                if position and position_size > 0:
                    margin_for_percent = Decimal(str(position.margin or 0))
                elif snapshot_position:
                    margin_for_percent = Decimal(str(snapshot_position.get("margin") or 0))
                if margin_for_percent <= 0 and margin_value > 0:
                    margin_for_percent = Decimal(str(margin_value))
                if margin_for_percent > 0:
                    pnl_percent = (pnl_unrealized / margin_for_percent) * Decimal("100")
                else:
                    pnl_percent = Decimal("0")
                self._state_machine.update_position_metrics(pnl_unrealized, pnl_percent)
                if (
                    self._pending_close
                    and self._pending_close_started > 0
                    and (now - self._pending_close_started)
                    > self._state_machine.context.waiting_exit_timeout
                ):
                    self._logger.warning(
                        "Close confirmation timed out; position still open"
                    )
                    self._pending_close = None
                    self._pending_close_started = 0.0
            else:
                self._no_position_count += 1
                if (
                    self._pending_close
                    and self._pending_close_started > 0
                    and (now - self._pending_close_started)
                    > self._state_machine.context.waiting_exit_timeout
                ):
                    self._logger.warning(
                        "Close confirmation timed out; clearing pending close info"
                    )
                    self._pending_close = None
                    self._pending_close_started = 0.0

                if self._no_position_count >= self._no_position_threshold:
                    if self._pending_close:
                        self._logger.info("Position closed confirmed by exchange")
                        self._finalize_close("position_closed_confirmed")
                    elif self._state_machine.has_position or self._state_machine.state in (
                        StrategyState.WAITING_EXIT,
                        StrategyState.RISK_OVERRIDE,
                    ):
                        self._logger.warning("Position missing; syncing state to closed")
                        self._finalize_close("position_missing")

            now = asyncio.get_event_loop().time()
            if not ws_orders_recent and now - self._last_orders_sync >= self._orders_sync_interval:
                orders = await self._executor.get_open_orders(self._execution_symbol)
                orders_payload = []
                for order in orders:
                    orders_payload.append(
                        {
                            "order_id": order.order_id,
                            "orderId": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "price": float(order.price) if order.price is not None else None,
                            "size": float(order.size),
                            "quantity": float(order.size),
                            "filled_size": float(order.filled_size),
                            "status": order.status.value,
                            "order_type": order.order_type.value,
                        }
                    )
                self._data_pool.update_orders(orders_payload)
                self._open_orders_cache = {
                    payload["order_id"]: payload
                    for payload in orders_payload
                    if payload.get("order_id")
                }
                self._last_orders_sync = now

            # Resolve pending entry orders
            if self._state_machine.state == StrategyState.WAITING_ENTRY:
                if position:
                    pos_ctx = PositionContext(
                        symbol=self._execution_symbol,
                        side=position.side,
                        margin=position.margin,
                    )
                    if self._state_machine.can_transition(StateTransition.AGENT_OPEN):
                        self._state_machine.transition(
                            StateTransition.AGENT_OPEN,
                            {"position": pos_ctx},
                        )
                    self._pending_entry_order_id = None
                else:
                    pending_id = self._pending_entry_order_id
                    open_orders = self._data_pool.get_snapshot().orders or []
                    if pending_id and any(
                        str(o.get("order_id") or o.get("orderId") or "") == pending_id
                        for o in open_orders
                    ):
                        return
                    # No position and no matching open order; keep waiting until timeout
        except Exception as e:
            self._logger.debug(f"Error updating position data: {e}")

    async def _handle_profit_signal(self, action: Any) -> None:
        """Handle profit threshold signal by asking AI to set stop loss.

        Args:
            action: RiskAction with move_stop_loss type
        """
        self._logger.info(f"Processing profit threshold signal: {action.reason}")

        # Get current snapshot and position
        snapshot = self._data_pool.get_snapshot()
        position = await self._executor.get_position(self._execution_symbol)

        if not position:
            self._logger.warning("No position found for profit signal")
            return
        if not self._should_invoke_agent(snapshot):
            return

        self._notify_agent_log("PROFIT", action.reason[:50], "Asking AI for stop loss")

        # Build profit data for AI
        profit_data = {
            "current_pnl_percent": action.data.get("current_pnl_percent", 0),
            "threshold_level": action.data.get("threshold_level", 10),
            "highest_pnl_percent": float(self._state_machine._context.position.highest_pnl_percent),
            "entry_price": float(position.entry_price),
            "position_side": position.side.value,
            "current_margin": float(position.margin),
        }

        # Ask AI agent to decide stop loss price
        try:
            decision = await self._agent.process_profit_signal(snapshot, profit_data)
            self._logger.info(
                f"AI profit decision: action={decision.command.action.value}, "
                f"stop_loss={decision.command.stop_loss_price}"
            )

            # Upload AI log (always when agent returns a decision)
            try:
                await self._executor.upload_ai_log(
                    stage="Profit Signal",
                    model=decision.model,
                    input_data=profit_data,
                    output={
                        "action": decision.command.action.value,
                        "stop_loss_price": decision.command.stop_loss_price,
                    },
                    explanation=decision.command.reason,
                )
            except Exception as e:
                self._logger.warning(f"Failed to upload AI log: {e}")

            # Execute move_stop_loss if AI decided
            if decision.command.action == AgentAction.MOVE_STOP_LOSS:
                stop_loss_price = decision.command.stop_loss_price
                if not stop_loss_price or stop_loss_price <= 0:
                    threshold_level = float(action.data.get("threshold_level", 0))
                    stop_loss_price = self._calculate_lock_in_stop_loss(
                        position,
                        threshold_level,
                    )
                    if stop_loss_price:
                        self._logger.info(
                            "AI stop loss missing; using lock-in fallback at "
                            f"{stop_loss_price:.6f}"
                        )
                if stop_loss_price:
                    await self._execute_move_stop_loss(position, stop_loss_price)
                else:
                    self._logger.warning("AI returned move_stop_loss without price")
                if self._state_machine.can_transition(StateTransition.AGENT_HOLD):
                    self._state_machine.transition(StateTransition.AGENT_HOLD)

            elif decision.command.action == AgentAction.CLOSE:
                # AI decided to close instead of moving stop loss
                if decision.command.side is None:
                    decision.command.side = position.side
                success = await self._execute_command(decision)
                if success:
                    self._logger.info("Close order placed from profit signal; awaiting confirmation")

            elif decision.command.action == AgentAction.OBSERVE:
                self._logger.info("AI decided to observe, not moving stop loss")
                if self._state_machine.can_transition(StateTransition.AGENT_HOLD):
                    self._state_machine.transition(StateTransition.AGENT_HOLD)

        except Exception as e:
            self._logger.error(f"Error processing profit signal: {e}")

    async def _execute_move_stop_loss(self, position: Any, stop_loss_price: float) -> None:
        """Execute stop loss order movement.

        Args:
            position: Current position
            stop_loss_price: New stop loss price
        """
        self._logger.info(f"Moving stop loss to {stop_loss_price:.4f}")

        try:
            # Cancel existing stop loss orders first
            orders = await self._executor.get_open_orders(self._execution_symbol)
            cancelled_count = 0
            for order in orders:
                if order.order_type.value in ("stop", "stop_market", "stop_loss"):
                    await self._executor.cancel_order(self._execution_symbol, order.order_id)
                    self._logger.info(f"Cancelled old stop loss order: {order.order_id}")
                    cancelled_count += 1

            plan_cancelled = 0
            cancel_stop_loss_plans = getattr(self._executor, "cancel_stop_loss_plans", None)
            if callable(cancel_stop_loss_plans):
                plan_cancelled = await cancel_stop_loss_plans(
                    self._execution_symbol,
                    position.side,
                )

            get_stop_loss_plans = getattr(self._executor, "get_stop_loss_plans", None)
            if callable(get_stop_loss_plans):
                remaining = await get_stop_loss_plans(self._execution_symbol, position.side)
                if remaining:
                    self._logger.warning(
                        "Stop loss plan cancel incomplete; skipping new stop loss placement. "
                        f"Remaining plan orders: {', '.join(remaining)}"
                    )
                    self._data_pool.add_operation(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "action": "move_stop_loss",
                            "side": position.side.value,
                            "stop_loss_price": stop_loss_price,
                            "result": "failed",
                            "error": "existing stop-loss plan orders not fully cancelled",
                            "source": "risk",
                            "type": "move_stop_loss",
                        }
                    )
                    return

            place_stop_loss_plan = getattr(self._executor, "place_stop_loss_plan", None)
            plan_order_id = None
            if callable(place_stop_loss_plan):
                plan_order_id = await place_stop_loss_plan(
                    symbol=self._execution_symbol,
                    side=position.side,
                    size=float(position.size),
                    trigger_price=stop_loss_price,
                )

            if plan_order_id:
                self._logger.info(
                    "Stop loss plan order placed: "
                    f"{plan_order_id} (cancelled {cancelled_count} stop orders, "
                    f"{plan_cancelled} plan orders)"
                )

                # Update position context only after successful placement
                if self._state_machine._context.position:
                    self._state_machine._context.position.stop_loss_price = stop_loss_price

                self._notify_agent_log(
                    "STOP_LOSS",
                    f"Target moved to {stop_loss_price:.4f}",
                    "Placed",
                )

                self._data_pool.add_operation(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "action": "move_stop_loss",
                        "side": position.side.value,
                        "stop_loss_price": stop_loss_price,
                        "result": "success",
                        "order_id": plan_order_id,
                        "source": "risk",
                        "type": "move_stop_loss",
                    }
                )
            else:
                self._logger.warning(
                    f"Stop loss move requested to {stop_loss_price:.4f}. "
                    f"Cancelled {cancelled_count} stop orders and {plan_cancelled} plan orders."
                )

                self._notify_agent_log(
                    "STOP_LOSS",
                    f"Target move failed at {stop_loss_price:.4f}",
                    "Not placed",
                )
                self._data_pool.add_operation(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "action": "move_stop_loss",
                        "side": position.side.value,
                        "stop_loss_price": stop_loss_price,
                        "result": "failed",
                        "error": "plan order not placed",
                        "source": "risk",
                        "type": "move_stop_loss",
                    }
                )

        except Exception as e:
            self._logger.error(f"Failed to move stop loss: {e}")
            self._data_pool.add_operation(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "move_stop_loss",
                    "stop_loss_price": stop_loss_price,
                    "result": "failed",
                    "error": str(e),
                    "source": "risk",
                    "type": "move_stop_loss",
                }
            )

    async def _check_risk(self) -> None:
        """Check risk rules and execute if triggered."""
        if not self._state_machine.has_position:
            return

        try:
            action = await self._risk_monitor.evaluate()
            if action:
                self._logger.warning(f"Risk action triggered: {action.reason}")
                self._notify_risk_event("WARNING", action.reason)

                # Handle move_stop_loss action (profit threshold signal)
                if action.action_type == "move_stop_loss":
                    if self._profit_signal_task and not self._profit_signal_task.done():
                        self._logger.info("Profit signal already in progress; skipping")
                        return
                    if self._state_machine.can_transition(StateTransition.PROFIT_THRESHOLD):
                        self._state_machine.transition(StateTransition.PROFIT_THRESHOLD)
                    self._profit_signal_task = asyncio.create_task(
                        self._handle_profit_signal(action)
                    )
                    return

                # Force close for high-priority risk actions
                if action.priority >= 80:
                    self._logger.critical(f"FORCE CLOSING POSITION: {action.reason}")
                    self._notify_risk_event("STOP_LOSS", f"Force close: {action.reason}")

                    if self._state_machine.can_transition(StateTransition.RISK_TRIGGERED):
                        self._state_machine.transition(StateTransition.RISK_TRIGGERED)

                    position = await self._executor.get_position(self._execution_symbol)
                    if position:
                        # Capture position info before closing for trade recording
                        entry_price = float(position.entry_price)
                        exit_price = self._get_current_price()
                        realized_pnl = float(position.unrealized_pnl)
                        trade_side = position.side.value
                        trade_size = float(position.size)
                        reason_upper = action.reason.upper()
                        risk_type = "stop_loss"
                        if "TRAILING STOP" in reason_upper:
                            risk_type = "trailing_stop"

                        order = await self._executor.close_position(
                            symbol=self._execution_symbol,
                            side=position.side,
                            size=None,
                        )
                        if order:
                            self._logger.info(f"Force closed position: {order.order_id}")
                            self._notify_agent_log(
                                "CLOSE", f"Force closed {position.side.value}", "Risk triggered"
                            )

                            self._set_pending_close(
                                {
                                    "action": "force_close",
                                    "side": trade_side,
                                    "size": trade_size,
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "pnl": realized_pnl,
                                    "reason": action.reason,
                                    "order_id": order.order_id,
                                    "source": "risk",
                                    "type": risk_type,
                                }
                            )

            # Auto breakeven stop loss check
            # When price moves 2% in profit direction, automatically move SL to 0.5% profit
            await self._check_auto_breakeven_sl()

        except Exception as e:
            self._logger.error(f"Error checking risk: {e}")

    async def _check_auto_breakeven_sl(self) -> None:
        """Check and execute auto breakeven stop loss.

        When price moves 2% in profit direction, automatically move SL to lock in 0.5% profit.
        This is executed automatically by the system, not by AI decision.
        """
        if self._breakeven_sl_moved:
            # Already moved for this position
            return

        try:
            position = await self._executor.get_position(self._execution_symbol)
            if not position:
                # No position, reset flag
                self._breakeven_sl_moved = False
                return

            entry_price = float(position.entry_price)
            current_price = self._get_current_price()

            if entry_price <= 0 or current_price <= 0:
                return

            # Calculate price movement percentage in profit direction
            if position.side == Side.LONG:
                price_movement_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                price_movement_pct = ((entry_price - current_price) / entry_price) * 100

            # Check if trigger threshold reached (2% profit)
            if price_movement_pct < self._breakeven_trigger_pct:
                return

            # Calculate breakeven+ stop loss price (lock in 0.5% profit)
            if position.side == Side.LONG:
                # Long: SL = entry * (1 + 0.5%) = entry * 1.005
                new_stop_loss = entry_price * (1 + self._breakeven_sl_lock_pct / 100)
            else:  # SHORT
                # Short: SL = entry * (1 - 0.5%) = entry * 0.995
                new_stop_loss = entry_price * (1 - self._breakeven_sl_lock_pct / 100)

            self._logger.info(
                f"AUTO BREAKEVEN: Price moved {price_movement_pct:.2f}% >= {self._breakeven_trigger_pct}%. "
                f"Moving SL to lock in {self._breakeven_sl_lock_pct}% profit: {new_stop_loss:.4f}"
            )

            # Execute the stop loss update (only for WEEX which supports plan orders)
            if hasattr(self._executor, "cancel_stop_loss_plans"):
                # Cancel existing SL plans
                await self._executor.cancel_stop_loss_plans(
                    self._execution_symbol, position.side
                )

                if hasattr(self._executor, "place_stop_loss_plan"):
                    # Place new breakeven+ SL
                    sl_plan_id = await self._executor.place_stop_loss_plan(
                        symbol=self._execution_symbol,
                        side=position.side,
                        size=float(position.size),
                        trigger_price=new_stop_loss,
                    )
                    if sl_plan_id:
                        self._breakeven_sl_moved = True
                        self._logger.info(
                            f"AUTO BREAKEVEN SL placed: {new_stop_loss:.4f} (plan: {sl_plan_id})"
                        )
                        self._notify_agent_log(
                            "BREAKEVEN",
                            f"SL moved to +{self._breakeven_sl_lock_pct}%",
                            f"Lock profit at {new_stop_loss:.4f}",
                        )

                        # Record operation
                        self._data_pool.add_operation(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "action": "auto_breakeven_sl",
                                "side": position.side.value,
                                "entry_price": entry_price,
                                "new_stop_loss": new_stop_loss,
                                "price_movement_pct": price_movement_pct,
                                "lock_pct": self._breakeven_sl_lock_pct,
                                "result": "success",
                            }
                        )
                    else:
                        self._logger.warning("Failed to place breakeven SL plan order")

        except Exception as e:
            self._logger.error(f"Error in auto breakeven SL check: {e}")

    async def _process_signal(self, signal: Signal) -> None:
        """Process a trading signal through the agent.

        Args:
            signal: Signal to process
        """
        try:
            snapshot = self._data_pool.get_snapshot()
            if snapshot.position and not self._state_machine.has_position:
                pos_ctx = self._position_context_from_snapshot(snapshot.position)
                if pos_ctx:
                    self._logger.info("Detected open position in data pool; syncing state")
                    if self._state_machine.can_transition(StateTransition.AGENT_OPEN):
                        self._state_machine.transition(
                            StateTransition.AGENT_OPEN,
                            {"position": pos_ctx},
                        )

            # Only process actionable signals
            if not signal.is_actionable:
                self._logger.info(
                    "Signal not actionable; AI not invoked: %s %s %s %s",
                    signal.signal_type.value,
                    signal.timeframe.value,
                    signal.strength.value,
                    signal.direction.value,
                )
                self._notify_agent_log(
                    "SKIP",
                    "Signal not actionable",
                    signal.signal_type.value,
                )
                return

            snapshot = self._ensure_snapshot_position(snapshot)

            # Check if we should process this signal based on current state
            if self._state_machine.is_idle:
                # IDLE state: process entry signals normally
                await self._process_entry_signal(signal)
            elif self._state_machine.has_position:
                # IN_POSITION state: check if signal suggests closing/reversing
                await self._process_position_signal(signal)
            else:
                # Other states (ANALYZING, WAITING_ENTRY, etc.): skip
                self._logger.info(
                    "Busy state (%s); AI not invoked for signal %s %s %s %s",
                    self._state_machine.state.value,
                    signal.signal_type.value,
                    signal.timeframe.value,
                    signal.strength.value,
                    signal.direction.value,
                )
                self._notify_agent_log(
                    "SKIP",
                    f"State={self._state_machine.state.value}",
                    signal.signal_type.value,
                )
        except Exception as e:
            self._logger.error(f"Error processing signal {signal.signal_type.value}: {e}")

    async def _process_position_signal(self, signal: Signal) -> None:
        """Process a signal while in position.

        When in position, we should consider:
        1. Opposite direction signals (suggest closing current position)
        2. Same direction signals (could suggest adding, but we'll let AI decide)

        Args:
            signal: Signal to process
        """
        # Get current position
        position = await self._executor.get_position(self._execution_symbol)
        if not position:
            self._logger.info(
                "No position found; skipping signal %s %s %s %s",
                signal.signal_type.value,
                signal.timeframe.value,
                signal.strength.value,
                signal.direction.value,
            )
            return
        self._data_pool.update_position(
            {
                "symbol": position.symbol,
                "side": position.side.value,
                "size": float(position.size),
                "entry_price": float(position.entry_price),
                "unrealized_pnl": float(position.unrealized_pnl),
                "margin": float(position.margin),
                "leverage": position.leverage,
            }
        )

        # Determine if signal is opposite to current position
        is_long_position = position.side == Side.LONG
        is_bearish_signal = signal.direction == SignalDirection.BEARISH
        is_bullish_signal = signal.direction == SignalDirection.BULLISH

        is_opposite_signal = (is_long_position and is_bearish_signal) or (
            not is_long_position and is_bullish_signal
        )

        # Only process strong opposite signals (weak signals are noise)
        if not is_opposite_signal:
            self._logger.info(
                "Same-direction signal; skipping AI: %s %s %s %s",
                signal.signal_type.value,
                signal.timeframe.value,
                signal.strength.value,
                signal.direction.value,
            )
            return

        entry_time = self._state_machine.context.position.entry_time
        if entry_time and self._min_hold_seconds > 0:
            held_seconds = (datetime.now() - entry_time).total_seconds()
            if held_seconds < self._min_hold_seconds:
                self._logger.info(
                    "Opposite signal ignored: hold time %.0fs < min %.0fs",
                    held_seconds,
                    self._min_hold_seconds,
                )
                self._notify_agent_log(
                    "HOLD",
                    "Min hold time",
                    f"{held_seconds:.0f}s < {self._min_hold_seconds:.0f}s",
                )
                return

        current_price = self._get_current_price()
        entry_price = float(position.entry_price) if position.entry_price else 0.0
        if current_price > 0 and entry_price > 0 and self._min_close_move_pct > 0:
            move_pct = abs((current_price - entry_price) / entry_price) * 100
            if move_pct < self._min_close_move_pct:
                self._logger.info(
                    "Opposite signal ignored: price move %.3f%% < min %.3f%%",
                    move_pct,
                    self._min_close_move_pct,
                )
                self._notify_agent_log(
                    "HOLD",
                    "Min price move",
                    f"{move_pct:.3f}% < {self._min_close_move_pct:.3f}%",
                )
                return

        if signal.strength == SignalStrength.WEAK:
            self._logger.info(
                "Weak opposite signal; skipping AI: %s %s %s %s",
                signal.signal_type.value,
                signal.timeframe.value,
                signal.strength.value,
                signal.direction.value,
            )
            return

        # Check debounce to avoid repeated signals
        if self._state_machine.should_debounce_signal(signal.signal_type.value):
            self._logger.info(
                "Signal debounced; skipping AI: %s %s %s %s",
                signal.signal_type.value,
                signal.timeframe.value,
                signal.strength.value,
                signal.direction.value,
            )
            return

        self._logger.info(
            f"Opposite signal while in position: {signal.signal_type.value} "
            f"(position: {position.side.value}, signal: {signal.direction.value})"
        )
        self._notify_signal(
            signal.signal_type.value,
            f"⚠️ Opposite signal! {signal.description}",
        )

        # Record signal for debounce
        self._state_machine.record_signal(signal.signal_type.value)

        # Get market snapshot
        snapshot = self._data_pool.get_snapshot()
        if not self._should_invoke_agent(snapshot):
            return

        # Build context for AI to decide whether to close/hold
        from ai_trading_team.core.signal_queue import (
            SignalType as OldSignalType,
        )
        from ai_trading_team.core.signal_queue import (
            StrategySignal,
        )

        old_type = OldSignalType.CUSTOM
        sanitized_signal = self._sanitize_signal_for_agent(signal)
        strategy_signal = StrategySignal(
            signal_type=old_type,
            data={
                **sanitized_signal,
                "context": "position_signal",
                "current_position_side": position.side.value,
                "current_pnl": float(position.unrealized_pnl),
            },
            priority=3,  # High priority for opposite signals
        )

        self._logger.info("Asking AI about opposite signal while in position...")
        self._notify_agent_log("OPPOSITE_SIGNAL", signal.signal_type.value, signal.description)

        # Get agent decision
        decision = await self._agent.process_signal(strategy_signal, snapshot)
        self._logger.info(
            f"Agent decision on opposite signal: {decision.command.action.value} - "
            f"{decision.command.reason[:100]}"
        )

        # Notify TUI about agent decision
        self._notify_agent_log(
            decision.command.action.value.upper(),
            "Response to opposite signal",
            decision.command.reason[:60],
        )

        # Upload AI log only for non-execution actions (execution logs are uploaded after orders)
        if decision.command.action not in (
            AgentAction.OPEN,
            AgentAction.ADD,
            AgentAction.REDUCE,
            AgentAction.CLOSE,
        ):
            try:
                await self._executor.upload_ai_log(
                    stage="Opposite Signal Processing",
                    model=decision.model,
                    input_data={
                        "signal_type": sanitized_signal.get("category", "signal_event"),
                        "signal_data": sanitized_signal,
                        "context": "position_signal",
                        "position_side": position.side.value,
                        "position_pnl": float(position.unrealized_pnl),
                    },
                    output={
                        "action": decision.command.action.value,
                        "reason": decision.command.reason,
                    },
                    explanation=decision.command.reason,
                )
            except Exception as e:
                self._logger.warning(f"Failed to upload AI log: {e}")

        # Handle agent decision
        if decision.command.action == AgentAction.CLOSE:
            self._logger.info("Agent decided to close position on opposite signal")
            await self._execute_close_position(position, decision)
        elif decision.command.action == AgentAction.REDUCE:
            self._logger.info("Agent decided to reduce position on opposite signal")
            await self._execute_reduce_position(position, decision)
        elif decision.command.action == AgentAction.ADD:
            self._logger.info("Agent decided to add to position")
            await self._execute_add_position(position, decision)
        else:
            # OBSERVE, HOLD, or other - do nothing
            self._logger.info(f"Agent decided to hold: {decision.command.reason[:60]}")
            self._notify_agent_log("HOLD", "Keeping position", decision.command.reason[:60])

    async def _execute_close_position(self, position: Any, decision: Any) -> None:
        """Execute position close based on agent decision.

        Args:
            position: Current position
            decision: Agent decision
        """
        try:
            if self._state_machine.can_transition(StateTransition.AGENT_CLOSE):
                self._state_machine.transition(StateTransition.AGENT_CLOSE)

            # Capture position info before closing for trade recording
            entry_price = float(position.entry_price)
            exit_price = self._get_current_price()
            realized_pnl = float(position.unrealized_pnl)
            trade_side = position.side.value
            trade_size = float(position.size)

            order = await self._executor.close_position(
                symbol=self._execution_symbol,
                side=position.side,
                size=None,  # Close full position
            )

            if order:
                self._logger.info(f"Position closed: {order.order_id}")
                self._notify_agent_log("CLOSE", "Position closed", f"PnL: {realized_pnl:.2f}")

                await self._upload_execution_log(
                    decision,
                    stage="Order Execution",
                    order=order,
                    extra_output={
                        "execution_action": "close",
                        "close_all": True,
                    },
                )

                self._set_pending_close(
                    {
                        "action": "close",
                        "side": trade_side,
                        "size": trade_size,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": realized_pnl,
                        "reason": decision.command.reason[:100],
                        "order_id": order.order_id,
                    }
                )

        except Exception as e:
            self._logger.error(f"Failed to close position: {e}")
            self._data_pool.add_operation(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "close",
                    "result": "failed",
                    "error": str(e),
                }
            )

    async def _execute_reduce_position(self, position: Any, decision: Any) -> None:
        """Execute partial position close based on agent decision.

        Args:
            position: Current position
            decision: Agent decision with size to reduce
        """
        try:
            reduce_size = decision.command.size
            if not reduce_size or reduce_size <= 0:
                self._logger.error("REDUCE action requires positive size")
                self._notify_agent_log("REDUCE", "Invalid size", "Skipping")
                return

            # Profit check: Only allow reduce when price moved ≥1% in favorable direction
            # Per prompts: 盈利≥1%时才允许平仓/减仓
            entry_price = float(position.entry_price)
            current_price = self._get_current_price()
            if entry_price > 0 and current_price > 0:
                if position.side == Side.LONG:
                    price_movement_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    price_movement_pct = ((entry_price - current_price) / entry_price) * 100

                MIN_PROFIT_MOVEMENT_PCT = 1.0  # 1% price movement required
                if price_movement_pct < MIN_PROFIT_MOVEMENT_PCT:
                    self._logger.warning(
                        f"REDUCE rejected: price movement {price_movement_pct:.2f}% < {MIN_PROFIT_MOVEMENT_PCT}%. "
                        f"Per prompts: only allow reduce when profit >= 1%"
                    )
                    self._notify_agent_log(
                        "REDUCE",
                        "Profit too low",
                        f"Movement {price_movement_pct:.2f}% < 1%",
                    )
                    return

            # Cap reduce size to position size
            position_size = float(position.size)
            if reduce_size >= position_size:
                self._logger.info("Reduce size >= position size, closing full position")
                await self._execute_close_position(position, decision)
                return

            # Calculate proportional P&L for the reduced portion
            entry_price = float(position.entry_price)
            exit_price = self._get_current_price()
            total_unrealized_pnl = float(position.unrealized_pnl)
            reduce_ratio = reduce_size / position_size
            partial_pnl = total_unrealized_pnl * reduce_ratio

            order = await self._executor.reduce_position(
                symbol=self._execution_symbol,
                side=position.side,
                size=reduce_size,
            )

            if order:
                self._logger.info(f"Position reduced by {reduce_size}: {order.order_id}")
                self._notify_agent_log(
                    "REDUCE",
                    f"Reduced by {reduce_size:.4f}",
                    f"Partial PnL: {partial_pnl:.2f}",
                )

                await self._upload_execution_log(
                    decision,
                    stage="Order Execution",
                    order=order,
                    extra_output={
                        "execution_action": "reduce",
                        "reduce_size": reduce_size,
                    },
                )

                # Record partial trade
                self._data_pool.record_trade(
                    pnl=partial_pnl,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    side=position.side.value,
                    size=reduce_size,
                )

                # Record operation
                self._data_pool.add_operation(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "action": "reduce",
                        "side": position.side.value,
                        "size": reduce_size,
                        "remaining_size": position_size - reduce_size,
                        "reason": decision.command.reason[:100],
                        "result": "success",
                    }
                )

        except Exception as e:
            self._logger.error(f"Failed to reduce position: {e}")
            self._data_pool.add_operation(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "reduce",
                    "result": "failed",
                    "error": str(e),
                }
            )

    async def _execute_add_position(self, position: Any, decision: Any) -> None:
        """Execute adding to existing position based on agent decision.

        Args:
            position: Current position
            decision: Agent decision with size to add
        """
        try:
            add_size = decision.command.size
            if not add_size or add_size <= 0:
                self._logger.error("ADD action requires positive size")
                self._notify_agent_log("ADD", "Invalid size", "Skipping")
                return

            # Profit check: Only allow add when profitable (per prompts: 只有盈利时才允许加仓)
            entry_price = float(position.entry_price)
            current_price = self._get_current_price()
            if entry_price > 0 and current_price > 0:
                if position.side == Side.LONG:
                    price_movement_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    price_movement_pct = ((entry_price - current_price) / entry_price) * 100

                if price_movement_pct <= 0:
                    self._logger.warning(
                        f"ADD rejected: position not profitable. "
                        f"Price movement {price_movement_pct:.2f}%. "
                        f"Per prompts: only add when profitable (顺势加仓)"
                    )
                    self._notify_agent_log(
                        "ADD",
                        "Not profitable",
                        f"Movement {price_movement_pct:.2f}% <= 0",
                    )
                    return

            # Add count limit: Max 2 adds (total 3 entries including initial)
            # Per prompts: 单方向最多加仓2次（含首次开仓共3笔）
            MAX_ADD_COUNT = 2
            current_add_count = self._session_manager.get_add_count(position.side.value)
            if current_add_count >= MAX_ADD_COUNT:
                self._logger.warning(
                    f"ADD rejected: max add count {MAX_ADD_COUNT} reached. "
                    f"Current adds: {current_add_count}"
                )
                self._notify_agent_log(
                    "ADD",
                    "Max adds reached",
                    f"{current_add_count}/{MAX_ADD_COUNT}",
                )
                return

            snapshot = self._data_pool.get_snapshot()
            volatility_pct = self._get_volatility_percent(snapshot)
            if volatility_pct is None:
                self._logger.info("ADD skipped: volatility unavailable")
                self._notify_agent_log("ADD", "Low volatility", "volatility unavailable")
                return
            if self._min_entry_move_pct > 0 and volatility_pct < self._min_entry_move_pct:
                self._logger.info(
                    "ADD skipped: volatility %.3f%% < min %.3f%%",
                    volatility_pct,
                    self._min_entry_move_pct,
                )
                self._notify_agent_log(
                    "ADD",
                    "Low volatility",
                    f"{volatility_pct:.3f}% < {self._min_entry_move_pct:.3f}%",
                )
                return

            take_profit_price = decision.command.take_profit_price
            if not take_profit_price or take_profit_price <= 0:
                self._logger.warning("ADD rejected: take_profit_price required")
                self._notify_risk_event(
                    "RR_LIMIT",
                    "ADD rejected: take_profit_price required for RR check",
                )
                return

            current_price = self._get_current_price()
            if current_price <= 0:
                self._logger.warning("ADD rejected: invalid current price")
                return
            stop_loss_offset = 0.01  # 1% price movement for stop loss (per prompts)
            if position.side == Side.LONG:
                stop_loss_price = current_price * (1 - stop_loss_offset)
            else:
                stop_loss_price = current_price * (1 + stop_loss_offset)
            rr = self._calculate_reward_risk(
                position.side,
                current_price,
                stop_loss_price,
                take_profit_price,
            )
            if rr is None or rr < 3.0:
                self._logger.warning(
                    "ADD rejected: RR %.2f < 3.0 (entry %.6f, stop %.6f, tp %.6f)",
                    rr or 0.0,
                    current_price,
                    stop_loss_price,
                    take_profit_price,
                )
                self._notify_risk_event(
                    "RR_LIMIT",
                    f"ADD rejected: RR {rr or 0.0:.2f} < 3.0",
                )
                return

            order = await self._executor.add_to_position(
                symbol=self._execution_symbol,
                side=position.side,
                size=add_size,
            )

            if order:
                self._logger.info(f"Added {add_size} to position: {order.order_id}")

                # After add, recalculate SL/TP based on new average entry price
                # Fetch updated position to get new average price
                updated_position = await self._executor.get_position(self._execution_symbol)
                if updated_position:
                    new_entry_price = float(updated_position.entry_price)
                    new_size = float(updated_position.size)

                    # Calculate new SL/TP based on average entry
                    sl_offset = 0.01  # 1% stop loss
                    if position.side == Side.LONG:
                        new_stop_loss = new_entry_price * (1 - sl_offset)
                    else:
                        new_stop_loss = new_entry_price * (1 + sl_offset)

                    # Update SL plan orders (only for WEEX which supports plan orders)
                    if hasattr(self._executor, "cancel_stop_loss_plans"):
                        await self._executor.cancel_stop_loss_plans(
                            self._execution_symbol, position.side
                        )

                        if hasattr(self._executor, "place_stop_loss_plan"):
                            sl_plan_id = await self._executor.place_stop_loss_plan(
                                symbol=self._execution_symbol,
                                side=position.side,
                                size=new_size,
                                trigger_price=new_stop_loss,
                            )
                            if sl_plan_id:
                                self._logger.info(
                                    f"Updated SL after add: {new_stop_loss:.4f} (plan: {sl_plan_id})"
                                )

                self._notify_agent_log(
                    "ADD",
                    f"Added {add_size:.4f}",
                    f"New size: {float(position.size) + add_size:.4f}",
                )

                await self._upload_execution_log(
                    decision,
                    stage="Order Execution",
                    order=order,
                    extra_output={
                        "execution_action": "add",
                        "add_size": add_size,
                    },
                )

                # Record operation
                self._data_pool.add_operation(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "action": "add",
                        "side": position.side.value,
                        "size": add_size,
                        "reason": decision.command.reason[:100],
                        "result": "success",
                    }
                )

        except Exception as e:
            self._logger.error(f"Failed to add to position: {e}")
            self._data_pool.add_operation(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "add",
                    "result": "failed",
                    "error": str(e),
                }
            )

    async def _process_entry_signal(self, signal: Signal) -> None:
        """Process an entry signal when IDLE.

        Args:
            signal: Signal to process
        """

        self._logger.info(f"Processing signal: {signal.signal_type.value}")
        self._logger.info(f"{signal.description}")

        # Notify TUI about the signal
        self._notify_signal(signal.signal_type.value, signal.description)

        # Transition to analyzing state
        self._state_machine.transition(
            StateTransition.ENTRY_SIGNAL,
            {"signal_type": signal.signal_type.value},
        )

        # Get market snapshot
        snapshot = self._data_pool.get_snapshot()
        snapshot = self._ensure_snapshot_position(snapshot)
        if not self._should_invoke_agent(snapshot):
            self._state_machine.transition(StateTransition.AGENT_OBSERVE)
            return

        # Convert Signal to StrategySignal format for agent
        from ai_trading_team.core.signal_queue import (
            SignalType as OldSignalType,
        )
        from ai_trading_team.core.signal_queue import (
            StrategySignal,
        )

        old_signal_type = OldSignalType.CUSTOM
        sanitized_signal = self._sanitize_signal_for_agent(signal)

        strategy_signal = StrategySignal(
            signal_type=old_signal_type,
            data={
                **sanitized_signal,
            },
            priority=2 if signal.strength.value == "moderate" else 3,
        )

        # Get agent decision
        self._logger.info("Calling AI agent for trading decision...")
        decision = await self._agent.process_signal(strategy_signal, snapshot)
        self._logger.info(
            f"Agent decision: {decision.command.action.value} - {decision.command.reason[:100]}"
        )

        # Notify TUI about agent decision
        self._notify_agent_log(
            decision.command.action.value.upper(),
            f"{decision.command.side.value if decision.command.side else 'N/A'} "
            f"size={decision.command.size or 'N/A'}",
            decision.command.reason[:60],
        )

        if decision.command.action in (AgentAction.OPEN, AgentAction.ADD):
            if decision.command.is_actionable():
                blocked, reason = self._should_block_entry(decision, snapshot)
                if blocked:
                    self._logger.info("%s", reason)
                    self._notify_agent_log("OBSERVE", "Entry blocked", reason)
                    try:
                        await self._executor.upload_ai_log(
                            stage="Signal Processing",
                            model=decision.model,
                            input_data={
                                "signal_type": sanitized_signal.get("category", "signal_event"),
                                "signal_data": sanitized_signal,
                                "market_snapshot": decision.market_snapshot,
                            },
                            output={
                                "action": "observe",
                                "side": decision.command.side.value
                                if decision.command.side
                                else None,
                                "size": decision.command.size,
                            },
                            explanation=reason,
                        )
                    except Exception as e:
                        self._logger.warning(f"Failed to upload AI log: {e}")
                    self._state_machine.transition(StateTransition.AGENT_OBSERVE)
                    return

        should_upload_decision_log = True
        if decision.command.action in (AgentAction.OPEN, AgentAction.ADD):
            if decision.command.is_actionable():
                should_upload_decision_log = False

        # Upload AI log only for non-execution actions (execution logs are uploaded after orders)
        if should_upload_decision_log:
            try:
                await self._executor.upload_ai_log(
                    stage="Signal Processing",
                    model=decision.model,
                    input_data={
                        "signal_type": sanitized_signal.get("category", "signal_event"),
                        "signal_data": sanitized_signal,
                        "market_snapshot": decision.market_snapshot,
                    },
                    output={
                        "action": decision.command.action.value,
                        "side": decision.command.side.value if decision.command.side else None,
                        "size": decision.command.size,
                    },
                    explanation=decision.command.reason,
                )
            except Exception as e:
                self._logger.warning(f"Failed to upload AI log: {e}")

        # Handle agent decision
        if decision.command.action == AgentAction.OBSERVE:
            self._state_machine.transition(StateTransition.AGENT_OBSERVE)
            self._notify_agent_log("OBSERVE", "No action taken", "Watching market")
        elif decision.command.action in (AgentAction.OPEN, AgentAction.ADD):
            if decision.command.is_actionable():
                success = await self._execute_command(decision)
                if success:
                    position = await self._executor.get_position(self._execution_symbol)
                    if position:
                        pos_ctx = PositionContext(
                            symbol=self._execution_symbol,
                            side=position.side,
                            margin=position.margin,
                        )
                        if self._state_machine.can_transition(StateTransition.AGENT_OPEN):
                            self._state_machine.transition(
                                StateTransition.AGENT_OPEN,
                                {"position": pos_ctx},
                            )
                        self._pending_entry_order_id = None
                    else:
                        # Order accepted but position not visible yet; wait for fill/confirmation
                        self._state_machine.transition(StateTransition.ORDER_PLACED)
                        self._logger.info(
                            "Order placed, waiting for position confirmation before marking failure"
                        )
                else:
                    self._pending_entry_order_id = None
                    self._state_machine.transition(StateTransition.ORDER_FAILED)
            else:
                self._state_machine.transition(StateTransition.AGENT_OBSERVE)
        elif decision.command.action == AgentAction.CLOSE:
            # CLOSE in ANALYZING state means agent wants to skip - treat as OBSERVE
            self._state_machine.transition(StateTransition.AGENT_OBSERVE)
            self._notify_agent_log("CLOSE", "No position to close", "Treating as observe")
        else:
            # Any other action (REDUCE, CANCEL, MOVE_STOP_LOSS) - treat as OBSERVE
            self._logger.debug(
                f"Unexpected action in entry signal: {decision.command.action.value}, treating as observe"
            )
            self._state_machine.transition(StateTransition.AGENT_OBSERVE)

    async def _execute_command(self, decision: AgentDecision) -> bool:
        """Execute agent command with risk controls.

        Implements:
        1. 750$ max margin limit check
        2. 30% stop loss order placement on exchange

        Args:
            decision: Agent decision with command to execute

        Returns:
            True if execution succeeded
        """
        command = decision.command
        self._logger.info(f"Executing command: {command.action.value}")

        errors = command.validate()
        if errors:
            self._logger.error(f"Command validation failed: {errors}")
            return False

        # Constants for risk control
        MAX_MARGIN_LIMIT = 750.0  # Maximum margin usage in USDT
        STOP_LOSS_PERCENT = 1.0  # Stop loss at 1% price movement (per prompts)

        try:
            if command.action == AgentAction.OPEN:
                if command.side and command.size:
                    # Get current account to check margin
                    account = await self._executor.get_account()
                    snapshot = self._data_pool.get_snapshot()
                    current_margin = float(account.used_margin)
                    if current_margin <= 0 and snapshot.position:
                        pos_margin = snapshot.position.get("margin", 0) if snapshot.position else 0
                        try:
                            pos_margin = float(pos_margin)
                        except (TypeError, ValueError):
                            pos_margin = 0.0
                        if pos_margin <= 0:
                            pos_size = snapshot.position.get("size", 0) if snapshot.position else 0
                            pos_entry = (
                                snapshot.position.get("entry_price", 0) if snapshot.position else 0
                            )
                            pos_leverage = (
                                snapshot.position.get("leverage", self._config.trading.leverage)
                                if snapshot.position
                                else self._config.trading.leverage
                            )
                            try:
                                pos_size = float(pos_size)
                                pos_entry = float(pos_entry)
                                pos_leverage = float(pos_leverage) if pos_leverage else 0
                            except (TypeError, ValueError):
                                pos_size = 0.0
                                pos_entry = 0.0
                                pos_leverage = 0.0
                            if pos_size > 0 and pos_entry > 0 and pos_leverage > 0:
                                pos_margin = (pos_size * pos_entry) / pos_leverage
                        current_margin = pos_margin
                    available_balance = float(account.available_balance)
                    max_margin_limit = min(MAX_MARGIN_LIMIT, available_balance)

                    # 1/20 Rule: Each open/add uses at most 1/20 (5%) of available balance
                    MAX_SINGLE_MARGIN_RATIO = 20  # Max single margin = available / 20
                    max_single_margin = available_balance / MAX_SINGLE_MARGIN_RATIO

                    # Get current price for margin calculation
                    current_price = float(
                        snapshot.ticker.get("last_price", 0) if snapshot.ticker else 0
                    )
                    if current_price <= 0:
                        self._logger.error("Cannot get current price for margin calculation")
                        return False

                    # Calculate margin for this order
                    # Margin = Position Value / Leverage = (Size * Price) / Leverage
                    leverage = self._config.trading.leverage

                    # Apply both limits: total 750 USDT and 1/20 per order
                    remaining_margin = max_margin_limit - current_margin
                    effective_margin = min(remaining_margin, max_single_margin)
                    max_size = (effective_margin * leverage) / current_price
                    if max_size <= 0:
                        self._logger.warning(
                            f"Order rejected: margin limit reached. "
                            f"Current: ${current_margin:.2f}, Limit: ${max_margin_limit:.2f}, "
                            f"Single order limit: ${max_single_margin:.2f}"
                        )
                        self._notify_risk_event(
                            "MARGIN_LIMIT",
                            f"Order rejected: margin limit ${max_margin_limit:.0f} reached",
                        )
                        return False

                    order_size = min(command.size, max_size)
                    if order_size < command.size:
                        self._logger.warning(
                            f"Order size adjusted to fit margin limit. "
                            f"Requested: {command.size:.6f}, Allowed: {order_size:.6f}, "
                            f"Limit: ${max_margin_limit:.2f}, 1/20 Rule: ${max_single_margin:.2f}"
                        )

                    order_margin = (order_size * current_price) / leverage
                    total_margin = current_margin + order_margin
                    if total_margin > max_margin_limit:
                        self._logger.warning(
                            f"Order rejected: would exceed margin limit. "
                            f"Current: ${current_margin:.2f}, Order: ${order_margin:.2f}, "
                            f"Total: ${total_margin:.2f}, Limit: ${max_margin_limit:.2f}"
                        )
                        self._notify_risk_event(
                            "MARGIN_LIMIT",
                            f"Order rejected: margin ${total_margin:.0f} exceeds ${max_margin_limit:.0f}",
                        )
                        return False

                    # Calculate stop loss price at 1% price movement (per prompts)
                    # For LONG: Stop Price = Entry * 0.99
                    # For SHORT: Stop Price = Entry * 1.01
                    stop_loss_offset = STOP_LOSS_PERCENT / 100  # 1% price movement
                    if command.side == Side.LONG:
                        stop_loss_price = current_price * (1 - stop_loss_offset)
                    else:  # SHORT
                        stop_loss_price = current_price * (1 + stop_loss_offset)

                    take_profit_price = command.take_profit_price
                    if not take_profit_price or take_profit_price <= 0:
                        self._logger.warning("Order rejected: take_profit_price required")
                        self._notify_risk_event(
                            "RR_LIMIT",
                            "Order rejected: take_profit_price required for RR check",
                        )
                        return False

                    rr = self._calculate_reward_risk(
                        command.side,
                        current_price,
                        stop_loss_price,
                        take_profit_price,
                    )
                    if rr is None or rr < 3.0:
                        self._logger.warning(
                            "Order rejected: RR %.2f < 3.0 (entry %.6f, stop %.6f, tp %.6f)",
                            rr or 0.0,
                            current_price,
                            stop_loss_price,
                            take_profit_price,
                        )
                        self._notify_risk_event(
                            "RR_LIMIT",
                            f"Order rejected: RR {rr or 0.0:.2f} < 3.0",
                        )
                        return False

                    self._logger.info(
                        f"Opening position: {command.side.value} {order_size} @ ~{current_price:.4f}, "
                        f"margin: ${order_margin:.2f}, stop loss: {stop_loss_price:.4f}, "
                        f"take profit: {take_profit_price:.4f}, RR: {rr:.2f}"
                    )

                    order = await self._executor.place_order(
                        symbol=self._execution_symbol,
                        side=command.side,
                        order_type=command.order_type or OrderType.MARKET,
                        size=order_size,
                        price=command.price,
                        action="open",
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                    )
                    self._logger.info(f"Opened position: {order.order_id}")
                    self._pending_entry_order_id = order.order_id or None

                    await self._upload_execution_log(
                        decision,
                        stage="Order Execution",
                        order=order,
                        extra_output={
                            "execution_action": "open",
                            "stop_loss_price": stop_loss_price,
                            "take_profit_price": take_profit_price,
                            "reward_risk_ratio": rr,
                            "margin_mode": 3,
                        },
                    )

                    operation = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "open",
                        "side": command.side.value,
                        "size": order_size,
                        "margin": order_margin,
                        "stop_loss_price": stop_loss_price,
                        "result": "success",
                        "order_id": order.order_id,
                    }
                    self._data_pool.add_operation(operation)
                    self._session_manager.add_operation(operation)
                    return True

            elif command.action == AgentAction.CLOSE and command.side:
                if self._state_machine.can_transition(StateTransition.AGENT_CLOSE):
                    self._state_machine.transition(StateTransition.AGENT_CLOSE)

                # Get position info before closing for trade recording
                position = await self._executor.get_position(self._execution_symbol)
                entry_price = float(position.entry_price) if position else 0.0
                exit_price = self._get_current_price()
                realized_pnl = float(position.unrealized_pnl) if position else 0.0
                trade_size = float(position.size) if position else command.size or 0.0

                close_order = await self._executor.close_position(
                    symbol=self._execution_symbol,
                    side=command.side,
                    size=command.size,
                )
                if close_order:
                    self._logger.info(f"Closed position: {close_order.order_id}")

                    await self._upload_execution_log(
                        decision,
                        stage="Order Execution",
                        order=close_order,
                        extra_output={
                            "execution_action": "close",
                            "close_size": command.size,
                        },
                    )

                    self._set_pending_close(
                        {
                            "action": "close",
                            "side": command.side.value,
                            "size": trade_size,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": realized_pnl,
                            "reason": decision.command.reason[:100],
                            "order_id": close_order.order_id,
                        }
                    )
                    return True

        except Exception as e:
            self._logger.error(f"Failed to execute command: {e}")

            operation = {
                "timestamp": datetime.now().isoformat(),
                "action": command.action.value,
                "side": command.side.value if command.side else None,
                "size": command.size,
                "result": "failed",
                "error": str(e),
            }
            self._data_pool.add_operation(operation)
            self._session_manager.add_operation(operation)

        return False

    async def stop(self) -> None:
        """Stop the trading bot and save state."""
        self._logger.info("Stopping trading bot...")
        self._running = False

        # Save final state before stopping
        try:
            self._sync_session_state()
            self._session_manager.save()
            self._logger.info(f"Session state saved to {self._session_manager.session_file}")
        except Exception as e:
            self._logger.error(f"Failed to save final state: {e}")

        # Display simulation summary in DRY_RUN mode
        if self._config.trading.dry_run and isinstance(self._executor, MockExecutor):
            summary = self._executor.get_summary()
            self._logger.info("=" * 60)
            self._logger.info("DRY_RUN Simulation Summary")
            self._logger.info("=" * 60)
            self._logger.info(f"Initial Balance: ${summary['initial_balance']:.2f}")
            self._logger.info(f"Current Balance: ${summary['current_balance']:.2f}")
            self._logger.info(f"Current Equity: ${summary['current_equity']:.2f}")
            self._logger.info(f"Realized P&L: ${summary['realized_pnl']:+.2f}")
            self._logger.info(f"Unrealized P&L: ${summary['unrealized_pnl']:+.2f}")
            self._logger.info(f"Total P&L: ${summary['total_pnl']:+.2f}")
            self._logger.info(f"Total Orders: {summary['total_orders']}")
            self._logger.info(f"Has Open Position: {summary['has_position']}")
            self._logger.info("=" * 60)

        # Stop data collection
        await self._data_manager.stop()

        await self._stop_weex_stream()

        # Disconnect from executor
        await self._executor.disconnect()

        self._logger.info("Trading bot stopped")


async def main_async(use_tui: bool = False) -> None:
    """Async main entry point.

    Args:
        use_tui: Whether to start with TUI interface
    """
    logger = setup_logging()
    config = Config.from_env()

    # Suppress console logging when using TUI
    if use_tui:
        # Remove console handlers to avoid interference with TUI
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if hasattr(handler, "stream"):
                root_logger.removeHandler(handler)
        # Keep file handler only
        logging.getLogger(__name__).setLevel(logging.WARNING)

    bot = TradingBot(config)

    # Handle shutdown gracefully
    loop = asyncio.get_event_loop()

    def shutdown_handler() -> None:
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        if use_tui and TUI_AVAILABLE and TradingApp is not None:
            # Start TUI with data pool integration
            tui_app = TradingApp(
                data_pool=bot._data_pool,
                symbol=bot._binance_symbol,
            )

            # Connect TUI to bot for signal/log notifications
            bot.set_tui(tui_app)

            # Start bot in background
            bot_task = asyncio.create_task(bot.start())

            # Run TUI (blocks until quit)
            await tui_app.run_async()

            # Stop bot when TUI exits
            await bot.stop()
            bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bot_task
        else:
            # Run without TUI
            await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.stop()


def main() -> None:
    """Application entry point."""
    import sys

    # Check for --tui flag
    use_tui = "--tui" in sys.argv or "-t" in sys.argv

    if use_tui and not TUI_AVAILABLE:
        print("TUI not available. Install with: uv sync --all-extras")
        sys.exit(1)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main_async(use_tui=use_tui))


if __name__ == "__main__":
    main()
