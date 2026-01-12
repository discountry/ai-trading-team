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
import logging
import signal
from datetime import datetime
from decimal import Decimal
from typing import Any

from ai_trading_team.agent.commands import AgentAction
from ai_trading_team.agent.schemas import AgentDecision
from ai_trading_team.agent.trader import LangChainTradingAgent
from ai_trading_team.config import Config
from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.core.session import SessionManager
from ai_trading_team.core.types import OrderType, Side
from ai_trading_team.data.manager import BinanceDataManager
from ai_trading_team.execution.binance.executor import BinanceExecutor
from ai_trading_team.execution.mock_executor import MockExecutor
from ai_trading_team.execution.weex.executor import WEEXExecutor
from ai_trading_team.logging import setup_logging
from ai_trading_team.risk.monitor import RiskMonitor
from ai_trading_team.risk.rules import DynamicTakeProfitRule, ForceStopLossRule, TrailingStopRule
from ai_trading_team.strategy.signals import (
    Signal,
    SignalAggregator,
    SignalDirection,
    SignalStrength,
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

        # Risk monitor with strategy-specific rules
        self._risk_monitor = RiskMonitor(self._data_pool, self._executor)
        self._setup_risk_rules()

        # Pending signals to process
        self._pending_signals: list[Signal] = []

        # State saving interval
        self._last_state_save = 0.0
        self._state_save_interval = 30.0  # Save every 30 seconds

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
        await self._data_manager.start(self._binance_symbol, kline_interval="1m")

        # Wait for data to be ready before proceeding
        if not self._data_manager.is_data_ready:
            self._logger.error("Data initialization failed - cannot start signal processing")
            return

        self._logger.info(f"Data ready: all timeframes loaded for {self._binance_symbol}")

        # Start background tasks
        asyncio.create_task(self._market_metrics_loop())
        asyncio.create_task(self._signal_update_loop())
        asyncio.create_task(self._state_save_loop())

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
                    self._data_pool.update_open_interest(
                        {
                            "open_interest": float(oi.open_interest),
                            "timestamp": oi.timestamp.isoformat(),
                        }
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to fetch OI: {e}")

                # Fetch mark price
                try:
                    mark = await rest_client.get_mark_price(self._binance_symbol)
                    self._data_pool.update_mark_price(mark)
                except Exception as e:
                    self._logger.debug(f"Failed to fetch mark price: {e}")

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
            if self._signal_aggregator.is_ready:
                self._logger.info("Signal aggregator ready - starting signal processing")
                break
            await asyncio.sleep(1)

        if not self._running:
            return

        # Track last kline update times to detect new data
        last_kline_counts: dict[str, int] = {}

        # Initialize last_refresh with current time to prevent immediate refresh
        current_time = asyncio.get_event_loop().time()

        # Refresh klines periodically for each timeframe
        timeframe_intervals = {
            "5m": 60,  # Check 5m klines every 60s
            "15m": 120,  # Check 15m klines every 2min
            "1h": 300,  # Check 1h klines every 5min
            "4h": 600,  # Check 4h klines every 10min
        }
        last_refresh: dict[str, float] = {tf: current_time for tf in timeframe_intervals}

        while self._running:
            try:
                current_time = asyncio.get_event_loop().time()

                # Refresh klines for each timeframe based on interval
                for interval, refresh_interval in timeframe_intervals.items():
                    if current_time - last_refresh[interval] >= refresh_interval:
                        last_refresh[interval] = current_time
                        await self._refresh_klines(interval)

                        # Update signals for this timeframe
                        timeframe = self._interval_to_timeframe(interval)
                        if timeframe:
                            signals = self._signal_aggregator.update(timeframe)
                            if signals:
                                self._pending_signals.extend(signals)

                # Also check for signals when 1m data updates (from WebSocket)
                snapshot = self._data_pool.get_snapshot()
                if snapshot.klines:
                    klines_1m = snapshot.klines.get("1m", [])
                    current_count = len(klines_1m)
                    if current_count != last_kline_counts.get("1m", 0):
                        last_kline_counts["1m"] = current_count
                        # Update all signal sources with latest data
                        signals = self._signal_aggregator.update()
                        if signals:
                            self._pending_signals.extend(signals)

                        # Update indicators in data pool for AI context
                        self._signal_aggregator.update_indicators()

            except Exception as e:
                self._logger.error(f"Error in signal update loop: {e}")

            await asyncio.sleep(5)  # Check every 5 seconds

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
            "5m": Timeframe.M5,
            "15m": Timeframe.M15,
            "1h": Timeframe.H1,
            "4h": Timeframe.H4,
        }
        return mapping.get(interval)

    async def _run_loop(self) -> None:
        """Main trading loop."""
        risk_check_interval = 1.0  # Check risk every 1 second
        last_risk_check = 0.0

        while self._running:
            try:
                current_time = asyncio.get_event_loop().time()

                # Check risk rules (highest priority, most frequent)
                if current_time - last_risk_check >= risk_check_interval:
                    last_risk_check = current_time
                    await self._check_risk()

                # Process pending signals (event-driven, not periodic)
                if self._pending_signals:
                    signal = self._pending_signals.pop(0)
                    await self._process_signal(signal)

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
            position = await self._executor.get_position(self._execution_symbol)
            if position:
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
            else:
                self._data_pool.update_position(None)

            account = await self._executor.get_account()
            self._data_pool.update_account(
                {
                    "balance": float(account.total_equity),
                    "available": float(account.available_balance),
                    "margin": float(account.used_margin),
                }
            )

            # Update trading statistics with current equity
            unrealized = float(position.unrealized_pnl) if position else 0.0
            self._data_pool.update_equity(
                current_equity=float(account.total_equity),
                unrealized_pnl=unrealized,
            )
        except Exception as e:
            self._logger.debug(f"Error updating position data: {e}")

    async def _handle_profit_signal(self, action: Any) -> None:
        """Handle profit threshold signal by asking AI to set stop loss.

        Args:
            action: RiskAction with move_stop_loss type
        """
        self._logger.info(f"Processing profit threshold signal: {action.reason}")
        self._notify_agent_log("PROFIT", action.reason[:50], "Asking AI for stop loss")

        # Get current snapshot and position
        snapshot = self._data_pool.get_snapshot()
        position = await self._executor.get_position(self._execution_symbol)

        if not position:
            self._logger.warning("No position found for profit signal")
            return

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

            # Execute move_stop_loss if AI decided
            if decision.command.action == AgentAction.MOVE_STOP_LOSS:
                if decision.command.stop_loss_price:
                    await self._execute_move_stop_loss(position, decision.command.stop_loss_price)
                else:
                    self._logger.warning("AI returned move_stop_loss without price")

            elif decision.command.action == AgentAction.CLOSE:
                # AI decided to close instead of moving stop loss
                success = await self._execute_command(decision)
                if success:
                    self._state_machine.transition(StateTransition.POSITION_CLOSED)
                    self._risk_monitor.reset_rules()

            elif decision.command.action == AgentAction.OBSERVE:
                self._logger.info("AI decided to observe, not moving stop loss")

            # Upload AI log
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

            # Note: Modifying stop loss on existing positions can require
            # exchange-specific conditional order APIs.
            # For now, we log the intended stop loss and update the state context
            # for the AI to be aware of the intended stop level.
            self._logger.warning(
                f"Stop loss move requested to {stop_loss_price:.4f}. "
                f"Note: Modifying stop loss on existing positions requires plan order API. "
                f"Cancelled {cancelled_count} existing stop orders."
            )

            # Update position context with new stop loss target for AI reference
            if self._state_machine._context.position:
                self._state_machine._context.position.stop_loss_price = stop_loss_price

            self._notify_agent_log("STOP_LOSS", f"Target moved to {stop_loss_price:.4f}", "Logged")

            self._data_pool.add_operation(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "move_stop_loss",
                    "side": position.side.value,
                    "stop_loss_price": stop_loss_price,
                    "result": "logged",
                    "note": "Plan order API required for actual stop modification",
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
                    await self._handle_profit_signal(action)
                    return

                # Force close for high-priority risk actions
                if action.priority >= 80:
                    self._logger.critical(f"FORCE CLOSING POSITION: {action.reason}")
                    self._notify_risk_event("STOP_LOSS", f"Force close: {action.reason}")

                    position = await self._executor.get_position(self._execution_symbol)
                    if position:
                        # Capture position info before closing for trade recording
                        entry_price = float(position.entry_price)
                        exit_price = self._get_current_price()
                        realized_pnl = float(position.unrealized_pnl)
                        trade_side = position.side.value
                        trade_size = float(position.size)

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

                            # Record the trade in trading statistics
                            self._data_pool.record_trade(
                                pnl=realized_pnl,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                side=trade_side,
                                size=trade_size,
                            )
                            self._logger.info(
                                f"Trade recorded: {trade_side} P&L=${realized_pnl:+.2f}"
                            )

                            self._data_pool.add_operation(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "action": "force_close",
                                    "side": position.side.value,
                                    "size": float(position.size),
                                    "pnl": realized_pnl,
                                    "result": "success",
                                    "reason": action.reason,
                                }
                            )

                            self._state_machine.transition(StateTransition.POSITION_CLOSED)
                            self._risk_monitor.reset_rules()

        except Exception as e:
            self._logger.error(f"Error checking risk: {e}")

    async def _process_signal(self, signal: Signal) -> None:
        """Process a trading signal through the agent.

        Args:
            signal: Signal to process
        """
        try:
            # Only process actionable signals
            if not signal.is_actionable:
                self._logger.debug(f"Signal not actionable: {signal.signal_type.value}")
                return

            # Check if we should process this signal based on current state
            if self._state_machine.is_idle:
                # IDLE state: process entry signals normally
                await self._process_entry_signal(signal)
            elif self._state_machine.has_position:
                # IN_POSITION state: check if signal suggests closing/reversing
                await self._process_position_signal(signal)
            else:
                # Other states (ANALYZING, WAITING_ENTRY, etc.): skip
                self._logger.debug(f"Busy state, skipping signal: {signal.signal_type.value}")
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
            self._logger.debug("No position found, skipping position signal")
            return

        # Determine if signal is opposite to current position
        is_long_position = position.side == Side.LONG
        is_bearish_signal = signal.direction == SignalDirection.BEARISH
        is_bullish_signal = signal.direction == SignalDirection.BULLISH

        is_opposite_signal = (is_long_position and is_bearish_signal) or (
            not is_long_position and is_bullish_signal
        )

        # Only process strong opposite signals (weak signals are noise)
        if not is_opposite_signal:
            self._logger.debug(
                f"Same direction signal while in position, skipping: {signal.signal_type.value}"
            )
            return

        if signal.strength == SignalStrength.WEAK:
            self._logger.debug(
                f"Weak opposite signal while in position, skipping: {signal.signal_type.value}"
            )
            return

        # Check debounce to avoid repeated signals
        if self._state_machine.should_debounce_signal(signal.signal_type.value):
            self._logger.debug(f"Signal debounced: {signal.signal_type.value}")
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

        # Build context for AI to decide whether to close/hold
        from ai_trading_team.core.signal_queue import (
            SignalType as OldSignalType,
        )
        from ai_trading_team.core.signal_queue import (
            StrategySignal,
        )

        # Create signal with position context
        if is_bearish_signal:
            old_type = OldSignalType.STRONG_BEARISH
        else:
            old_type = OldSignalType.STRONG_BULLISH
        strategy_signal = StrategySignal(
            signal_type=old_type,
            data={
                "new_signal_type": signal.signal_type.value,
                "direction": signal.direction.value,
                "strength": signal.strength.value,
                "timeframe": signal.timeframe.value,
                "source": signal.source,
                "description": signal.description,
                "context": "opposite_signal_while_in_position",
                "current_position_side": position.side.value,
                "current_pnl": float(position.unrealized_pnl),
                **signal.data,
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

        # Upload AI decision log
        try:
            await self._executor.upload_ai_log(
                stage="Opposite Signal Processing",
                model=decision.model,
                input_data={
                    "signal_type": signal.signal_type.value,
                    "signal_data": signal.data,
                    "context": "opposite_signal_while_in_position",
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

                # Record trade
                self._data_pool.record_trade(
                    pnl=realized_pnl,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    side=trade_side,
                    size=trade_size,
                )

                # Transition state machine
                self._state_machine.transition(StateTransition.POSITION_CLOSED)

                # Reset risk monitor
                self._risk_monitor.reset_rules()

                # Record operation
                self._data_pool.add_operation(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "action": "close",
                        "side": trade_side,
                        "size": trade_size,
                        "reason": decision.command.reason[:100],
                        "result": "success",
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

            order = await self._executor.add_to_position(
                symbol=self._execution_symbol,
                side=position.side,
                size=add_size,
            )

            if order:
                self._logger.info(f"Added {add_size} to position: {order.order_id}")
                self._notify_agent_log(
                    "ADD",
                    f"Added {add_size:.4f}",
                    f"New size: {float(position.size) + add_size:.4f}",
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

        self._logger.info("Calling AI agent for trading decision...")

        # Transition to analyzing state
        self._state_machine.transition(
            StateTransition.ENTRY_SIGNAL,
            {"signal_type": signal.signal_type.value},
        )

        # Get market snapshot
        snapshot = self._data_pool.get_snapshot()

        # Convert Signal to StrategySignal format for agent
        from ai_trading_team.core.signal_queue import (
            SignalType as OldSignalType,
        )
        from ai_trading_team.core.signal_queue import (
            StrategySignal,
        )

        # Map new signal direction to old signal type
        if signal.direction == SignalDirection.BULLISH:
            old_signal_type = OldSignalType.STRONG_BULLISH
        elif signal.direction == SignalDirection.BEARISH:
            old_signal_type = OldSignalType.STRONG_BEARISH
        else:
            old_signal_type = OldSignalType.CONFLICTING_SIGNALS

        strategy_signal = StrategySignal(
            signal_type=old_signal_type,
            data={
                "new_signal_type": signal.signal_type.value,
                "direction": signal.direction.value,
                "strength": signal.strength.value,
                "timeframe": signal.timeframe.value,
                "source": signal.source,
                "description": signal.description,
                **signal.data,
            },
            priority=2 if signal.strength.value == "moderate" else 3,
        )

        # Get agent decision
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
                        self._state_machine.transition(
                            StateTransition.AGENT_OPEN,
                            {"position": pos_ctx},
                        )
                    else:
                        # Position not found after successful order - treat as failed
                        self._state_machine.transition(StateTransition.ORDER_FAILED)
                else:
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

        # Upload AI log
        try:
            await self._executor.upload_ai_log(
                stage="Signal Processing",
                model=decision.model,
                input_data={
                    "signal_type": signal.signal_type.value,
                    "signal_data": signal.data,
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
        STOP_LOSS_PERCENT = 30.0  # Stop loss at 30% margin loss

        try:
            if command.action == AgentAction.OPEN:
                if command.side and command.size:
                    # Get current account to check margin
                    account = await self._executor.get_account()
                    current_margin = float(account.used_margin)

                    # Get current price for margin calculation
                    snapshot = self._data_pool.get_snapshot()
                    current_price = float(
                        snapshot.ticker.get("last_price", 0) if snapshot.ticker else 0
                    )
                    if current_price <= 0:
                        self._logger.error("Cannot get current price for margin calculation")
                        return False

                    # Calculate margin for this order
                    # Margin = Position Value / Leverage = (Size * Price) / Leverage
                    leverage = self._config.trading.leverage
                    order_margin = (command.size * current_price) / leverage

                    # Check if adding this position would exceed margin limit
                    total_margin = current_margin + order_margin
                    if total_margin > MAX_MARGIN_LIMIT:
                        self._logger.warning(
                            f"Order rejected: would exceed margin limit. "
                            f"Current: ${current_margin:.2f}, Order: ${order_margin:.2f}, "
                            f"Total: ${total_margin:.2f}, Limit: ${MAX_MARGIN_LIMIT}"
                        )
                        self._notify_risk_event(
                            "MARGIN_LIMIT",
                            f"Order rejected: ${total_margin:.0f} exceeds ${MAX_MARGIN_LIMIT} limit",
                        )
                        return False

                    # Calculate stop loss price at 30% margin loss
                    # For LONG: Stop Price = Entry * (1 - 30% / Leverage)
                    # For SHORT: Stop Price = Entry * (1 + 30% / Leverage)
                    stop_loss_offset = STOP_LOSS_PERCENT / 100 / leverage
                    if command.side == Side.LONG:
                        stop_loss_price = current_price * (1 - stop_loss_offset)
                    else:  # SHORT
                        stop_loss_price = current_price * (1 + stop_loss_offset)

                    self._logger.info(
                        f"Opening position: {command.side.value} {command.size} @ ~{current_price:.4f}, "
                        f"margin: ${order_margin:.2f}, stop loss: {stop_loss_price:.4f}"
                    )

                    order = await self._executor.place_order(
                        symbol=self._execution_symbol,
                        side=command.side,
                        order_type=command.order_type or OrderType.MARKET,
                        size=command.size,
                        price=command.price,
                        action="open",
                        stop_loss_price=stop_loss_price,
                    )
                    self._logger.info(f"Opened position: {order.order_id}")

                    operation = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "open",
                        "side": command.side.value,
                        "size": command.size,
                        "margin": order_margin,
                        "stop_loss_price": stop_loss_price,
                        "result": "success",
                        "order_id": order.order_id,
                    }
                    self._data_pool.add_operation(operation)
                    self._session_manager.add_operation(operation)
                    return True

            elif command.action == AgentAction.CLOSE and command.side:
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

                    # Record the trade in trading statistics
                    self._data_pool.record_trade(
                        pnl=realized_pnl,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        side=command.side.value,
                        size=trade_size,
                    )
                    self._logger.info(
                        f"Trade recorded: {command.side.value} P&L=${realized_pnl:+.2f}"
                    )

                    operation = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "close",
                        "side": command.side.value,
                        "size": command.size,
                        "pnl": realized_pnl,
                        "result": "success",
                        "order_id": close_order.order_id,
                    }
                    self._data_pool.add_operation(operation)
                    self._session_manager.add_operation(operation)
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
