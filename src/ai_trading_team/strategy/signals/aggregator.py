"""Signal Aggregator - Manages signal sources and collects signals.

This replaces the old orchestrator's periodic evaluation with event-driven signals.
The aggregator:
1. Manages multiple signal sources
2. Updates sources with new data
3. Collects emitted signals
4. Optionally detects confluence (multiple aligned signals)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ai_trading_team.core.data_pool import DataPool
from ai_trading_team.indicators.volatility import VolatilityAnalyzer
from ai_trading_team.strategy.signals.base import SignalSource
from ai_trading_team.strategy.signals.bollinger_breakout import BollingerBreakoutSignal
from ai_trading_team.strategy.signals.funding_rate import FundingRateSignal
from ai_trading_team.strategy.signals.liquidation import LiquidationSignal
from ai_trading_team.strategy.signals.ls_ratio import LongShortRatioSignal
from ai_trading_team.strategy.signals.ma_crossover import MACrossoverSignal
from ai_trading_team.strategy.signals.macd_crossover import MACDCrossoverSignal
from ai_trading_team.strategy.signals.open_interest import OpenInterestSignal
from ai_trading_team.strategy.signals.order_status import OrderStatusSignal
from ai_trading_team.strategy.signals.pnl_change import PnLChangeSignal
from ai_trading_team.strategy.signals.risk_signal import RiskSignal
from ai_trading_team.strategy.signals.rsi_extreme import RSIExtremeSignal
from ai_trading_team.strategy.signals.types import (
    ALL_TIMEFRAMES,
    Signal,
    SignalDirection,
    SignalStrength,
    SignalType,
    Timeframe,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalWindow:
    """Configuration for signal confluence detection."""

    # Time window to look for aligned signals
    window_seconds: int = 300  # 5 minutes

    # Minimum signals needed for confluence
    min_signals_for_confluence: int = 2

    # Weights for different signal types (for confluence scoring)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "ma_crossover_60": 1.5,  # MA signals weighted higher
            "macd_crossover": 1.5,
            "rsi_extreme": 1.2,
            "bollinger_breakout": 1.2,
            "funding_rate": 1.0,
            "long_short_ratio": 1.0,
            "open_interest": 0.8,
            "liquidation": 1.3,
        }
    )


class SignalAggregator:
    """Aggregates signals from multiple sources.

    Key differences from old orchestrator:
    - No periodic scoring - only collects actual signals
    - Signals are event-driven (state changes only)
    - Optional confluence detection for aligned signals
    - Multi-timeframe support (5m, 15m, 1h, 4h)
    """

    def __init__(
        self,
        data_pool: DataPool,
        symbol: str,
        confluence_config: SignalWindow | None = None,
    ) -> None:
        """Initialize signal aggregator.

        Args:
            data_pool: Shared data pool
            symbol: Trading symbol
            confluence_config: Optional confluence detection config
        """
        self._data_pool = data_pool
        self._symbol = symbol
        self._confluence_config = confluence_config or SignalWindow()

        # Signal sources
        self._sources: list[SignalSource] = []

        # Recent signals (for confluence detection)
        self._recent_signals: deque[Signal] = deque(maxlen=100)

        # Callbacks for new signals
        self._signal_callbacks: list[Any] = []

        # Cooldown for confluence signals (prevent spam)
        self._last_confluence_time: datetime | None = None
        self._confluence_cooldown = timedelta(seconds=60)

        # Data readiness tracking
        self._is_ready = False
        self._required_kline_intervals = ["15m", "1h", "4h"]
        self._min_klines_required = 60  # Need at least 60 klines for MA60

        # Volatility analyzer for data completeness check
        self._volatility_analyzer = VolatilityAnalyzer(
            history_size=100,
            min_samples=20,
            volatility_multiplier=0.8,
        )

        # Initialize default signal sources
        self._init_default_sources()

    def _init_default_sources(self) -> None:
        """Initialize default signal sources per USER_INSTRUCTIONS.md."""
        # ===========================================
        # Technical Indicator Signals (5m, 15m, 1h, 4h)
        # ===========================================

        # SMA(60) crossover on all timeframes
        self.add_source(
            MACrossoverSignal(
                ma_period=60,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # RSI(14) extremes on all timeframes
        self.add_source(
            RSIExtremeSignal(
                oversold_threshold=30.0,
                overbought_threshold=70.0,
                rsi_period=14,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # MACD(12, 26, 9) crossover on all timeframes
        self.add_source(
            MACDCrossoverSignal(
                fast_period=12,
                slow_period=26,
                signal_period=9,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # Bollinger Bands(20, 2) breakout on all timeframes
        self.add_source(
            BollingerBreakoutSignal(
                period=20,
                std_dev=2.0,
                timeframes=ALL_TIMEFRAMES,
            )
        )

        # ===========================================
        # Market Data Signals
        # ===========================================

        # Funding rate shifts
        self.add_source(FundingRateSignal())

        # Long/Short ratio change >5% in 5 minutes
        self.add_source(
            LongShortRatioSignal(
                change_threshold_percent=5.0,
                window_minutes=5,
            )
        )

        # Open Interest change >5% in 5 minutes
        self.add_source(
            OpenInterestSignal(
                change_threshold_percent=5.0,
                window_minutes=5,
            )
        )

        # Liquidation >=100k USD (major at >=1M)
        self.add_source(
            LiquidationSignal(
                min_value_usd=100_000.0,
                major_value_usd=1_000_000.0,
            )
        )

        # ===========================================
        # Account/Risk Signals
        # ===========================================

        # P&L change ±5%
        self.add_source(
            PnLChangeSignal(
                threshold_percent=5.0,
            )
        )

        # Risk module triggers
        self.add_source(RiskSignal())

        # Order status changes
        self.add_source(OrderStatusSignal())

        logger.info(f"Initialized {len(self._sources)} signal sources")
        for source in self._sources:
            logger.debug(f"  - {source.name}: timeframes={[tf.value for tf in source.timeframes]}")

    def _check_data_ready(self, snapshot: Any) -> bool:
        """Check if all required data is available.

        Args:
            snapshot: Current data snapshot

        Returns:
            True if data is ready for signal processing
        """
        if self._is_ready:
            return True

        # Check klines for all required intervals
        if not snapshot.klines:
            return False

        for interval in self._required_kline_intervals:
            klines = snapshot.klines.get(interval, [])
            if len(klines) < self._min_klines_required:
                logger.debug(
                    f"Data not ready: {interval} has {len(klines)} klines, "
                    f"need {self._min_klines_required}"
                )
                return False

        # Check ticker data
        if not snapshot.ticker:
            logger.debug("Data not ready: no ticker data")
            return False

        # Update volatility analyzer with ATR data
        atr_15m = self._calculate_atr_pct(snapshot.klines.get("15m", []), 14)
        atr_1h = self._calculate_atr_pct(snapshot.klines.get("1h", []), 14)
        atr_4h = self._calculate_atr_pct(snapshot.klines.get("4h", []), 14)
        self._volatility_analyzer.update(atr_15m, atr_1h, atr_4h)

        # Check volatility data completeness
        sample_count = len(self._volatility_analyzer._composite_history)
        min_samples = self._volatility_analyzer._min_samples
        if sample_count < min_samples:
            logger.debug(
                f"Data not ready: volatility samples {sample_count}/{min_samples}"
            )
            return False

        # All checks passed
        self._is_ready = True
        logger.info(
            f"Signal aggregator ready: all {len(self._required_kline_intervals)} "
            f"timeframes have sufficient data, volatility samples={sample_count}"
        )
        return True

    @property
    def is_ready(self) -> bool:
        """Check if aggregator is ready for signal processing."""
        return self._is_ready

    def set_ready(self, ready: bool = True) -> None:
        """Manually set ready state.

        Args:
            ready: Ready state to set
        """
        if ready and not self._is_ready:
            logger.info("Signal aggregator manually set to ready")
        self._is_ready = ready

    def add_source(self, source: SignalSource) -> None:
        """Add a signal source.

        Args:
            source: SignalSource to add
        """
        self._sources.append(source)
        logger.debug(f"Added signal source: {source.name}")

    def remove_source(self, name: str) -> bool:
        """Remove a signal source by name.

        Args:
            name: Source name to remove

        Returns:
            True if removed
        """
        for i, source in enumerate(self._sources):
            if source.name == name:
                self._sources.pop(i)
                return True
        return False

    def on_signal(self, callback: Any) -> None:
        """Register a callback for new signals.

        Args:
            callback: Function to call with new Signal
        """
        self._signal_callbacks.append(callback)

    def update(self, timeframe: Timeframe | None = None) -> list[Signal]:
        """Update all sources and collect new signals.

        This should be called when new data arrives for a timeframe.
        Signals will not be generated until data is ready.

        Args:
            timeframe: Specific timeframe to update, or None for all

        Returns:
            List of new signals emitted
        """
        snapshot = self._data_pool.get_snapshot()

        # Check if data is ready before processing signals
        if not self._check_data_ready(snapshot):
            return []

        signals: list[Signal] = []

        timeframes = [timeframe] if timeframe else list(Timeframe)

        for tf in timeframes:
            for source in self._sources:
                if not source.enabled:
                    continue

                signal = source.update(snapshot, tf)
                if signal:
                    signals.append(signal)
                    self._recent_signals.append(signal)

                    # Notify callbacks
                    for callback in self._signal_callbacks:
                        try:
                            callback(signal)
                        except Exception as e:
                            logger.error(f"Signal callback error: {e}")

        # Check for confluence
        if len(signals) > 0:
            confluence = self._check_confluence()
            if confluence:
                signals.append(confluence)
                self._recent_signals.append(confluence)

        return signals

    def _check_confluence(self) -> Signal | None:
        """Check for signal confluence (multiple aligned signals).

        Returns:
            Confluence signal if detected
        """
        # Check cooldown
        now = datetime.now()
        if (
            self._last_confluence_time
            and now - self._last_confluence_time < self._confluence_cooldown
        ):
            return None

        config = self._confluence_config
        cutoff = now - timedelta(seconds=config.window_seconds)

        # Filter recent signals within window, excluding confluence signals themselves
        recent = [
            s for s in self._recent_signals if s.timestamp > cutoff and s.source != "aggregator"
        ]

        if len(recent) < config.min_signals_for_confluence:
            return None

        # Count bullish vs bearish signals
        bullish_score = 0.0
        bearish_score = 0.0

        for signal in recent:
            weight = config.weights.get(signal.source, 1.0)
            if signal.strength == SignalStrength.STRONG:
                weight *= 1.5
            elif signal.strength == SignalStrength.WEAK:
                weight *= 0.5

            if signal.direction == SignalDirection.BULLISH:
                bullish_score += weight
            elif signal.direction == SignalDirection.BEARISH:
                bearish_score += weight

        # Check for confluence
        min_score = config.min_signals_for_confluence

        if bullish_score >= min_score and bullish_score > bearish_score * 1.5:
            self._last_confluence_time = now
            return Signal(
                signal_type=SignalType.BULLISH_CONFLUENCE,
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.STRONG
                if bullish_score >= min_score * 2
                else SignalStrength.MODERATE,
                timeframe=Timeframe.H1,  # Confluence is multi-timeframe
                source="aggregator",
                data={
                    "bullish_score": bullish_score,
                    "bearish_score": bearish_score,
                    "signal_count": len(recent),
                    "signals": [s.signal_type.value for s in recent if s.is_bullish],
                },
                description=(
                    f"BULLISH CONFLUENCE: {bullish_score:.1f} bullish vs "
                    f"{bearish_score:.1f} bearish from {len(recent)} signals"
                ),
            )

        elif bearish_score >= min_score and bearish_score > bullish_score * 1.5:
            self._last_confluence_time = now
            return Signal(
                signal_type=SignalType.BEARISH_CONFLUENCE,
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.STRONG
                if bearish_score >= min_score * 2
                else SignalStrength.MODERATE,
                timeframe=Timeframe.H1,
                source="aggregator",
                data={
                    "bullish_score": bullish_score,
                    "bearish_score": bearish_score,
                    "signal_count": len(recent),
                    "signals": [s.signal_type.value for s in recent if s.is_bearish],
                },
                description=(
                    f"BEARISH CONFLUENCE: {bearish_score:.1f} bearish vs "
                    f"{bullish_score:.1f} bullish from {len(recent)} signals"
                ),
            )

        return None

    def get_recent_signals(
        self,
        seconds: int = 300,
        direction: SignalDirection | None = None,
    ) -> list[Signal]:
        """Get recent signals within a time window.

        Args:
            seconds: Time window in seconds
            direction: Optional filter by direction

        Returns:
            List of matching signals
        """
        cutoff = datetime.now() - timedelta(seconds=seconds)
        signals = [s for s in self._recent_signals if s.timestamp > cutoff]

        if direction:
            signals = [s for s in signals if s.direction == direction]

        return signals

    def get_source_states(self) -> dict[str, Any]:
        """Get current state of all sources for debugging.

        Returns:
            Dictionary of source states
        """
        states = {}
        for source in self._sources:
            states[source.name] = {
                "enabled": source.enabled,
                "timeframes": [tf.value for tf in source.timeframes],
                "states": {tf.value: str(source.get_state(tf)) for tf in source.timeframes},
            }
        return states

    def reset(self) -> None:
        """Reset all source states and ready status."""
        for source in self._sources:
            source.reset()
        self._recent_signals.clear()
        self._is_ready = False
        self._last_confluence_time = None
        logger.info("Signal aggregator reset")

    def update_indicators(self) -> None:
        """Update data pool with computed indicator values.

        This method computes indicators from signal sources and stores them
        in the data pool for the AI context.
        """
        snapshot = self._data_pool.get_snapshot()

        if not self._check_data_ready(snapshot):
            return

        timeframes = ("15m", "1h", "4h")
        sma_periods = (5, 20, 60)
        for interval in timeframes:
            interval_klines = snapshot.klines.get(interval, []) if snapshot.klines else []
            if not interval_klines:
                continue

            closes = [float(k.get("close", 0)) for k in interval_klines]
            if not closes or all(c == 0 for c in closes):
                continue

            current_price = closes[-1]

            rsi = self._calculate_rsi(closes, 14)
            if rsi is not None:
                self._data_pool.update_indicator(f"RSI_14_{interval}", round(rsi, 2))
                if interval == "1h":
                    self._data_pool.update_indicator("RSI_14", round(rsi, 2))

            for period in sma_periods:
                if len(closes) < period:
                    continue
                sma_value = sum(closes[-period:]) / period
                position = "above" if current_price > sma_value else "below"
                distance_percent = (
                    ((current_price - sma_value) / sma_value) * 100 if sma_value > 0 else 0.0
                )
                sma_payload = {
                    "value": round(sma_value, 6),
                    "price_position": position,
                    "distance_percent": round(distance_percent, 2),
                }
                self._data_pool.update_indicator(f"SMA_{period}_{interval}", sma_payload)
                if interval == "1h" and period == 60:
                    self._data_pool.update_indicator("SMA_60", sma_payload)

            macd = self._calculate_macd(closes, 12, 26, 9)
            if macd:
                macd_value, signal_value, histogram = macd
                macd_payload = {
                    "macd": round(macd_value, 6),
                    "signal": round(signal_value, 6),
                    "histogram": round(histogram, 6),
                }
                self._data_pool.update_indicator(
                    f"MACD_12_26_9_{interval}",
                    macd_payload,
                )
                if interval == "1h":
                    self._data_pool.update_indicator("MACD_12_26_9", macd_payload)

            bb = self._calculate_bollinger_bands(closes, 20, 2.0)
            if bb:
                upper, middle, lower = bb
                width_percent = (upper - lower) / middle * 100 if middle > 0 else 0.0
                bb_payload = {
                    "upper": round(upper, 6),
                    "middle": round(middle, 6),
                    "lower": round(lower, 6),
                    "width_percent": round(width_percent, 2),
                    "position": round((current_price - lower) / (upper - lower), 2)
                    if upper != lower
                    else 0.5,
                }
                self._data_pool.update_indicator(f"BB_20_{interval}", bb_payload)
                if interval == "1h":
                    self._data_pool.update_indicator("BB_20", bb_payload)

        atr_values: list[float] = []
        weights = {"15m": 0.5, "1h": 0.3, "4h": 0.2}
        weighted_sum = 0.0
        total_weight = 0.0
        for interval, weight in weights.items():
            interval_klines = snapshot.klines.get(interval, []) if snapshot.klines else []
            atr_pct = self._calculate_atr_pct(interval_klines, 14)
            if atr_pct is None:
                continue
            atr_pct = round(atr_pct, 4)
            atr_values.append(atr_pct)
            weighted_sum += atr_pct * weight
            total_weight += weight
            self._data_pool.update_indicator(f"ATR_14_{interval}", atr_pct)

        if atr_values and total_weight > 0:
            composite_atr = round(weighted_sum / total_weight, 4)
            self._data_pool.update_indicator("ATR_14_COMPOSITE", composite_atr)

    def _calculate_rsi(self, closes: list[float], period: int) -> float | None:
        """Calculate RSI from close prices."""
        if len(closes) < period + 1:
            return None

        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_ema(self, values: list[float], period: int) -> list[float]:
        """Calculate EMA for a list of values."""
        if len(values) < period:
            return []

        multiplier = 2 / (period + 1)
        ema = [sum(values[:period]) / period]

        for value in values[period:]:
            ema.append((value - ema[-1]) * multiplier + ema[-1])

        return ema

    def _calculate_macd(
        self,
        closes: list[float],
        fast_period: int,
        slow_period: int,
        signal_period: int,
    ) -> tuple[float, float, float] | None:
        """Calculate MACD, Signal, and Histogram."""
        if len(closes) < slow_period + signal_period:
            return None

        fast_ema = self._calculate_ema(closes, fast_period)
        slow_ema = self._calculate_ema(closes, slow_period)

        if not fast_ema or not slow_ema:
            return None

        offset = slow_period - fast_period
        if len(fast_ema) <= offset:
            return None

        fast_aligned = fast_ema[offset:]
        if len(fast_aligned) != len(slow_ema):
            min_len = min(len(fast_aligned), len(slow_ema))
            fast_aligned = fast_aligned[-min_len:]
            slow_ema = slow_ema[-min_len:]

        macd_line = [f - s for f, s in zip(fast_aligned, slow_ema, strict=False)]
        if len(macd_line) < signal_period:
            return None

        signal_line = self._calculate_ema(macd_line, signal_period)
        if not signal_line:
            return None

        macd_value = macd_line[-1]
        signal_value = signal_line[-1]
        histogram = macd_value - signal_value

        return macd_value, signal_value, histogram

    def _calculate_bollinger_bands(
        self, closes: list[float], period: int, std_dev: float
    ) -> tuple[float, float, float] | None:
        """Calculate Bollinger Bands."""
        if len(closes) < period:
            return None

        period_closes = closes[-period:]
        middle = sum(period_closes) / len(period_closes)
        variance = sum((c - middle) ** 2 for c in period_closes) / len(period_closes)
        std = variance**0.5

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    def _calculate_atr_pct(self, klines: list[dict[str, Any]], period: int) -> float | None:
        """Calculate ATR as a percent of price."""
        if len(klines) < period + 1:
            return None

        true_ranges: list[float] = []
        for i in range(1, len(klines)):
            high = float(klines[i].get("high", 0))
            low = float(klines[i].get("low", 0))
            prev_close = float(klines[i - 1].get("close", 0))
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        atr = sum(true_ranges[-period:]) / period
        last_close = float(klines[-1].get("close", 0))
        if last_close <= 0 or atr <= 0:
            return None
        return atr / last_close * 100
