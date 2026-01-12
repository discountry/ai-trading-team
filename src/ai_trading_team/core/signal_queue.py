"""Signal queue with timestamp-based ordering."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from threading import RLock
from typing import Any


class SignalType(str, Enum):
    """Strategy signal types."""

    # RSI signals
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"

    # MACD signals
    MACD_BULLISH_CROSS = "macd_bullish_cross"
    MACD_BEARISH_CROSS = "macd_bearish_cross"

    # MA crossover signals (short/long MA)
    MA_BULLISH_CROSS = "ma_bullish_cross"
    MA_BEARISH_CROSS = "ma_bearish_cross"

    # MA position signals (price relative to MA60)
    PRICE_ABOVE_MA = "price_above_ma"  # Price above 1H MA60 -> bullish
    PRICE_BELOW_MA = "price_below_ma"  # Price below 1H MA60 -> bearish

    # Price level signals
    PRICE_SUPPORT = "price_support"
    PRICE_RESISTANCE = "price_resistance"

    # Funding rate signals
    FUNDING_POSITIVE = "funding_positive"  # Positive funding -> lean short
    FUNDING_NEGATIVE = "funding_negative"  # Negative funding -> lean long
    FUNDING_EXTREME_POSITIVE = "funding_extreme_positive"  # Very high funding
    FUNDING_EXTREME_NEGATIVE = "funding_extreme_negative"  # Very low funding

    # Long/Short ratio signals
    LONGS_DOMINANT = "longs_dominant"  # More longs -> lean short
    SHORTS_DOMINANT = "shorts_dominant"  # More shorts -> lean long

    # Open Interest signals
    OI_INCREASING_CONSOLIDATION = (
        "oi_increasing_consolidation"  # OI up + price flat -> big move coming
    )
    OI_DIVERGENCE_BULLISH = "oi_divergence_bullish"  # Price down + OI down -> bullish
    OI_DIVERGENCE_BEARISH = "oi_divergence_bearish"  # Price up + OI down -> bearish

    # Volatility signals
    VOLATILITY_LOW = "volatility_low"  # ATR/BB squeeze -> avoid trading
    VOLATILITY_EXPANDING = "volatility_expanding"  # Volatility increasing
    BOLLINGER_UPPER_TOUCH = "bollinger_upper_touch"  # Price at upper band
    BOLLINGER_LOWER_TOUCH = "bollinger_lower_touch"  # Price at lower band

    # Risk/Profit signals (internal)
    PROFIT_THRESHOLD_REACHED = "profit_threshold_reached"  # 10% profit increase
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"  # 25% loss - force close
    TRAILING_STOP_TRIGGERED = "trailing_stop_triggered"  # Trailing stop hit

    # Composite signals (multi-factor)
    STRONG_BULLISH = "strong_bullish"  # Multiple bullish factors aligned
    STRONG_BEARISH = "strong_bearish"  # Multiple bearish factors aligned
    CONFLICTING_SIGNALS = "conflicting_signals"  # Mixed signals - observe

    # Custom
    CUSTOM = "custom"


@dataclass
class StrategySignal:
    """Strategy signal with metadata."""

    signal_type: SignalType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent

    def __lt__(self, other: "StrategySignal") -> bool:
        """Compare by priority, then timestamp."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.timestamp < other.timestamp  # Earlier timestamp first


class SignalQueue:
    """Thread-safe signal queue with deduplication.

    Prevents signal overlap by tracking recent signals.
    """

    def __init__(self, max_size: int = 100, dedup_window_seconds: float = 5.0) -> None:
        """Initialize signal queue.

        Args:
            max_size: Maximum queue size
            dedup_window_seconds: Time window for signal deduplication
        """
        self._queue: Queue[StrategySignal] = Queue(maxsize=max_size)
        self._lock = RLock()
        self._recent_signals: list[tuple[SignalType, datetime]] = []
        self._dedup_window = dedup_window_seconds

    def _cleanup_recent(self) -> None:
        """Remove old signals from dedup tracking."""
        now = datetime.now()
        self._recent_signals = [
            (sig_type, ts)
            for sig_type, ts in self._recent_signals
            if (now - ts).total_seconds() < self._dedup_window
        ]

    def _is_duplicate(self, signal: StrategySignal) -> bool:
        """Check if signal is a duplicate within the dedup window."""
        self._cleanup_recent()
        return any(sig_type == signal.signal_type for sig_type, _ in self._recent_signals)

    def push(self, signal: StrategySignal) -> bool:
        """Push a signal to the queue.

        Args:
            signal: Strategy signal to push

        Returns:
            True if pushed, False if duplicate or queue full
        """
        with self._lock:
            if self._is_duplicate(signal):
                return False

            try:
                self._queue.put_nowait(signal)
                self._recent_signals.append((signal.signal_type, signal.timestamp))
                return True
            except Exception:
                return False

    def pop(self, timeout: float | None = None) -> StrategySignal | None:
        """Pop a signal from the queue.

        Args:
            timeout: Max seconds to wait (None = non-blocking)

        Returns:
            Signal or None if empty/timeout
        """
        try:
            if timeout is None:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def peek(self) -> StrategySignal | None:
        """Peek at the next signal without removing it."""
        with self._lock:
            if self._queue.empty():
                return None
            # Get and put back (not ideal but Queue doesn't have peek)
            try:
                signal = self._queue.get_nowait()
                self._queue.put(signal)
                return signal
            except Empty:
                return None

    def clear(self) -> None:
        """Clear all signals from the queue."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            self._recent_signals.clear()

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
