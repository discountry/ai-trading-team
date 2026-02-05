"""Dynamic Volatility Analyzer.

Calculates dynamic volatility thresholds based on historical ATR data.
Different symbols (BTC, DOGE, etc.) will have different baseline volatility.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class VolatilityState:
    """Current volatility analysis state."""

    current_atr_pct: float  # Current ATR as % of price
    avg_atr_pct: float  # Average ATR over history window
    percentile: float  # Current ATR percentile (0-100)
    is_above_average: bool  # True if current > average (good for trading)
    volatility_ratio: float  # current / average ratio
    recommendation: str  # "trade" or "wait"
    threshold: float  # Dynamic threshold (avg * multiplier)


class VolatilityAnalyzer:
    """Dynamic volatility analyzer for adaptive trading thresholds.

    Instead of using fixed thresholds like 0.5%, this analyzer:
    1. Tracks historical ATR values for each symbol
    2. Calculates a dynamic threshold based on the symbol's average volatility
    3. Recommends trading only when current volatility exceeds the average

    This adapts to different symbols:
    - BTC: Lower baseline volatility (~0.2-0.4%)
    - DOGE: Higher baseline volatility (~0.8-1.5%)
    """

    def __init__(
        self,
        history_size: int = 100,
        min_samples: int = 20,
        volatility_multiplier: float = 0.8,
    ) -> None:
        """Initialize volatility analyzer.

        Args:
            history_size: Number of ATR samples to keep for history
            min_samples: Minimum samples needed before making recommendations
            volatility_multiplier: Multiplier for average to set threshold
                                   0.8 means threshold = 80% of average
        """
        self._history_size = history_size
        self._min_samples = min_samples
        self._multiplier = volatility_multiplier

        # Separate history per timeframe
        self._atr_history: dict[str, deque[float]] = {
            "15m": deque(maxlen=history_size),
            "1h": deque(maxlen=history_size),
            "4h": deque(maxlen=history_size),
        }

        # Composite ATR history (weighted combination)
        self._composite_history: deque[float] = deque(maxlen=history_size)

    def update(
        self,
        atr_15m: float | None = None,
        atr_1h: float | None = None,
        atr_4h: float | None = None,
    ) -> None:
        """Update ATR history with new values.

        Args:
            atr_15m: 15-minute ATR as percentage of price
            atr_1h: 1-hour ATR as percentage of price
            atr_4h: 4-hour ATR as percentage of price
        """
        if atr_15m is not None and atr_15m > 0:
            self._atr_history["15m"].append(atr_15m)
        if atr_1h is not None and atr_1h > 0:
            self._atr_history["1h"].append(atr_1h)
        if atr_4h is not None and atr_4h > 0:
            self._atr_history["4h"].append(atr_4h)

        # Calculate composite (weighted average)
        composite = self._calculate_composite(atr_15m, atr_1h, atr_4h)
        if composite is not None:
            self._composite_history.append(composite)

    def _calculate_composite(
        self,
        atr_15m: float | None,
        atr_1h: float | None,
        atr_4h: float | None,
    ) -> float | None:
        """Calculate weighted composite ATR.

        Weights: 15m=0.5, 1h=0.3, 4h=0.2
        """
        weights = {"15m": 0.5, "1h": 0.3, "4h": 0.2}
        values = {"15m": atr_15m, "1h": atr_1h, "4h": atr_4h}

        weighted_sum = 0.0
        total_weight = 0.0

        for tf, weight in weights.items():
            val = values[tf]
            if val is not None and val > 0:
                weighted_sum += val * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return None

    def analyze(self, current_composite: float | None = None) -> VolatilityState:
        """Analyze current volatility state.

        Args:
            current_composite: Current composite ATR (if not provided, uses last value)

        Returns:
            VolatilityState with analysis results
        """
        # Use provided value or last recorded value
        if current_composite is None:
            if self._composite_history:
                current_composite = self._composite_history[-1]
            else:
                # No data yet, return conservative state
                return VolatilityState(
                    current_atr_pct=0.0,
                    avg_atr_pct=0.0,
                    percentile=0.0,
                    is_above_average=False,
                    volatility_ratio=0.0,
                    recommendation="wait",
                    threshold=0.0,
                )

        # Calculate statistics
        history = list(self._composite_history)
        sample_count = len(history)

        if sample_count < self._min_samples:
            # Not enough data, be conservative
            return VolatilityState(
                current_atr_pct=current_composite,
                avg_atr_pct=current_composite,
                percentile=50.0,
                is_above_average=False,
                volatility_ratio=1.0,
                recommendation="wait",
                threshold=current_composite,
            )

        # Calculate average and percentile
        avg_atr = sum(history) / len(history)
        sorted_history = sorted(history)

        # Find percentile of current value
        position = sum(1 for x in sorted_history if x <= current_composite)
        percentile = (position / len(sorted_history)) * 100

        # Calculate dynamic threshold
        threshold = avg_atr * self._multiplier

        # Determine if volatility is sufficient for trading
        is_above_average = current_composite >= threshold
        volatility_ratio = current_composite / avg_atr if avg_atr > 0 else 0.0

        recommendation = "trade" if is_above_average else "wait"

        return VolatilityState(
            current_atr_pct=current_composite,
            avg_atr_pct=avg_atr,
            percentile=percentile,
            is_above_average=is_above_average,
            volatility_ratio=volatility_ratio,
            recommendation=recommendation,
            threshold=threshold,
        )

    def get_timeframe_stats(self, timeframe: str) -> dict[str, float]:
        """Get statistics for a specific timeframe.

        Args:
            timeframe: "15m", "1h", or "4h"

        Returns:
            Dict with current, average, min, max, percentile
        """
        history = list(self._atr_history.get(timeframe, []))

        if not history:
            return {
                "current": 0.0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "percentile": 0.0,
                "sample_count": 0,
            }

        current = history[-1]
        avg = sum(history) / len(history)
        sorted_h = sorted(history)
        position = sum(1 for x in sorted_h if x <= current)
        percentile = (position / len(sorted_h)) * 100

        return {
            "current": current,
            "average": avg,
            "min": min(history),
            "max": max(history),
            "percentile": percentile,
            "sample_count": len(history),
        }

    def format_for_prompt(self) -> str:
        """Format volatility analysis for AI prompt.

        Returns:
            Human-readable volatility analysis string
        """
        state = self.analyze()

        if len(self._composite_history) < self._min_samples:
            return (
                f"波动率分析: 数据采集中 ({len(self._composite_history)}/{self._min_samples}), "
                f"当前综合ATR={state.current_atr_pct:.4f}%"
            )

        lines = [
            "【波动率分析】",
            f"当前综合ATR: {state.current_atr_pct:.4f}%",
            f"历史平均ATR: {state.avg_atr_pct:.4f}%",
            f"动态阈值: {state.threshold:.4f}% (平均值×{self._multiplier})",
            f"当前波动百分位: {state.percentile:.1f}%",
            f"波动率比值: {state.volatility_ratio:.2f}x",
            f"交易建议: {'✅ 波动充足可交易' if state.is_above_average else '⏸️ 波动不足建议观望'}",
        ]

        # Add per-timeframe details
        for tf in ["15m", "1h", "4h"]:
            stats = self.get_timeframe_stats(tf)
            if stats["sample_count"] > 0:
                lines.append(
                    f"  {tf}: 当前{stats['current']:.4f}% / "
                    f"均值{stats['average']:.4f}% / "
                    f"百分位{stats['percentile']:.0f}%"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export analyzer state to dict.

        Returns:
            Dict with all volatility metrics
        """
        state = self.analyze()
        return {
            "current_atr_pct": state.current_atr_pct,
            "avg_atr_pct": state.avg_atr_pct,
            "threshold": state.threshold,
            "percentile": state.percentile,
            "volatility_ratio": state.volatility_ratio,
            "is_above_average": state.is_above_average,
            "recommendation": state.recommendation,
            "sample_count": len(self._composite_history),
            "min_samples": self._min_samples,
            "timeframes": {tf: self.get_timeframe_stats(tf) for tf in ["15m", "1h", "4h"]},
        }
