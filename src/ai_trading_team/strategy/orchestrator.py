"""Strategy Orchestrator - Multi-factor signal aggregation and decision making.

This module implements the core strategy orchestration logic that:
1. Combines multiple single-factor strategy signals
2. Weighs signals based on priority and market context
3. Integrates with the state machine for lifecycle management
4. Generates composite trading decisions

Trading Philosophy Integration:
- MA60 position as primary trend indicator
- RSI for overbought/oversold conditions
- Funding rate for sentiment (contrarian)
- Long/short ratio for crowd positioning (contrarian)
- Volatility filter to avoid choppy markets
- Market bias: Currently bearish (BTC 82000-92000 range)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ai_trading_team.core.data_pool import DataPool, DataSnapshot
from ai_trading_team.core.signal_queue import SignalQueue, SignalType, StrategySignal
from ai_trading_team.core.types import Side
from ai_trading_team.strategy.base import Strategy
from ai_trading_team.strategy.factors.funding_rate import FundingRateStrategy
from ai_trading_team.strategy.factors.long_short_ratio import LongShortRatioStrategy
from ai_trading_team.strategy.factors.ma_position import MAPositionStrategy
from ai_trading_team.strategy.factors.rsi_oversold import RSIOversoldStrategy
from ai_trading_team.strategy.factors.volatility import VolatilityStrategy
from ai_trading_team.strategy.state_machine import (
    PositionContext,
    StateTransition,
    StrategyState,
    StrategyStateMachine,
)

logger = logging.getLogger(__name__)


class MarketBias(str, Enum):
    """Overall market bias."""

    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"


class SignalStrength(str, Enum):
    """Signal strength classification."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    CONFLICTING = "conflicting"


@dataclass
class FactorScore:
    """Score from a single factor."""

    factor_name: str
    signal_type: SignalType | None
    bullish_score: float  # -1 to 1
    weight: float  # Factor importance weight
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeSignal:
    """Aggregated signal from multiple factors."""

    timestamp: datetime
    total_score: float  # -1 (bearish) to 1 (bullish)
    strength: SignalStrength
    suggested_side: Side | None
    factor_scores: list[FactorScore]
    primary_signal: SignalType | None
    volatility_ok: bool
    market_bias: MarketBias


class StrategyOrchestrator:
    """Orchestrates multiple strategy factors and manages trading state.

    This is the main strategy controller that:
    1. Initializes and manages all factor strategies
    2. Aggregates signals with weighted scoring
    3. Manages the trading state machine
    4. Decides when to involve the agent
    5. Tracks operation history
    """

    # Factor weights (sum to 1.0)
    FACTOR_WEIGHTS = {
        "ma_position": 0.30,  # Primary trend indicator
        "rsi": 0.20,  # Momentum
        "funding_rate": 0.15,  # Sentiment (contrarian)
        "long_short_ratio": 0.15,  # Crowd positioning (contrarian)
        "volatility": 0.20,  # Volatility filter
    }

    # Thresholds for signal generation
    STRONG_SIGNAL_THRESHOLD = 0.6
    MODERATE_SIGNAL_THRESHOLD = 0.3
    WEAK_SIGNAL_THRESHOLD = 0.1

    def __init__(
        self,
        data_pool: DataPool,
        signal_queue: SignalQueue,
        symbol: str,
        default_market_bias: MarketBias = MarketBias.BEARISH,
        kline_interval_1h: str = "1h",
        kline_interval_1m: str = "1m",
    ) -> None:
        """Initialize Strategy Orchestrator.

        Args:
            data_pool: Shared data pool
            signal_queue: Signal queue for emitting signals
            symbol: Trading symbol
            default_market_bias: Default market bias (bearish per STRATEGY.md)
            kline_interval_1h: 1-hour kline interval key
            kline_interval_1m: 1-minute kline interval key
        """
        self._data_pool = data_pool
        self._signal_queue = signal_queue
        self._symbol = symbol
        self._default_market_bias = default_market_bias
        self._kline_1h = kline_interval_1h
        self._kline_1m = kline_interval_1m

        # For rate-limiting logs (only log when something changes)
        self._last_logged_score: float | None = None
        self._last_logged_reason: str | None = None

        # Initialize state machine
        self._state_machine = StrategyStateMachine(
            symbol=symbol,
            cooldown_seconds=60,
            force_stop_loss_percent=Decimal("25"),  # 25% force stop-loss
            profit_signal_threshold=Decimal("10"),  # 10% profit signals
        )

        # Initialize factor strategies
        self._ma_position = MAPositionStrategy(
            data_pool, signal_queue, ma_period=60, kline_interval=kline_interval_1h
        )
        self._rsi = RSIOversoldStrategy(
            data_pool, signal_queue, oversold_threshold=30.0, overbought_threshold=70.0
        )
        self._funding = FundingRateStrategy(data_pool, signal_queue)
        self._long_short = LongShortRatioStrategy(data_pool, signal_queue)
        self._volatility = VolatilityStrategy(
            data_pool, signal_queue, kline_interval=kline_interval_1h
        )

        # All factors for iteration
        self._factors: list[Strategy] = [
            self._ma_position,
            self._rsi,
            self._funding,
            self._long_short,
            self._volatility,
        ]

    @property
    def state(self) -> StrategyState:
        """Current trading state."""
        return self._state_machine.state

    @property
    def state_machine(self) -> StrategyStateMachine:
        """Access to state machine."""
        return self._state_machine

    @property
    def has_position(self) -> bool:
        """Check if currently holding a position."""
        return self._state_machine.has_position

    @property
    def is_ready_for_signals(self) -> bool:
        """Check if ready to process new entry signals."""
        return self._state_machine.is_idle

    def evaluate(self) -> StrategySignal | None:
        """Evaluate all factors and generate composite signal if appropriate.

        This is the main entry point called periodically by the trading loop.

        Returns:
            Composite signal if conditions warrant agent involvement
        """
        # Check for timeouts first
        self._state_machine.check_timeout()

        snapshot = self._data_pool.get_snapshot()

        # Evaluate all factors
        composite = self._evaluate_factors(snapshot)

        # Update state machine with position P&L if in position
        if self._state_machine.has_position and snapshot.position:
            pnl_trigger = self._check_position_pnl(snapshot)
            if pnl_trigger:
                self._state_machine.transition(pnl_trigger)

        # Generate signal based on state and composite evaluation
        return self._generate_signal(composite, snapshot)

    def _evaluate_factors(self, snapshot: DataSnapshot) -> CompositeSignal:
        """Evaluate all factor strategies and compute composite score.

        Args:
            snapshot: Current market data snapshot

        Returns:
            CompositeSignal with aggregated analysis
        """
        factor_scores: list[FactorScore] = []
        primary_signal: SignalType | None = None
        highest_priority = -1

        # Evaluate each factor
        for factor in self._factors:
            signal = factor.check_conditions(snapshot)
            score = self._compute_factor_score(factor, signal, snapshot)
            factor_scores.append(score)

            # Track highest priority signal
            if signal and signal.priority > highest_priority:
                primary_signal = signal.signal_type
                highest_priority = signal.priority

        # Compute weighted total score
        total_score = sum(score.bullish_score * score.weight for score in factor_scores)

        # Apply market bias adjustment
        bias_adjustment = self._get_bias_adjustment()
        total_score += bias_adjustment

        # Clamp to [-1, 1]
        total_score = max(-1.0, min(1.0, total_score))

        # Determine signal strength
        abs_score = abs(total_score)
        if self._has_conflicting_signals(factor_scores):
            strength = SignalStrength.CONFLICTING
        elif abs_score >= self.STRONG_SIGNAL_THRESHOLD:
            strength = SignalStrength.STRONG
        elif abs_score >= self.MODERATE_SIGNAL_THRESHOLD:
            strength = SignalStrength.MODERATE
        elif abs_score >= self.WEAK_SIGNAL_THRESHOLD:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.WEAK

        # Determine suggested side
        suggested_side: Side | None = None
        if strength != SignalStrength.CONFLICTING and abs_score >= self.WEAK_SIGNAL_THRESHOLD:
            suggested_side = Side.LONG if total_score > 0 else Side.SHORT

        # Check volatility
        volatility_ok = not self._volatility.is_low_volatility(snapshot)

        # Determine market bias from factors
        market_bias = self._determine_market_bias(total_score)

        return CompositeSignal(
            timestamp=datetime.now(),
            total_score=total_score,
            strength=strength,
            suggested_side=suggested_side,
            factor_scores=factor_scores,
            primary_signal=primary_signal,
            volatility_ok=volatility_ok,
            market_bias=market_bias,
        )

    def _compute_factor_score(
        self,
        factor: Strategy,
        signal: StrategySignal | None,
        snapshot: DataSnapshot,
    ) -> FactorScore:
        """Compute bullish/bearish score for a factor.

        Args:
            factor: The strategy factor
            signal: Signal generated (if any)
            snapshot: Current snapshot

        Returns:
            FactorScore with bullish_score in [-1, 1]
        """
        weight = self.FACTOR_WEIGHTS.get(factor.name.split("_")[0].lower(), 0.1)
        bullish_score = 0.0
        data: dict[str, Any] = {}

        if factor.name == "ma_position":
            # Check if price is above/below MA
            bias = self._ma_position.get_current_bias(snapshot)
            if bias == "bullish":
                bullish_score = 1.0
            elif bias == "bearish":
                bullish_score = -1.0
            data["bias"] = bias

        elif factor.name == "RSI_Oversold_Overbought":
            # Check RSI value from indicators
            indicators = snapshot.indicators or {}
            rsi = indicators.get("RSI_14")
            if rsi is not None:
                if rsi <= 30:
                    bullish_score = 0.8  # Oversold -> bullish
                elif rsi >= 70:
                    bullish_score = -0.8  # Overbought -> bearish
                elif rsi < 50:
                    bullish_score = 0.2
                elif rsi > 50:
                    bullish_score = -0.2
                data["rsi"] = rsi

        elif factor.name == "funding_rate":
            bias = self._funding.get_current_bias(snapshot)
            if bias == "bullish":
                bullish_score = 0.6  # Negative funding -> bullish
            elif bias == "bearish":
                bullish_score = -0.6  # Positive funding -> bearish
            data["bias"] = bias

        elif factor.name == "long_short_ratio":
            bias = self._long_short.get_current_bias(snapshot)
            if bias == "bullish":
                bullish_score = 0.5  # Shorts dominant (contrarian)
            elif bias == "bearish":
                bullish_score = -0.5  # Longs dominant (contrarian)
            data["bias"] = bias

        elif factor.name == "volatility":
            # Volatility doesn't have directional bias
            # but low volatility should reduce confidence
            if self._volatility.is_low_volatility(snapshot):
                bullish_score = 0.0  # Neutral, but flagged elsewhere
                data["low_volatility"] = True
            data["volatility_state"] = (
                "low" if self._volatility.is_low_volatility(snapshot) else "normal"
            )

        return FactorScore(
            factor_name=factor.name,
            signal_type=signal.signal_type if signal else None,
            bullish_score=bullish_score,
            weight=weight,
            data=data,
        )

    def _has_conflicting_signals(self, scores: list[FactorScore]) -> bool:
        """Check if factor scores are conflicting.

        Args:
            scores: List of factor scores

        Returns:
            True if signals are conflicting
        """
        bullish_count = sum(1 for s in scores if s.bullish_score > 0.3)
        bearish_count = sum(1 for s in scores if s.bullish_score < -0.3)

        # Conflicting if roughly equal number of strong signals
        return bullish_count >= 2 and bearish_count >= 2

    def _get_bias_adjustment(self) -> float:
        """Get market bias adjustment based on default bias.

        Returns:
            Adjustment value to add to total score
        """
        # Per STRATEGY.md: overall bearish bias
        bias_adjustments = {
            MarketBias.STRONGLY_BEARISH: -0.2,
            MarketBias.BEARISH: -0.1,
            MarketBias.NEUTRAL: 0.0,
            MarketBias.BULLISH: 0.1,
            MarketBias.STRONGLY_BULLISH: 0.2,
        }
        return bias_adjustments.get(self._default_market_bias, 0.0)

    def _determine_market_bias(self, score: float) -> MarketBias:
        """Determine market bias from composite score.

        Args:
            score: Composite score

        Returns:
            MarketBias enum
        """
        if score >= 0.6:
            return MarketBias.STRONGLY_BULLISH
        elif score >= 0.3:
            return MarketBias.BULLISH
        elif score <= -0.6:
            return MarketBias.STRONGLY_BEARISH
        elif score <= -0.3:
            return MarketBias.BEARISH
        return MarketBias.NEUTRAL

    def _check_position_pnl(self, snapshot: DataSnapshot) -> StateTransition | None:
        """Check position P&L and return any triggered transitions.

        Args:
            snapshot: Current snapshot

        Returns:
            StateTransition if triggered, None otherwise
        """
        if not snapshot.position:
            return None

        # Explicitly check for missing or invalid margin
        margin_raw = snapshot.position.get("margin")
        if margin_raw is None:
            return None

        margin = Decimal(str(margin_raw))
        if margin <= 0:
            return None

        pnl = Decimal(str(snapshot.position.get("unrealized_pnl", 0)))
        pnl_percent = (pnl / margin) * 100

        return self._state_machine.update_position(pnl, pnl_percent)

    def _generate_signal(
        self,
        composite: CompositeSignal,
        snapshot: DataSnapshot,
    ) -> StrategySignal | None:
        """Generate signal based on composite evaluation and state.

        Args:
            composite: Composite signal analysis
            snapshot: Current snapshot

        Returns:
            Signal to process or None
        """
        current_state = self._state_machine.state

        # Only generate entry signals when IDLE
        if current_state == StrategyState.IDLE:
            return self._generate_entry_signal(composite, snapshot)

        # Handle profit threshold state
        elif current_state == StrategyState.PROFIT_SIGNAL:
            return self._generate_profit_signal(snapshot)

        # In position but not at profit threshold - continue monitoring
        elif current_state == StrategyState.IN_POSITION:
            # Could generate exit signals here based on factor reversal
            return None

        return None

    def _generate_entry_signal(
        self,
        composite: CompositeSignal,
        snapshot: DataSnapshot,
    ) -> StrategySignal | None:
        """Generate entry signal if conditions warrant.

        Args:
            composite: Composite analysis
            snapshot: Current snapshot

        Returns:
            Entry signal or None
        """
        # Determine skip reason (if any)
        skip_reason: str | None = None
        if not composite.volatility_ok:
            skip_reason = "low_volatility"
        elif composite.strength in (SignalStrength.WEAK, SignalStrength.CONFLICTING):
            skip_reason = f"strength={composite.strength.value}"

        # Only log if score changed significantly or reason changed
        score_changed = (
            self._last_logged_score is None
            or abs(composite.total_score - self._last_logged_score) >= 0.05
        )
        reason_changed = skip_reason != self._last_logged_reason

        if score_changed or reason_changed:
            logger.info(
                f"Strategy: score={composite.total_score:+.2f}, "
                f"strength={composite.strength.value}, side={composite.suggested_side}, "
                f"volatility_ok={composite.volatility_ok}"
            )
            if skip_reason:
                logger.info(f"Skipping signal: {skip_reason} (need moderate+)")
            self._last_logged_score = composite.total_score
            self._last_logged_reason = skip_reason

        # Don't trade in low volatility
        if not composite.volatility_ok:
            return None

        # Need at least moderate signal strength
        if composite.strength in (SignalStrength.WEAK, SignalStrength.CONFLICTING):
            return None

        # Determine signal type
        if composite.suggested_side == Side.LONG:
            signal_type = SignalType.STRONG_BULLISH
        elif composite.suggested_side == Side.SHORT:
            signal_type = SignalType.STRONG_BEARISH
        else:
            return None

        # Check debounce - don't send same signal type within 5 minutes
        if self._state_machine.should_debounce_signal(signal_type.value):
            remaining = (
                self._state_machine.context.signal_debounce_seconds
                - (
                    datetime.now()
                    - (self._state_machine.context.last_signal_time or datetime.now())
                ).total_seconds()
            )
            logger.debug(f"Signal debounced: {signal_type.value} (wait {remaining:.0f}s more)")
            return None

        # Record signal for debounce tracking
        self._state_machine.record_signal(signal_type.value)

        # Create composite signal
        signal = StrategySignal(
            signal_type=signal_type,
            data={
                "composite_score": composite.total_score,
                "strength": composite.strength.value,
                "suggested_side": composite.suggested_side.value
                if composite.suggested_side
                else None,
                "market_bias": composite.market_bias.value,
                "volatility_ok": composite.volatility_ok,
                "factor_analysis": [
                    {
                        "factor": fs.factor_name,
                        "score": fs.bullish_score,
                        "weight": fs.weight,
                        "signal": fs.signal_type.value if fs.signal_type else None,
                    }
                    for fs in composite.factor_scores
                ],
                "primary_signal": (
                    composite.primary_signal.value if composite.primary_signal else None
                ),
            },
            priority=3 if composite.strength == SignalStrength.STRONG else 2,
        )

        # Transition state machine
        self._state_machine.transition(
            StateTransition.ENTRY_SIGNAL,
            {
                "signal_type": signal_type.value,
                "signal_data": signal.data,
            },
        )

        # Queue and return
        logger.info(
            f"🚀 Entry signal generated: {signal_type.value}, "
            f"side={composite.suggested_side}, strength={composite.strength.value}"
        )
        self._signal_queue.push(signal)
        return signal

    def _generate_profit_signal(self, snapshot: DataSnapshot) -> StrategySignal | None:
        """Generate profit threshold signal for agent decision.

        Args:
            snapshot: Current snapshot

        Returns:
            Profit signal for agent
        """
        if not snapshot.position:
            return None

        signal = StrategySignal(
            signal_type=SignalType.PROFIT_THRESHOLD_REACHED,
            data={
                "position_side": snapshot.position.get("side"),
                "entry_price": snapshot.position.get("entry_price"),
                "current_pnl_percent": float(
                    self._state_machine.context.position.unrealized_pnl_percent
                ),
                "highest_pnl_percent": float(
                    self._state_machine.context.position.highest_pnl_percent
                ),
                "recommendation": "Consider taking profit or moving stop-loss",
            },
            priority=2,
        )

        self._signal_queue.push(signal)
        return signal

    def handle_agent_decision(
        self,
        decision: str,
        position_context: PositionContext | None = None,
    ) -> bool:
        """Handle agent decision and update state machine.

        Args:
            decision: Agent decision (open, close, observe, hold)
            position_context: Position context if opening

        Returns:
            True if transition succeeded
        """
        current_state = self._state_machine.state

        if current_state == StrategyState.WAITING_ENTRY:
            if decision == "open" and position_context:
                return self._state_machine.transition(
                    StateTransition.AGENT_OPEN,
                    {"position": position_context},
                )
            elif decision == "observe":
                return self._state_machine.transition(StateTransition.AGENT_OBSERVE)

        elif current_state == StrategyState.PROFIT_SIGNAL:
            if decision == "close":
                return self._state_machine.transition(StateTransition.AGENT_CLOSE)
            elif decision in ("hold", "observe"):
                return self._state_machine.transition(StateTransition.AGENT_HOLD)

        elif current_state == StrategyState.WAITING_EXIT and decision == "close":
            if self._state_machine.can_transition(StateTransition.AGENT_CLOSE):
                return self._state_machine.transition(StateTransition.AGENT_CLOSE)
            return False

        return False

    def on_order_filled(self, position_context: PositionContext) -> bool:
        """Handle order filled event.

        Args:
            position_context: New position context

        Returns:
            True if transition succeeded
        """
        return self._state_machine.transition(
            StateTransition.ORDER_FILLED,
            {"position": position_context},
        )

    def on_position_closed(self) -> bool:
        """Handle position closed event.

        Returns:
            True if transition succeeded
        """
        return self._state_machine.transition(StateTransition.POSITION_CLOSED)

    def on_order_failed(self) -> bool:
        """Handle order failed event.

        Returns:
            True if transition succeeded
        """
        return self._state_machine.transition(StateTransition.ORDER_FAILED)

    def force_risk_close(self) -> bool:
        """Trigger force close due to risk control.

        Returns:
            True if transition succeeded
        """
        return self._state_machine.transition(StateTransition.RISK_TRIGGERED)

    def get_status(self) -> dict[str, Any]:
        """Get current orchestrator status for logging/monitoring.

        Returns:
            Status dictionary
        """
        return {
            "state": self._state_machine.state.value,
            "state_info": self._state_machine.get_state_info(),
            "has_position": self.has_position,
            "is_ready": self.is_ready_for_signals,
            "default_bias": self._default_market_bias.value,
            "factors_enabled": [f.name for f in self._factors if f.enabled],
        }
