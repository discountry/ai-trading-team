"""Trading agent main logic."""

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ai_trading_team.agent.commands import AgentAction, AgentCommand
from ai_trading_team.agent.schemas import AgentDecision
from ai_trading_team.config import Config
from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.core.signal_queue import StrategySignal
from ai_trading_team.core.types import OrderType, Side
from ai_trading_team.indicators.volatility import VolatilityAnalyzer

logger = logging.getLogger(__name__)


class LangChainTradingAgent:
    """LangChain-based trading agent.

    Uses Claude via langchain-anthropic to process signals and generate commands.
    """

    def __init__(self, config: Config, symbol: str) -> None:
        """Initialize trading agent.

        Args:
            config: Application configuration
            symbol: Trading symbol (e.g., "cmt_btcusdt")
        """
        self._config = config
        self._symbol = symbol
        self._llm_timeout_seconds = 60.0
        self._llm = self._create_llm()
        self._prompts = self._load_prompts()
        # Dynamic volatility analyzer - adapts to each symbol's characteristics
        self._volatility_history_size = 100
        self._volatility_min_samples = 20
        self._volatility_multiplier = 0.8
        self._volatility_analyzer = VolatilityAnalyzer(
            history_size=self._volatility_history_size,
            min_samples=self._volatility_min_samples,
            volatility_multiplier=self._volatility_multiplier,  # Threshold = 80% of average
        )
        self._oi_history: dict[str, deque[tuple[datetime, float]]] = {
            "15m": deque(maxlen=30),
            "1h": deque(maxlen=30),
            "4h": deque(maxlen=30),
        }
        self._oi_current: tuple[datetime, float] | None = None
        self._oi_windows: tuple[tuple[str, int], ...] = (("15m", 15), ("1h", 60), ("4h", 240))
        self._oi_series_points = 5

    def _load_prompts(self) -> dict[str, str]:
        """Load prompts based on config.trading.prompt_style.

        Returns:
            Dict with SYSTEM_PROMPT, DECISION_PROMPT, PROFIT_SIGNAL_PROMPT
        """
        style = self._config.trading.prompt_style

        if style == "long":
            from ai_trading_team.agent.prompts_long import (
                DECISION_PROMPT,
                PROFIT_SIGNAL_PROMPT,
                SYSTEM_PROMPT,
            )
        elif style == "short":
            from ai_trading_team.agent.prompts import (
                DECISION_PROMPT,
                PROFIT_SIGNAL_PROMPT,
                SYSTEM_PROMPT,
            )
        else:  # neutral (default)
            from ai_trading_team.agent.prompts_neutral import (
                DECISION_PROMPT,
                PROFIT_SIGNAL_PROMPT,
                SYSTEM_PROMPT,
            )

        logger.info(f"Loaded prompt style: {style}")

        return {
            "system": SYSTEM_PROMPT,
            "decision": DECISION_PROMPT,
            "profit_signal": PROFIT_SIGNAL_PROMPT,
        }

    def _create_llm(self) -> Any:
        """Create LangChain LLM client."""
        from langchain_anthropic import ChatAnthropic
        from pydantic import SecretStr

        return ChatAnthropic(
            model_name="MiniMax-M2.1",
            api_key=SecretStr(self._config.api.anthropic_api_key),
            base_url=self._config.api.anthropic_base_url,
            max_tokens_to_sample=2048,
            timeout=self._llm_timeout_seconds,
            stop=None,
        )

    async def process_signal(
        self,
        signal: StrategySignal,
        snapshot: DataSnapshot,
    ) -> AgentDecision:
        """Process a strategy signal and generate a decision.

        Args:
            signal: Strategy signal that triggered the agent
            snapshot: Current market data snapshot

        Returns:
            Agent decision with command and metadata
        """
        start_time = time.time()

        # Format context for the prompt
        context = self.format_context(signal, snapshot)

        # Create messages
        messages = [
            SystemMessage(content=self._prompts["system"]),
            HumanMessage(content=self._prompts["decision"].format(**context)),
        ]

        # Retry logic for JSON parsing failures
        max_retries = 2
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                # Invoke the LLM
                response = await asyncio.wait_for(
                    self._llm.ainvoke(messages),
                    timeout=self._llm_timeout_seconds,
                )
                raw_content = response.content

                # Extract text content from response (handles thinking blocks, etc.)
                text_content = self._extract_text_content(raw_content)

                # Parse the response
                command = self.parse_response(text_content)

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Get token usage if available
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = response.usage_metadata.get("input_tokens", 0)
                    completion_tokens = response.usage_metadata.get("output_tokens", 0)

                decision = AgentDecision(
                    signal_type=signal.signal_type.value,
                    signal_data=signal.data,
                    market_snapshot=self._snapshot_to_dict(snapshot),
                    command=command,
                    timestamp=datetime.now(),
                    model="MiniMax-M2.1",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )

                if attempt > 0:
                    logger.info(f"Agent succeeded on retry {attempt}")

                logger.info(
                    f"Agent decision: action={command.action.value}, "
                    f"side={command.side}, reason={command.reason[:50]}..."
                )

                return decision

            except TimeoutError:
                latency_ms = (time.time() - start_time) * 1000
                logger.error("Agent timeout after 60 seconds")
                return AgentDecision(
                    signal_type=signal.signal_type.value,
                    signal_data=signal.data,
                    market_snapshot=self._snapshot_to_dict(snapshot),
                    command=AgentCommand(
                        action=AgentAction.OBSERVE,
                        symbol=self._symbol,
                        reason="Agent timeout after 60 seconds. Observing for safety.",
                    ),
                    timestamp=datetime.now(),
                    model="MiniMax-M2.1",
                    latency_ms=latency_ms,
                )
            except ValueError as e:
                # JSON parsing errors - retry if attempts remaining
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Agent JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying..."
                    )
                    continue
                # Fall through to return OBSERVE after all retries exhausted
            except Exception as e:
                # Other errors - don't retry
                logger.error(f"Agent processing error: {e}")
                latency_ms = (time.time() - start_time) * 1000
                return AgentDecision(
                    signal_type=signal.signal_type.value,
                    signal_data=signal.data,
                    market_snapshot=self._snapshot_to_dict(snapshot),
                    command=AgentCommand(
                        action=AgentAction.OBSERVE,
                        symbol=self._symbol,
                        reason=f"Agent error: {e}. Observing for safety.",
                    ),
                    timestamp=datetime.now(),
                    model="MiniMax-M2.1",
                    latency_ms=latency_ms,
                )

        # All retries exhausted for JSON parsing errors
        logger.error(f"Agent JSON parse failed after {max_retries + 1} attempts: {last_error}")
        latency_ms = (time.time() - start_time) * 1000
        return AgentDecision(
            signal_type=signal.signal_type.value,
            signal_data=signal.data,
            market_snapshot=self._snapshot_to_dict(snapshot),
            command=AgentCommand(
                action=AgentAction.OBSERVE,
                symbol=self._symbol,
                reason=f"Agent error after {max_retries + 1} attempts: {last_error}. Observing for safety.",
            ),
            timestamp=datetime.now(),
            model="MiniMax-M2.1",
            latency_ms=latency_ms,
        )

    def _extract_text_content(self, content: Any) -> str:
        """Extract text content from LLM response.

        Handles different response formats:
        - Plain string
        - List of content blocks (thinking, text, etc.)
        - Dict with text field

        Args:
            content: Raw LLM response content

        Returns:
            Extracted text content as string
        """
        # Plain string
        if isinstance(content, str):
            return content

        # List of content blocks (e.g., from extended thinking models)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Look for 'text' type blocks or direct text field (not thinking)
                    if block.get("type") == "text" or (
                        "text" in block and block.get("type") != "thinking"
                    ):
                        text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return "\n".join(text_parts) if text_parts else str(content)

        # Dict with text field
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return str(content)

        return str(content)

    def parse_response(self, response: str) -> AgentCommand:
        """Parse LLM response into structured command.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed agent command

        Raises:
            ValueError: If response cannot be parsed
        """
        import re

        # Extract JSON from response
        try:
            # First, try to extract JSON from markdown code blocks
            # Match ```json ... ``` or ``` ... ```
            code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
            code_blocks = re.findall(code_block_pattern, response)

            json_str = None
            data = None

            # Try each code block to find valid JSON
            for block in code_blocks:
                block = block.strip()
                if block.startswith("{"):
                    try:
                        data = json.loads(block)
                        json_str = block
                        break
                    except json.JSONDecodeError:
                        continue

            # If no valid JSON in code blocks, try to find raw JSON
            if data is None:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")

                json_str = response[json_start:json_end]
                data = json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, response: {response[:500]}...")
            raise ValueError(f"Failed to parse JSON: {e}") from e

        def _parse_float(value: Any) -> float | None:
            if value is None:
                return None
            if isinstance(value, int | float):
                return float(value)
            if isinstance(value, str):
                raw = value.strip().lower()
                if raw in {"", "none", "null", "n/a", "na", "market"}:
                    return None
                try:
                    return float(raw)
                except ValueError:
                    return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        # Parse action
        action_str = data.get("action", "observe").lower()
        try:
            action = AgentAction(action_str)
        except ValueError:
            action = AgentAction.OBSERVE

        # Parse side
        side_str = data.get("side")
        side: Side | None = None
        if side_str:
            with contextlib.suppress(ValueError):
                side = Side(side_str.lower())

        # Parse order type
        order_type_str = data.get("order_type")
        order_type: OrderType | None = None
        if order_type_str:
            try:
                order_type = OrderType(order_type_str.lower())
            except ValueError:
                order_type = OrderType.MARKET

        return AgentCommand(
            action=action,
            symbol=data.get("symbol", self._symbol),
            side=side,
            size=_parse_float(data.get("size")),
            price=_parse_float(data.get("price")),
            order_type=order_type,
            stop_loss_price=_parse_float(data.get("stop_loss_price")),
            take_profit_price=_parse_float(data.get("take_profit_price", data.get("take_profit"))),
            reason=data.get("reason", "No reason provided"),
        )

    def format_context(
        self,
        signal: StrategySignal,
        snapshot: DataSnapshot,
    ) -> dict[str, Any]:
        """Format signal and snapshot into prompt context.

        Args:
            signal: Strategy signal
            snapshot: Market data snapshot

        Returns:
            Context dict for prompt formatting
        """
        # Format ticker
        ticker_str = "N/A"
        if snapshot.ticker:
            ticker = snapshot.ticker
            ticker_str = (
                f"Last: {ticker.get('last_price', 'N/A')}, "
                f"Bid: {ticker.get('bid_price', 'N/A')}, "
                f"Ask: {ticker.get('ask_price', 'N/A')}, "
                f"24h Change: {ticker.get('price_change_percent', 'N/A')}%"
            )

        # Format klines for ALL timeframes (multi-timeframe analysis)
        klines_str = "N/A"
        if snapshot.klines:
            klines_parts = []
            timeframe_order = ["15m", "1h", "4h"]
            for interval in timeframe_order:
                klines = snapshot.klines.get(interval, [])
                if klines:
                    last_5 = klines[-5:]
                    tf_lines = [f"【{interval}】 (最近5根K线):"]
                    for k in last_5:
                        tf_lines.append(
                            f"  O:{k.get('open', 0):.4f} H:{k.get('high', 0):.4f} "
                            f"L:{k.get('low', 0):.4f} C:{k.get('close', 0):.4f}"
                        )
                    klines_parts.append("\n".join(tf_lines))
            klines_str = "\n".join(klines_parts) if klines_parts else "N/A"

        # Format orderbook
        orderbook_str = "N/A"
        if snapshot.orderbook:
            ob = snapshot.orderbook
            bids = ob.get("bids", [])[:3]
            asks = ob.get("asks", [])[:3]
            orderbook_str = f"Bids: {bids}, Asks: {asks}"

        # Format indicators
        indicators_str = "N/A"
        if snapshot.indicators:
            indicators_str = self._format_indicators(snapshot.indicators)

        # Format ATR (multi-timeframe) and update volatility analyzer
        atr_str, atr_15m, atr_1h, atr_4h = self._extract_atr_values(snapshot)
        self._volatility_analyzer.update(atr_15m, atr_1h, atr_4h)
        volatility_analysis = self._volatility_analyzer.format_for_prompt()

        # Format funding rate
        funding_str = "N/A"
        if snapshot.funding_rate:
            fr = snapshot.funding_rate
            rate = fr.get("funding_rate", 0)
            funding_str = f"Rate: {float(rate) * 100:.4f}%"

        # Format long/short ratio
        ls_ratio_str = "N/A"
        if snapshot.long_short_ratio:
            ls = snapshot.long_short_ratio
            long_r = ls.get("long_ratio", 0.5)
            short_r = ls.get("short_ratio", 0.5)
            ls_ratio_str = f"Long: {float(long_r) * 100:.1f}%, Short: {float(short_r) * 100:.1f}%"

        open_interest_str, open_interest_series_str = self._format_open_interest_context(snapshot)

        # Format position
        position_str = "无持仓"
        if snapshot.position:
            pos = snapshot.position
            pnl_value = pos.get("unrealized_pnl")
            try:
                pnl_numeric = float(pnl_value) if pnl_value is not None else None
            except (TypeError, ValueError):
                pnl_numeric = None

            if pnl_numeric is None or abs(pnl_numeric) < 1e-12:
                current_price = None
                if snapshot.ticker:
                    try:
                        current_price = float(snapshot.ticker.get("last_price", 0))
                    except (TypeError, ValueError):
                        current_price = None
                try:
                    entry_price = float(pos.get("entry_price", 0))
                    size = float(pos.get("size", 0))
                except (TypeError, ValueError):
                    entry_price = 0.0
                    size = 0.0

                side_value = str(pos.get("side", "")).lower()
                if (
                    current_price
                    and entry_price > 0
                    and size > 0
                    and side_value
                    in (
                        "long",
                        "short",
                    )
                ):
                    if side_value == "long":
                        pnl_numeric = (current_price - entry_price) * size
                    else:
                        pnl_numeric = (entry_price - current_price) * size

            position_str = (
                f"Symbol: {pos.get('symbol')}, "
                f"Side: {pos.get('side')}, "
                f"Size: {pos.get('size')}, "
                f"Entry: {pos.get('entry_price')}, "
                f"PnL: {pnl_numeric if pnl_numeric is not None else pos.get('unrealized_pnl')}, "
                f"Margin: {pos.get('margin')}"
            )

        # Format orders
        orders_str = "无挂单"
        if snapshot.orders:
            orders_str = "\n".join(
                f"ID:{o.get('order_id')}, Side:{o.get('side')}, "
                f"Price:{o.get('price')}, Size:{o.get('size')}"
                for o in snapshot.orders
            )

        # Format account with trading config
        account_str = "N/A"
        if snapshot.account:
            acc = snapshot.account
            available = float(acc.get("available", 0))
            used_margin = float(acc.get("margin", 0))
            # 单次开仓保证金 = min(可用余额/20, 750-已用保证金)
            single_margin = min(available / 20, 750 - used_margin)
            single_margin = max(single_margin, 0)  # 不能为负
            account_str = (
                f"Balance: {acc.get('balance')} USDT, "
                f"Available: {available:.2f} USDT, "
                f"Used Margin: {used_margin:.2f} USDT\n"
                f"Leverage: 20x (固定), "
                f"单次最大保证金: {single_margin:.2f} USDT (可用余额/20)"
            )

        # Format recent operations (last 10)
        ops_str = "无历史操作记录"
        if snapshot.recent_operations:
            ops_lines = []
            for op in snapshot.recent_operations[-10:]:
                ops_lines.append(
                    f"[{op.get('timestamp', 'N/A')}] "
                    f"{op.get('action', 'N/A')} {op.get('side', '')} "
                    f"Size:{op.get('size', 'N/A')} "
                    f"Result:{op.get('result', 'N/A')}"
                )
            ops_str = "\n".join(ops_lines) if ops_lines else ops_str

        # Extract signal data for multi-factor analysis
        signal_type = signal.data.get("category", signal.signal_type.value)

        return {
            "signal_type": signal_type,
            "signal_data": json.dumps(signal.data, default=str),
            "ticker": ticker_str,
            "klines": klines_str,
            "orderbook": orderbook_str,
            "indicators": indicators_str,
            "atr": atr_str,
            "volatility_analysis": volatility_analysis,
            "funding_rate": funding_str,
            "long_short_ratio": ls_ratio_str,
            "open_interest": open_interest_str,
            "open_interest_series": open_interest_series_str,
            "position": position_str,
            "orders": orders_str,
            "account": account_str,
            "recent_operations": ops_str,
        }

    def _reset_volatility_analyzer(self) -> None:
        self._volatility_analyzer = VolatilityAnalyzer(
            history_size=self._volatility_history_size,
            min_samples=self._volatility_min_samples,
            volatility_multiplier=self._volatility_multiplier,
        )

    def backfill_volatility(self, snapshot: DataSnapshot) -> int:
        """Backfill volatility analyzer with historical kline ATR series."""
        if not snapshot.klines:
            return 0

        series_by_tf: dict[str, list[tuple[datetime, float]]] = {}
        for tf in ("15m", "1h", "4h"):
            series = self._calculate_atr_series_pct(snapshot.klines.get(tf, []), 14)
            if series:
                series_by_tf[tf] = series

        if not series_by_tf:
            return 0

        base_tf = "4h" if "4h" in series_by_tf else "1h" if "1h" in series_by_tf else "15m"

        self._reset_volatility_analyzer()

        base_series = series_by_tf[base_tf]
        indices = {tf: 0 for tf in series_by_tf}
        last_vals: dict[str, float | None] = {tf: None for tf in series_by_tf}

        for ts, _val in base_series:
            for tf, series in series_by_tf.items():
                while indices[tf] < len(series) and series[indices[tf]][0] <= ts:
                    last_vals[tf] = series[indices[tf]][1]
                    indices[tf] += 1
            self._volatility_analyzer.update(
                last_vals.get("15m"),
                last_vals.get("1h"),
                last_vals.get("4h"),
            )

        return len(self._volatility_analyzer._composite_history)

    def update_volatility(self, snapshot: DataSnapshot) -> None:
        """Update volatility analyzer with latest ATR values."""
        _, atr_15m, atr_1h, atr_4h = self._extract_atr_values(snapshot)
        self._volatility_analyzer.update(atr_15m, atr_1h, atr_4h)

    def update_open_interest(self, oi_data: dict[str, Any] | None) -> None:
        value, ts = self._extract_open_interest(oi_data)
        if value is None:
            return
        if ts is None:
            ts = datetime.now()
        self._oi_current = (ts, value)

    def backfill_open_interest(self, history_by_period: dict[str, list[dict[str, Any]]]) -> int:
        """Backfill open interest history with REST data."""
        if not history_by_period:
            return 0
        total = 0
        for period, history in history_by_period.items():
            if period not in self._oi_history:
                continue
            entries: list[tuple[datetime, float]] = []
            for item in history:
                value, ts = self._extract_open_interest(item)
                if value is None or ts is None:
                    continue
                entries.append((ts, value))
            if not entries:
                continue
            entries.sort(key=lambda x: x[0])
            self._oi_history[period].clear()
            for ts, value in entries:
                self._oi_history[period].append((ts, value))
            total += len(entries)
        return total

    def volatility_ready(self) -> bool:
        """Return True when volatility analyzer has enough samples."""
        return len(self._volatility_analyzer._composite_history) >= self._volatility_analyzer._min_samples

    def volatility_sample_status(self) -> tuple[int, int]:
        """Return (sample_count, min_samples) for volatility analyzer."""
        return (
            len(self._volatility_analyzer._composite_history),
            self._volatility_analyzer._min_samples,
        )

    def _extract_atr_values(
        self, snapshot: DataSnapshot
    ) -> tuple[str, float | None, float | None, float | None]:
        atr_str = "N/A"
        atr_15m, atr_1h, atr_4h = None, None, None
        if snapshot.indicators:
            atr_str = self._format_atr(snapshot.indicators)
            atr_15m = snapshot.indicators.get("ATR_14_15m")
            atr_1h = snapshot.indicators.get("ATR_14_1h")
            atr_4h = snapshot.indicators.get("ATR_14_4h")
        if atr_str == "N/A" and snapshot.klines:
            atr_str = self._format_atr_from_klines(snapshot.klines)
            atr_15m = self._calculate_atr_pct(snapshot.klines.get("15m", []), 14)
            atr_1h = self._calculate_atr_pct(snapshot.klines.get("1h", []), 14)
            atr_4h = self._calculate_atr_pct(snapshot.klines.get("4h", []), 14)
        return atr_str, atr_15m, atr_1h, atr_4h

    def _extract_open_interest(
        self, oi_data: dict[str, Any] | None
    ) -> tuple[float | None, datetime | None]:
        if not oi_data:
            return None, None
        value = oi_data.get("open_interest")
        if value is None:
            value = oi_data.get("openInterest")
        if value is None:
            value = oi_data.get("sumOpenInterest")
        if value is None:
            value = oi_data.get("oi")
        if value is None:
            return None, None
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None, None
        if value_f <= 0:
            return None, None
        ts = self._parse_timestamp(oi_data.get("timestamp") or oi_data.get("time"))
        return value_f, ts

    def _parse_timestamp(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, int | float):
            return datetime.fromtimestamp(value / 1000 if value > 1e12 else value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                numeric = float(raw)
            except ValueError:
                numeric = None
            if numeric is not None:
                return datetime.fromtimestamp(numeric / 1000 if numeric > 1e12 else numeric)
            try:
                return datetime.fromisoformat(raw)
            except (TypeError, ValueError):
                return None
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    def _format_open_interest_context(self, snapshot: DataSnapshot) -> tuple[str, str]:
        if snapshot.open_interest:
            self.update_open_interest(snapshot.open_interest)
        current = self._oi_current or self._latest_oi_from_history()
        if not current:
            return "N/A", "N/A"
        ts, value = current
        ts_str = ts.isoformat() if ts else ""
        oi_str = f"{value:,.0f}"
        if ts_str:
            oi_str = f"{oi_str} ({ts_str})"
        series = self._format_open_interest_series(ts, value)
        return oi_str, series

    def _format_open_interest_series(self, now_ts: datetime, current: float) -> str:
        parts: list[str] = []
        for label, minutes in self._oi_windows:
            history = self._oi_history.get(label)
            values: list[float | None] = [current]
            for i in range(1, self._oi_series_points + 1):
                cutoff = now_ts - timedelta(minutes=minutes * i)
                if history:
                    values.append(self._find_oi_value(history, cutoff))
                else:
                    values.append(None)
            changes: list[str] = []
            for i in range(self._oi_series_points):
                latest = values[i]
                prior = values[i + 1]
                if latest is None or prior is None or prior <= 0:
                    changes.append("N/A")
                    continue
                change_pct = ((latest - prior) / prior) * 100
                changes.append(f"{change_pct:+.2f}%")
            parts.append(f"{label}: {', '.join(changes)}")
        return " | ".join(parts) if parts else "N/A"

    def _find_oi_value(
        self,
        history: deque[tuple[datetime, float]],
        cutoff: datetime,
    ) -> float | None:
        for ts, value in reversed(history):
            if ts <= cutoff:
                return value
        return None

    def _latest_oi_from_history(self) -> tuple[datetime, float] | None:
        latest: tuple[datetime, float] | None = None
        for history in self._oi_history.values():
            if not history:
                continue
            ts, value = history[-1]
            if latest is None or ts > latest[0]:
                latest = (ts, value)
        return latest

    def _format_indicators(self, indicators: dict[str, Any]) -> str:
        """Format indicators dict for AI context.

        Handles both simple values and nested dict values,
        producing human-readable output.

        Args:
            indicators: Raw indicators dict from snapshot

        Returns:
            Formatted string for prompt context
        """
        lines = []

        for key, value in indicators.items():
            if isinstance(value, dict):
                # Format structured indicator data
                parts = []
                for k, v in value.items():
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.2f}")
                    else:
                        parts.append(f"{k}={v}")
                lines.append(f"{key}: {', '.join(parts)}")
            elif isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines) if lines else "N/A"

    def _format_atr(self, indicators: dict[str, Any]) -> str:
        atr_values: list[str] = []
        for interval in ("15m", "1h", "4h"):
            key = f"ATR_14_{interval}"
            value = indicators.get(key)
            if isinstance(value, int | float):
                atr_values.append(f"{interval}:{value:.4f}%")
        composite = indicators.get("ATR_14_COMPOSITE")
        if isinstance(composite, int | float):
            atr_values.append(f"composite:{composite:.4f}%")
        return ", ".join(atr_values) if atr_values else "N/A"

    def _format_atr_from_klines(self, klines: dict[str, list[dict[str, Any]]]) -> str:
        weights = {"15m": 0.5, "1h": 0.3, "4h": 0.2}
        atr_values: list[str] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for interval, weight in weights.items():
            series = klines.get(interval, [])
            atr_pct = self._calculate_atr_pct(series, 14)
            if atr_pct is None:
                continue
            atr_values.append(f"{interval}:{atr_pct:.4f}%")
            weighted_sum += atr_pct * weight
            total_weight += weight
        if total_weight > 0:
            atr_values.append(f"composite:{(weighted_sum / total_weight):.4f}%")
        return ", ".join(atr_values) if atr_values else "N/A"

    def _calculate_atr_pct(
        self,
        klines: list[dict[str, Any]],
        period: int,
    ) -> float | None:
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

    def _kline_timestamp(self, kline: dict[str, Any]) -> datetime | None:
        ts = kline.get("close_time") or kline.get("open_time")
        return ts if isinstance(ts, datetime) else None

    def _calculate_atr_series_pct(
        self,
        klines: list[dict[str, Any]],
        period: int,
    ) -> list[tuple[datetime, float]]:
        if len(klines) < period + 1:
            return []

        true_ranges: list[float] = []
        atr_series: list[tuple[datetime, float]] = []
        tr_sum = 0.0

        for i in range(1, len(klines)):
            high = float(klines[i].get("high", 0))
            low = float(klines[i].get("low", 0))
            prev_close = float(klines[i - 1].get("close", 0))
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            tr_sum += tr

            if len(true_ranges) > period:
                tr_sum -= true_ranges[-period - 1]

            if len(true_ranges) < period:
                continue

            atr = tr_sum / period
            last_close = float(klines[i].get("close", 0))
            if last_close <= 0 or atr <= 0:
                continue
            ts = self._kline_timestamp(klines[i])
            if ts is None:
                continue
            atr_series.append((ts, atr / last_close * 100))

        return atr_series

    def _snapshot_to_dict(self, snapshot: DataSnapshot) -> dict[str, Any]:
        """Convert snapshot to serializable dict."""
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "ticker": snapshot.ticker,
            "klines_count": sum(len(v) for v in (snapshot.klines or {}).values()),
            "indicators": snapshot.indicators,
            "position": snapshot.position,
            "orders_count": len(snapshot.orders or []),
        }

    async def process_profit_signal(
        self,
        snapshot: DataSnapshot,
        profit_data: dict[str, Any],
    ) -> AgentDecision:
        """Process a profit threshold signal and decide stop loss price.

        Args:
            snapshot: Current market data snapshot
            profit_data: Profit signal data containing threshold info

        Returns:
            Agent decision with stop loss command
        """
        start_time = time.time()

        # Format position info
        position_str = "无持仓"
        if snapshot.position:
            pos = snapshot.position
            position_str = (
                f"Symbol: {pos.get('symbol')}, "
                f"Side: {pos.get('side')}, "
                f"Size: {pos.get('size')}, "
                f"Entry: {pos.get('entry_price')}, "
                f"PnL: {pos.get('unrealized_pnl')}, "
                f"Margin: {pos.get('margin')}"
            )

        # Format market summary
        market_summary = "N/A"
        if snapshot.ticker:
            ticker = snapshot.ticker
            market_summary = (
                f"Current Price: {ticker.get('last_price', 'N/A')}, "
                f"24h Change: {ticker.get('price_change_percent', 'N/A')}%"
            )

        # Format indicators
        indicators_str = "N/A"
        if snapshot.indicators:
            indicators_str = self._format_indicators(snapshot.indicators)

        open_interest_str, open_interest_series_str = self._format_open_interest_context(snapshot)

        # Build context for profit signal prompt
        context = {
            "position": position_str,
            "current_pnl_percent": profit_data.get("current_pnl_percent", 0),
            "threshold_level": profit_data.get("threshold_level", 10),
            "highest_pnl_percent": profit_data.get("highest_pnl_percent", 0),
            "position_side": profit_data.get("position_side", "unknown"),
            "entry_price": profit_data.get("entry_price", 0),
            "market_summary": market_summary,
            "indicators": indicators_str,
            "open_interest": open_interest_str,
            "open_interest_series": open_interest_series_str,
            "symbol": self._symbol,
        }

        # Create messages
        messages = [
            SystemMessage(content=self._prompts["system"]),
            HumanMessage(content=self._prompts["profit_signal"].format(**context)),
        ]

        try:
            # Invoke the LLM
            response = await asyncio.wait_for(
                self._llm.ainvoke(messages),
                timeout=self._llm_timeout_seconds,
            )
            raw_content = response.content

            # Extract text content from response (handles thinking blocks, etc.)
            text_content = self._extract_text_content(raw_content)

            # Parse the response
            command = self.parse_response(text_content)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Get token usage if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = response.usage_metadata.get("input_tokens", 0)
                completion_tokens = response.usage_metadata.get("output_tokens", 0)

            decision = AgentDecision(
                signal_type="profit_threshold",
                signal_data=profit_data,
                market_snapshot=self._snapshot_to_dict(snapshot),
                command=command,
                timestamp=datetime.now(),
                model="MiniMax-M2.1",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            logger.info(
                f"Profit signal decision: action={command.action.value}, "
                f"stop_loss_price={command.stop_loss_price}, "
                f"reason={command.reason[:50]}..."
            )

            return decision

        except TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("Profit signal timeout after 60 seconds")
            return AgentDecision(
                signal_type="profit_threshold",
                signal_data=profit_data,
                market_snapshot=self._snapshot_to_dict(snapshot),
                command=AgentCommand(
                    action=AgentAction.OBSERVE,
                    symbol=self._symbol,
                    reason="Agent timeout after 60 seconds. Observing for safety.",
                ),
                timestamp=datetime.now(),
                model="MiniMax-M2.1",
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Profit signal processing error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return AgentDecision(
                signal_type="profit_threshold",
                signal_data=profit_data,
                market_snapshot=self._snapshot_to_dict(snapshot),
                command=AgentCommand(
                    action=AgentAction.OBSERVE,
                    symbol=self._symbol,
                    reason=f"Agent error: {e}. Observing for safety.",
                ),
                timestamp=datetime.now(),
                model="MiniMax-M2.1",
                latency_ms=latency_ms,
            )
