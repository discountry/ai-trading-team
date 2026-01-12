"""Trading agent main logic."""

import contextlib
import json
import logging
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ai_trading_team.agent.commands import AgentAction, AgentCommand
from ai_trading_team.agent.prompts import DECISION_PROMPT, PROFIT_SIGNAL_PROMPT, SYSTEM_PROMPT
from ai_trading_team.agent.schemas import AgentDecision
from ai_trading_team.config import Config
from ai_trading_team.core.data_pool import DataSnapshot
from ai_trading_team.core.signal_queue import StrategySignal
from ai_trading_team.core.types import OrderType, Side

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
        self._llm = self._create_llm()

    def _create_llm(self) -> Any:
        """Create LangChain LLM client."""
        from langchain_anthropic import ChatAnthropic
        from pydantic import SecretStr

        return ChatAnthropic(
            model_name="MiniMax-M2.1",
            api_key=SecretStr(self._config.api.anthropic_api_key),
            base_url=self._config.api.anthropic_base_url,
            max_tokens_to_sample=2048,
            timeout=None,
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
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=DECISION_PROMPT.format(**context)),
        ]

        try:
            # Invoke the LLM
            response = await self._llm.ainvoke(messages)
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

            logger.info(
                f"Agent decision: action={command.action.value}, "
                f"side={command.side}, reason={command.reason[:50]}..."
            )

            return decision

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            # Return observe command on error
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
            size=float(data["size"]) if data.get("size") else None,
            price=float(data["price"]) if data.get("price") else None,
            order_type=order_type,
            stop_loss_price=float(data["stop_loss_price"]) if data.get("stop_loss_price") else None,
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
            timeframe_order = ["5m", "15m", "1h", "4h"]
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

        # Format position
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
            available = acc.get("available", 0)
            max_margin = available * self._config.trading.max_position_percent / 100
            account_str = (
                f"Balance: {acc.get('balance')} USDT, "
                f"Available: {available} USDT, "
                f"Used Margin: {acc.get('margin')} USDT\n"
                f"Leverage: {self._config.trading.leverage}x, "
                f"Max Position Percent: {self._config.trading.max_position_percent}%, "
                f"Max Usable Margin: {max_margin:.2f} USDT"
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
        signal_data = signal.data
        factor_analysis = "N/A"
        if "factor_analysis" in signal_data:
            factor_lines = []
            for fa in signal_data.get("factor_analysis", []):
                factor_lines.append(
                    f"- {fa.get('factor', 'N/A')}: "
                    f"score={fa.get('score', 0):.2f}, "
                    f"weight={fa.get('weight', 0):.2f}"
                )
            factor_analysis = "\n".join(factor_lines) if factor_lines else "N/A"

        return {
            "signal_type": signal.signal_type.value,
            "signal_data": json.dumps(signal.data, default=str),
            "signal_strength": signal_data.get("strength", "unknown"),
            "suggested_side": signal_data.get("suggested_side", "N/A"),
            "composite_score": signal_data.get("composite_score", "N/A"),
            "market_bias": signal_data.get("market_bias", "neutral"),
            "volatility_ok": signal_data.get("volatility_ok", True),
            "factor_analysis": factor_analysis,
            "ticker": ticker_str,
            "klines": klines_str,
            "orderbook": orderbook_str,
            "indicators": indicators_str,
            "funding_rate": funding_str,
            "long_short_ratio": ls_ratio_str,
            "position": position_str,
            "orders": orders_str,
            "account": account_str,
            "recent_operations": ops_str,
        }

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
            "symbol": self._symbol,
        }

        # Create messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=PROFIT_SIGNAL_PROMPT.format(**context)),
        ]

        try:
            # Invoke the LLM
            response = await self._llm.ainvoke(messages)
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
