"""AI API module for Weex SDK."""

from typing import Any, Dict, Optional

from weex_sdk.client import AsyncWeexClient, WeexClient


class AIAPI:
    """AI API methods."""

    def __init__(self, client: WeexClient) -> None:
        """Initialize AI API.

        Args:
            client: WeexClient instance
        """
        self.client = client

    def upload_ai_log(
        self,
        stage: str,
        model: str,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        explanation: str,
        order_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Upload AI log for order execution.

        Args:
            stage: Stage identifier (e.g., 'Decision Making', 'Strategy Generation')
            model: Model name (e.g., 'GPT-5-mini', 'GPT-5-turbo')
            input_data: Input parameters (JSON dict)
            output: Output results (JSON dict)
            explanation: Concise explanatory summary of AI's behavior (max 500 words)
            order_id: Order ID (optional)

        Returns:
            Upload response with success status

        Raises:
            WeexAPIError: On API errors

        Example:
            >>> client = WeexClient(api_key, secret_key, passphrase)
            >>> client.ai.upload_ai_log(
            ...     stage="Decision Making",
            ...     model="GPT-5-mini",
            ...     input_data={"prompt": "Analyze BTC trend"},
            ...     output={"signal": "Buy", "confidence": 0.82},
            ...     explanation="AI analyzed market data and generated buy signal"
            ... )
        """
        data: Dict[str, Any] = {
            "stage": stage,
            "model": model,
            "input": input_data,
            "output": output,
            "explanation": explanation,
        }

        if order_id is not None:
            data["orderId"] = order_id

        return self.client.post("/capi/v2/order/uploadAiLog", data=data)


class AsyncAIAPI:
    """Async AI API methods."""

    def __init__(self, client: AsyncWeexClient) -> None:
        """Initialize Async AI API.

        Args:
            client: AsyncWeexClient instance
        """
        self.client = client

    async def upload_ai_log(
        self,
        stage: str,
        model: str,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        explanation: str,
        order_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Upload AI log for order execution (async).

        Args:
            stage: Stage identifier
            model: Model name
            input_data: Input parameters (JSON dict)
            output: Output results (JSON dict)
            explanation: Explanatory summary (max 500 words)
            order_id: Order ID (optional)

        Returns:
            Upload response with success status

        Raises:
            WeexAPIError: On API errors
        """
        data: Dict[str, Any] = {
            "stage": stage,
            "model": model,
            "input": input_data,
            "output": output,
            "explanation": explanation,
        }

        if order_id is not None:
            data["orderId"] = order_id

        return await self.client.post("/capi/v2/order/uploadAiLog", data=data)
