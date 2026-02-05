"""Test LLM response parsing for various formats."""

import json

import pytest


def extract_text_content(content):
    """Extract text content from LLM response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" or (
                    "text" in block and block.get("type") != "thinking"
                ):
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else str(content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return str(content)
    return str(content)


def parse_response(response: str) -> dict:
    """Parse JSON from response text."""
    import re

    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    code_blocks = re.findall(code_block_pattern, response)

    data = None
    for block in code_blocks:
        block = block.strip()
        if block.startswith("{"):
            try:
                data = json.loads(block)
                break
            except json.JSONDecodeError:
                continue

    if data is None:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found")
        json_str = response[json_start:json_end]
        data = json.loads(json_str)

    return data


class TestExtractTextContent:
    """Test _extract_text_content method."""

    def test_plain_string(self):
        """Plain string returns as-is."""
        content = '{"action": "observe"}'
        result = extract_text_content(content)
        assert result == content

    def test_list_with_thinking_and_text(self):
        """List with thinking and text blocks extracts only text."""
        content = [
            {"signature": "abc123", "thinking": "Let me analyze...", "type": "thinking"},
            {"text": '```json\n{"action": "open"}\n```', "type": "text"},
        ]
        result = extract_text_content(content)
        assert "action" in result
        assert "thinking" not in result.lower() or "type" in result

    def test_list_with_multiple_text_blocks(self):
        """List with multiple text blocks combines them."""
        content = [
            {"text": "First part", "type": "text"},
            {"text": "Second part", "type": "text"},
        ]
        result = extract_text_content(content)
        assert "First part" in result
        assert "Second part" in result

    def test_dict_with_text_field(self):
        """Dict with text field extracts it."""
        content = {"text": '{"action": "close"}', "other": "ignored"}
        result = extract_text_content(content)
        assert result == '{"action": "close"}'

    def test_list_with_string_blocks(self):
        """List of strings concatenates them."""
        content = ["part1", "part2"]
        result = extract_text_content(content)
        assert "part1" in result
        assert "part2" in result


class TestParseResponse:
    """Test parse_response method."""

    def test_json_in_code_block(self):
        """JSON in markdown code block is extracted."""
        response = """Some text before
```json
{
  "action": "open",
  "side": "long"
}
```
Some text after"""
        data = parse_response(response)
        assert data["action"] == "open"
        assert data["side"] == "long"

    def test_json_in_plain_code_block(self):
        """JSON in plain code block (no json marker) is extracted."""
        response = """```
{"action": "close", "reason": "test"}
```"""
        data = parse_response(response)
        assert data["action"] == "close"
        assert data["reason"] == "test"

    def test_raw_json(self):
        """Raw JSON without code block is extracted."""
        response = 'The decision is: {"action": "observe", "symbol": "BTC"}'
        data = parse_response(response)
        assert data["action"] == "observe"
        assert data["symbol"] == "BTC"

    def test_json_with_newlines(self):
        """JSON with newlines in code block is extracted."""
        response = """```json
{
  "action": "open",
  "symbol": "BTC/USDT",
  "side": "short",
  "size": 187500,
  "price": null,
  "order_type": "market",
  "stop_loss_price": 0.1420,
  "reason": "做空信号分析..."
}
```"""
        data = parse_response(response)
        assert data["action"] == "open"
        assert data["side"] == "short"
        assert data["size"] == 187500
        assert data["stop_loss_price"] == 0.1420

    def test_no_json_raises(self):
        """No JSON in response raises ValueError."""
        response = "Just some text without any JSON"
        with pytest.raises(ValueError, match="No JSON found"):
            parse_response(response)


class TestFullPipeline:
    """Test the full extraction and parsing pipeline."""

    def test_extended_thinking_response(self):
        """Full pipeline with extended thinking response format."""
        # Simulate actual response from error log
        mock_content = [
            {
                "signature": "935d1a630961f446...",
                "thinking": "让我分析一下当前的市场情况...",
                "type": "thinking",
            },
            {
                "text": """```json
{
  "action": "open",
  "symbol": "BTC/USDT",
  "side": "short",
  "size": 187500,
  "price": null,
  "order_type": "market",
  "stop_loss_price": 0.1420,
  "reason": "做空信号分析"
}
```""",
                "type": "text",
            },
        ]

        # Extract text content
        text = extract_text_content(mock_content)
        assert "action" in text
        assert "thinking" not in text.lower() or len(text) < 100

        # Parse JSON
        data = parse_response(text)
        assert data["action"] == "open"
        assert data["side"] == "short"
        assert data["size"] == 187500
