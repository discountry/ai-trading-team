"""Local file log writer."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_trading_team.audit.models import AgentLog, OrderLog


class LocalLogWriter:
    """Writes audit logs to local JSON files."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir or Path("logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _get_daily_file(self, prefix: str) -> Path:
        """Get log file path for today."""
        date_str = datetime.now().strftime("%Y%m%d")
        return self._log_dir / f"{prefix}_{date_str}.jsonl"

    def write_agent_log(self, log: AgentLog) -> None:
        """Write agent decision log."""
        file_path = self._get_daily_file("agent")
        self._append_json(file_path, log.to_dict())

    def write_order_log(self, log: OrderLog) -> None:
        """Write order execution log."""
        file_path = self._get_daily_file("orders")
        self._append_json(file_path, log.to_dict())

    def _append_json(self, file_path: Path, data: dict[str, Any]) -> None:
        """Append JSON line to file."""
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def read_agent_logs(self, date: datetime | None = None) -> list[AgentLog]:
        """Read agent logs for a date (default: today)."""
        if date is None:
            date = datetime.now()

        file_path = self._log_dir / f"agent_{date.strftime('%Y%m%d')}.jsonl"
        if not file_path.exists():
            return []

        logs = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    logs.append(self._dict_to_agent_log(data))
        return logs

    def _dict_to_agent_log(self, data: dict[str, Any]) -> AgentLog:
        """Convert dictionary to AgentLog."""
        return AgentLog(
            log_id=data["log_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signal_type=data.get("signal_type", ""),
            signal_data=data.get("signal_data", {}),
            market_data=data.get("market_data", {}),
            indicators=data.get("indicators", {}),
            position=data.get("position"),
            orders=data.get("orders", []),
            action=data.get("action", ""),
            command=data.get("command", {}),
            reason=data.get("reason", ""),
            model=data.get("model", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            latency_ms=data.get("latency_ms", 0.0),
        )
