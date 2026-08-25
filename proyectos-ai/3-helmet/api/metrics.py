import json
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _MetricsState:
    total_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    request_latencies: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class MetricsTracker:
    """Thread-safe metrics tracker with Prometheus export."""

    def __init__(self) -> None:
        self._state = _MetricsState()
        self._lock = threading.Lock()

    def record_request(self, latency_ms: float, tokens: int = 0) -> None:
        with self._lock:
            self._state.total_requests += 1
            self._state.total_latency_ms += latency_ms
            self._state.total_tokens += tokens
            self._state.request_latencies.append(latency_ms)

    @property
    def total_requests(self) -> int:
        with self._lock:
            return self._state.total_requests

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if self._state.total_requests == 0:
                return 0.0
            return self._state.total_latency_ms / self._state.total_requests

    @property
    def avg_tokens(self) -> int:
        with self._lock:
            if self._state.total_requests == 0:
                return 0
            return int(self._state.total_tokens / self._state.total_requests)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_requests": self.total_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_tokens": self.avg_tokens,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_prometheus(self) -> str:
        lines: list[str] = []
        uptime = time.time() - self._state.start_time
        with self._lock:
            lines.append("# HELP helmet_requests_total Total number of requests")
            lines.append("# TYPE helmet_requests_total counter")
            lines.append(f"helmet_requests_total {self._state.total_requests}")
            lines.append("")
            lines.append("# HELP helmet_latency_ms Average latency in milliseconds")
            lines.append("# TYPE helmet_latency_ms gauge")
            avg = (
                self._state.total_latency_ms / self._state.total_requests
                if self._state.total_requests > 0
                else 0.0
            )
            lines.append(f"helmet_latency_ms {avg:.2f}")
            lines.append("")
            lines.append("# HELP helmet_tokens_avg Average tokens per request")
            lines.append("# TYPE helmet_tokens_avg gauge")
            avg_t = (
                int(self._state.total_tokens / self._state.total_requests)
                if self._state.total_requests > 0
                else 0
            )
            lines.append(f"helmet_tokens_avg {avg_t}")
            lines.append("")
            lines.append("# HELP helmet_uptime_seconds Time since server start")
            lines.append("# TYPE helmet_uptime_seconds gauge")
            lines.append(f"helmet_uptime_seconds {uptime:.1f}")
        return "\n".join(lines)


metrics = MetricsTracker()
