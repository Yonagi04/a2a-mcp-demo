"""
Centralized network logging utility for MCP-A2A multi-agent system.
Provides structured logging for all network interactions.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class NetworkLogType(Enum):
    A2A = "a2a"
    MCP = "mcp"
    LLM = "llm"


class NetworkLogSubtype(Enum):
    AGENT_CARD = "agent_card"
    SEND_MESSAGE = "send_message"
    TOOL_CALL = "tool_call"
    DELEGATION = "delegation"
    DISCOVERY = "discovery"
    MODEL_CALL = "model_call"
    A2A_HANDLE = "a2a_handle"


class NetworkLogDirection(Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


@dataclass
class NetworkLogEntry:
    timestamp: str = ""
    type: str = ""
    subtype: str = ""
    direction: str = ""
    source: str = ""
    destination: str = ""
    request_id: str = ""
    request_query: Optional[str] = None
    request_data: dict = field(default_factory=dict)
    response_time_seconds: float = 0.0
    response_status: str = "pending"
    response_result: Any = None
    error: Optional[str] = None


@dataclass
class LlmLogEntry:
    """大模型调用日志条目"""
    timestamp: str = ""
    type: str = "llm"
    subtype: str = "model_call"
    source: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    response_status: str = "pending"
    error: Optional[str] = None


@dataclass
class A2AHandleLogEntry:
    """Remote Agent A2A 处理日志条目"""
    timestamp: str = ""
    type: str = "a2a"
    subtype: str = "a2a_handle"
    source: str = ""
    destination: str = ""
    request_id: str = ""
    message: str = ""
    handling_time_seconds: float = 0.0
    response_status: str = "pending"
    error: Optional[str] = None


class NetworkLogger:
    """
    Centralized logger for network interactions in the MCP-A2A system.
    Uses structured JSON logging for machine-parseable output.
    """

    def __init__(self, name: str = "network"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _make_entry(
        self,
        type_: NetworkLogType,
        subtype: NetworkLogSubtype,
        direction: NetworkLogDirection,
        source: str,
        destination: str,
        request_id: str,
        request_query: Optional[str] = None,
        request_data: Optional[dict] = None,
    ) -> NetworkLogEntry:
        return NetworkLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            type=type_.value,
            subtype=subtype.value,
            direction=direction.value,
            source=source,
            destination=destination,
            request_id=request_id,
            request_query=request_query,
            request_data=request_data or {},
        )

    def _log_entry(self, entry: NetworkLogEntry):
        self.logger.info(json.dumps(asdict(entry), ensure_ascii=False, default=str))

    def log_request(
        self,
        type_: NetworkLogType,
        subtype: NetworkLogSubtype,
        direction: NetworkLogDirection,
        source: str,
        destination: str,
        request_id: str,
        request_query: Optional[str] = None,
        request_data: Optional[dict] = None,
    ) -> NetworkLogEntry:
        entry = self._make_entry(
            type_, subtype, direction, source, destination,
            request_id, request_query, request_data
        )
        self._log_entry(entry)
        return entry

    def log_response(
        self,
        entry: NetworkLogEntry,
        response_time_seconds: float,
        response_status: str,
        response_result: Any = None,
        error: Optional[str] = None,
    ):
        entry.response_time_seconds = response_time_seconds
        entry.response_status = response_status
        entry.response_result = response_result
        entry.error = error
        self._log_entry(entry)

    def log_a2a_request(
        self,
        source: str,
        destination: str,
        request_id: str,
        message: str,
        request_data: Optional[dict] = None,
    ) -> NetworkLogEntry:
        return self.log_request(
            type_=NetworkLogType.A2A,
            subtype=NetworkLogSubtype.SEND_MESSAGE,
            direction=NetworkLogDirection.OUTBOUND,
            source=source,
            destination=destination,
            request_id=request_id,
            request_query=message,
            request_data=request_data,
        )

    def log_a2a_response(
        self,
        entry: NetworkLogEntry,
        response_time_seconds: float,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
    ):
        self.log_response(
            entry=entry,
            response_time_seconds=response_time_seconds,
            response_status="success" if success else "failure",
            response_result=result,
            error=error,
        )

    def log_mcp_request(
        self,
        source: str,
        destination: str,
        request_id: str,
        tool_name: str,
        arguments: Optional[dict] = None,
    ) -> NetworkLogEntry:
        return self.log_request(
            type_=NetworkLogType.MCP,
            subtype=NetworkLogSubtype.TOOL_CALL,
            direction=NetworkLogDirection.OUTBOUND,
            source=source,
            destination=destination,
            request_id=request_id,
            request_query=tool_name,
            request_data={"arguments": arguments} if arguments else {},
        )

    def log_mcp_response(
        self,
        entry: NetworkLogEntry,
        response_time_seconds: float,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
    ):
        self.log_response(
            entry=entry,
            response_time_seconds=response_time_seconds,
            response_status="success" if success else "failure",
            response_result=result,
            error=error,
        )

    def _log_llm_entry(self, entry: LlmLogEntry):
        self.logger.info(json.dumps(asdict(entry), ensure_ascii=False, default=str))

    def log_llm_call(
        self,
        source: str,
        model: str,
        latency_seconds: float,
        response_status: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
    ):
        entry = LlmLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            type="llm",
            subtype="model_call",
            source=source,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency_seconds,
            response_status=response_status,
            error=error,
        )
        self._log_llm_entry(entry)

    def _log_a2a_handle_entry(self, entry: A2AHandleLogEntry):
        self.logger.info(json.dumps(asdict(entry), ensure_ascii=False, default=str))

    def log_a2a_handle_request(
        self,
        source: str,
        destination: str,
        request_id: str,
        message: str,
    ) -> A2AHandleLogEntry:
        entry = A2AHandleLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            type="a2a",
            subtype="a2a_handle",
            source=source,
            destination=destination,
            request_id=request_id,
            message=message,
        )
        self._log_a2a_handle_entry(entry)
        return entry

    def log_a2a_handle_response(
        self,
        entry: A2AHandleLogEntry,
        handling_time_seconds: float,
        response_status: str,
        error: Optional[str] = None,
    ):
        entry.handling_time_seconds = handling_time_seconds
        entry.response_status = response_status
        entry.error = error
        self._log_a2a_handle_entry(entry)


# Global singleton instance
network_logger = NetworkLogger()