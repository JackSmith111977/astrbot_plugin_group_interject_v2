# -*- coding: utf-8 -*-
"""Executors for a decision: quick text or an optional safe tool loop.

The executor only sends.  Memory writing is owned by the runtime so that the
"one successful send -> exactly one memory write" invariant stays in one place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .model import (
    TOOL_LOOP_TIMEOUT_SECONDS,
    Decision,
    ExecutionResult,
)
from .prompt import clean_reply

SendCall = Callable[[str, str], Awaitable[bool]]
"""``(umo, text) -> sent``."""

ToolLoopCall = Callable[[Any, str, str], Awaitable[str]]
"""``(event, system_prompt, user_prompt) -> final text``."""


class Executor:
    """Execute one decision, at most once, through quick text or a tool loop."""

    def __init__(
        self,
        *,
        send: SendCall,
        tool_loop: ToolLoopCall | None = None,
        logger=None,
    ) -> None:
        self._send = send
        self._tool_loop = tool_loop
        from .logger_fallback import fallback_logger
        self.logger = logger or fallback_logger()

    async def execute(
        self,
        decision: Decision,
        *,
        umo: str,
        event: Any = None,
        tool_system_prompt: str = "",
        tool_user_prompt: str = "",
        bot_names: tuple[str, ...] = (),
        speaker_names: tuple[str, ...] = (),
    ) -> ExecutionResult:
        """Run one decision.

        Args:
            decision: A reply decision that already passed confidence/tool checks.
            umo: Session origin for sending.
            event: Captured group event for the tool loop; may be None.
            tool_system_prompt: System prompt for the tool loop.
            tool_user_prompt: User prompt for the tool loop.
            bot_names: Persona names to strip from tool output.
            speaker_names: Window speaker names to strip from tool output.

        Returns:
            ExecutionResult.  ``sent`` is true only when the platform accepted
            the message (or when the caller explicitly simulates dry-run).
        """

        used_tools = False
        text = decision.text
        if decision.mode == "agent" and decision.needs_tools:
            if self._tool_loop is None or event is None:
                return ExecutionResult(False, reason="tool_unavailable")
            try:
                raw = await asyncio.wait_for(
                    self._tool_loop(event, tool_system_prompt, tool_user_prompt),
                    timeout=TOOL_LOOP_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                return ExecutionResult(False, reason="tool_timeout")
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("[group-interject-v2] tool loop failed")
                return ExecutionResult(False, reason="tool_error")
            used_tools = True
            text = clean_reply(
                str(raw or ""),
                max_chars=0,
                bot_names=bot_names,
                speaker_names=speaker_names,
            ) or ""
            if not text:
                return ExecutionResult(False, reason="empty_tool_text", used_tools=True)

        if not text.strip():
            return ExecutionResult(False, reason="empty_text", used_tools=used_tools)
        try:
            sent = bool(await self._send(umo, text))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("[group-interject-v2] send failed")
            return ExecutionResult(False, reason="send_exception", used_tools=used_tools)
        if not sent:
            return ExecutionResult(False, reason="send_false", used_tools=used_tools)
        return ExecutionResult(True, text=text, used_tools=used_tools)


__all__ = ["Executor", "SendCall", "ToolLoopCall"]
