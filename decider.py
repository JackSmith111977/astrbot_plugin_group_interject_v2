# -*- coding: utf-8 -*-
"""One lightweight LLM decision call per admitted window.

The transport is injected so the decider can be tested with a fake callable and
so the plugin can use AstrBot's ``context.llm_generate`` without importing it
here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace

from .model import JUDGE_TIMEOUT_SECONDS, Decision, Situation
from .prompt import (
    build_system_prompt,
    build_user_prompt,
    clean_reply,
    parse_decision,
    render_situation_window,
)

LLMCall = Callable[[str, str, str], Awaitable[str]]
"""``(umo, system_prompt, user_prompt) -> raw completion``."""


class Decider:
    """Build the prompt, call the model once, parse and clean the reply."""

    def __init__(
        self,
        llm: LLMCall,
        *,
        clock: Callable[[], float] = time.time,
        logger=None,
        timeout_seconds: float = JUDGE_TIMEOUT_SECONDS,
    ) -> None:
        self._llm = llm
        self._clock = clock
        from .logger_fallback import fallback_logger
        self.logger = logger or fallback_logger()
        self.timeout_seconds = float(timeout_seconds)

    async def decide(
        self,
        umo: str,
        situation: Situation,
        *,
        group_name: str,
        persona_name: str = "",
        persona_prompt: str = "",
        extra_rules: str = "",
    ) -> Decision:
        """Run one decision; all failures become ``silent``.

        Args:
            umo: Unified message origin for provider resolution.
            situation: Current snapshot.
            group_name: Display name for the prompt.
            persona_name: Selected persona name.
            persona_prompt: Selected persona prompt.
            extra_rules: User-appended style rules.

        Returns:
            A parsed and text-cleaned Decision.  Never raises for model errors.
        """

        system_prompt = build_system_prompt(persona_prompt, extra_rules)
        user_prompt = build_user_prompt(
            group_name=group_name,
            now=situation.now,
            persona_name=persona_name or "你",
            window_text=render_situation_window(situation),
            situation=situation,
        )
        try:
            raw = await asyncio.wait_for(
                self._llm(umo, system_prompt, user_prompt),
                timeout=max(0.01, self.timeout_seconds),
            )
        except asyncio.TimeoutError:
            return Decision.silent("timeout")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("[group-interject-v2] decision llm failed")
            return Decision.silent("llm_error")

        decision = parse_decision(str(raw or ""))
        if not decision.is_reply:
            return decision

        bot_names = (persona_name,) if persona_name else ()
        speaker_names = tuple(
            {
                msg.sender_name
                for msg in situation.window
                if msg.sender_name and not msg.is_bot
            }
        )
        cleaned = clean_reply(
            decision.text,
            max_chars=0,
            bot_names=bot_names,
            speaker_names=speaker_names,
        )
        if not cleaned:
            return Decision.silent("empty_text")
        return replace(decision, text=cleaned)


__all__ = ["Decider", "LLMCall"]
