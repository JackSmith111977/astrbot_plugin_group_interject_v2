# -*- coding: utf-8 -*-
"""Per-group runtime: timeline, episodes, debounce and the v2 pipeline.

The runtime is the only place that mutates group state.  It enforces:

- one in-flight decision per group;
- one decision per judged window;
- one send per decision;
- one memory write per successful send;
- a post-send re-arm requirement (new messages + cooldown + episode turns).

No AstrBot imports; all transport capabilities are injected.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import policy
from .context import render_directed, render_summaries
from .model import (
    AGENT_BUSY_LEASE_SECONDS,
    IMAGE_CAPTION_SHORT_WAIT_SECONDS,
    IMAGE_CAPTION_WAIT_SECONDS,
    LAST_EVENT_MAX_AGE_SECONDS,
    SUMMARY_BATCH_MESSAGES,
    SUMMARY_MAX_RECORDS,
    SUMMARY_RETRY_SECONDS,
    SUMMARY_TRIGGER_CHARS,
    SUMMARY_TRIGGER_MESSAGES,
    SUPPRESSED_SECONDS,
    Admission,
    Config,
    Decision,
    Episode,
    ExecutionResult,
    Msg,
    Situation,
    State,
    SummaryRecord,
    fresh_non_bot,
    group_filter_reason,
    non_bot_count,
    slow_rest_seconds,
    trim_messages,
)
from .prompt import (
    build_tool_system_prompt,
    build_tool_user_prompt,
    render_situation_window,
)
from .text import build_bare_at_context, in_quiet_hours, message_body

LOG_PREFIX = "[group-interject-v2]"

SKIP_LOG_INTERVAL_SECONDS = 30
SKIP_HISTORY_LIMIT = 200
DECISION_HISTORY_LIMIT = 20
RECENT_INTERJECTION_LIMIT = 10


class DeciderLike(Protocol):
    async def decide(
        self,
        umo: str,
        situation: Situation,
        *,
        group_name: str,
        persona_name: str = "",
        persona_prompt: str = "",
        extra_rules: str = "",
    ) -> Decision: ...


class ExecutorLike(Protocol):
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
    ) -> ExecutionResult: ...


class SummarizerLike(Protocol):
    async def summarize(
        self,
        umo: str,
        group_name: str,
        messages: list[Msg],
    ) -> str | None: ...


class CaptionerLike(Protocol):
    """Background image-caption writer used by the runtime."""

    def submit(self, umo: str, msg: Msg, group_id: str = "") -> None: ...

    async def wait_ready(
        self,
        umo: str,
        refs: Sequence[str],
        timeout: float = IMAGE_CAPTION_WAIT_SECONDS,
    ) -> None: ...

    async def close(self) -> None: ...


PersonaCall = Callable[[str], Awaitable[tuple[str, str]]]
MemoryCall = Callable[[str, str, str, str, str], Awaitable[None]]
ToolsAvailable = Callable[[], bool]


@dataclass
class GroupState:
    """In-memory state for one group; nothing is persisted."""

    umo: str
    group_id: str = ""
    group_name: str = ""
    window: deque[Msg] = field(default_factory=deque)
    episode: Episode = field(default_factory=Episode)
    state: State = State.QUIET
    pending: asyncio.Task | None = None
    epoch: int = 0
    deciding: bool = False
    reschedule: bool = False
    last_judge_at: float = 0.0
    last_judged_ts: float = 0.0
    last_bot_activity_ts: float = 0.0
    active_agents: int = 0
    agent_busy_until: float = 0.0
    suppressed_until: float = 0.0
    last_event: Any = None
    last_event_ts: float = 0.0
    self_id: str = ""
    self_name: str = ""
    first_seen_ts: float = 0.0
    last_skip_log_ts: float = 0.0
    last_skip_log_reason: str = ""
    summary_task: asyncio.Task | None = None
    recent_interjections: deque[tuple[float, str]] = field(default_factory=deque)
    decisions: deque[dict[str, object]] = field(default_factory=deque)
    skip_events: deque[tuple[float, str]] = field(default_factory=deque)


class Runtime:
    """Per-group episode state machine and orchestration."""

    def __init__(
        self,
        config: Config,
        *,
        decider: DeciderLike,
        executor: ExecutorLike,
        get_persona: PersonaCall,
        clock: Callable[[], float] = time.time,
        logger=None,
        tools_available: ToolsAvailable | None = None,
        memory_writer: MemoryCall | None = None,
        summarizer: SummarizerLike | None = None,
        captioner: CaptionerLike | None = None,
    ) -> None:
        self.config = config
        self._decider = decider
        self._executor = executor
        self._get_persona = get_persona
        self._clock = clock
        from .logger_fallback import fallback_logger
        self.logger = logger or fallback_logger()
        self._tools_available = tools_available or (lambda: False)
        self._memory_writer = memory_writer
        self._summarizer = summarizer
        self._captioner = captioner
        self._states: dict[str, GroupState] = {}
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    async def terminate(self) -> None:
        """Cancel all pending tasks and clear state."""

        self._closed = True
        tasks: list[asyncio.Task] = []
        count = len(self._states)
        for state in self._states.values():
            state.epoch += 1
            if state.pending is not None and not state.pending.done():
                state.pending.cancel()
                tasks.append(state.pending)
            state.pending = None
            if state.summary_task is not None and not state.summary_task.done():
                state.summary_task.cancel()
                tasks.append(state.summary_task)
            state.summary_task = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        captioner = self._captioner
        self._captioner = None
        if captioner is not None:
            try:
                await captioner.close()
            except Exception:
                self.logger.exception("%s captioner close failed", LOG_PREFIX)
        self._states.clear()
        self.logger.info("%s terminated (%d group states cleared)", LOG_PREFIX, count)

    # -- ingestion ---------------------------------------------------------

    def ingest(
        self,
        *,
        umo: str,
        group_id: str,
        group_name: str = "",
        sender_id: str = "",
        sender_name: str = "",
        text: str = "",
        directed: bool = False,
        is_bot: bool = False,
        addressed_to_bot: bool = False,
        decision_id: str = "",
        self_id: str = "",
        self_name: str = "",
        event: Any = None,
        schedule_decision: bool = True,
        image_refs: Sequence[str] = (),
    ) -> None:
        """Consume one group message.

        This method only mutates memory and schedules a debounce task; it never
        awaits network I/O, so the AstrBot handler can call it safely.
        Set ``schedule_decision=False`` to record context without treating this
        event as a fresh interjection trigger (used for third-party pokes).
        """

        cfg = self.config
        if self._closed or not cfg.enable:
            return
        umo = str(umo or "").strip()
        if not umo:
            return
        if group_filter_reason(cfg, group_id, umo):
            return
        state = self._state(umo)
        state.group_id = str(group_id or state.group_id)
        if group_name:
            state.group_name = str(group_name)
        if self_id:
            state.self_id = str(self_id)
        if self_name:
            state.self_name = str(self_name)
        if event is not None:
            state.last_event = event
            state.last_event_ts = self._clock()

        now = self._clock()
        text = str(text or "").strip()
        sender_id = str(sender_id or "")
        if is_bot:
            self._append(state, now, sender_id, sender_name, text, is_bot=True)
            state.last_bot_activity_ts = now
            state.episode.touch(now)
            state.episode.human_since_last_bot = 0
            self._trim(state, now)
            self._maybe_schedule_summary(state, now)
            return
        if sender_id and sender_id in cfg.ignore_senders:
            self._record_skip(state, "ignored_sender", now, window=len(state.window))
            return

        msg = self._append(
            state,
            now,
            sender_id,
            sender_name,
            text,
            directed=directed,
            image_refs=image_refs,
        )
        self._submit_caption(umo, msg, state.group_id)
        self._trim(state, now)
        self._advance_episode(state, now)
        state.episode.human_since_last_bot += 1
        self._maybe_schedule_summary(state, now)
        if state.first_seen_ts <= 0:
            state.first_seen_ts = now

        if policy.should_suppress(
            text,
            addressed_to_bot=bool(directed or addressed_to_bot),
            last_bot_activity_ts=state.last_bot_activity_ts,
            now=now,
        ):
            self._suppress(state, now, text, decision_id)
            return

        if directed or addressed_to_bot:
            self._cancel_pending(state)
            if (
                state.episode.turn_count > 0
                and state.episode.last_sent_at > 0
                and now - state.episode.last_sent_at
                <= self.config.preset.episode_window_seconds
            ):
                state.episode.mark_response(now)
                state.state = State.ENGAGED
            self.logger.debug(
                "%s decision=%s stage=route result=directed group=%s",
                LOG_PREFIX,
                decision_id or "-",
                state.group_id or umo,
            )
            return

        if state.suppressed_until > now:
            return
        if state.state is State.SUPPRESSED:
            state.state = State.OBSERVING

        if (
            state.episode.turn_count > 0
            and not state.episode.responded
            and state.episode.last_sent_at > 0
            and state.episode.followup_active(
                now, self.config.preset.episode_window_seconds
            )
        ):
            bot_names = tuple(
                name for name in (state.self_name, *self.config.wake_names) if name
            )
            if policy.is_continuation(
                text,
                bot_text=state.episode.last_sent_text,
                bot_names=bot_names,
            ):
                state.episode.mark_response(now)
                state.state = State.ENGAGED
                self.logger.info(
                    "%s decision=%s stage=feedback result=continuation group=%s "
                    "turn=%d",
                    LOG_PREFIX,
                    decision_id or "-",
                    state.group_id or umo,
                    state.episode.turn_count,
                )

        if state.state in (State.QUIET, State.ENGAGED):
            state.state = State.OBSERVING
        if schedule_decision:
            self._schedule(state)

    # -- hooks -------------------------------------------------------------

    def note_agent_begin(self, umo: str, group_id: str = "") -> None:
        if self._closed or not self.config.enable:
            return
        if group_filter_reason(self.config, group_id, umo):
            return
        state = self._state(str(umo or "").strip())
        state.group_id = str(group_id or state.group_id)
        state.active_agents += 1
        state.agent_busy_until = self._clock() + AGENT_BUSY_LEASE_SECONDS
        state.last_bot_activity_ts = self._clock()

    def note_agent_done(self, umo: str, group_id: str = "") -> None:
        if self._closed or not self.config.enable:
            return
        if group_filter_reason(self.config, group_id, umo):
            return
        state = self._state(str(umo or "").strip())
        state.active_agents = max(0, state.active_agents - 1)
        if state.active_agents == 0:
            state.agent_busy_until = 0.0
        state.last_bot_activity_ts = self._clock()

    def record_bot_message(self, umo: str, group_id: str, text: str) -> None:
        """Record a bot message emitted by the main Agent (after_message_sent)."""

        if self._closed or not self.config.enable:
            return
        if group_filter_reason(self.config, group_id, umo):
            return
        text = str(text or "").strip()
        if not text:
            return
        state = self._state(str(umo or "").strip())
        now = self._clock()
        preset = self.config.preset
        state.last_bot_activity_ts = now
        state.episode.touch(now)
        state.episode.human_since_last_bot = 0
        if state.episode.turn_count >= preset.episode_max_turns:
            state.episode.next_extra_at = max(
                state.episode.next_extra_at,
                now + max(1, int(preset.episode_rest_seconds)),
            )
        self._append(state, now, "", "", text, is_bot=True)
        self._trim(state, now)
        if (
            state.episode.awaiting_response
            and state.episode.last_sent_at > 0
            and now - state.episode.last_sent_at
            > self.config.preset.episode_window_seconds
        ):
            state.episode.awaiting_response = False
            if state.state is State.ENGAGED:
                state.state = State.OBSERVING

    # -- public read-only helpers -----------------------------------------

    def last_bot_context(self, umo: str, max_seconds: int = 300) -> str:
        state = self._states.get(str(umo or "").strip())
        if state is None or not state.episode.last_sent_text:
            return ""
        if self._clock() - state.episode.last_sent_at > max(1, int(max_seconds)):
            return ""
        return state.episode.last_sent_text

    def bare_at_context(self, umo: str, max_messages: int = 6) -> str:
        state = self._states.get(str(umo or "").strip())
        if state is None:
            return ""
        now = self._clock()
        window = trim_messages(
            list(state.window),
            now,
            self.config.preset.window_minutes,
        )
        recent = build_bare_at_context(window, max_messages=max_messages)
        cold = render_summaries(list(state.episode.cold), max_chars=600)
        if cold and recent:
            return f"{cold}\n{recent}"
        return cold or recent

    def recent_interjections(
        self,
        umo: str,
        *,
        limit: int = 3,
        max_age_seconds: float = 1800.0,
    ) -> list[tuple[float, str]]:
        state = self._states.get(str(umo or "").strip())
        if state is None:
            return []
        now = self._clock()
        items = [
            (ts, text)
            for ts, text in state.recent_interjections
            if now - float(ts) <= max(1.0, float(max_age_seconds))
        ]
        return items[-max(1, int(limit)) :]

    def stats_snapshot(self) -> list[dict[str, object]]:
        """Read-only per-group telemetry for the status view and logs."""

        now = self._clock()
        items: list[dict[str, object]] = []
        for umo, state in self._states.items():
            window = trim_messages(
                list(state.window),
                now,
                self.config.preset.window_minutes,
            )
            last_sent = state.episode.last_sent_at
            cooldown_left = 0.0
            if last_sent > 0:
                cooldown_left = max(
                    0.0,
                    self.config.preset.cooldown_seconds - (now - last_sent),
                )
                if state.episode.responded:
                    cooldown_left = max(
                        0.0,
                        self.config.preset.followup_cooldown_seconds
                        - (now - last_sent),
                    )
            last_skip = None
            if state.skip_events:
                ts, reason = state.skip_events[-1]
                last_skip = {"reason": reason, "ago_seconds": round(now - ts, 1)}
            items.append(
                {
                    "umo": umo,
                    "group_id": state.group_id,
                    "group_name": state.group_name,
                    "state": state.state.value,
                    "window_messages": len(window),
                    "window_capacity": max(1, self.config.preset.window_size),
                    "episode_active": state.episode.active,
                    "episode_turns": state.episode.turn_count,
                    "episode_burst_limit": self.config.preset.episode_max_turns,
                    "episode_extra_turns": state.episode.extra_turns,
                    "episode_remaining_turns": state.episode.remaining_turns(
                        self.config.preset.episode_max_turns
                    ),
                    "episode_human_since_last_bot": state.episode.human_since_last_bot,
                    "episode_reentry_messages_remaining": max(
                        0,
                        self.config.preset.episode_reentry_messages
                        - state.episode.human_since_last_bot,
                    ),
                    "episode_next_extra_in_seconds": round(
                        max(
                            0.0,
                            (
                                state.episode.next_extra_at
                                or (
                                    state.episode.last_sent_at
                                    + self.config.preset.episode_rest_seconds
                                    if state.episode.last_sent_at > 0
                                    else now
                                    + self.config.preset.episode_rest_seconds
                                )
                            )
                            - now,
                        ),
                        1,
                    ),
                    "episode_slow": (
                        state.episode.turn_count
                        >= self.config.preset.episode_max_turns
                        and (
                            state.episode.next_extra_at > now
                            or state.episode.human_since_last_bot
                            < self.config.preset.episode_reentry_messages
                        )
                    ),
                    "episode_responded": state.episode.responded,
                    "episode_age_seconds": (
                        round(now - state.episode.started_at)
                        if state.episode.started_at > 0
                        else 0
                    ),
                    "followup_active": state.episode.followup_active(
                        now, self.config.preset.episode_window_seconds
                    ),
                    "last_judge_ago_seconds": (
                        round(now - state.last_judge_at)
                        if state.last_judge_at > 0
                        else None
                    ),
                    "last_sent_ago_seconds": (
                        round(now - last_sent) if last_sent > 0 else None
                    ),
                    "cooldown_remaining_seconds": round(cooldown_left, 1),
                    "suppressed_remaining_seconds": round(
                        max(0.0, state.suppressed_until - now), 1
                    ),
                    "agent_busy": self._agent_busy(state, now),
                    "deciding": state.deciding,
                    "pending": bool(
                        state.pending is not None and not state.pending.done()
                    ),
                    "last_skip": last_skip,
                    "skip_reasons": self._skip_reason_counts(state, now),
                    "recent_decisions": self._recent_decisions(state, now),
                    "cold_summaries": len(state.episode.cold),
                    "summary_chars": sum(
                        len(record.text) for record in state.episode.cold
                    ),
                    "pending_spill": len(state.episode.pending_spill),
                    "summary_calls": state.episode.summary_calls,
                    "last_summary_ago_seconds": (
                        round(now - state.episode.last_summary_at)
                        if state.episode.last_summary_at > 0
                        else None
                    ),
                    "last_summary_status": state.episode.last_summary_status,
                    "summary_inflight": bool(
                        state.summary_task is not None
                        and not state.summary_task.done()
                    ),
                    "recent_summaries": [
                        {
                            "ago_seconds": round(now - record.created_at),
                            "source_count": record.source_count,
                            "chars": len(record.text),
                        }
                        for record in list(state.episode.cold)[-3:]
                    ],
                }
            )
        items.sort(key=lambda item: str(item.get("group_name") or item.get("umo")))
        return items

    async def probe(self, umo: str) -> dict[str, object]:
        """Run one read-only judgment for a group window (no send, no mutation)."""

        state = self._states.get(str(umo or "").strip())
        if state is None:
            return {
                "ok": False,
                "error": "该群还没有消息窗口（插件尚未收到该群消息，或已被白名单过滤）",
            }
        now = self._clock()
        situation = self._situation(state, now)
        admission = policy.admit(
            self.config,
            situation,
            group_id=state.group_id,
            umo=state.umo,
            phase="judge",
        )
        result: dict[str, object] = {
            "ok": True,
            "umo": state.umo,
            "group_id": state.group_id,
            "group_name": state.group_name,
            "state": state.state.value,
            "admission": {
                "ok": admission.ok,
                "reason": admission.reason,
                "remaining_seconds": admission.remaining_seconds,
            },
            "window_messages": len(situation.window),
            "episode_turns": situation.episode_turns,
            "episode_burst_limit": self.config.preset.episode_max_turns,
            "episode_extra_turns": situation.episode_extra_turns,
            "episode_human_since_last_bot": situation.episode_human_since_last_bot,
            "episode_reentry_messages_remaining": max(
                0,
                self.config.preset.episode_reentry_messages
                - situation.episode_human_since_last_bot,
            ),
            "episode_next_extra_in_seconds": round(
                max(
                    0.0,
                    (
                        situation.episode_next_extra_at
                        or (
                            situation.episode_last_sent_at
                            + self.config.preset.episode_rest_seconds
                            if situation.episode_last_sent_at > 0
                            else situation.now
                            + self.config.preset.episode_rest_seconds
                        )
                    )
                    - situation.now,
                ),
                1,
            ),
            "fresh_messages": len(situation.fresh_messages),
            "cold_summaries": len(situation.summaries),
            "summary_chars": sum(len(item.text) for item in situation.summaries),
        }
        if not admission.ok:
            return result
        persona_name, persona_prompt = await self._safe_persona(state.umo)
        started = self._clock()
        decision = await self._decider.decide(
            state.umo,
            situation,
            group_name=state.group_name or state.group_id or state.umo,
            persona_name=persona_name,
            persona_prompt=persona_prompt,
            extra_rules=self.config.extra_rules,
        )
        decision = policy.finalize_decision(
            decision,
            self.config,
            tools_available=bool(self._tools_available()),
        )
        result.update(
            {
                "elapsed": round(self._clock() - started, 2),
                "persona": persona_name,
                "decision": {
                    "action": decision.action,
                    "mode": decision.mode,
                    "text": decision.text,
                    "confidence": decision.confidence,
                    "continuity": decision.continuity,
                    "needs_tools": decision.needs_tools,
                    "reason": decision.reason,
                },
                "would_send": decision.is_reply,
            }
        )
        return result

    # -- internal: episode / suppression ----------------------------------

    def _state(self, umo: str) -> GroupState:
        state = self._states.get(umo)
        if state is None:
            state = GroupState(
                umo=umo,
                window=deque(maxlen=max(1, self.config.preset.window_size)),
                first_seen_ts=self._clock(),
            )
            self._states[umo] = state
        return state

    def _advance_episode(self, state: GroupState, now: float) -> None:
        preset = self.config.preset
        if not state.episode.active:
            state.episode.start(now)
            if state.state is State.QUIET:
                state.state = State.OBSERVING
            return
        if now - state.episode.last_activity_at > preset.episode_gap_seconds:
            old_turns = state.episode.turn_count
            self._cancel_summary(state)
            state.episode.close(now)
            state.episode.start(now)
            state.state = State.OBSERVING
            self.logger.info(
                "%s stage=episode state=close group=%s turns=%d",
                LOG_PREFIX,
                state.group_id or state.umo,
                old_turns,
            )
            return
        state.episode.touch(now)
        if (
            state.episode.awaiting_response
            and state.episode.last_sent_at > 0
            and now - state.episode.last_sent_at > preset.episode_window_seconds
        ):
            state.episode.awaiting_response = False
            if state.state is State.ENGAGED:
                state.state = State.OBSERVING

    def _suppress(
        self,
        state: GroupState,
        now: float,
        text: str,
        decision_id: str,
    ) -> None:
        state.suppressed_until = max(
            state.suppressed_until,
            now + SUPPRESSED_SECONDS,
        )
        state.state = State.SUPPRESSED
        state.episode.awaiting_response = False
        self._cancel_pending(state)
        self.logger.info(
            "%s decision=%s stage=feedback result=suppressed group=%s "
            "until=%s preview=%r",
            LOG_PREFIX,
            decision_id or "-",
            state.group_id or state.umo,
            int(state.suppressed_until),
            str(text or "")[:60],
        )

    # -- internal: scheduling / pipeline ----------------------------------

    def _schedule(self, state: GroupState) -> None:
        if self._closed:
            return
        if state.suppressed_until > self._clock():
            return
        if state.deciding:
            state.reschedule = True
            return
        self._cancel_pending(state)
        state.epoch += 1
        state.pending = asyncio.create_task(
            self._debounce(state, state.epoch),
            name=f"group-interject-v2:{state.umo}",
        )

    async def _debounce(self, state: GroupState, epoch: int) -> None:
        try:
            await asyncio.sleep(max(0.0, self.config.preset.debounce_seconds))
            if self._closed or state.epoch != epoch:
                return
            await self._run_decision(state, epoch)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception(
                "%s debounce failed group=%s",
                LOG_PREFIX,
                state.umo,
            )
        finally:
            if state.pending is asyncio.current_task():
                state.pending = None
            if (
                state.reschedule
                and not self._closed
                and state.epoch == epoch
            ):
                state.reschedule = False
                self._schedule(state)

    async def _run_decision(self, state: GroupState, epoch: int) -> None:
        if self._closed or state.deciding:
            return
        state.deciding = True
        decision_id = uuid.uuid4().hex[:8]
        started = self._clock()
        try:
            now = self._clock()
            situation = self._situation(state, now)
            admission = policy.admit(
                self.config,
                situation,
                group_id=state.group_id,
                umo=state.umo,
                phase="judge",
            )
            if not admission.ok:
                self._record_skip(
                    state,
                    admission.reason,
                    now,
                    window=len(situation.window),
                    extra=(
                        f"remaining={admission.remaining_seconds} "
                        f"msgs={admission.remaining_messages}"
                    ),
                )
                state.state = (
                    State.SUPPRESSED
                    if state.suppressed_until > now
                    else State.OBSERVING
                )
                return

            state.state = State.CANDIDATE
            captioner = self._captioner
            if captioner is not None:
                refs = tuple(
                    ref
                    for msg in situation.window
                    for ref in msg.image_refs
                    if ref
                )
                if refs:
                    recent_has_unresolved_image = any(
                        msg.image_refs and not msg.image_caption
                        for msg in situation.window
                    )
                    wait_seconds = (
                        IMAGE_CAPTION_WAIT_SECONDS
                        if recent_has_unresolved_image
                        else IMAGE_CAPTION_SHORT_WAIT_SECONDS
                    )
                    try:
                        await captioner.wait_ready(
                            state.umo,
                            refs,
                            timeout=wait_seconds,
                        )
                    except Exception:
                        self.logger.warning(
                            "%s caption wait failed group=%s",
                            LOG_PREFIX,
                            state.group_id or state.umo,
                            exc_info=True,
                        )
                    if self._closed or state.epoch != epoch:
                        return
                    now = self._clock()
                    situation = self._situation(state, now)
            persona_name, persona_prompt = await self._safe_persona(state.umo)
            if persona_name:
                state.self_name = persona_name
            decision = await self._decider.decide(
                state.umo,
                situation,
                group_name=state.group_name or state.group_id or state.umo,
                persona_name=persona_name,
                persona_prompt=persona_prompt,
                extra_rules=self.config.extra_rules,
            )
            decision = policy.finalize_decision(
                decision,
                self.config,
                tools_available=bool(self._tools_available()),
            )
            elapsed = self._clock() - started
            if not decision.is_reply:
                state.last_judge_at = self._clock()
                state.last_judged_ts = state.last_judge_at
                state.state = State.OBSERVING
                self._record_decision(
                    state,
                    result="silent",
                    reason=decision.reason,
                    confidence=decision.confidence,
                    elapsed=elapsed,
                )
                return

            now = self._clock()
            pre_send = policy.admit(
                self.config,
                self._situation(state, now),
                group_id=state.group_id,
                umo=state.umo,
                phase="send",
            )
            if not pre_send.ok:
                state.last_judge_at = now
                state.last_judged_ts = now
                state.state = State.OBSERVING
                self._record_decision(
                    state,
                    result="aborted",
                    reason=pre_send.reason,
                    confidence=decision.confidence,
                    elapsed=elapsed,
                    text=decision.text,
                )
                return

            if self.config.dry_run:
                execution = ExecutionResult(
                    sent=True,
                    text=decision.text,
                    reason="dry_run",
                    used_tools=decision.mode == "agent" and decision.needs_tools,
                    dry_run=True,
                )
            else:
                bot_names = (persona_name,) if persona_name else ()
                speaker_names = tuple(
                    {
                        msg.sender_name
                        for msg in situation.window
                        if msg.sender_name and not msg.is_bot
                    }
                )
                event = self._usable_event(state, now)
                execution = await self._executor.execute(
                    decision,
                    umo=state.umo,
                    event=event,
                    tool_system_prompt=build_tool_system_prompt(
                        persona_prompt,
                        self.config.extra_rules,
                    ),
                    tool_user_prompt=build_tool_user_prompt(
                        group_name=state.group_name or state.group_id or state.umo,
                        now=now,
                        persona_name=persona_name or "你",
                        window_text=render_situation_window(situation),
                        reason=decision.reason,
                    ),
                    bot_names=bot_names,
                    speaker_names=speaker_names,
                )

            if not execution.sent:
                state.last_judge_at = now
                state.last_judged_ts = now
                state.state = State.OBSERVING
                self._record_decision(
                    state,
                    result="error",
                    reason=execution.reason,
                    confidence=decision.confidence,
                    elapsed=elapsed,
                    text=decision.text,
                )
                return

            await self._after_sent(
                state,
                decision,
                execution,
                persona_name=persona_name,
                elapsed=elapsed,
            )
            state.last_judge_at = self._clock()
            state.last_judged_ts = state.last_judge_at
            state.state = State.ENGAGED
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception(
                "%s pipeline failed group=%s",
                LOG_PREFIX,
                state.umo,
            )
            now = self._clock()
            state.last_judge_at = now
            state.last_judged_ts = now
            state.state = State.OBSERVING
            self._record_decision(
                state,
                result="error",
                reason="pipeline_exception",
                elapsed=self._clock() - started,
            )
        finally:
            state.deciding = False

    async def _after_sent(
        self,
        state: GroupState,
        decision: Decision,
        execution: ExecutionResult,
        *,
        persona_name: str,
        elapsed: float,
    ) -> None:
        now = self._clock()
        preset = self.config.preset
        prior_turns = state.episode.turn_count
        state.episode.on_send(now, execution.text)
        if state.episode.turn_count >= preset.episode_max_turns:
            if prior_turns >= preset.episode_max_turns:
                state.episode.extra_turns += 1
            state.episode.next_extra_at = now + slow_rest_seconds(
                preset.episode_rest_seconds,
                state.episode.extra_turns,
            )
        state.last_bot_activity_ts = now
        state.last_judged_ts = now
        state.last_judge_at = now
        if not execution.dry_run:
            state.recent_interjections.append((now, execution.text))
            while len(state.recent_interjections) > RECENT_INTERJECTION_LIMIT:
                state.recent_interjections.popleft()
        state.state = State.ENGAGED
        self._record_decision(
            state,
            result="sent",
            reason=execution.reason,
            confidence=decision.confidence,
            elapsed=elapsed,
            text=execution.text,
            sent=True,
            used_tools=execution.used_tools,
            dry_run=execution.dry_run,
        )
        if execution.dry_run or self._memory_writer is None:
            return
        try:
            await asyncio.wait_for(
                self._memory_writer(
                    state.umo,
                    state.group_id,
                    execution.text,
                    state.self_id,
                    state.self_name or persona_name,
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "%s memory bridge timeout group=%s",
                LOG_PREFIX,
                state.group_id or state.umo,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.warning(
                "%s memory bridge failed group=%s",
                LOG_PREFIX,
                state.group_id or state.umo,
                exc_info=True,
            )

    def _usable_event(self, state: GroupState, now: float) -> Any:
        if state.last_event is None:
            return None
        if now - state.last_event_ts > LAST_EVENT_MAX_AGE_SECONDS:
            return None
        return state.last_event

    def _situation(self, state: GroupState, now: float) -> Situation:
        preset = self.config.preset
        window = trim_messages(list(state.window), now, preset.window_minutes)
        last = window[-1] if window else None
        last_sent = state.episode.last_sent_at
        fresh = (
            fresh_non_bot(window, last_sent)
            if last_sent > 0
            else [msg for msg in window if not msg.is_bot]
        )
        episode_active = (
            state.episode.active
            and state.episode.started_at > 0
            and now - state.episode.last_activity_at <= preset.episode_gap_seconds
        )
        return Situation(
            now=now,
            state=state.state,
            window=tuple(window),
            fresh_messages=tuple(fresh),
            last_message=last,
            last_bot_activity_ts=state.last_bot_activity_ts,
            agent_busy=self._agent_busy(state, now),
            last_judge_at=state.last_judge_at,
            last_judged_ts=state.last_judged_ts,
            suppressed_until=state.suppressed_until,
            quiet_hours=in_quiet_hours(self.config.quiet_hours, now),
            episode_active=episode_active,
            followup_active=state.episode.followup_active(
                now,
                preset.episode_window_seconds,
            ),
            episode_turns=state.episode.turn_count,
            episode_responded=state.episode.responded,
            episode_last_sent_at=last_sent,
            episode_last_sent_text=state.episode.last_sent_text,
            episode_extra_turns=state.episode.extra_turns,
            episode_next_extra_at=state.episode.next_extra_at,
            episode_human_since_last_bot=state.episode.human_since_last_bot,
            summaries=tuple(state.episode.cold),
        )

    def _agent_busy(self, state: GroupState, now: float) -> bool:
        if state.active_agents <= 0:
            return False
        if now >= state.agent_busy_until:
            self.logger.warning(
                "%s agent_busy lease expired group=%s stale_count=%d",
                LOG_PREFIX,
                state.group_id or state.umo,
                state.active_agents,
            )
            state.active_agents = 0
            state.agent_busy_until = 0.0
            return False
        return True

    # -- internal: window / records ---------------------------------------

    def _append(
        self,
        state: GroupState,
        now: float,
        sender_id: str,
        sender_name: str,
        text: str,
        *,
        is_bot: bool = False,
        directed: bool = False,
        image_refs: Sequence[str] = (),
    ) -> Msg:
        # Never rely on deque maxlen silently dropping old messages: evicted
        # messages must go through cold-summary spill first.
        if state.window.maxlen is not None and len(state.window) >= state.window.maxlen:
            self._spill(state, [state.window.popleft()])
        msg = Msg(
            ts=now,
            sender_id=str(sender_id or ""),
            sender_name=str(sender_name or ""),
            text=str(text or ""),
            is_bot=is_bot,
            directed=directed,
            image_refs=tuple(image_refs),
        )
        state.window.append(msg)
        return msg

    def _submit_caption(self, umo: str, msg: Msg, group_id: str) -> None:
        captioner = self._captioner
        if captioner is None or not msg.image_refs:
            return
        try:
            captioner.submit(umo, msg, group_id)
        except Exception:
            self.logger.exception(
                "%s caption submit failed group=%s",
                LOG_PREFIX,
                group_id or umo,
            )

    def _trim(self, state: GroupState, now: float) -> None:
        window_minutes = self.config.preset.window_minutes
        if window_minutes <= 0:
            return
        cutoff = now - window_minutes * 60
        kept: list[Msg] = []
        spilled: list[Msg] = []
        for msg in state.window:
            if msg.ts < cutoff:
                spilled.append(msg)
            else:
                kept.append(msg)
        if spilled:
            self._spill(state, spilled)
        if len(kept) != len(state.window):
            state.window = deque(kept, maxlen=max(1, self.config.preset.window_size))

    def _spill(self, state: GroupState, messages: Sequence[Msg]) -> None:
        if not messages:
            return
        state.episode.pending_spill.extend(messages)
        while len(state.episode.pending_spill) > 200:
            state.episode.pending_spill.popleft()

    def _maybe_schedule_summary(self, state: GroupState, now: float) -> None:
        """Schedule one cold-summary task if the spill batch is large enough."""

        if self._summarizer is None or self._closed or not self.config.enable:
            return
        episode = state.episode
        if not episode.active or not episode.pending_spill:
            return
        if state.summary_task is not None:
            if not state.summary_task.done():
                return
            state.summary_task = None
        chars = sum(len(message_body(msg)) for msg in episode.pending_spill)
        if (
            len(episode.pending_spill) < SUMMARY_TRIGGER_MESSAGES
            and chars < SUMMARY_TRIGGER_CHARS
        ):
            return
        if now - episode.last_summary_attempt_at < SUMMARY_RETRY_SECONDS:
            return
        state.summary_task = asyncio.create_task(
            self._summarize_spill(state, episode, now),
            name=f"group-interject-v2:summary:{state.umo}",
        )

    async def _summarize_spill(
        self,
        state: GroupState,
        episode: Episode,
        now: float,
    ) -> None:
        batch = list(episode.pending_spill)[:SUMMARY_BATCH_MESSAGES]
        if not batch:
            return
        episode.last_summary_status = "running"
        try:
            summary = await self._summarizer.summarize(  # type: ignore[union-attr]
                state.umo,
                state.group_name or state.group_id or state.umo,
                batch,
            )
            if self._closed or state.episode is not episode:
                return
            if not summary:
                episode.last_summary_status = "empty"
                episode.last_summary_attempt_at = self._clock()
                return
            record = SummaryRecord(
                text=str(summary),
                created_at=self._clock(),
                covered_from_ts=float(batch[0].ts),
                covered_to_ts=float(batch[-1].ts),
                source_count=len(batch),
            )
            episode.cold.append(record)
            for _ in range(len(batch)):
                if episode.pending_spill:
                    episode.pending_spill.popleft()
            episode.summary_calls += 1
            episode.last_summary_at = self._clock()
            episode.last_summary_status = "ok"
            self.logger.info(
                "%s stage=summary result=ok group=%s messages=%d cold=%d chars=%d",
                LOG_PREFIX,
                state.group_id or state.umo,
                len(batch),
                len(episode.cold),
                len(record.text),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception(
                "%s summary failed group=%s",
                LOG_PREFIX,
                state.group_id or state.umo,
            )
            episode.last_summary_status = "error"
            episode.last_summary_attempt_at = self._clock()
        finally:
            if state.summary_task is asyncio.current_task():
                state.summary_task = None
            if (
                not self._closed
                and state.episode is episode
                and episode.active
            ):
                self._maybe_schedule_summary(state, self._clock())

    def _cancel_summary(self, state: GroupState) -> None:
        task = state.summary_task
        state.summary_task = None
        if task is not None and not task.done():
            task.cancel()

    def directed_context(self, umo: str, *, exclude_last: bool = True) -> str:
        """Build the compact DirectedView for an @bot/name-wake request."""

        state = self._states.get(str(umo or "").strip())
        if state is None:
            return ""
        now = self._clock()
        self_lines = [
            text
            for ts, text in state.recent_interjections
            if now - float(ts) <= 1800 and str(text).strip()
        ]
        return render_directed(
            state.episode,
            list(state.window),
            self_lines=self_lines,
            exclude_last=exclude_last,
        )

    def update_config(self, config: Config) -> None:
        """Apply a hot config change without resetting Episode/frequency state."""

        self.config = config
        preset = config.preset
        now = self._clock()
        for state in self._states.values():
            size = max(1, preset.window_size)
            while len(state.window) > size:
                self._spill(state, [state.window.popleft()])
            if state.window.maxlen != size:
                state.window = deque(state.window, maxlen=size)
            self._trim(state, now)
            if (
                state.episode.turn_count >= preset.episode_max_turns
                and state.episode.next_extra_at <= 0
            ):
                state.episode.next_extra_at = now + max(
                    1,
                    int(preset.episode_rest_seconds),
                )
            if not config.enable:
                self._cancel_pending(state)
                self._cancel_summary(state)
            else:
                self._maybe_schedule_summary(state, now)
        self.logger.info(
            "%s config hot-applied mode=%s dry_run=%s enable=%s",
            LOG_PREFIX,
            config.mode,
            config.dry_run,
            config.enable,
        )

    def _cancel_pending(self, state: GroupState) -> None:
        if state.deciding:
            state.epoch += 1
            state.reschedule = False
            return
        state.epoch += 1
        if state.pending is not None and not state.pending.done():
            state.pending.cancel()
        state.pending = None

    def _record_skip(
        self,
        state: GroupState,
        reason: str,
        now: float,
        *,
        window: int = 0,
        extra: str = "",
    ) -> None:
        state.skip_events.append((now, reason))
        while state.skip_events and now - state.skip_events[0][0] >= 3600:
            state.skip_events.popleft()
        while len(state.skip_events) > SKIP_HISTORY_LIMIT:
            state.skip_events.popleft()
        should_log = (
            reason != state.last_skip_log_reason
            or now - state.last_skip_log_ts >= SKIP_LOG_INTERVAL_SECONDS
        )
        state.last_skip_log_ts = now
        state.last_skip_log_reason = reason
        if not should_log:
            return
        self.logger.info(
            "%s stage=admission result=skip reason=%s group=%s window=%d "
            "state=%s %s",
            LOG_PREFIX,
            reason,
            state.group_id or state.umo,
            window,
            state.state.value,
            extra,
        )

    def _record_decision(
        self,
        state: GroupState,
        *,
        result: str,
        reason: str,
        confidence: float,
        elapsed: float,
        text: str = "",
        sent: bool = False,
        used_tools: bool = False,
        dry_run: bool | None = None,
    ) -> None:
        now = self._clock()
        dry_run_value = self.config.dry_run if dry_run is None else bool(dry_run)
        item = {
            "ts": now,
            "result": result,
            "reason": reason,
            "confidence": round(float(confidence), 2),
            "text": " ".join(str(text or "").split())[:80],
            "sent": bool(sent),
            "used_tools": bool(used_tools),
            "dry_run": dry_run_value,
            "state": state.state.value,
            "episode_turns": state.episode.turn_count,
        }
        state.decisions.append(item)
        while len(state.decisions) > DECISION_HISTORY_LIMIT:
            state.decisions.popleft()
        self.logger.info(
            "%s decision=%s result=%s reason=%s group=%s state=%s "
            "confidence=%.2f window=%d episode_turns=%d elapsed=%.1fs "
            "sent=%s tools=%s dry_run=%s text=%r",
            LOG_PREFIX,
            uuid.uuid4().hex[:8],
            result,
            reason or "-",
            state.group_id or state.umo,
            state.state.value,
            float(confidence),
            len(state.window),
            state.episode.turn_count,
            float(elapsed),
            str(bool(sent)).lower(),
            str(bool(used_tools)).lower(),
            str(bool(dry_run_value)).lower(),
            " ".join(str(text or "").split())[:80],
        )

    def _skip_reason_counts(
        self,
        state: GroupState,
        now: float,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ts, reason in state.skip_events:
            if now - ts < 3600:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    def _recent_decisions(
        self,
        state: GroupState,
        now: float,
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for entry in list(state.decisions)[-10:]:
            item = dict(entry)
            ts = item.pop("ts", None)
            if isinstance(ts, (int, float)):
                item["ago_seconds"] = round(now - float(ts), 1)
            items.append(item)
        return items

    async def _safe_persona(self, umo: str) -> tuple[str, str]:
        try:
            name, prompt = await self._get_persona(umo)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.warning(
                "%s persona resolution failed",
                LOG_PREFIX,
                exc_info=True,
            )
            return "", ""
        return str(name or ""), str(prompt or "")


__all__ = ["GroupState", "LOG_PREFIX", "Runtime"]
