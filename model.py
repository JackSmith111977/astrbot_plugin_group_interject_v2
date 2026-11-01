# -*- coding: utf-8 -*-
"""Domain models and configuration for the v2 group-interject plugin.

This module deliberately does not import AstrBot.  The runtime can therefore be
unit-tested with a fake clock, fake LLM and fake sender.

The v2 design replaces the old collection of per-feature gates with one
Episode state machine and one ``Decision`` contract.  Frequency is controlled
by the message/time window, a post-send re-arm rule, cooldowns and an episode
turn budget.  There is intentionally no hourly output quota.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """Internal, non-user-facing tuning values selected by ``Config.mode``."""

    window_size: int
    window_minutes: int
    min_messages: int
    debounce_seconds: float
    cooldown_seconds: int
    judge_cooldown_seconds: int
    followup_cooldown_seconds: int
    episode_window_seconds: int
    episode_gap_seconds: int
    episode_max_turns: int
    episode_rest_seconds: int
    episode_reentry_messages: int
    min_confidence: float
    bot_quiet_seconds: int
    last_message_max_age_seconds: int


PRESETS: dict[str, Preset] = {
    "quiet": Preset(
        window_size=60,
        window_minutes=20,
        min_messages=4,
        debounce_seconds=4.0,
        cooldown_seconds=900,
        judge_cooldown_seconds=120,
        followup_cooldown_seconds=60,
        episode_window_seconds=180,
        episode_gap_seconds=1200,
        episode_max_turns=2,
        episode_rest_seconds=1800,
        episode_reentry_messages=40,
        min_confidence=0.72,
        bot_quiet_seconds=45,
        last_message_max_age_seconds=180,
    ),
    "balanced": Preset(
        window_size=40,
        window_minutes=15,
        min_messages=3,
        debounce_seconds=2.5,
        cooldown_seconds=420,
        judge_cooldown_seconds=60,
        followup_cooldown_seconds=45,
        episode_window_seconds=150,
        episode_gap_seconds=900,
        episode_max_turns=3,
        episode_rest_seconds=900,
        episode_reentry_messages=25,
        min_confidence=0.55,
        bot_quiet_seconds=25,
        last_message_max_age_seconds=120,
    ),
    "active": Preset(
        window_size=30,
        window_minutes=10,
        min_messages=2,
        debounce_seconds=2.0,
        cooldown_seconds=240,
        judge_cooldown_seconds=45,
        followup_cooldown_seconds=30,
        episode_window_seconds=120,
        episode_gap_seconds=600,
        episode_max_turns=4,
        episode_rest_seconds=600,
        episode_reentry_messages=15,
        min_confidence=0.48,
        bot_quiet_seconds=15,
        last_message_max_age_seconds=90,
    ),
}

DEFAULT_MODE = "balanced"
MODES = tuple(PRESETS)

# Optional custom tuning, applied on top of one of the presets above.  Only the
# high-leverage frequency fields are exposed; internal pacing stays in Preset so
# that the custom layer cannot become a second gate stack.
CUSTOM_FIELD_LIMITS: dict[str, tuple[int | float, int | float, int | float]] = {
    "window_size": (10, 120, 5),
    "window_minutes": (3, 60, 1),
    "min_messages": (1, 10, 1),
    "cooldown_seconds": (30, 3600, 30),
    "judge_cooldown_seconds": (10, 600, 10),
    "min_confidence": (0.20, 0.95, 0.01),
    "episode_max_turns": (1, 6, 1),
    "episode_rest_seconds": (60, 3600, 60),
    "episode_reentry_messages": (1, 100, 1),
}

# Slow mode is a soft extension of the burst cap: each extra turn stretches the
# next re-entry interval, capped at 30 minutes.  These are internal pacing
# constants, not user-facing knobs.
EPISODE_SLOW_STEP_SECONDS = 300
EPISODE_SLOW_MAX_SECONDS = 1800

# Hard safety limits, not user-facing knobs.
MAX_FINAL_TEXT_CHARS = 1800
MAX_PROMPT_CHARS = 60
JUDGE_TIMEOUT_SECONDS = 30
TOOL_LOOP_TIMEOUT_SECONDS = 90
TOOL_CALL_TIMEOUT_SECONDS = 20
SUPPRESSED_SECONDS = 1800
AGENT_BUSY_LEASE_SECONDS = 900
LAST_EVENT_MAX_AGE_SECONDS = 120

# Cold context (internal constants; no public config knobs).
SUMMARY_MAX_RECORDS = 3
SUMMARY_MAX_CHARS = 500
SUMMARY_INPUT_MAX_CHARS = 3000
SUMMARY_TRIGGER_MESSAGES = 8
SUMMARY_TRIGGER_CHARS = 1000
SUMMARY_BATCH_MESSAGES = 12
SUMMARY_RETRY_SECONDS = 60
SUMMARY_TIMEOUT_SECONDS = 20
AMBIENT_SUMMARY_MAX_CHARS = 1200
DIRECTED_CONTEXT_MAX_CHARS = 900
DIRECTED_HOT_MESSAGES = 3

# Image descriptions are an internal enhancement.  They are generated in the
# background, cached briefly, and never exposed as public configuration.
IMAGE_CAPTION_MAX_IMAGES = 2
IMAGE_CAPTION_TIMEOUT_SECONDS = 20.0
IMAGE_CAPTION_WAIT_SECONDS = 12.0
IMAGE_CAPTION_SHORT_WAIT_SECONDS = 1.5
IMAGE_CAPTION_CACHE_SECONDS = 1800
IMAGE_CAPTION_FAILURE_THRESHOLD = 3
IMAGE_CAPTION_FAILURE_COOLDOWN_SECONDS = 300


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomTuning:
    """Optional overrides layered on top of the selected preset."""

    enabled: bool = False
    window_size: int = 40
    window_minutes: int = 15
    min_messages: int = 3
    cooldown_seconds: int = 420
    judge_cooldown_seconds: int = 60
    min_confidence: float = 0.55
    episode_max_turns: int = 3
    episode_rest_seconds: int = 900
    episode_reentry_messages: int = 25

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        base: Preset | None = None,
    ) -> CustomTuning:
        raw = raw if isinstance(raw, Mapping) else {}
        base = base or PRESETS[DEFAULT_MODE]
        window_size = _as_int(
            raw.get("window_size"),
            base.window_size,
            *CUSTOM_FIELD_LIMITS["window_size"][:2],
        )
        min_messages = _as_int(
            raw.get("min_messages"),
            base.min_messages,
            *CUSTOM_FIELD_LIMITS["min_messages"][:2],
        )
        return cls(
            enabled=_as_bool(raw.get("enabled"), False),
            window_size=window_size,
            window_minutes=_as_int(
                raw.get("window_minutes"),
                base.window_minutes,
                *CUSTOM_FIELD_LIMITS["window_minutes"][:2],
            ),
            min_messages=min(min_messages, window_size),
            cooldown_seconds=_as_int(
                raw.get("cooldown_seconds"),
                base.cooldown_seconds,
                *CUSTOM_FIELD_LIMITS["cooldown_seconds"][:2],
            ),
            judge_cooldown_seconds=_as_int(
                raw.get("judge_cooldown_seconds"),
                base.judge_cooldown_seconds,
                *CUSTOM_FIELD_LIMITS["judge_cooldown_seconds"][:2],
            ),
            min_confidence=_as_float(
                raw.get("min_confidence"),
                base.min_confidence,
                *CUSTOM_FIELD_LIMITS["min_confidence"][:2],
            ),
            episode_max_turns=_as_int(
                raw.get("episode_max_turns"),
                base.episode_max_turns,
                *CUSTOM_FIELD_LIMITS["episode_max_turns"][:2],
            ),
            episode_rest_seconds=_as_int(
                raw.get("episode_rest_seconds"),
                base.episode_rest_seconds,
                *CUSTOM_FIELD_LIMITS["episode_rest_seconds"][:2],
            ),
            episode_reentry_messages=_as_int(
                raw.get("episode_reentry_messages"),
                base.episode_reentry_messages,
                *CUSTOM_FIELD_LIMITS["episode_reentry_messages"][:2],
            ),
        )

    def apply(self, base: Preset) -> Preset:
        return replace(
            base,
            window_size=self.window_size,
            window_minutes=self.window_minutes,
            min_messages=self.min_messages,
            cooldown_seconds=self.cooldown_seconds,
            judge_cooldown_seconds=self.judge_cooldown_seconds,
            min_confidence=self.min_confidence,
            episode_max_turns=self.episode_max_turns,
            episode_rest_seconds=self.episode_rest_seconds,
            episode_reentry_messages=self.episode_reentry_messages,
        )


@dataclass(frozen=True)
class Config:
    """Public plugin configuration (12 flat fields + nested ``custom``)."""

    enable: bool = True
    mode: str = DEFAULT_MODE
    dry_run: bool = True
    provider_id: str = ""
    group_whitelist: tuple[str, ...] = ()
    group_blacklist: tuple[str, ...] = ()
    ignore_senders: tuple[str, ...] = ()
    wake_names: tuple[str, ...] = ()
    quiet_hours: str = ""
    extra_rules: str = ""
    image_caption_enable: bool = True
    memory_bridge_enable: bool = True
    custom: CustomTuning = field(default_factory=CustomTuning)

    @property
    def preset(self) -> Preset:
        base = PRESETS.get(self.mode, PRESETS[DEFAULT_MODE])
        return self.custom.apply(base) if self.custom.enabled else base

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> Config:
        raw = raw if isinstance(raw, Mapping) else {}
        default = cls()
        mode = _as_str(raw.get("mode"), default.mode).lower()
        if mode not in PRESETS:
            mode = default.mode
        custom = CustomTuning.from_mapping(
            raw.get("custom"),
            PRESETS.get(mode, PRESETS[DEFAULT_MODE]),
        )
        return cls(
            enable=_as_bool(raw.get("enable"), default.enable),
            mode=mode,
            dry_run=_as_bool(raw.get("dry_run"), default.dry_run),
            provider_id=_as_str(raw.get("provider_id"), default.provider_id),
            group_whitelist=_as_str_list(raw.get("group_whitelist")),
            group_blacklist=_as_str_list(raw.get("group_blacklist")),
            ignore_senders=_as_str_list(raw.get("ignore_senders")),
            wake_names=_as_str_list(raw.get("wake_names")),
            quiet_hours=_as_str(raw.get("quiet_hours"), default.quiet_hours),
            extra_rules=_as_str(raw.get("extra_rules"), default.extra_rules),
            image_caption_enable=_as_bool(
                raw.get("image_caption_enable"), default.image_caption_enable
            ),
            memory_bridge_enable=_as_bool(
                raw.get("memory_bridge_enable"), default.memory_bridge_enable
            ),
            custom=custom,
        )


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "是", "开")
    return default


def _as_str(value: Any, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _as_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        parsed = int(default)
    return max(int(low), min(int(high), parsed))


def _as_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = float(default)
    return max(float(low), min(float(high), parsed))


def _as_str_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part for part in re.split(r"[,，;；\s]+", value) if part)
    if isinstance(value, Iterable) and not isinstance(value, dict | bytes):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def group_filter_reason(cfg: Config, group_id: str, umo: str) -> str | None:
    """Return a blacklist/whitelist rejection reason, or None when allowed."""

    keys = {str(group_id or "").strip(), str(umo or "").strip()} - {""}
    if keys & set(cfg.group_blacklist):
        return "blacklist"
    if cfg.group_whitelist and not (keys & set(cfg.group_whitelist)):
        return "not_in_whitelist"
    return None


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class State(str, Enum):
    QUIET = "QUIET"
    OBSERVING = "OBSERVING"
    CANDIDATE = "CANDIDATE"
    ENGAGED = "ENGAGED"
    SUPPRESSED = "SUPPRESSED"


# ---------------------------------------------------------------------------
# Messages and episodes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Msg:
    """One formatted group message."""

    ts: float
    sender_id: str
    sender_name: str
    text: str
    is_bot: bool = False
    directed: bool = False
    image_refs: tuple[str, ...] = ()
    image_caption: str = ""


@dataclass(frozen=True, slots=True)
class SummaryRecord:
    """One compressed, older slice of an Episode (cold context)."""

    text: str
    created_at: float
    covered_from_ts: float
    covered_to_ts: float
    source_count: int


@dataclass(slots=True)
class Episode:
    """A topic segment: a sequence of messages separated by long silence."""

    started_at: float = 0.0
    last_activity_at: float = 0.0
    turn_count: int = 0
    extra_turns: int = 0
    next_extra_at: float = 0.0
    human_since_last_bot: int = 0
    last_sent_at: float = 0.0
    last_sent_text: str = ""
    responded: bool = False
    awaiting_response: bool = False
    closed_at: float = 0.0
    cold: deque[SummaryRecord] = field(
        default_factory=lambda: deque(maxlen=SUMMARY_MAX_RECORDS)
    )
    pending_spill: deque[Msg] = field(default_factory=deque)
    summary_calls: int = 0
    last_summary_at: float = 0.0
    last_summary_attempt_at: float = 0.0
    last_summary_status: str = ""

    @property
    def active(self) -> bool:
        return self.started_at > 0 and self.closed_at <= 0

    def start(self, now: float) -> None:
        self.started_at = now
        self.last_activity_at = now
        self.turn_count = 0
        self.extra_turns = 0
        self.next_extra_at = 0.0
        self.human_since_last_bot = 0
        self.last_sent_at = 0.0
        self.last_sent_text = ""
        self.responded = False
        self.awaiting_response = False
        self.closed_at = 0.0
        self.cold.clear()
        self.pending_spill.clear()
        self.summary_calls = 0
        self.last_summary_at = 0.0
        self.last_summary_attempt_at = 0.0
        self.last_summary_status = ""

    def close(self, now: float) -> None:
        if self.started_at > 0 and self.closed_at <= 0:
            self.closed_at = now
        self.awaiting_response = False

    def touch(self, now: float) -> None:
        if self.started_at <= 0:
            self.start(now)
        if self.closed_at > 0:
            self.start(now)
        self.last_activity_at = now

    def mark_response(self, now: float) -> None:
        self.touch(now)
        self.responded = True

    def on_send(self, now: float, text: str) -> None:
        if not self.active:
            self.start(now)
        self.turn_count += 1
        self.human_since_last_bot = 0
        self.last_sent_at = now
        self.last_sent_text = str(text or "")
        self.responded = False
        self.awaiting_response = True
        self.last_activity_at = now

    def followup_active(self, now: float, window_seconds: int) -> bool:
        return (
            self.turn_count > 0
            and self.active
            and self.last_sent_at > 0
            and now - self.last_sent_at <= max(0, int(window_seconds))
        )

    def remaining_turns(self, max_turns: int) -> int:
        return max(0, int(max_turns) - int(self.turn_count))


# ---------------------------------------------------------------------------
# Situation / decision / execution contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Situation:
    """Immutable snapshot handed to admission and the decider."""

    now: float
    state: State
    window: tuple[Msg, ...]
    fresh_messages: tuple[Msg, ...]
    last_message: Msg | None
    last_bot_activity_ts: float
    agent_busy: bool
    last_judge_at: float
    last_judged_ts: float
    suppressed_until: float
    quiet_hours: bool
    episode_active: bool
    followup_active: bool
    episode_turns: int
    episode_responded: bool
    episode_last_sent_at: float
    episode_last_sent_text: str
    episode_extra_turns: int = 0
    episode_next_extra_at: float = 0.0
    episode_human_since_last_bot: int = 0
    summaries: tuple[SummaryRecord, ...] = ()


@dataclass(frozen=True)
class Decision:
    """The single contract between decider and executor."""

    action: str = "silent"
    mode: str = "quick"
    text: str = ""
    confidence: float = 0.0
    continuity: bool = False
    needs_tools: bool = False
    reason: str = ""

    @property
    def is_reply(self) -> bool:
        return self.action == "reply" and bool(self.text.strip())

    @classmethod
    def silent(cls, reason: str = "") -> Decision:
        return cls(action="silent", reason=str(reason or "")[:160])


@dataclass(frozen=True)
class Admission:
    """Result of the single admission implementation."""

    ok: bool
    reason: str = ""
    remaining_seconds: float = 0.0
    remaining_messages: int = 0


@dataclass(frozen=True)
class ExecutionResult:
    """Result of one executor call.  Runtime enforces at most one send."""

    sent: bool
    text: str = ""
    reason: str = ""
    used_tools: bool = False
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers shared with runtime
# ---------------------------------------------------------------------------


def non_bot_count(messages: Sequence[Msg]) -> int:
    return sum(1 for msg in messages if not msg.is_bot)


def fresh_non_bot(messages: Sequence[Msg], after_ts: float) -> list[Msg]:
    return [msg for msg in messages if not msg.is_bot and msg.ts > after_ts]


def trim_messages(messages: Sequence[Msg], now: float, window_minutes: int) -> list[Msg]:
    if window_minutes <= 0:
        return list(messages)
    cutoff = now - window_minutes * 60
    return [msg for msg in messages if msg.ts >= cutoff]


def slow_rest_seconds(base_rest: int, extra_turns: int) -> int:
    """Return the next slow-mode interval after ``extra_turns`` bonus sends."""

    base = max(0, int(base_rest))
    stepped = base + max(0, int(extra_turns)) * EPISODE_SLOW_STEP_SECONDS
    return min(stepped, max(base, EPISODE_SLOW_MAX_SECONDS))


__all__ = [
    "Admission",
    "AGENT_BUSY_LEASE_SECONDS",
    "AMBIENT_SUMMARY_MAX_CHARS",
    "CUSTOM_FIELD_LIMITS",
    "Config",
    "CustomTuning",
    "DEFAULT_MODE",
    "DIRECTED_CONTEXT_MAX_CHARS",
    "DIRECTED_HOT_MESSAGES",
    "EPISODE_SLOW_MAX_SECONDS",
    "EPISODE_SLOW_STEP_SECONDS",
    "Decision",
    "Episode",
    "ExecutionResult",
    "JUDGE_TIMEOUT_SECONDS",
    "LAST_EVENT_MAX_AGE_SECONDS",
    "MAX_FINAL_TEXT_CHARS",
    "MAX_PROMPT_CHARS",
    "MODES",
    "Msg",
    "PRESETS",
    "Preset",
    "SUPPRESSED_SECONDS",
    "SUMMARY_BATCH_MESSAGES",
    "SUMMARY_INPUT_MAX_CHARS",
    "SUMMARY_MAX_CHARS",
    "SUMMARY_MAX_RECORDS",
    "SUMMARY_RETRY_SECONDS",
    "SUMMARY_TIMEOUT_SECONDS",
    "SUMMARY_TRIGGER_CHARS",
    "SUMMARY_TRIGGER_MESSAGES",
    "Situation",
    "SummaryRecord",
    "State",
    "TOOL_CALL_TIMEOUT_SECONDS",
    "TOOL_LOOP_TIMEOUT_SECONDS",
    "fresh_non_bot",
    "group_filter_reason",
    "non_bot_count",
    "slow_rest_seconds",
    "trim_messages",
]
