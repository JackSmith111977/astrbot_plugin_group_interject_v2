# -*- coding: utf-8 -*-
"""Pure state-machine policy: admission, final decision checks and feedback.

There is exactly one admission implementation.  The old plugin had two copies
(a coarse pre-check and a full gate); this module is the replacement.
"""

from __future__ import annotations

from .model import (
    Admission,
    Config,
    Decision,
    Preset,
    Situation,
    group_filter_reason,
    non_bot_count,
)
from .text import is_noise_complaint, looks_like_continuation, looks_like_noise


def admit(
    cfg: Config,
    situation: Situation,
    *,
    group_id: str,
    umo: str,
    phase: str = "judge",
) -> Admission:
    """Check whether the current situation may produce/execute a decision.

    Args:
        cfg: Plugin configuration.
        situation: Immutable snapshot of the group window and episode.
        group_id: Group number, used for whitelist/blacklist.
        umo: Unified message origin, also accepted by the filters.
        phase: ``judge`` for the decision call or ``send`` for the pre-send
            recheck.  The send phase skips the "new message since last judge"
            checks because the decision was already made for this window.

    Returns:
        Admission with the first failing reason, or ``ok=True``.
    """

    preset = cfg.preset
    now = situation.now
    if not cfg.enable:
        return Admission(False, "disabled")
    reason = group_filter_reason(cfg, group_id, umo)
    if reason:
        return Admission(False, reason)
    if situation.suppressed_until > now:
        return Admission(
            False,
            "suppressed",
            round(situation.suppressed_until - now, 1),
        )
    if situation.quiet_hours:
        return Admission(False, "quiet_hours")
    if situation.agent_busy:
        return Admission(False, "agent_busy")
    if (
        situation.last_bot_activity_ts > 0
        and now - situation.last_bot_activity_ts < preset.bot_quiet_seconds
    ):
        return Admission(
            False,
            "bot_quiet",
            round(preset.bot_quiet_seconds - (now - situation.last_bot_activity_ts), 1),
        )

    if phase == "judge":
        if (
            situation.last_judge_at > 0
            and now - situation.last_judge_at < preset.judge_cooldown_seconds
        ):
            return Admission(
                False,
                "judge_cooldown",
                round(
                    preset.judge_cooldown_seconds
                    - (now - situation.last_judge_at),
                    1,
                ),
            )
        if (
            situation.last_judged_ts > 0
            and situation.last_message is not None
            and situation.last_message.ts <= situation.last_judged_ts
            and (
                situation.episode_last_sent_at <= 0
                or situation.last_judged_ts > situation.episode_last_sent_at
            )
        ):
            return Admission(False, "window_judged")

    window = list(situation.window)
    if not window:
        return Admission(False, "empty_window")
    if non_bot_count(window) < preset.min_messages:
        return Admission(False, "min_messages")
    last = situation.last_message or window[-1]
    if last.is_bot:
        return Admission(False, "last_is_bot")
    if last.directed:
        return Admission(False, "last_directed")
    if now - last.ts > preset.last_message_max_age_seconds:
        return Admission(
            False,
            "last_message_stale",
            round(preset.last_message_max_age_seconds - (now - last.ts), 1),
        )
    if looks_like_noise(window):
        return Admission(False, "noise")

    if situation.episode_active:
        if situation.followup_active and not situation.episode_responded:
            return Admission(
                False,
                "followup_no_reply",
                round(
                    preset.episode_window_seconds
                    - (now - situation.episode_last_sent_at),
                    1,
                ),
            )
        if situation.episode_turns >= preset.episode_max_turns:
            slow = slow_gate(situation, preset)
            if slow is not None:
                return slow

    followup_ready = (
        situation.followup_active and situation.episode_responded
    )
    effective_cooldown = (
        preset.followup_cooldown_seconds if followup_ready else preset.cooldown_seconds
    )
    if (
        situation.episode_last_sent_at > 0
        and now - situation.episode_last_sent_at < effective_cooldown
    ):
        return Admission(
            False,
            "followup_wait" if followup_ready else "cooldown",
            round(effective_cooldown - (now - situation.episode_last_sent_at), 1),
        )

    if phase == "judge" and situation.episode_last_sent_at > 0:
        required = 1 if followup_ready else preset.min_messages
        if len(situation.fresh_messages) < required:
            return Admission(False, "window_rearm")

    return Admission(True)


def slow_gate(situation: Situation, preset: Preset) -> Admission | None:
    """Return a slow-mode rejection, or None when one extra turn is allowed."""

    if (
        not situation.episode_active
        or situation.episode_turns < preset.episode_max_turns
    ):
        return None

    next_at = float(situation.episode_next_extra_at)
    if next_at <= 0:
        anchor = (
            float(situation.episode_last_sent_at)
            if situation.episode_last_sent_at > 0
            else float(situation.now)
        )
        next_at = anchor + max(1, int(preset.episode_rest_seconds))

    missing_messages = max(
        0,
        int(preset.episode_reentry_messages)
        - max(0, int(situation.episode_human_since_last_bot)),
    )
    if situation.now >= next_at and missing_messages == 0:
        return None
    return Admission(
        False,
        "episode_slow",
        round(max(0.0, next_at - situation.now), 1),
        missing_messages,
    )


def finalize_decision(
    decision: Decision,
    cfg: Config,
    *,
    tools_available: bool,
) -> Decision:
    """Apply confidence/tool safety rules to a parsed decision."""

    if not decision.is_reply:
        return Decision.silent(decision.reason or "silent")
    if decision.confidence < cfg.preset.min_confidence:
        return Decision.silent(
            f"below_confidence({decision.confidence:.2f}<{cfg.preset.min_confidence:.2f})"
        )
    if decision.needs_tools and not tools_available:
        return Decision.silent("tools_unavailable")
    if decision.mode == "agent" and not tools_available:
        return Decision.silent("tools_unavailable")
    return decision


def should_suppress(
    text: str,
    *,
    addressed_to_bot: bool,
    last_bot_activity_ts: float,
    now: float,
) -> bool:
    """Whether a message is a directed/recent complaint that should mute us."""

    if not is_noise_complaint(text):
        return False
    recent_bot = (
        last_bot_activity_ts > 0 and now - last_bot_activity_ts <= 120
    )
    return bool(addressed_to_bot or recent_bot)


def is_continuation(
    text: str,
    *,
    bot_text: str,
    bot_names: tuple[str, ...],
) -> bool:
    """Heuristic for unlocking the engaged follow-up path without an @."""

    return looks_like_continuation(
        text,
        bot_text=bot_text,
        bot_names=bot_names,
    )


__all__ = [
    "admit",
    "finalize_decision",
    "is_continuation",
    "should_suppress",
    "slow_gate",
]
