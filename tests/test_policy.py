"""Admission and decision-policy tests."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2 import policy  # noqa: E402
from astrbot_plugin_group_interject_v2.model import (  # noqa: E402
    Config,
    Decision,
    Msg,
    Situation,
    State,
)

NOW = 1_700_000_000.0
UMO = "aiocqhttp:GroupMessage:100000001"
GID = "100000001"


def _msg(
    ts: float,
    text: str | None = None,
    *,
    is_bot: bool = False,
    directed: bool = False,
) -> Msg:
    return Msg(
        ts,
        "1",
        "群友",
        text if text is not None else f"消息{int(ts)}",
        is_bot=is_bot,
        directed=directed,
    )


def situation(
    *,
    now: float = NOW,
    window: tuple[Msg, ...] | None = None,
    fresh: tuple[Msg, ...] | None = None,
    **kwargs: object,
) -> Situation:
    if window is None:
        window = (
            _msg(now - 30),
            _msg(now - 20),
            _msg(now - 10),
        )
    values: dict[str, object] = {
        "now": now,
        "state": State.OBSERVING,
        "window": window,
        "fresh_messages": fresh if fresh is not None else window,
        "last_message": window[-1] if window else None,
        "last_bot_activity_ts": 0.0,
        "agent_busy": False,
        "last_judge_at": 0.0,
        "last_judged_ts": 0.0,
        "suppressed_until": 0.0,
        "quiet_hours": False,
        "episode_active": True,
        "followup_active": False,
        "episode_turns": 0,
        "episode_responded": False,
        "episode_last_sent_at": 0.0,
        "episode_last_sent_text": "",
    }
    values.update(kwargs)
    return Situation(**values)  # type: ignore[arg-type]


class TestAdmission(unittest.TestCase):
    def test_ok(self) -> None:
        result = policy.admit(Config(), situation(), group_id=GID, umo=UMO)
        self.assertTrue(result.ok)

    def test_hard_blocks(self) -> None:
        cfg = Config()
        cases = [
            ({"enable": False}, "disabled"),
            ({"group_blacklist": (GID,)}, "blacklist"),
            ({"group_whitelist": ("1",)}, "not_in_whitelist"),
        ]
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                c = replace(cfg, **overrides)  # type: ignore[arg-type]
                self.assertEqual(
                    policy.admit(c, situation(), group_id=GID, umo=UMO).reason,
                    reason,
                )

    def test_suppress_quiet_agent_bot(self) -> None:
        self.assertEqual(
            policy.admit(
                Config(),
                situation(suppressed_until=NOW + 10),
                group_id=GID,
                umo=UMO,
            ).reason,
            "suppressed",
        )
        self.assertEqual(
            policy.admit(
                Config(quiet_hours="00:00-23:59"),
                situation(quiet_hours=True),
                group_id=GID,
                umo=UMO,
            ).reason,
            "quiet_hours",
        )
        self.assertEqual(
            policy.admit(
                Config(),
                situation(agent_busy=True),
                group_id=GID,
                umo=UMO,
            ).reason,
            "agent_busy",
        )
        self.assertEqual(
            policy.admit(
                Config(),
                situation(last_bot_activity_ts=NOW - 5),
                group_id=GID,
                umo=UMO,
            ).reason,
            "bot_quiet",
        )

    def test_message_quality(self) -> None:
        self.assertEqual(
            policy.admit(Config(), situation(window=()), group_id=GID, umo=UMO).reason,
            "empty_window",
        )
        self.assertEqual(
            policy.admit(
                Config(),
                situation(
                    window=(
                        _msg(NOW - 30),
                        _msg(NOW - 20),
                        _msg(NOW - 15),
                        _msg(NOW - 5, is_bot=True),
                    )
                ),
                group_id=GID,
                umo=UMO,
            ).reason,
            "last_is_bot",
        )
        self.assertEqual(
            policy.admit(
                Config(),
                situation(
                    window=(
                        _msg(NOW - 30),
                        _msg(NOW - 20),
                        _msg(NOW - 15),
                        _msg(NOW - 5, directed=True),
                    ),
                ),
                group_id=GID,
                umo=UMO,
            ).reason,
            "last_directed",
        )
        self.assertEqual(
            policy.admit(
                Config(),
                situation(window=(_msg(NOW - 200), _msg(NOW - 190), _msg(NOW - 180))),
                group_id=GID,
                umo=UMO,
            ).reason,
            "last_message_stale",
        )
        repeated = tuple(_msg(NOW - 10 + i, "复读") for i in range(3))
        self.assertEqual(
            policy.admit(
                Config(),
                situation(window=repeated),
                group_id=GID,
                umo=UMO,
            ).reason,
            "noise",
        )

    def test_judge_cooldown_and_same_window(self) -> None:
        self.assertEqual(
            policy.admit(
                Config(),
                situation(last_judge_at=NOW - 10),
                group_id=GID,
                umo=UMO,
            ).reason,
            "judge_cooldown",
        )
        window = (_msg(NOW - 30), _msg(NOW - 20), _msg(NOW - 10))
        self.assertEqual(
            policy.admit(
                Config(),
                situation(
                    window=window,
                    last_judged_ts=NOW - 5,
                ),
                group_id=GID,
                umo=UMO,
            ).reason,
            "window_judged",
        )

    def test_cooldown_and_window_rearm(self) -> None:
        self.assertEqual(
            policy.admit(
                Config(),
                situation(episode_last_sent_at=NOW - 100),
                group_id=GID,
                umo=UMO,
            ).reason,
            "cooldown",
        )
        window = (_msg(NOW - 30), _msg(NOW - 20), _msg(NOW - 10))
        self.assertEqual(
            policy.admit(
                Config(),
                situation(
                    window=window,
                    fresh=(window[0],),
                    episode_last_sent_at=NOW - 500,
                ),
                group_id=GID,
                umo=UMO,
            ).reason,
            "window_rearm",
        )

    def test_followup_and_turn_cap(self) -> None:
        self.assertEqual(
            policy.admit(
                Config(),
                situation(
                    followup_active=True,
                    episode_turns=1,
                    episode_last_sent_at=NOW - 60,
                ),
                group_id=GID,
                umo=UMO,
            ).reason,
            "followup_no_reply",
        )
        self.assertEqual(
            policy.admit(
                Config(),
                situation(
                    followup_active=True,
                    episode_turns=3,
                    episode_responded=True,
                ),
                group_id=GID,
                umo=UMO,
            ).reason,
            "episode_slow",
        )
        result = policy.admit(
            Config(),
            situation(
                followup_active=True,
                episode_turns=1,
                episode_responded=True,
                episode_last_sent_at=NOW - 50,
                fresh=(_msg(NOW - 5),),
            ),
            group_id=GID,
            umo=UMO,
        )
        self.assertTrue(result.ok)

    def test_slow_gate_requires_time_and_messages(self) -> None:
        cfg = Config(mode="active")
        blocked_time = policy.admit(
            cfg,
            situation(
                episode_turns=4,
                episode_responded=True,
                episode_next_extra_at=NOW + 100,
                episode_human_since_last_bot=15,
            ),
            group_id=GID,
            umo=UMO,
        )
        self.assertEqual(blocked_time.reason, "episode_slow")
        self.assertEqual(blocked_time.remaining_seconds, 100)
        self.assertEqual(blocked_time.remaining_messages, 0)

        blocked_messages = policy.admit(
            cfg,
            situation(
                episode_turns=4,
                episode_responded=True,
                episode_next_extra_at=NOW - 1,
                episode_human_since_last_bot=14,
            ),
            group_id=GID,
            umo=UMO,
        )
        self.assertEqual(blocked_messages.reason, "episode_slow")
        self.assertEqual(blocked_messages.remaining_messages, 1)

        allowed = policy.admit(
            cfg,
            situation(
                episode_turns=4,
                episode_responded=True,
                episode_next_extra_at=NOW - 1,
                episode_human_since_last_bot=15,
            ),
            group_id=GID,
            umo=UMO,
        )
        self.assertTrue(allowed.ok)

    def test_send_phase_skips_new_window_requirement(self) -> None:
        result = policy.admit(
            Config(),
            situation(
                episode_last_sent_at=NOW - 500,
                last_judged_ts=NOW - 400,
            ),
            group_id=GID,
            umo=UMO,
            phase="send",
        )
        self.assertTrue(result.ok)


class TestFinalize(unittest.TestCase):
    def test_confidence_threshold(self) -> None:
        cfg = Config(mode="balanced")
        low = Decision(action="reply", text="hi", confidence=0.2)
        self.assertFalse(policy.finalize_decision(low, cfg, tools_available=True).is_reply)
        high = Decision(action="reply", text="hi", confidence=0.9)
        self.assertTrue(policy.finalize_decision(high, cfg, tools_available=True).is_reply)

    def test_tools_unavailable(self) -> None:
        decision = Decision(
            action="reply",
            text="查一下",
            confidence=0.9,
            mode="agent",
            needs_tools=True,
        )
        self.assertFalse(
            policy.finalize_decision(decision, Config(), tools_available=False).is_reply
        )


class TestFeedback(unittest.TestCase):
    def test_complaint_requires_context(self) -> None:
        self.assertFalse(
            policy.should_suppress(
                "别吵",
                addressed_to_bot=False,
                last_bot_activity_ts=0,
                now=NOW,
            )
        )
        self.assertTrue(
            policy.should_suppress(
                "别吵",
                addressed_to_bot=True,
                last_bot_activity_ts=0,
                now=NOW,
            )
        )
        self.assertTrue(
            policy.should_suppress(
                "太吵了",
                addressed_to_bot=False,
                last_bot_activity_ts=NOW - 30,
                now=NOW,
            )
        )

    def test_continuation(self) -> None:
        self.assertTrue(
            policy.is_continuation(
                "奖励吃鱼片",
                bot_text="奖励吃鱼片",
                bot_names=("鱼片",),
            )
        )
        self.assertFalse(
            policy.is_continuation(
                "今天天气不错我们去公园吧",
                bot_text="比赛结果出来了",
                bot_names=("鱼片",),
            )
        )


if __name__ == "__main__":
    unittest.main()
