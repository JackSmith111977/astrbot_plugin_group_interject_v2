"""Executor and decider adapter tests."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2.decider import Decider  # noqa: E402
from astrbot_plugin_group_interject_v2.executor import Executor  # noqa: E402
from astrbot_plugin_group_interject_v2.model import (  # noqa: E402
    Decision,
    Msg,
    Situation,
    State,
)

NOW = 1_700_000_000.0
UMO = "aiocqhttp:GroupMessage:100000001"
LOG = None  # 本地测试不需要真实 logger


def _situation() -> Situation:
    window = (
        Msg(NOW - 30, "1", "甲", "今天好累"),
        Msg(NOW - 20, "2", "乙", "确实"),
        Msg(NOW - 10, "1", "甲", "下班了"),
    )
    return Situation(
        now=NOW,
        state=State.OBSERVING,
        window=window,
        fresh_messages=window,
        last_message=window[-1],
        last_bot_activity_ts=0.0,
        agent_busy=False,
        last_judge_at=0.0,
        last_judged_ts=0.0,
        suppressed_until=0.0,
        quiet_hours=False,
        episode_active=True,
        followup_active=False,
        episode_turns=0,
        episode_responded=False,
        episode_last_sent_at=0.0,
        episode_last_sent_text="",
    )


class TestExecutor(unittest.IsolatedAsyncioTestCase):
    async def test_quick_send(self) -> None:
        sent: list[tuple[str, str]] = []

        async def send(umo: str, text: str) -> bool:
            sent.append((umo, text))
            return True

        executor = Executor(send=send, logger=LOG)
        result = await executor.execute(
            Decision(action="reply", text="哈哈", confidence=0.9),
            umo=UMO,
        )
        self.assertTrue(result.sent)
        self.assertFalse(result.used_tools)
        self.assertEqual(sent, [(UMO, "哈哈")])

    async def test_tool_loop_used_and_cleaned(self) -> None:
        sent: list[str] = []
        calls: list[tuple] = []

        async def send(_umo: str, text: str) -> bool:
            sent.append(text)
            return True

        async def tool_loop(event, system_prompt, user_prompt):
            calls.append((event, system_prompt, user_prompt))
            return "鱼片：查到了"

        executor = Executor(send=send, tool_loop=tool_loop, logger=LOG)
        result = await executor.execute(
            Decision(
                action="reply",
                mode="agent",
                needs_tools=True,
                text="查一下",
                confidence=0.9,
            ),
            umo=UMO,
            event=object(),
            tool_system_prompt="sys",
            tool_user_prompt="user",
            bot_names=("鱼片",),
        )
        self.assertTrue(result.sent)
        self.assertTrue(result.used_tools)
        self.assertEqual(sent, ["查到了"])
        self.assertEqual(len(calls), 1)

    async def test_tool_unavailable_and_failure(self) -> None:
        sent: list[str] = []

        async def send(_umo: str, text: str) -> bool:
            sent.append(text)
            return True

        executor = Executor(send=send, logger=LOG)
        result = await executor.execute(
            Decision(
                action="reply",
                mode="agent",
                needs_tools=True,
                text="查",
                confidence=0.9,
            ),
            umo=UMO,
            event=None,
        )
        self.assertFalse(result.sent)
        self.assertEqual(result.reason, "tool_unavailable")

        async def broken_tool_loop(event, system_prompt, user_prompt):
            raise RuntimeError("boom")

        broken = Executor(send=send, tool_loop=broken_tool_loop, logger=LOG)
        result = await broken.execute(
            Decision(
                action="reply",
                mode="agent",
                needs_tools=True,
                text="查",
                confidence=0.9,
            ),
            umo=UMO,
            event=object(),
        )
        self.assertFalse(result.sent)
        self.assertEqual(result.reason, "tool_error")
        self.assertEqual(sent, [])

    async def test_send_failure(self) -> None:
        async def send(_umo: str, _text: str) -> bool:
            return False

        executor = Executor(send=send, logger=LOG)
        result = await executor.execute(
            Decision(action="reply", text="hi", confidence=0.9),
            umo=UMO,
        )
        self.assertFalse(result.sent)
        self.assertEqual(result.reason, "send_false")


class TestDecider(unittest.IsolatedAsyncioTestCase):
    async def test_parse_and_clean(self) -> None:
        calls: list[tuple] = []

        async def llm(umo, system_prompt, user_prompt):
            calls.append((umo, system_prompt, user_prompt))
            return '{"action":"reply","text":"鱼片：你好呀","confidence":0.9}'

        decider = Decider(llm, logger=LOG)
        decision = await decider.decide(
            UMO,
            _situation(),
            group_name="测试群",
            persona_name="鱼片",
            persona_prompt="人格",
        )
        self.assertTrue(decision.is_reply)
        self.assertEqual(decision.text, "你好呀")
        self.assertEqual(len(calls), 1)
        self.assertIn("测试群", calls[0][2])

    async def test_timeout_and_exception_become_silent(self) -> None:
        async def slow_llm(umo, system_prompt, user_prompt):
            await asyncio.sleep(1.0)
            return "{}"

        decider = Decider(slow_llm, logger=LOG, timeout_seconds=0.01)
        decision = await decider.decide(UMO, _situation(), group_name="群")
        self.assertFalse(decision.is_reply)
        self.assertEqual(decision.reason, "timeout")

        async def broken_llm(umo, system_prompt, user_prompt):
            raise RuntimeError("boom")

        decider = Decider(broken_llm, logger=LOG)
        decision = await decider.decide(UMO, _situation(), group_name="群")
        self.assertFalse(decision.is_reply)
        self.assertEqual(decision.reason, "llm_error")


if __name__ == "__main__":
    unittest.main()
