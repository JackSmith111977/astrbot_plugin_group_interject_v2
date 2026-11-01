"""Cold-summary spill, FIFO and summarizer tests."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2.model import (  # noqa: E402
    Config,
    Decision,
    Episode,
    ExecutionResult,
    Msg,
    SummaryRecord,
)
from astrbot_plugin_group_interject_v2.runtime import Runtime  # noqa: E402
from astrbot_plugin_group_interject_v2.summarizer import (  # noqa: E402
    Summarizer,
    clean_summary,
)

UMO = "aiocqhttp:GroupMessage:100000001"
GID = "100000001"
NOW = 1_700_000_000.0
LOG = None  # 本地测试不需要真实 logger


class FakeClock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SilentDecider:
    async def decide(self, umo, situation, **kwargs):
        return Decision.silent("test")


class FakeExecutor:
    async def execute(self, decision, **kwargs):
        return ExecutionResult(True, text=decision.text)


class FakeSummarizer:
    def __init__(self, text: str | None = "较早话题摘要") -> None:
        self.text = text
        self.calls: list[list] = []

    async def summarize(self, umo: str, group_name: str, messages: list) -> str | None:
        self.calls.append(list(messages))
        return self.text


def make_runtime(
    summarizer: FakeSummarizer | None = None,
    config: Config | None = None,
) -> tuple[Runtime, FakeClock]:
    clock = FakeClock()

    async def persona(_umo: str) -> tuple[str, str]:
        return "鱼片", "人格"

    runtime = Runtime(
        config
        or Config(
            mode="active",
            dry_run=True,
            group_whitelist=(GID,),
            wake_names=("鱼片",),
        ),
        decider=SilentDecider(),
        executor=FakeExecutor(),
        get_persona=persona,
        clock=clock,
        logger=LOG,
        tools_available=lambda: False,
        summarizer=summarizer,
    )
    return runtime, clock


async def add_messages(
    runtime: Runtime,
    count: int,
    prefix: str = "消息",
    *,
    yield_after: bool = True,
) -> None:
    for index in range(count):
        runtime.ingest(
            umo=UMO,
            group_id=GID,
            sender_id="10001",
            sender_name="群友甲",
            text=f"{prefix}{index}-{NOW + index}",
            self_id="2278168275",
        )
    state = runtime._states[UMO]
    if state.pending is not None:
        state.pending.cancel()
        state.pending = None
    if yield_after:
        await asyncio.sleep(0)


async def flush_summary(runtime: Runtime) -> None:
    state = runtime._states[UMO]
    for _ in range(30):
        await asyncio.sleep(0)
        task = state.summary_task
        if task is not None and not task.done():
            await task
        await asyncio.sleep(0)
        if state.summary_task is None:
            return


class TestEpisodeSummary(unittest.TestCase):
    def test_episode_start_clears_cold(self) -> None:
        episode = Episode()
        episode.start(NOW)
        episode.cold.append(
            SummaryRecord("x", NOW, NOW, NOW, 1)
        )
        episode.pending_spill.append(object())  # type: ignore[arg-type]
        episode.start(NOW + 10)
        self.assertEqual(len(episode.cold), 0)
        self.assertEqual(len(episode.pending_spill), 0)

    def test_clean_summary(self) -> None:
        self.assertEqual(clean_summary("```\n摘要：在聊股票。\n```"), "在聊股票。")
        self.assertIsNone(clean_summary(""))
        self.assertIsNone(clean_summary("   "))
        self.assertLessEqual(len(clean_summary("字" * 900) or ""), 500)


class TestSummarizer(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_returns_none(self) -> None:
        async def slow(umo, system_prompt, user_prompt):
            await asyncio.sleep(1)
            return "摘要"

        result = await Summarizer(slow, timeout_seconds=0.01).summarize(
            UMO,
            "群",
            [],
        )
        self.assertIsNone(result)

    async def test_normal_summary(self) -> None:
        async def llm(umo, system_prompt, user_prompt):
            self.assertIn("待压缩", user_prompt)
            return "话题：股票；未解决：明天走势。"

        result = await Summarizer(llm).summarize(
            UMO,
            "群",
            [
                Msg(NOW, "1", "甲", "今天在聊股票"),
                Msg(NOW + 1, "2", "乙", "明天可能还有一波"),
            ],
        )
        self.assertEqual(result, "话题：股票；未解决：明天走势。")

        empty = await Summarizer(llm).summarize(UMO, "群", [])
        self.assertIsNone(empty, "empty input should not call the model")


class TestRuntimeSummary(unittest.IsolatedAsyncioTestCase):
    async def test_spill_then_cold_summary(self) -> None:
        summarizer = FakeSummarizer("摘要一")
        runtime, _clock = make_runtime(summarizer)
        await add_messages(runtime, 38)
        await flush_summary(runtime)
        state = runtime._states[UMO]
        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(len(state.episode.cold), 1)
        self.assertEqual(len(state.episode.pending_spill), 0)
        self.assertEqual(state.episode.last_summary_status, "ok")
        await runtime.terminate()

    async def test_cold_fifo_max_three(self) -> None:
        summarizer = FakeSummarizer("摘要")
        runtime, _clock = make_runtime(summarizer)
        for batch in range(4):
            await add_messages(runtime, 38, prefix=f"批次{batch}-")
            await flush_summary(runtime)
        state = runtime._states[UMO]
        self.assertEqual(len(state.episode.cold), 3)
        self.assertGreaterEqual(len(summarizer.calls), 4)
        await runtime.terminate()

    async def test_failure_keeps_pending_and_does_not_break(self) -> None:
        summarizer = FakeSummarizer(None)
        runtime, _clock = make_runtime(summarizer)
        await add_messages(runtime, 38)
        await flush_summary(runtime)
        state = runtime._states[UMO]
        self.assertGreaterEqual(len(state.episode.pending_spill), 1)
        self.assertEqual(list(state.episode.cold), [])
        self.assertEqual(state.episode.last_summary_status, "empty")
        await runtime.terminate()

    async def test_update_config_shrinks_window_into_spill(self) -> None:
        runtime, _clock = make_runtime(
            config=Config(
                mode="quiet",
                dry_run=True,
                group_whitelist=(GID,),
            )
        )
        await add_messages(runtime, 35)
        state = runtime._states[UMO]
        self.assertEqual(len(state.window), 35)
        runtime.update_config(
            Config(mode="active", dry_run=True, group_whitelist=(GID,))
        )
        self.assertEqual(len(state.window), 30)
        self.assertEqual(len(state.episode.pending_spill), 5)
        await runtime.terminate()

    async def test_disable_cancels_summary_and_keeps_pending(self) -> None:
        summarizer = FakeSummarizer("不应调用")
        runtime, _clock = make_runtime(summarizer)
        await add_messages(runtime, 38, yield_after=False)
        state = runtime._states[UMO]
        runtime.update_config(Config(enable=False, group_whitelist=(GID,)))
        await asyncio.sleep(0.01)
        self.assertEqual(summarizer.calls, [])
        self.assertEqual(len(state.episode.pending_spill), 8)
        self.assertIsNone(state.summary_task)
        await runtime.terminate()

    async def test_empty_at_context_includes_cold_summary(self) -> None:
        runtime, clock = make_runtime()
        await add_messages(runtime, 3)
        state = runtime._states[UMO]
        state.episode.cold.append(
            SummaryRecord("冷摘要：之前聊过股票", clock(), NOW, NOW, 3)
        )
        text = runtime.bare_at_context(UMO, 6)
        self.assertIn("冷摘要", text)
        await runtime.terminate()

    async def test_summary_does_not_change_admission(self) -> None:
        runtime, clock = make_runtime(FakeSummarizer("摘要"))
        await add_messages(runtime, 3)
        state = runtime._states[UMO]
        situation = runtime._situation(state, clock())
        before = situation.window
        state.episode.cold.append(
            SummaryRecord("冷摘要", clock(), NOW, NOW, 1)
        )
        after = runtime._situation(state, clock())
        self.assertEqual(before, after.window)
        self.assertEqual(len(after.summaries), 1)
        await runtime.terminate()


if __name__ == "__main__":
    unittest.main()
