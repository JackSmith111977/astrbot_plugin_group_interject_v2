"""Runtime pipeline tests with fake clock/decider/executor/memory."""

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
    ExecutionResult,
    State,
)
from astrbot_plugin_group_interject_v2.runtime import Runtime  # noqa: E402

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


class FakeDecider:
    def __init__(self, decisions: list[Decision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[tuple[str, object]] = []

    async def decide(self, umo, situation, **kwargs):
        self.calls.append((umo, situation))
        if not self.decisions:
            return Decision.silent("no_more")
        return self.decisions.pop(0)


class BlockingDecider(FakeDecider):
    def __init__(self, decision: Decision) -> None:
        super().__init__([decision])
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.attempts = 0

    async def decide(self, umo, situation, **kwargs):
        self.attempts += 1
        self.started.set()
        await self.release.wait()
        return await super().decide(umo, situation, **kwargs)


class FakeExecutor:
    def __init__(self, sent: bool = True) -> None:
        self.sent = sent
        self.calls: list[Decision] = []

    async def execute(self, decision, **kwargs):
        self.calls.append(decision)
        return ExecutionResult(
            sent=self.sent,
            text=decision.text if self.sent else "",
            reason="ok" if self.sent else "send_false",
        )


class FakeCaptioner:
    def __init__(self, caption: str = "一只猫趴在键盘上") -> None:
        self.caption = caption
        self.submitted: list[tuple[str, object, str]] = []
        self.waited: list[tuple[str, tuple[str, ...], float]] = []
        self.closed = False

    def submit(self, umo: str, msg, group_id: str = "") -> None:
        self.submitted.append((umo, msg, group_id))
        if self.caption:
            msg.image_caption = self.caption

    async def wait_ready(self, umo, refs, timeout: float = 1.5) -> None:
        self.waited.append((umo, tuple(refs), timeout))

    async def close(self) -> None:
        self.closed = True


def make_runtime(
    *,
    decisions: list[Decision] | None = None,
    executor: FakeExecutor | None = None,
    dry_run: bool = False,
    mode: str = "active",
    captioner: FakeCaptioner | None = None,
) -> tuple[Runtime, FakeClock, FakeDecider, FakeExecutor, list[tuple]]:
    clock = FakeClock()
    decider: FakeDecider = FakeDecider(decisions or [])
    executor = executor or FakeExecutor()
    memory: list[tuple] = []

    async def persona(_umo: str) -> tuple[str, str]:
        return "鱼片", "测试人格"

    async def memory_writer(*args) -> None:
        memory.append(args)

    runtime = Runtime(
        Config(
            mode=mode,
            dry_run=dry_run,
            group_whitelist=(GID,),
            wake_names=("鱼片", "鲸鲸"),
        ),
        decider=decider,
        executor=executor,
        get_persona=persona,
        clock=clock,
        logger=LOG,
        tools_available=lambda: True,
        memory_writer=memory_writer,
        captioner=captioner,
    )
    return runtime, clock, decider, executor, memory


async def add_messages(runtime: Runtime, clock: FakeClock, *texts: str) -> None:
    for text in texts:
        runtime.ingest(
            umo=UMO,
            group_id=GID,
            group_name="测试群",
            sender_id="10001",
            sender_name="群友甲",
            text=text,
            directed=False,
            self_id="2278168275",
        )
    state = runtime._states[UMO]
    if state.pending is not None:
        state.pending.cancel()
        state.pending = None
    await asyncio.sleep(0)


async def run_once(runtime: Runtime) -> None:
    state = runtime._states[UMO]
    await runtime._run_decision(state, state.epoch)


class TestRuntimePipeline(unittest.IsolatedAsyncioTestCase):
    async def test_reply_sends_once_and_writes_memory_once(self) -> None:
        rt, clock, decider, executor, memory = make_runtime(
            decisions=[
                Decision(
                    action="reply",
                    text="哈哈，确实",
                    confidence=0.9,
                )
            ]
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(len(memory), 1)
        state = rt._states[UMO]
        self.assertEqual(state.episode.turn_count, 1)
        self.assertEqual(state.state, State.ENGAGED)
        self.assertTrue(state.episode.last_sent_text)
        self.assertEqual(len(decider.calls), 1)
        await rt.terminate()

    async def test_image_caption_is_submitted_and_visible_to_decider(self) -> None:
        captioner = FakeCaptioner()
        rt, clock, decider, _executor, _memory = make_runtime(
            decisions=[Decision.silent("media_only")],
            captioner=captioner,
        )
        rt.ingest(
            umo=UMO,
            group_id=GID,
            group_name="测试群",
            sender_id="10001",
            sender_name="群友甲",
            text="[图片]",
            self_id="2278168275",
            image_refs=("https://example.test/a.png",),
        )
        state = rt._states[UMO]
        if state.pending is not None:
            state.pending.cancel()
            state.pending = None
        await asyncio.sleep(0)
        self.assertEqual(len(captioner.submitted), 1)
        self.assertEqual(state.window[-1].image_refs, ("https://example.test/a.png",))
        self.assertEqual(state.window[-1].image_caption, "一只猫趴在键盘上")

        await add_messages(rt, clock, "好好玩", "强奸德战")
        await run_once(rt)
        self.assertTrue(captioner.waited)
        self.assertTrue(
            any(
                msg.image_caption == "一只猫趴在键盘上"
                for msg in decider.calls[0][1].window
            )
        )
        await rt.terminate()
        self.assertTrue(captioner.closed)

    async def test_unresolved_recent_image_uses_long_caption_wait(self) -> None:
        captioner = FakeCaptioner(caption="")
        rt, clock, decider, _executor, _memory = make_runtime(
            decisions=[Decision.silent("media_pending")],
            captioner=captioner,
        )
        rt.ingest(
            umo=UMO,
            group_id=GID,
            group_name="测试群",
            sender_id="10001",
            sender_name="群友甲",
            text="[图片]",
            self_id="2278168275",
            image_refs=("https://example.test/pending.png",),
        )
        state = rt._states[UMO]
        if state.pending is not None:
            state.pending.cancel()
            state.pending = None
        await asyncio.sleep(0)
        self.assertFalse(state.window[-1].image_caption)
        await add_messages(rt, clock, "图呢")
        await run_once(rt)
        self.assertTrue(captioner.waited)
        self.assertEqual(captioner.waited[0][2], 12.0)
        self.assertFalse(decider.calls[0][1].window[-1].image_caption)
        await rt.terminate()

    async def test_ingest_without_schedule_records_context_only(self) -> None:
        rt, clock, decider, _executor, _memory = make_runtime()
        rt.ingest(
            umo=UMO,
            group_id=GID,
            group_name="测试群",
            sender_id="10002",
            sender_name="群友乙",
            text="[戳了 @123456（别人）]",
            self_id="2278168275",
            schedule_decision=False,
        )
        state = rt._states[UMO]
        self.assertEqual(len(state.window), 1)
        self.assertIsNone(state.pending)
        self.assertEqual(decider.calls, [])
        rt.ingest(
            umo=UMO,
            group_id=GID,
            group_name="测试群",
            sender_id="10001",
            sender_name="群友甲",
            text="正常消息",
            self_id="2278168275",
        )
        self.assertIsNotNone(state.pending)
        if state.pending is not None:
            state.pending.cancel()
            state.pending = None
        await asyncio.sleep(0)
        await rt.terminate()

    async def test_dry_run_does_not_send_or_write_memory(self) -> None:
        rt, clock, _decider, executor, memory = make_runtime(
            decisions=[Decision(action="reply", text="好", confidence=0.9)],
            dry_run=True,
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(executor.calls, [])
        self.assertEqual(memory, [])
        state = rt._states[UMO]
        self.assertEqual(state.episode.turn_count, 1)
        self.assertTrue(state.decisions[-1]["dry_run"])
        await rt.terminate()

    async def test_send_failure_does_not_advance_episode(self) -> None:
        rt, clock, _decider, executor, memory = make_runtime(
            decisions=[Decision(action="reply", text="好", confidence=0.9)],
            executor=FakeExecutor(sent=False),
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(memory, [])
        state = rt._states[UMO]
        self.assertEqual(state.episode.turn_count, 0)
        self.assertEqual(state.state, State.OBSERVING)
        await rt.terminate()

    async def test_silent_window_is_not_rejudged_without_new_messages(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime(
            decisions=[
                Decision.silent("not_natural"),
                Decision(action="reply", text="接一句", confidence=0.9),
            ]
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(len(decider.calls), 1)
        clock.advance(50)
        await run_once(rt)
        self.assertEqual(len(decider.calls), 1)
        self.assertEqual(executor.calls, [])
        await add_messages(rt, clock, "那早点睡")
        await run_once(rt)
        self.assertEqual(len(decider.calls), 2)
        self.assertEqual(len(executor.calls), 1)
        await rt.terminate()

    async def test_post_send_rearm_requires_new_messages(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime(
            decisions=[
                Decision(action="reply", text="第一句", confidence=0.9),
                Decision(action="reply", text="第二句", confidence=0.9),
            ]
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        clock.advance(500)
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        await add_messages(rt, clock, "新消息一")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        await add_messages(rt, clock, "新消息二", "新消息三")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(len(decider.calls), 2)
        await rt.terminate()

    async def test_one_in_flight_decision_per_group(self) -> None:
        decider = BlockingDecider(
            Decision(action="reply", text="只发一次", confidence=0.9)
        )
        rt, clock, _, executor, _memory = make_runtime()
        rt._decider = decider
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        state = rt._states[UMO]
        first = asyncio.create_task(rt._run_decision(state, state.epoch))
        await decider.started.wait()
        await rt._run_decision(state, state.epoch)
        self.assertEqual(decider.attempts, 1)
        decider.release.set()
        await first
        self.assertEqual(len(decider.calls), 1)
        self.assertEqual(len(executor.calls), 1)
        await rt.terminate()

    async def test_complaint_suppresses_and_expires(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime(
            decisions=[Decision(action="reply", text="接一句", confidence=0.9)]
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        rt.ingest(
            umo=UMO,
            group_id=GID,
            sender_id="10001",
            sender_name="群友甲",
            text="别插话了，太吵",
            directed=True,
            addressed_to_bot=True,
            self_id="2278168275",
        )
        state = rt._states[UMO]
        if state.pending is not None:
            state.pending.cancel()
            state.pending = None
        self.assertEqual(state.state, State.SUPPRESSED)
        await run_once(rt)
        self.assertEqual(executor.calls, [])
        clock.advance(1801)
        await add_messages(rt, clock, "新话题一", "新话题二", "新话题三")
        await run_once(rt)
        self.assertEqual(len(decider.calls), 1)
        self.assertEqual(len(executor.calls), 1)
        await rt.terminate()

    async def test_agent_busy_lease_expires(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime(
            decisions=[Decision(action="reply", text="接一句", confidence=0.9)]
        )
        rt.note_agent_begin(UMO, GID)
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(decider.calls, [])
        clock.advance(901)
        await add_messages(rt, clock, "新话题一", "新话题二", "新话题三")
        await run_once(rt)
        self.assertEqual(len(decider.calls), 1)
        self.assertEqual(len(executor.calls), 1)
        await rt.terminate()

    async def test_probe_does_not_send_or_advance_state(self) -> None:
        rt, clock, decider, executor, memory = make_runtime(
            decisions=[Decision(action="reply", text="试判", confidence=0.9)]
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        state = rt._states[UMO]
        before = state.episode.turn_count
        result = await rt.probe(UMO)
        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("would_send"))
        self.assertIn("episode_burst_limit", result)
        self.assertIn("episode_human_since_last_bot", result)
        self.assertEqual(executor.calls, [])
        self.assertEqual(memory, [])
        self.assertEqual(state.episode.turn_count, before)
        self.assertEqual(len(decider.calls), 1)
        await rt.terminate()

    async def test_stats_has_no_hourly_quota_fields(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime()
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        stats = rt.stats_snapshot()
        self.assertEqual(len(stats), 1)
        self.assertNotIn("interjects_last_hour", stats[0])
        self.assertNotIn("judges_last_hour", stats[0])
        self.assertIn("episode_turns", stats[0])
        self.assertIn("episode_extra_turns", stats[0])
        self.assertIn("episode_human_since_last_bot", stats[0])
        self.assertIn("episode_slow", stats[0])
        await rt.terminate()

    async def test_human_progress_resets_on_bot_and_episode_close(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime()
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        state = rt._states[UMO]
        self.assertEqual(state.episode.human_since_last_bot, 3)
        rt.record_bot_message(UMO, GID, "我接一句")
        self.assertEqual(state.episode.human_since_last_bot, 0)

        state.episode.turn_count = 4
        state.episode.next_extra_at = 0.0
        rt.record_bot_message(UMO, GID, "已经说满四句")
        self.assertGreaterEqual(state.episode.next_extra_at - clock.now, 600)

        state.episode.extra_turns = 2
        state.episode.human_since_last_bot = 9
        clock.advance(700)
        await add_messages(rt, clock, "新话题")
        self.assertEqual(state.episode.extra_turns, 0)
        self.assertEqual(state.episode.human_since_last_bot, 1)
        await rt.terminate()

    async def test_slow_gate_allows_extra_after_time_and_messages(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime(
            decisions=[Decision(action="reply", text=f"第{i}句", confidence=0.9) for i in range(6)]
        )
        for i in range(4):
            await add_messages(rt, clock, f"消息{i}a", f"消息{i}b", f"消息{i}c")
            await run_once(rt)
            if i < 3:
                clock.advance(300)
        state = rt._states[UMO]
        self.assertEqual(state.episode.turn_count, 4)
        self.assertEqual(state.episode.extra_turns, 0)
        self.assertGreater(state.episode.next_extra_at, clock.now)

        clock.advance(130)
        await add_messages(rt, clock, *[f"新消息{i}" for i in range(15)])
        await run_once(rt)
        self.assertEqual(len(executor.calls), 4)
        self.assertEqual(state.skip_events[-1][1], "episode_slow")

        clock.advance(300)
        await add_messages(rt, clock, "保持话题一")
        clock.advance(170)
        await add_messages(rt, clock, "保持话题二")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 5)
        self.assertEqual(state.episode.extra_turns, 1)
        self.assertGreaterEqual(state.episode.next_extra_at - clock.now, 899)

        clock.advance(130)
        await add_messages(rt, clock, *[f"又一波{i}" for i in range(15)])
        await run_once(rt)
        self.assertEqual(len(executor.calls), 5)
        self.assertEqual(state.skip_events[-1][1], "episode_slow")

        clock.advance(300)
        await add_messages(rt, clock, "保持话题三")
        clock.advance(470)
        await add_messages(rt, clock, "保持话题四")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 6)
        self.assertEqual(state.episode.extra_turns, 2)
        self.assertEqual(len(decider.calls), 6)
        await rt.terminate()

    async def test_update_config_backfills_missing_slow_anchor(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime()
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        state = rt._states[UMO]
        state.episode.turn_count = 4
        state.episode.next_extra_at = 0.0
        rt.update_config(Config(mode="active", group_whitelist=(GID,)))
        self.assertGreater(state.episode.next_extra_at, clock.now)
        self.assertGreaterEqual(
            state.episode.next_extra_at - clock.now,
            600 - 1,
        )
        await rt.terminate()

    async def test_terminate_cancels_pending(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime()
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        # add_messages cancels the pending task; schedule one again explicitly.
        rt._schedule(rt._states[UMO])
        await rt.terminate()
        self.assertEqual(rt._states, {})

    async def test_followup_requires_response(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime(
            decisions=[
                Decision(action="reply", text="第一句", confidence=0.9),
                Decision(action="reply", text="第二句", confidence=0.9),
            ]
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        clock.advance(50)
        await add_messages(rt, clock, "今天天气不错我们去公园吧")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(len(decider.calls), 1)
        await rt.terminate()

    async def test_followup_sends_when_responded(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime(
            decisions=[
                Decision(action="reply", text="第一句", confidence=0.9),
                Decision(action="reply", text="第二句", confidence=0.9),
            ]
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        clock.advance(50)
        await add_messages(rt, clock, "哈哈")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(len(decider.calls), 2)
        await rt.terminate()

    async def test_memory_failure_does_not_block_send(self) -> None:
        rt, clock, _decider, executor, _memory = make_runtime(
            decisions=[Decision(action="reply", text="好", confidence=0.9)]
        )

        async def broken_memory(*args) -> None:
            raise RuntimeError("memory down")

        rt._memory_writer = broken_memory
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(rt._states[UMO].episode.turn_count, 1)
        await rt.terminate()

    async def test_dry_run_does_not_publish_recent_interjections(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime(
            decisions=[Decision(action="reply", text="好", confidence=0.9)],
            dry_run=True,
        )
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        await run_once(rt)
        self.assertEqual(rt.recent_interjections(UMO), [])
        await rt.terminate()

    async def test_group_whitelist_ignores_other_groups(self) -> None:
        rt, clock, decider, executor, _memory = make_runtime()
        rt.ingest(
            umo="aiocqhttp:GroupMessage:999",
            group_id="999",
            sender_id="1",
            sender_name="别人",
            text="今天好累",
        )
        await asyncio.sleep(0)
        self.assertEqual(rt._states, {})
        self.assertEqual(decider.calls, [])
        self.assertEqual(executor.calls, [])
        await rt.terminate()

    async def test_probe_blocked_does_not_call_llm(self) -> None:
        rt, clock, decider, _executor, _memory = make_runtime()
        result = await rt.probe(UMO)
        self.assertFalse(result.get("ok"))
        self.assertEqual(decider.calls, [])
        await rt.terminate()

    async def test_stats_include_summary_fields(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime()
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        stats = rt.stats_snapshot()
        self.assertIn("cold_summaries", stats[0])
        self.assertIn("pending_spill", stats[0])
        self.assertIn("summary_calls", stats[0])
        await rt.terminate()

    async def test_update_config_disables_pending(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime()
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        rt._schedule(rt._states[UMO])
        rt.update_config(Config(enable=False, group_whitelist=(GID,)))
        self.assertIsNone(rt._states[UMO].pending)
        self.assertFalse(rt.config.enable)
        await rt.terminate()

    async def test_directed_context_runtime(self) -> None:
        rt, clock, _decider, _executor, _memory = make_runtime()
        await add_messages(rt, clock, "今天好累", "确实", "下班了")
        text = rt.directed_context(UMO, exclude_last=True)
        self.assertIsInstance(text, str)
        await rt.terminate()


if __name__ == "__main__":
    unittest.main()
