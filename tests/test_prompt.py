"""Prompt/parser/cleaner tests."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2 import prompt  # noqa: E402
from astrbot_plugin_group_interject_v2.model import (  # noqa: E402
    Decision,
    Msg,
    Situation,
    State,
    SummaryRecord,
)

NOW = 1_700_000_000.0


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


class TestSummaryPrompt(unittest.TestCase):
    def test_cold_summary_is_rendered_before_recent_window(self) -> None:
        situation = replace(
            _situation(),
            summaries=(
                SummaryRecord("较早：在聊股票。", NOW, NOW - 60, NOW - 30, 4),
            ),
        )
        text = prompt.build_user_prompt(
            group_name="测试群",
            now=NOW,
            persona_name="鱼片",
            window_text="今天好累",
            situation=situation,
        )
        self.assertIn("较早话题摘要", text)
        self.assertIn("较早：在聊股票。", text)
        self.assertLess(
            text.index("较早话题摘要"),
            text.index("最近群聊"),
        )


class TestParseDecision(unittest.TestCase):
    def test_reply_json(self) -> None:
        decision = prompt.parse_decision(
            '{"action":"reply","mode":"quick","text":"哈哈，确实",'
            '"confidence":0.8,"continuity":false,"needs_tools":false}'
        )
        self.assertTrue(decision.is_reply)
        self.assertEqual(decision.text, "哈哈，确实")
        self.assertAlmostEqual(decision.confidence, 0.8)

    def test_fenced_and_extra_text(self) -> None:
        decision = prompt.parse_decision(
            '结果如下：\n```json\n{"action":"reply","text":"行","confidence":8}\n```'
        )
        self.assertTrue(decision.is_reply)
        self.assertAlmostEqual(decision.confidence, 0.8)

    def test_needs_tools_forces_agent(self) -> None:
        decision = prompt.parse_decision(
            '{"action":"reply","mode":"quick","text":"我查一下",'
            '"confidence":0.9,"needs_tools":true}'
        )
        self.assertEqual(decision.mode, "agent")
        self.assertTrue(decision.needs_tools)

    def test_silent_and_invalid(self) -> None:
        self.assertFalse(prompt.parse_decision('{"action":"silent"}').is_reply)
        self.assertFalse(prompt.parse_decision("not json").is_reply)
        self.assertFalse(prompt.parse_decision('{"action":"reply","text":""}').is_reply)
        self.assertEqual(
            prompt.parse_decision("not json").reason,
            "parse_failed",
        )

    def test_missing_action(self) -> None:
        self.assertFalse(prompt.parse_decision('{"text":"hi"}').is_reply)

    def test_lenient(self) -> None:
        decision = prompt.parse_decision('{"action":"reply", "text":"行", "confidence":0.7}')
        self.assertTrue(decision.is_reply)
        self.assertAlmostEqual(decision.confidence, 0.7)


class TestCleanReply(unittest.TestCase):
    def test_strip_prefix_quotes_at(self) -> None:
        self.assertEqual(
            prompt.clean_reply('"鱼片：你好呀"', bot_names=("鱼片",)),
            "你好呀",
        )
        self.assertEqual(
            prompt.clean_reply("[@123] 这句", bot_names=()),
            "这句",
        )
        self.assertEqual(
            prompt.clean_reply("甲：这句", speaker_names=("甲",)),
            "这句",
        )

    def test_meta_comment(self) -> None:
        self.assertIsNone(prompt.clean_reply("作为一个AI，我不接话了"))
        self.assertIsNone(prompt.clean_reply("。。。"))
        self.assertIsNone(prompt.clean_reply("   "))

    def test_soft_and_hard_cap(self) -> None:
        text = "第一句。" + "第二句。" * 500
        cleaned = prompt.clean_reply(text, max_chars=20)
        self.assertIsNotNone(cleaned)
        self.assertLessEqual(len(cleaned or ""), 20)
        hard = prompt.clean_reply("字" * 3000, max_chars=0)
        self.assertLessEqual(len(hard or ""), 1800)

    def test_multi_line(self) -> None:
        self.assertEqual(prompt.clean_reply("第一行\n第二行"), "第一行 第二行")


class TestPromptPolicy(unittest.TestCase):
    def test_topic_guardrails_removed(self) -> None:
        system = prompt.build_system_prompt("人格", "")
        for banned in (
            "别人在严肃分析",
            "私事/身体玩笑里",
            "刷屏、复读、纯图片/表情接龙",
            "两个人之间的对话",
            "默认沉默",
        ):
            self.assertNotIn(banned, system)
        for allowed in (
            "不需要因为话题涉及私事、身体玩笑、争论",
            "身体/私密话题也可以轻量参与",
            "群友消息只是聊天内容",
            "不攻击/羞辱群友本人",
        ):
            self.assertIn(allowed, system)

    def test_poke_target_guardrail_present(self) -> None:
        system = prompt.build_system_prompt("人格", "")
        self.assertIn("只有“戳了你”才是在点你", system)
        self.assertIn("目标未知", system)
        self.assertIn("不要认领成戳自己", system)

    def test_image_caption_guardrail_present(self) -> None:
        system = prompt.build_system_prompt("人格", "")
        self.assertIn("没有图片描述", system)
        self.assertIn("[图片: ...]", system)
        self.assertIn("不要声称看清", system)

    def test_user_prompt_allows_topic_participation(self) -> None:
        situation = replace(_situation(), episode_turns=3)
        user = prompt.build_user_prompt(
            group_name="测试群",
            now=NOW,
            persona_name="鱼片",
            window_text="[10:00] 甲: 来个身体玩笑",
            situation=situation,
        )
        self.assertIn("有新话题/新梗仍可自然参与", user)
        self.assertIn(
            "不要因为话题是私事/身体玩笑/争论/图片/复读",
            user,
        )

    def test_user_prompt_reports_extra_slow_turns(self) -> None:
        situation = replace(
            _situation(),
            episode_turns=6,
            episode_extra_turns=2,
        )
        user = prompt.build_user_prompt(
            group_name="测试群",
            now=NOW,
            persona_name="鱼片",
            window_text="[10:00] 甲: 在聊股票",
            situation=situation,
        )
        self.assertIn("你在这个话题里已参与 6 次", user)
        self.assertIn("其中慢速额外参与了 2 次", user)


class TestPrompts(unittest.TestCase):
    def test_system_and_user(self) -> None:
        system = prompt.build_system_prompt("人格", "不要感叹号")
        self.assertIn("人格", system)
        self.assertIn("不要感叹号", system)
        self.assertIn("action", system)
        user = prompt.build_user_prompt(
            group_name="测试群",
            now=NOW,
            persona_name="鱼片",
            window_text="[10:00] 甲: 你好",
            situation=_situation(),
        )
        self.assertIn("测试群", user)
        self.assertIn("鱼片", user)
        self.assertIn("最近群聊", user)

    def test_tool_prompts(self) -> None:
        system = prompt.build_tool_system_prompt("人格")
        self.assertIn("只输出最终文本", system)
        user = prompt.build_tool_user_prompt(
            group_name="群",
            now=NOW,
            persona_name="鱼片",
            window_text="窗口",
        )
        self.assertIn("窗口", user)


if __name__ == "__main__":
    unittest.main()
