"""@bot-only directed detection and context projection tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2.context import (  # noqa: E402
    render_directed,
    render_summaries,
)
from astrbot_plugin_group_interject_v2.model import (  # noqa: E402
    DIRECTED_CONTEXT_MAX_CHARS,
    Episode,
    Msg,
    SummaryRecord,
)
from astrbot_plugin_group_interject_v2.text import (  # noqa: E402
    format_message_chain,
    has_bot_reference,
    is_poke_at_other,
    poke_event,
    raw_poke_event,
)

NOW = 1_700_000_000.0
SELF_ID = "2278168275"


class _Type:
    def __init__(self, value: str) -> None:
        self.value = value


class Comp:
    def __init__(self, kind: str, **kwargs) -> None:
        self.type = _Type(kind)
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestDirectedDetection(unittest.TestCase):
    def test_at_another_member_is_not_directed(self) -> None:
        chain = [
            Comp("At", qq="123456", name="别人"),
            Comp("Plain", text=" 你好"),
        ]
        self.assertFalse(has_bot_reference(chain, SELF_ID))
        text, directed = format_message_chain(chain, SELF_ID)
        self.assertFalse(directed)
        self.assertIn("[@别人(别人)]", text)

    def test_at_bot_is_directed(self) -> None:
        chain = [
            Comp("At", qq=SELF_ID, name="鱼片"),
            Comp("Plain", text=" 在吗"),
        ]
        self.assertTrue(has_bot_reference(chain, SELF_ID))
        _text, directed = format_message_chain(chain, SELF_ID)
        self.assertTrue(directed)

    def test_quote_bot_is_directed(self) -> None:
        chain = [
            Comp("Reply", sender_id=SELF_ID, message_str="上一句"),
            Comp("Plain", text=" 接着聊"),
        ]
        self.assertTrue(has_bot_reference(chain, SELF_ID))
        _text, directed = format_message_chain(chain, SELF_ID)
        self.assertTrue(directed)

    def test_quote_other_is_not_directed(self) -> None:
        chain = [
            Comp("Reply", sender_id="123456", message_str="别人说的"),
            Comp("Plain", text=" 回复"),
        ]
        self.assertFalse(has_bot_reference(chain, SELF_ID))
        _text, directed = format_message_chain(chain, SELF_ID)
        self.assertFalse(directed)

    def test_poke_other_is_labeled_and_not_directed(self) -> None:
        text, directed = format_message_chain(
            [Comp("Poke", id="123456")],
            SELF_ID,
        )
        self.assertFalse(directed)
        self.assertEqual(text, "[戳了 @123456（别人）]")

    def test_poke_bot_is_labeled_you(self) -> None:
        text, directed = format_message_chain(
            [Comp("Poke", id=SELF_ID)],
            SELF_ID,
        )
        self.assertFalse(directed)
        self.assertEqual(text, "[戳了你]")

    def test_poke_without_target_is_unknown(self) -> None:
        text, _directed = format_message_chain([Comp("Poke")], SELF_ID)
        self.assertEqual(text, "[戳一戳（目标未知）]")

    def test_poke_helpers(self) -> None:
        self.assertEqual(
            poke_event([Comp("Poke", id="123456")]),
            (True, "123456"),
        )
        self.assertEqual(poke_event([Comp("Plain", text="hi")]), (False, ""))
        self.assertEqual(
            raw_poke_event(
                {
                    "post_type": "notice",
                    "sub_type": "poke",
                    "target_id": "123456",
                }
            ),
            (True, "123456"),
        )
        self.assertEqual(raw_poke_event({"sub_type": "poke"}), (True, ""))
        self.assertEqual(raw_poke_event(None), (False, ""))
        self.assertTrue(is_poke_at_other(True, "123456", SELF_ID))
        self.assertFalse(is_poke_at_other(True, SELF_ID, SELF_ID))
        self.assertTrue(is_poke_at_other(True, "", SELF_ID))
        self.assertFalse(is_poke_at_other(False, "", SELF_ID))


class TestDirectedContext(unittest.TestCase):
    def test_excludes_current_message_and_includes_cold(self) -> None:
        episode = Episode()
        episode.start(NOW)
        episode.cold.append(
            SummaryRecord(
                text="较早话题：在聊股票。",
                created_at=NOW,
                covered_from_ts=NOW - 100,
                covered_to_ts=NOW - 90,
                source_count=5,
            )
        )
        window = [
            Msg(NOW - 20, "1", "甲", "最近一句"),
            Msg(NOW - 10, "2", "乙", "当前 @bot", directed=True),
        ]
        rendered = render_directed(
            episode,
            window,
            self_lines=["我刚才说的话"],
            exclude_last=True,
        )
        self.assertIn("较早话题", rendered)
        self.assertNotIn("当前 @bot", rendered)
        self.assertIn("我刚才说的话", rendered)
        self.assertLessEqual(len(rendered), DIRECTED_CONTEXT_MAX_CHARS)

    def test_summary_budget_keeps_newest(self) -> None:
        records = [
            SummaryRecord(
                text=f"旧摘要{i}-" + "x" * 400,
                created_at=NOW + i,
                covered_from_ts=NOW,
                covered_to_ts=NOW + 1,
                source_count=1,
            )
            for i in range(3)
        ]
        rendered = render_summaries(records, max_chars=600)
        self.assertLessEqual(len(rendered), 600 + 20)
        self.assertIn("旧摘要2", rendered)


if __name__ == "__main__":
    unittest.main()
