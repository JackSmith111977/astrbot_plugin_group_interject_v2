"""Core model and text helper tests (no AstrBot imports)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2 import text  # noqa: E402
from astrbot_plugin_group_interject_v2.ingress import Ingress  # noqa: E402
from astrbot_plugin_group_interject_v2.model import (  # noqa: E402
    Config,
    Decision,
    Episode,
    Msg,
    State,
    group_filter_reason,
    non_bot_count,
    slow_rest_seconds,
)

NOW = 1_700_000_000.0


class _FakeType:
    def __init__(self, value: str) -> None:
        self.value = value


class Comp:
    def __init__(self, kind: str, **kwargs) -> None:
        self.type = _FakeType(kind)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Image:
    def __init__(self, url: str = "", file: str = "") -> None:
        self.url = url
        self.file = file


class _MessageEvent:
    def __init__(self, components: list[object]) -> None:
        self._components = components

    def get_messages(self) -> list[object]:
        return self._components


class TestConfig(unittest.TestCase):
    def test_defaults_have_no_quota_fields(self) -> None:
        keys = set(asdict(Config()).keys())
        self.assertEqual(
            keys,
            {
                "enable",
                "mode",
                "dry_run",
                "provider_id",
                "group_whitelist",
                "group_blacklist",
                "ignore_senders",
                "wake_names",
                "quiet_hours",
                "extra_rules",
                "image_caption_enable",
                "memory_bridge_enable",
                "custom",
            },
        )
        self.assertNotIn("max_interjects_per_hour", keys)
        self.assertNotIn("judge_max_per_hour", keys)

    def test_privacy_toggles(self) -> None:
        cfg = Config()
        self.assertTrue(cfg.image_caption_enable)
        self.assertTrue(cfg.memory_bridge_enable)
        off = Config.from_mapping(
            {"image_caption_enable": False, "memory_bridge_enable": "off"}
        )
        self.assertFalse(off.image_caption_enable)
        self.assertFalse(off.memory_bridge_enable)

    def test_mode_presets(self) -> None:
        self.assertEqual(Config(mode="balanced").preset.cooldown_seconds, 420)
        self.assertLess(
            Config(mode="active").preset.min_messages,
            Config(mode="quiet").preset.min_messages,
        )
        self.assertEqual(Config(mode="active").preset.episode_rest_seconds, 600)
        self.assertEqual(Config(mode="active").preset.episode_reentry_messages, 15)
        self.assertEqual(Config(mode="balanced").preset.episode_rest_seconds, 900)
        self.assertEqual(Config(mode="quiet").preset.episode_reentry_messages, 40)
        self.assertEqual(Config.from_mapping({"mode": "ACTIVE"}).mode, "active")
        self.assertEqual(Config.from_mapping({"mode": "nope"}).mode, "balanced")

    def test_mapping_and_group_filter(self) -> None:
        cfg = Config.from_mapping(
            {
                "enable": "false",
                "group_whitelist": "100000001, 100000002",
                "group_blacklist": ["100000002"],
                "ignore_senders": [123],
                "wake_names": ["鱼片", "鲸鲸"],
            }
        )
        self.assertFalse(cfg.enable)
        self.assertEqual(cfg.wake_names, ("鱼片", "鲸鲸"))
        self.assertEqual(group_filter_reason(cfg, "100000002", "x"), "blacklist")
        self.assertEqual(group_filter_reason(cfg, "100000001", "x"), None)
        self.assertEqual(group_filter_reason(cfg, "1", "x"), "not_in_whitelist")

    def test_slow_rest_escalates_and_caps(self) -> None:
        self.assertEqual(slow_rest_seconds(600, 0), 600)
        self.assertEqual(slow_rest_seconds(600, 1), 900)
        self.assertEqual(slow_rest_seconds(600, 5), 1800)
        self.assertEqual(slow_rest_seconds(1800, 3), 1800)
        self.assertEqual(slow_rest_seconds(3600, 3), 3600)

    def test_episode_helpers(self) -> None:
        episode = Episode()
        episode.start(NOW)
        self.assertEqual(episode.extra_turns, 0)
        self.assertEqual(episode.next_extra_at, 0.0)
        self.assertEqual(episode.human_since_last_bot, 0)
        episode.human_since_last_bot = 5
        episode.mark_response(NOW + 1)
        self.assertTrue(episode.responded)
        episode.on_send(NOW + 2, "hi")
        self.assertEqual(episode.turn_count, 1)
        self.assertEqual(episode.human_since_last_bot, 0)
        self.assertTrue(episode.followup_active(NOW + 10, 150))
        self.assertFalse(episode.followup_active(NOW + 200, 150))
        episode.close(NOW + 300)
        self.assertFalse(episode.active)
        self.assertEqual(episode.remaining_turns(3), 2)

    def test_decision_silent(self) -> None:
        self.assertFalse(Decision.silent("x").is_reply)
        self.assertTrue(Decision(action="reply", text="hi").is_reply)
        self.assertFalse(Decision(action="reply", text="  ").is_reply)


class TestText(unittest.TestCase):
    def test_format_message_chain_directed_and_other(self) -> None:
        chain = [
            Comp("Plain", text="你好"),
            Comp("At", qq="2278168275", name="鱼片"),
            Comp("At", qq="1234", name="别人"),
        ]
        rendered, directed = text.format_message_chain(chain, "2278168275")
        self.assertTrue(directed)
        self.assertIn("[@鱼片]", rendered)
        self.assertIn("[@别人(别人)]", rendered)

    def test_quote_to_bot_is_directed(self) -> None:
        rendered, directed = text.format_message_chain(
            [Comp("Reply", sender_id="2278168275", message_str="上一句")],
            "2278168275",
        )
        self.assertTrue(directed)
        self.assertIn("引用 你", rendered)

    def test_window_format(self) -> None:
        messages = [
            Msg(NOW, "1", "甲", "第一句"),
            Msg(NOW + 1, "", "", "我接过一句", is_bot=True),
        ]
        rendered = text.format_window(messages)
        self.assertIn("甲#1", rendered)
        self.assertIn("你:", rendered)
        self.assertEqual(non_bot_count(messages), 1)

    def test_noise_and_wake(self) -> None:
        repeated = [Msg(NOW + i, "1", "甲", "复读") for i in range(3)]
        self.assertTrue(text.looks_like_noise(repeated))
        self.assertIsNone(text.match_wake_name("鱼片昨天说", ["鱼片"]))
        self.assertEqual(text.match_wake_name("鱼片，在吗", ["鱼片"]), "鱼片")
        self.assertEqual(text.match_wake_name("[引用 你] 鱼片 在吗", ["鱼片"]), "鱼片")

    def test_quiet_hours(self) -> None:
        # 2023-11-14 23:00 UTC-ish; use local construction-independent check.
        self.assertTrue(text.in_quiet_hours("00:00-23:59", NOW))
        self.assertFalse(text.in_quiet_hours("", NOW))
        self.assertEqual(len(text.parse_quiet_hours("01:00-02:00, 23:00-00:30")), 2)

    def test_bare_at_context(self) -> None:
        messages = [
            Msg(NOW, "1", "甲", "今天好累"),
            Msg(NOW + 1, "", "", "我也有点", is_bot=True),
        ]
        context = text.build_bare_at_context(messages, max_messages=6)
        self.assertIn("今天好累", context)
        self.assertIn("你：我也有点", context)

    def test_window_and_bare_at_include_image_caption(self) -> None:
        messages = [
            Msg(NOW, "1", "甲", "[图片]", image_caption="一只猫趴在键盘上"),
            Msg(NOW + 1, "2", "乙", "好好玩"),
        ]
        rendered = text.format_window(messages)
        self.assertIn("[图片: 一只猫趴在键盘上]", rendered)
        context = text.build_bare_at_context(messages)
        self.assertIn("一只猫趴在键盘上", context)
        self.assertEqual(
            text.message_body(messages[1]),
            "好好玩",
        )

    def test_image_reference_keywords(self) -> None:
        keywords = ("图呢", "照片", "截屏", "刚发的")
        self.assertTrue(text.looks_like_image_reference("图呢", keywords))
        self.assertTrue(text.looks_like_image_reference("刚发的照片", keywords))
        self.assertFalse(text.looks_like_image_reference("今天行情怎么样", keywords))

    def test_collect_image_refs(self) -> None:
        ingress = Ingress(Config())
        event = _MessageEvent(
            [
                Image(url="u1"),
                Image(url="", file="f2"),
                Image(url="u1"),
                Image(url="u3"),
            ]
        )
        self.assertEqual(ingress.collect_image_refs(event), ("u1", "f2"))
        self.assertEqual(
            ingress.collect_image_refs(_MessageEvent([Image(url="u1")]), limit=1),
            ("u1",),
        )
        self.assertEqual(ingress.collect_image_refs(_MessageEvent([])), ())

    def test_force_attach_cached_image_for_empty_at(self) -> None:
        class FakeCompImage:
            def __init__(self, url: str = "", file: str = "") -> None:
                self.url = url
                self.file = file

            @classmethod
            def fromURL(cls, url: str) -> "FakeCompImage":
                return cls(url=url)

            @classmethod
            def fromFileSystem(cls, path: str) -> "FakeCompImage":
                return cls(file=path)

        class FakeEvent:
            def __init__(self) -> None:
                self.message_obj = SimpleNamespace(message=[])
                self.message_str = ""
                self.unified_msg_origin = "umo"

            def get_messages(self) -> list[object]:
                return []

        ingress = Ingress(Config())
        ingress.cache_image("umo", url="https://example.test/a.png")
        event = FakeEvent()
        with patch.dict(
            sys.modules,
            {
                "astrbot.api.message_components": SimpleNamespace(
                    Image=FakeCompImage
                )
            },
        ):
            self.assertTrue(
                ingress.attach_cached_image(
                    event,
                    "umo",
                    "[只 @ 了你]",
                    force=True,
                )
            )
        self.assertEqual(len(event.message_obj.message), 1)
        self.assertIn("刚才消息里的图片", event.message_str)


if __name__ == "__main__":
    unittest.main()
