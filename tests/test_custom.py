"""Custom preset-override tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2 import webui  # noqa: E402
from astrbot_plugin_group_interject_v2.model import (  # noqa: E402
    PRESETS,
    Config,
)


class TestCustomPreset(unittest.TestCase):
    def test_disabled_keeps_preset(self) -> None:
        cfg = Config.from_mapping(
            {
                "mode": "active",
                "custom": {
                    "enabled": False,
                    "window_size": 12,
                    "min_messages": 1,
                },
            }
        )
        self.assertEqual(cfg.preset, PRESETS["active"])
        self.assertEqual(cfg.preset.window_size, 30)

    def test_enabled_overrides_only_core_fields(self) -> None:
        cfg = Config.from_mapping(
            {
                "mode": "active",
                "custom": {
                    "enabled": True,
                    "window_size": 20,
                    "window_minutes": 8,
                    "min_messages": 1,
                    "cooldown_seconds": 120,
                    "judge_cooldown_seconds": 30,
                    "min_confidence": 0.4,
                    "episode_max_turns": 5,
                    "episode_rest_seconds": 120,
                    "episode_reentry_messages": 7,
                },
            }
        )
        preset = cfg.preset
        self.assertEqual(preset.window_size, 20)
        self.assertEqual(preset.window_minutes, 8)
        self.assertEqual(preset.min_messages, 1)
        self.assertEqual(preset.cooldown_seconds, 120)
        self.assertEqual(preset.judge_cooldown_seconds, 30)
        self.assertAlmostEqual(preset.min_confidence, 0.4)
        self.assertEqual(preset.episode_max_turns, 5)
        self.assertEqual(preset.episode_rest_seconds, 120)
        self.assertEqual(preset.episode_reentry_messages, 7)
        # Internal pacing still comes from the selected preset.
        self.assertEqual(preset.debounce_seconds, PRESETS["active"].debounce_seconds)
        self.assertEqual(
            preset.followup_cooldown_seconds,
            PRESETS["active"].followup_cooldown_seconds,
        )

    def test_values_are_clamped_for_runtime(self) -> None:
        cfg = Config.from_mapping(
            {
                "custom": {
                    "enabled": True,
                    "window_size": 999,
                    "min_messages": 99,
                    "min_confidence": -1,
                    "episode_rest_seconds": 1,
                    "episode_reentry_messages": 999,
                }
            }
        )
        self.assertEqual(cfg.preset.window_size, 120)
        self.assertEqual(cfg.preset.min_messages, 10)
        self.assertAlmostEqual(cfg.preset.min_confidence, 0.2)
        self.assertEqual(cfg.preset.episode_rest_seconds, 60)
        self.assertEqual(cfg.preset.episode_reentry_messages, 100)

    def test_default_config_unchanged(self) -> None:
        cfg = Config()
        self.assertFalse(cfg.custom.enabled)
        self.assertEqual(cfg.preset, PRESETS["balanced"])


class TestCustomWebui(unittest.TestCase):
    def test_nested_payload(self) -> None:
        payload = {
            "mode": "balanced",
            "custom": {
                "enabled": True,
                "window_size": 20,
                "window_minutes": 8,
                "min_messages": 2,
                "cooldown_seconds": 120,
                "judge_cooldown_seconds": 30,
                "min_confidence": 0.4,
                "episode_max_turns": 4,
                "episode_rest_seconds": 120,
                "episode_reentry_messages": 8,
            },
        }
        values, errors = webui.coerce_payload(payload)
        self.assertEqual(errors, [])
        self.assertEqual(values["custom"]["min_messages"], 2)
        self.assertEqual(values["custom"]["window_size"], 20)
        self.assertEqual(values["custom"]["episode_rest_seconds"], 120)
        self.assertEqual(values["custom"]["episode_reentry_messages"], 8)
        self.assertTrue(values["custom"]["enabled"])

    def test_nested_range_and_cross_errors(self) -> None:
        values, errors = webui.coerce_payload(
            {
                "custom": {
                    "enabled": True,
                    "window_size": 5,
                    "min_messages": 10,
                    "min_confidence": 2,
                }
            }
        )
        self.assertNotIn("custom", values)
        self.assertTrue(any("window_size" in item for item in errors))
        self.assertTrue(any("min_confidence" in item for item in errors))

        _values, errors = webui.coerce_payload(
            {
                "custom": {
                    "enabled": True,
                    "window_size": 20,
                    "min_messages": 10,
                }
            }
        )
        self.assertEqual(errors, [])

    def test_nested_unknown_key(self) -> None:
        values, errors = webui.coerce_payload(
            {"custom": {"enabled": True, "nope": 1}}
        )
        self.assertNotIn("custom", values)
        self.assertTrue(any("未知参数" in item for item in errors))

    def test_nested_defaults(self) -> None:
        merged = webui.merge_with_defaults(
            {"custom": {"enabled": True, "window_size": 22}}
        )
        self.assertEqual(merged["custom"]["window_size"], 22)
        self.assertEqual(merged["custom"]["window_minutes"], 15)
        self.assertEqual(merged["custom"]["min_confidence"], 0.55)
        self.assertEqual(merged["custom"]["episode_rest_seconds"], 900)
        self.assertEqual(merged["custom"]["episode_reentry_messages"], 25)


if __name__ == "__main__":
    unittest.main()
