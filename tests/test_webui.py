"""WebUI config validation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2 import webui  # noqa: E402


class TestSchema(unittest.TestCase):
    def test_public_fields_and_custom_object(self) -> None:
        schema = webui.load_schema()
        self.assertEqual(
            set(schema),
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
        fields = webui.schema_fields(schema)
        mode = next(item for item in fields if item["key"] == "mode")
        self.assertEqual(mode["options"], ["quiet", "balanced", "active"])
        provider = next(item for item in fields if item["key"] == "provider_id")
        self.assertEqual(provider["special"], "select_provider")
        custom = next(item for item in fields if item["key"] == "custom")
        self.assertEqual(custom["type"], "object")
        self.assertEqual(len(custom["items"]), 10)
        by_key = {item["key"]: item for item in custom["items"]}
        self.assertEqual(by_key["min_confidence"]["min"], 0.2)
        self.assertEqual(by_key["window_size"]["max"], 120)
        self.assertEqual(by_key["episode_rest_seconds"]["min"], 60)
        self.assertEqual(by_key["episode_reentry_messages"]["max"], 100)

    def test_merge_defaults(self) -> None:
        values = webui.merge_with_defaults({"mode": "quiet"})
        self.assertEqual(values["mode"], "quiet")
        self.assertEqual(values["dry_run"], True)


class TestCoerce(unittest.TestCase):
    def test_full_payload(self) -> None:
        values, errors = webui.coerce_payload(
            {
                "enable": "true",
                "mode": "active",
                "dry_run": False,
                "provider_id": "p1",
                "group_whitelist": "100000001, 123",
                "group_blacklist": ["999"],
                "ignore_senders": [1, "2"],
                "wake_names": "鱼片\n鲸鲸",
                "quiet_hours": "01:00-08:00",
                "extra_rules": "短句",
            }
        )
        self.assertEqual(errors, [])
        self.assertEqual(values["group_whitelist"], ["100000001", "123"])
        self.assertEqual(values["group_blacklist"], ["999"])
        self.assertEqual(values["ignore_senders"], ["1", "2"])
        self.assertEqual(values["wake_names"], ["鱼片", "鲸鲸"])
        self.assertEqual(values["mode"], "active")
        self.assertFalse(values["dry_run"])

    def test_unknown_key(self) -> None:
        _values, errors = webui.coerce_payload({"nope": 1})
        self.assertTrue(errors)

    def test_invalid_mode_and_quiet_hours(self) -> None:
        _values, errors = webui.coerce_payload({"mode": "ultra"})
        self.assertTrue(any("mode" in item for item in errors))
        _values, errors = webui.coerce_payload({"quiet_hours": "25:00-26:00"})
        self.assertTrue(any("quiet_hours" in item for item in errors))

    def test_empty_and_bad_types(self) -> None:
        _values, errors = webui.coerce_payload({"group_whitelist": 1})
        self.assertTrue(errors)
        values, errors = webui.coerce_payload({"provider_id": None})
        self.assertEqual(errors, [])
        self.assertEqual(values["provider_id"], "")


if __name__ == "__main__":
    unittest.main()
