"""Static checks for the config/monitor control view structure."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2 import webui  # noqa: E402

VIEW = _PLUGIN_ROOT / "views" / "status"


class TestViewStructure(unittest.TestCase):
    def test_config_is_grouped_into_tabs_and_sections(self) -> None:
        html = (VIEW / "index.html").read_text(encoding="utf-8")
        js = (VIEW / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="config-tabs"', html)
        self.assertIn('id="config-panes"', html)
        self.assertNotIn("config-grid", html)
        self.assertNotIn("state.loaded", js)
        self.assertNotIn("const bridge = window.AstrBotPluginView", js)
        self.assertIn("waitForBridge", js)
        self.assertIn("DOMContentLoaded", js)
        for label in ("运行", "范围", "交互"):
            self.assertIn(f'label: "{label}"', js)
        for field in webui.load_schema():
            self.assertIn(field, js)
        for marker in ("renderObjectField", "custom-tuning", "fillCustomFromPreset"):
            self.assertIn(marker, js)
        for marker in ("episode_extra_turns", "episode_reentry_messages_remaining"):
            self.assertIn(marker, js)

    def test_monitor_still_has_summary_and_groups(self) -> None:
        html = (VIEW / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="summary"', html)
        self.assertIn('id="groups"', html)
        self.assertIn('id="probe"', html)


if __name__ == "__main__":
    unittest.main()
