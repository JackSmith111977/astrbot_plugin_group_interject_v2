"""仓库合规守卫：市场元数据一致性 + 红线静态扫描。

对应 AstrBot 官方 CI 的关键检查与插件市场审核红线：
- metadata 四必填、版本三方一致、名称与仓库目录一致；
- 禁止内部部署路径、禁止 requests、禁止 eval/exec 动态执行；
- 配置 schema 与元数据可解析。

本文件不依赖 astrbot 与 PyYAML，可在任意 Python 环境运行。
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".html", ".js", ".css"}
REQUIRED_FIELDS = ("name", "display_name", "desc", "version", "author", "repo")

BANNED_SNIPPETS = (
    "/opt/qqbot",
    "deploy/deploy_group_interject",
    "docs/plugins/",
    "docs/research/",
)
BANNED_CODE = (
    "from astrbot import",
    "import requests",
    "from requests import",
)
DYNAMIC_EXEC = re.compile(r"\b(?:eval|exec)\s*\(")

ALLOWED_PLATFORMS = {
    "aiocqhttp",
    "qq_official",
    "qq_official_webhook",
    "telegram",
    "wecom",
    "wecom_ai_bot",
    "lark",
    "dingtalk",
    "discord",
    "slack",
    "kook",
    "vocechat",
    "weixin_official_account",
    "weixin_oc",
    "satori",
    "misskey",
    "line",
    "matrix",
    "mattermost",
}


def _metadata_text() -> str:
    return (ROOT / "metadata.yaml").read_text(encoding="utf-8")


def _field(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}:[ \t]*(.*)$", _metadata_text(), re.MULTILINE
    )
    return match.group(1).strip() if match else ""


def _iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


class MetadataTests(unittest.TestCase):
    def test_required_fields(self) -> None:
        for name in REQUIRED_FIELDS:
            self.assertTrue(_field(name), f"metadata.yaml 缺少必填字段 {name}")

    def test_name_matches_directory(self) -> None:
        self.assertEqual(_field("name"), ROOT.name)

    def test_version_is_semver(self) -> None:
        self.assertRegex(_field("version"), r"^\d+\.\d+\.\d+$")

    def test_version_matches_entrypoint(self) -> None:
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        match = re.search(r'^PLUGIN_VERSION\s*=\s*"([^"]+)"', main, re.MULTILINE)
        self.assertIsNotNone(match, "main.py 缺少 PLUGIN_VERSION")
        self.assertEqual(_field("version"), match.group(1))

    def test_repo_url(self) -> None:
        repo = _field("repo")
        self.assertTrue(repo.startswith("https://github.com/"), repo)
        self.assertTrue(repo.endswith("/" + ROOT.name), repo)

    def test_astrbot_version_pep440(self) -> None:
        spec = _field("astrbot_version").strip('"').strip("'")
        self.assertTrue(spec, "metadata.yaml 缺少 astrbot_version")
        self.assertTrue(spec.startswith((">=", ">", "~=", "==")), spec)
        self.assertFalse(spec.startswith("v"), spec)

    def test_support_platforms_keys(self) -> None:
        match = re.search(
            r"^support_platforms:\s*\n((?:[ \t]*-[ \t]*\S+[ \t]*\n)+)",
            _metadata_text(),
            re.MULTILINE,
        )
        self.assertIsNotNone(match, "support_platforms 未声明")
        values = re.findall(r"-[ \t]*(\S+)", match.group(1))
        self.assertTrue(values, "support_platforms 为空")
        for value in values:
            self.assertIn(value, ALLOWED_PLATFORMS, f"未知平台 key: {value}")

    def test_logo_declared_and_present(self) -> None:
        logo = _field("logo")
        self.assertTrue(logo, "metadata.yaml 缺少 logo")
        self.assertTrue((ROOT / logo).is_file(), "logo 文件不存在")


class SchemaTests(unittest.TestCase):
    def test_schema_fields(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        required = {
            "enable",
            "mode",
            "dry_run",
            "provider_id",
            "image_caption_enable",
            "memory_bridge_enable",
            "group_whitelist",
            "group_blacklist",
            "ignore_senders",
            "wake_names",
            "quiet_hours",
            "extra_rules",
            "custom",
        }
        self.assertEqual(set(schema), required)
        self.assertIs(schema["image_caption_enable"]["default"], True)
        self.assertIs(schema["memory_bridge_enable"]["default"], True)
        self.assertIs(schema["dry_run"]["default"], True)


class RedLineTests(unittest.TestCase):
    def test_no_internal_paths(self) -> None:
        for path in _iter_text_files():
            if path == SELF:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for snippet in BANNED_SNIPPETS:
                self.assertNotIn(
                    snippet,
                    text,
                    f"{path.relative_to(ROOT)} 含内部信息：{snippet}",
                )

    def test_no_banned_code(self) -> None:
        for path in ROOT.rglob("*.py"):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path == SELF:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for snippet in BANNED_CODE:
                self.assertNotIn(
                    snippet,
                    text,
                    f"{path.relative_to(ROOT)} 命中红线写法：{snippet}",
                )
            self.assertIsNone(
                DYNAMIC_EXEC.search(text),
                f"{path.relative_to(ROOT)} 使用了动态执行 eval/exec",
            )


if __name__ == "__main__":
    unittest.main()
