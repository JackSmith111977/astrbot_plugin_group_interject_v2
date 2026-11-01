"""Image-caption pipeline tests (no AstrBot imports)."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))

from astrbot_plugin_group_interject_v2.caption import (  # noqa: E402
    ImageCaptioner,
    clean_caption,
)
from astrbot_plugin_group_interject_v2.model import Msg  # noqa: E402

UMO = "aiocqhttp:GroupMessage:100000001"
NOW = 1_700_000_000.0


def _msg(*refs: str) -> Msg:
    return Msg(NOW, "1", "甲", "[图片]", image_refs=tuple(refs))


async def _settle(captioner: ImageCaptioner, refs: tuple[str, ...]) -> None:
    """Give background tasks a chance to reach the inflight map, then wait."""

    await asyncio.sleep(0.05)
    await captioner.wait_ready(UMO, refs, timeout=1.0)
    await asyncio.sleep(0.01)


class TestImageCaptioner(unittest.IsolatedAsyncioTestCase):
    async def test_success_writes_caption_to_message(self) -> None:
        async def call(_umo: str, _ref: str) -> str:
            return "图片描述：一只猫趴在键盘上"

        captioner = ImageCaptioner(call, timeout_seconds=1.0)
        msg = _msg("https://example.test/a.png")
        captioner.submit(UMO, msg, "100000001")
        await _settle(captioner, ("https://example.test/a.png",))
        self.assertEqual(msg.image_caption, "一只猫趴在键盘上")
        await captioner.close()

    async def test_timeout_keeps_placeholder(self) -> None:
        async def call(_umo: str, _ref: str) -> str:
            await asyncio.sleep(1)
            return "太慢了"

        captioner = ImageCaptioner(call, timeout_seconds=0.02)
        msg = _msg("u-timeout")
        captioner.submit(UMO, msg)
        await _settle(captioner, ("u-timeout",))
        self.assertEqual(msg.image_caption, "")
        await captioner.close()

    async def test_same_ref_is_deduped_and_cached(self) -> None:
        calls: list[str] = []

        async def call(_umo: str, ref: str) -> str:
            calls.append(ref)
            return "同一张图"

        captioner = ImageCaptioner(call, timeout_seconds=1.0)
        first = _msg("u-shared")
        second = _msg("u-shared")
        captioner.submit(UMO, first)
        captioner.submit(UMO, second)
        await _settle(captioner, ("u-shared",))
        self.assertEqual(calls, ["u-shared"])
        self.assertEqual(first.image_caption, "同一张图")
        self.assertEqual(second.image_caption, "同一张图")

        third = _msg("u-shared")
        captioner.submit(UMO, third)
        await _settle(captioner, ("u-shared",))
        self.assertEqual(calls, ["u-shared"])
        self.assertEqual(third.image_caption, "同一张图")
        await captioner.close()

    async def test_max_images_limit(self) -> None:
        calls: list[str] = []

        async def call(_umo: str, ref: str) -> str:
            calls.append(ref)
            return f"图-{ref}"

        captioner = ImageCaptioner(call, max_images=2, timeout_seconds=1.0)
        msg = _msg("u1", "u2", "u3")
        captioner.submit(UMO, msg)
        await _settle(captioner, ("u1", "u2", "u3"))
        self.assertEqual(calls, ["u1", "u2"])
        self.assertIn("图-u1", msg.image_caption)
        self.assertIn("图-u2", msg.image_caption)
        await captioner.close()

    async def test_caption_failure_does_not_raise(self) -> None:
        async def call(_umo: str, _ref: str) -> str:
            raise RuntimeError("vision down")

        captioner = ImageCaptioner(call, timeout_seconds=1.0)
        msg = _msg("u-fail")
        captioner.submit(UMO, msg)
        await _settle(captioner, ("u-fail",))
        self.assertEqual(msg.image_caption, "")
        await captioner.close()

    async def test_close_cancels_work(self) -> None:
        started = asyncio.Event()

        async def call(_umo: str, _ref: str) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        captioner = ImageCaptioner(call, timeout_seconds=10.0)
        captioner.submit(UMO, _msg("u-slow"))
        await started.wait()
        await captioner.close()
        self.assertTrue(captioner._closed)  # noqa: SLF001 - explicit lifecycle check


class TestCleanCaption(unittest.TestCase):
    def test_prefix_and_brackets(self) -> None:
        self.assertEqual(clean_caption("图片描述： 一只猫 [x]"), "一只猫 ［x］")

    def test_empty_and_unusable(self) -> None:
        self.assertEqual(clean_caption(""), "")
        self.assertEqual(clean_caption("无法识别"), "")
        self.assertEqual(clean_caption("N/A"), "")

    def test_truncates(self) -> None:
        caption = clean_caption("猫" * 100)
        self.assertLessEqual(len(caption), 60)
        self.assertTrue(caption.endswith("…"))


if __name__ == "__main__":
    unittest.main()
