# -*- coding: utf-8 -*-
"""Ingress helpers: recent image cache, bare-@ context and bot-line hints.

These helpers touch AstrBot message components but deliberately keep all
network/event-queue decisions in ``main.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .model import Config
from .text import looks_like_image_reference, one_line

LOG_PREFIX = "[group-interject-v2]"

DEFAULT_IMAGE_REFERENCE_KEYWORDS = (
    "这张图",
    "那张图",
    "这个图",
    "那个图",
    "图片",
    "截图",
    "图里",
    "图上的",
    "图里说",
    "图中",
    "图呢",
    "照片",
    "截屏",
    "刚发的",
)


@dataclass
class CachedImage:
    ts: float
    url: str = ""
    file: str = ""


class Ingress:
    """Per-group image cache and small event-injection helpers."""

    def __init__(
        self,
        config: Config,
        *,
        logger=None,
    ) -> None:
        self.config = config
        from .logger_fallback import fallback_logger
        self.logger = logger or fallback_logger()
        self._images: dict[str, CachedImage] = {}

    def cache_event_images(self, event: Any, umo: str, sender_id: str) -> None:
        """Remember the most recent image in a group for later "this image" asks."""

        try:
            self_id = str(event.get_self_id() or "")
            if self_id and sender_id == self_id:
                return
            for component in event.get_messages():
                if type(component).__name__ != "Image":
                    continue
                url = str(getattr(component, "url", "") or "")
                file = str(getattr(component, "file", "") or "")
                if url or file:
                    self.cache_image(umo, url=url, file=file)
                    return
        except Exception:
            self.logger.exception("%s cache image failed", LOG_PREFIX)

    def cache_image(
        self,
        umo: str,
        *,
        url: str = "",
        file: str = "",
        max_seconds: int = 600,
    ) -> None:
        umo = str(umo or "").strip()
        if not umo or not (url or file):
            return
        now = time.time()
        self._images[umo] = CachedImage(ts=now, url=str(url or ""), file=str(file or ""))
        cutoff = now - max(1, int(max_seconds))
        self._images = {
            key: value for key, value in self._images.items() if value.ts >= cutoff
        }

    def cached_image(self, umo: str, max_seconds: int = 300) -> CachedImage | None:
        item = self._images.get(str(umo or "").strip())
        if item is None:
            return None
        if time.time() - item.ts > max(1, int(max_seconds)):
            return None
        return item

    def attach_cached_image(
        self,
        event: Any,
        umo: str,
        text: str,
        *,
        force: bool = False,
    ) -> bool:
        """Append the cached image when a directed message references an image."""

        try:
            if self.has_media_message(event):
                return False
            if not force and not looks_like_image_reference(
                text,
                DEFAULT_IMAGE_REFERENCE_KEYWORDS,
            ):
                return False
            cached = self.cached_image(umo, 300)
            if cached is None:
                return False
            ref = cached.url or cached.file
            if not ref:
                return False
            from astrbot.api.message_components import Image as CompImage

            if ref.startswith(("http://", "https://")):
                component = CompImage.fromURL(ref)
            elif ref.startswith("base64://"):
                component = CompImage(file=ref)
            else:
                component = CompImage.fromFileSystem(ref)
            event.message_obj.message.append(component)
            note = "（这张图是刚才消息里的图片）"
            event.message_str = (
                str(getattr(event, "message_str", "") or "") + " " + note
            ).strip()
            self.logger.info(
                "%s stage=image state=attach source=%s",
                LOG_PREFIX,
                "url" if cached.url else "file",
            )
            return True
        except Exception:
            self.logger.exception("%s attach cached image failed", LOG_PREFIX)
            return False

    def inject_bare_at_context(self, event: Any, runtime: Any, max_messages: int = 6) -> bool:
        """Inject recent group context into an empty @ message."""

        try:
            if str(getattr(event, "message_str", "") or "").strip():
                return False
            if self.has_media_message(event):
                return False
            snippet = runtime.bare_at_context(
                event.unified_msg_origin,
                max_messages,
            )
            if not snippet:
                return False
            from astrbot.api.message_components import Plain

            event.message_obj.message.append(Plain(text=snippet))
            event.message_str = snippet
            self.logger.info(
                "%s stage=bare_at state=inject chars=%d",
                LOG_PREFIX,
                len(snippet),
            )
            return True
        except Exception:
            self.logger.exception("%s bare at context injection failed", LOG_PREFIX)
            return False

    def inject_bot_line_hint(self, event: Any, runtime: Any, max_seconds: int = 300) -> bool:
        """Tell the main Agent what the plugin last said in this group."""

        try:
            if event.get_extra("_group_interject_v2_bot_hint"):
                return False
            if not str(getattr(event, "message_str", "") or "").strip():
                return False
            context = runtime.last_bot_context(
                event.unified_msg_origin,
                max_seconds,
            )
            if not context:
                return False
            from astrbot.api.message_components import Plain

            note = (
                f"（你刚才在群里插话说：「{one_line(context)[:120]}」；"
                "如果对方是在接你的话，就接着聊，别重复上一句。）"
            )
            event.message_obj.message.append(Plain(text=note))
            event.message_str = (
                str(getattr(event, "message_str", "") or "") + " " + note
            ).strip()
            event.set_extra("_group_interject_v2_bot_hint", True)
            self.logger.info(
                "%s stage=handoff state=inject chars=%d",
                LOG_PREFIX,
                len(note),
            )
            return True
        except Exception:
            self.logger.exception("%s bot-line hint injection failed", LOG_PREFIX)
            return False

    @staticmethod
    def collect_image_refs(event: Any, limit: int = 2) -> tuple[str, ...]:
        """Return top-level Image refs (URL preferred, then file), capped."""

        refs: list[str] = []
        try:
            for component in event.get_messages():
                if type(component).__name__ != "Image":
                    continue
                ref = str(
                    getattr(component, "url", "")
                    or getattr(component, "file", "")
                    or ""
                ).strip()
                if ref and ref not in refs:
                    refs.append(ref)
                    if len(refs) >= max(1, int(limit)):
                        break
        except Exception:
            return ()
        return tuple(refs)

    @staticmethod
    def has_media_message(event: Any) -> bool:
        try:
            for component in event.get_messages():
                if type(component).__name__ in {"Image", "Record", "File", "Video"}:
                    return True
        except Exception:
            return False
        return False


__all__ = ["DEFAULT_IMAGE_REFERENCE_KEYWORDS", "Ingress", "LOG_PREFIX"]
