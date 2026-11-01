# -*- coding: utf-8 -*-
"""Background image descriptions for the v2 interjection pipeline.

The decider is deliberately text-only: a caption is generated once, written back
to the in-memory :class:`Msg`, and then reused by the normal window renderer and
the cold summarizer.  This module has no AstrBot imports; the transport call is
injected by ``main.py`` so it can be unit-tested with a fake model.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence

from .model import (
    IMAGE_CAPTION_CACHE_SECONDS,
    IMAGE_CAPTION_FAILURE_COOLDOWN_SECONDS,
    IMAGE_CAPTION_FAILURE_THRESHOLD,
    IMAGE_CAPTION_MAX_IMAGES,
    IMAGE_CAPTION_TIMEOUT_SECONDS,
    Msg,
)
from .text import one_line

LOG_PREFIX = "[group-interject-v2]"

CaptionCall = Callable[[str, str], Awaitable[str]]
"""``(umo, image_ref) -> short caption``."""

_PREFIX = re.compile(r"^\s*(?:图片描述|图片内容|描述|caption)\s*[:：]\s*", re.I)
_UNUSABLE = {
    "无法识别",
    "无法判断",
    "看不清",
    "看不到",
    "unknown",
    "n/a",
    "none",
}
_CAPTION_MAX_CHARS = 60


def clean_caption(raw: object) -> str:
    """Normalize one model caption into a safe, short single line."""

    text = one_line(str(raw or ""))
    if not text:
        return ""
    text = _PREFIX.sub("", text).strip()
    text = text.strip("`*_# \t\"'“”「」")
    text = " ".join(text.split())
    if not text:
        return ""
    if text.lower() in _UNUSABLE:
        return ""
    text = text.replace("[", "［").replace("]", "］")
    if len(text) > _CAPTION_MAX_CHARS:
        text = text[: _CAPTION_MAX_CHARS - 1].rstrip() + "…"
    return text


class ImageCaptioner:
    """Async, bounded, in-memory image caption helper.

    It is intentionally fire-and-forget: ``submit`` never awaits a network call.
    The runtime may call :meth:`wait_ready` briefly before a decision so a
    caption that is already in flight can be included in the prompt.
    """

    def __init__(
        self,
        caption_call: CaptionCall,
        *,
        max_images: int = IMAGE_CAPTION_MAX_IMAGES,
        timeout_seconds: float = IMAGE_CAPTION_TIMEOUT_SECONDS,
        cache_ttl_seconds: int = IMAGE_CAPTION_CACHE_SECONDS,
        max_entries: int = 128,
        max_concurrency: int = 2,
        failure_threshold: int = IMAGE_CAPTION_FAILURE_THRESHOLD,
        failure_cooldown_seconds: int = IMAGE_CAPTION_FAILURE_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.time,
        logger=None,
    ) -> None:
        self._caption_call = caption_call
        self.max_images = max(1, int(max_images))
        self.timeout_seconds = max(0.05, float(timeout_seconds))
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self.max_concurrency = max(1, int(max_concurrency))
        self.failure_threshold = max(1, int(failure_threshold))
        self.failure_cooldown_seconds = max(1, int(failure_cooldown_seconds))
        self._clock = clock
        from .logger_fallback import fallback_logger
        self.logger = logger or fallback_logger()
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._cache: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()
        self._inflight: dict[tuple[str, str], asyncio.Task[str]] = {}
        self._tasks: set[asyncio.Task[object]] = set()
        self._closed = False
        self._failures = 0
        self._disabled_until = 0.0
        self._last_failure = ""

    # -- public API --------------------------------------------------------

    def submit(self, umo: str, msg: Msg, group_id: str = "") -> None:
        """Schedule captioning for one message; never blocks or raises."""

        if self._closed or msg.image_caption:
            return
        refs = tuple(
            dict.fromkeys(
                str(ref).strip()
                for ref in (msg.image_refs or ())
                if str(ref).strip()
            )
        )[: self.max_images]
        if not refs:
            return
        now = self._clock()
        if now < self._disabled_until:
            self.logger.debug(
                "%s stage=caption state=skip reason=circuit_open group=%s",
                LOG_PREFIX,
                group_id or "-",
            )
            return
        try:
            task = asyncio.create_task(
                self._fill(umo, msg, refs, group_id),
                name=f"group-interject-v2:caption:{group_id or umo}",
            )
        except RuntimeError:
            # No running event loop; the caller can simply keep ``[图片]``.
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_ready(
        self,
        umo: str,
        refs: Sequence[str],
        timeout: float = 1.5,
    ) -> None:
        """Wait briefly for captions already in flight for the given refs."""

        if self._closed:
            return
        tasks: list[asyncio.Task[str]] = []
        for ref in dict.fromkeys(str(item) for item in refs):
            task = self._inflight.get((umo, ref))
            if task is not None and not task.done():
                tasks.append(task)
        if not tasks:
            return
        _done, pending = await asyncio.wait(
            tasks,
            timeout=max(0.0, float(timeout)),
        )
        if pending:
            self.logger.debug(
                "%s stage=caption state=wait result=timeout pending=%d",
                LOG_PREFIX,
                len(pending),
            )

    async def close(self) -> None:
        """Cancel background work; safe to call more than once."""

        self._closed = True
        tasks = list(self._tasks)
        tasks.extend(
            task for task in self._inflight.values() if task not in tasks
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._inflight.clear()
        self._cache.clear()

    # -- internals ---------------------------------------------------------

    async def _fill(
        self,
        umo: str,
        msg: Msg,
        refs: tuple[str, ...],
        group_id: str,
    ) -> None:
        started = self._clock()
        try:
            captions = await asyncio.gather(*(self._get(umo, ref) for ref in refs))
            caption = clean_caption("；".join(item for item in captions if item))
            if not caption or self._closed:
                self.logger.info(
                    "%s stage=caption state=empty group=%s images=%d reason=%s "
                    "elapsed=%.1fs",
                    LOG_PREFIX,
                    group_id or "-",
                    len(refs),
                    self._last_failure or "empty",
                    self._clock() - started,
                )
                return
            msg.image_caption = caption
            self._last_failure = ""
            self.logger.info(
                "%s stage=caption state=ok group=%s images=%d chars=%d elapsed=%.1fs",
                LOG_PREFIX,
                group_id or "-",
                len(refs),
                len(caption),
                self._clock() - started,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("%s stage=caption state=error group=%s", LOG_PREFIX, group_id or "-")

    async def _get(self, umo: str, ref: str) -> str:
        if self._closed:
            return ""
        key = (umo, ref)
        now = self._clock()
        cached = self._cache.get(key)
        if cached is not None and now - cached[0] <= self.cache_ttl_seconds:
            self._cache.move_to_end(key)
            return cached[1]

        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._fetch(key, umo, ref))
            self._inflight[key] = task
            task.add_done_callback(
                lambda _task, cache_key=key: self._inflight.pop(cache_key, None)
            )
        try:
            return await task
        except asyncio.CancelledError:
            raise
        except Exception:
            return ""

    async def _fetch(self, key: tuple[str, str], umo: str, ref: str) -> str:
        async with self._sem:
            try:
                raw = await asyncio.wait_for(
                    self._caption_call(umo, ref),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._record_failure("timeout")
                return ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_failure(type(exc).__name__)
                return ""
            caption = clean_caption(raw)
            if not caption:
                self._record_failure("empty")
                return ""
            self._failures = 0
            self._last_failure = ""
            self._cache[key] = (self._clock(), caption)
            self._cache.move_to_end(key)
            self._trim_cache()
            return caption

    def _record_failure(self, reason: str) -> None:
        self._last_failure = reason or "error"
        self._failures += 1
        if self._failures < self.failure_threshold:
            self.logger.debug(
                "%s stage=caption state=fail reason=%s failures=%d",
                LOG_PREFIX,
                reason or "-",
                self._failures,
            )
            return
        self._failures = 0
        self._disabled_until = self._clock() + self.failure_cooldown_seconds
        self.logger.warning(
            "%s stage=caption state=circuit_open reason=%s cooldown=%ds",
            LOG_PREFIX,
            reason or "-",
            self.failure_cooldown_seconds,
        )

    def _trim_cache(self) -> None:
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)


__all__ = ["CaptionCall", "ImageCaptioner", "clean_caption"]
