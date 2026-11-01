# -*- coding: utf-8 -*-
"""把插话侧发出的消息写入与主 Agent 相同的长期记忆。

当前线上长期记忆由 ``astrbot_plugin_livingmemory`` 持有：主 Agent 的
``on_llm_response`` 会把 assistant 消息写入其 ConversationStore，而插件
直接 ``context.send_message`` 不经过主 Agent 的 LLM 响应钩子，因此如果不
桥接，长期记忆里只有群友/主 Agent，没有插件插话。

设计取舍：
- 不 import LivingMemory 的内部类，按属性探测其公开对象，版本不匹配时自动
  降级为 no-op，不影响插话；
- 只写 ``role=assistant`` 的消息，不写记忆摘要；摘要仍由 LivingMemory 自己
  在后续主 Agent 回复时统一生成，避免双重总结；
- 写入异步执行且吞异常，失败只记 warning，不阻塞消息发送。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

LOG_PREFIX = "[group-interject-v2]"
_LIVINGMEMORY_NAMES = {
    "astrbot_plugin_livingmemory",
    "livingmemory",
}


class LivingMemoryBridge:
    """按需发现 LivingMemory 插件并把 bot 发言写入其会话存储。"""

    def __init__(
        self,
        context: Any,
        *,
        logger=None,
    ) -> None:
        self.context = context
        from .logger_fallback import fallback_logger
        self.logger = logger or fallback_logger()
        self._plugin: Any = None
        self._manager: Any = None
        self._next_retry_at = 0.0
        self._warned = False

    def _resolve(self) -> None:
        if self._manager is not None:
            return
        now = time.monotonic()
        if now < self._next_retry_at:
            return
        self._next_retry_at = now + 5.0
        try:
            stars = list(self.context.get_all_stars())
        except Exception:
            stars = []
        for meta in stars:
            name = str(getattr(meta, "name", "") or "").strip().lower()
            if name not in _LIVINGMEMORY_NAMES:
                continue
            plugin = getattr(meta, "star_cls", None)
            handler = getattr(plugin, "event_handler", None)
            manager = getattr(handler, "conversation_manager", None)
            if manager is not None:
                self._plugin = plugin
                self._manager = manager
                return

    @property
    def available(self) -> bool:
        self._resolve()
        return self._manager is not None

    async def record_assistant(
        self,
        *,
        umo: str,
        group_id: str,
        text: str,
        sender_id: str = "",
        sender_name: str = "",
        platform: str = "",
    ) -> bool:
        """把一条插件插话写入 LivingMemory 的会话消息。

        Returns:
            True 表示已写入；False 表示 LivingMemory 不存在/未初始化/写入失败。
        """
        self._resolve()
        manager = self._manager
        if manager is None:
            return False
        add_message = getattr(manager, "add_message", None)
        if not callable(add_message):
            return False
        try:
            await add_message(
                session_id=umo,
                role="assistant",
                content=str(text or ""),
                sender_id=str(sender_id or "bot"),
                sender_name=str(sender_name or ""),
                group_id=str(group_id or ""),
                platform=str(platform or umo.split(":", 1)[0]),
                is_bot_message=True,
                metadata={"source": "group_interject_v2"},
            )
            return True
        except asyncio.CancelledError:  # pragma: no cover - 由调用方处理
            raise
        except Exception as exc:
            if not self._warned:
                self._warned = True
                self.logger.warning(
                    "%s livingmemory bridge write failed (%s); "
                    "后续失败不再重复告警",
                    LOG_PREFIX,
                    type(exc).__name__,
                )
            return False


__all__ = ["LivingMemoryBridge"]
