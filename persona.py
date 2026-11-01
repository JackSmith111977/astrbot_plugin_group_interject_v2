# -*- coding: utf-8 -*-
"""人格解析：让插话侧与主 Agent 使用同一份“最终生效人格”。

AstrBot 主 Agent 通过 ``persona_manager.resolve_selected_persona`` 解析
「会话覆盖 / 会话强制 / 默认配置」后的最终人格；插话侧如果只取
``get_default_persona_v3``，在会话切换过人格时会出现两个身份。

本模块用注入对象而非直接导入 AstrBot，便于单测；任何一步失败都回退到
默认人格，保证插话功能不被 AstrBot 版本差异拖垮。
"""

from __future__ import annotations

from typing import Any


def _persona_fields(persona: Any) -> tuple[str, str]:
    """从 Personality/dict 里提取 (name, prompt)，不假定具体类型。"""
    if not isinstance(persona, dict):
        return "", ""
    return str(persona.get("name") or ""), str(persona.get("prompt") or "")


def _platform_from_umo(umo: str) -> str:
    """``aiocqhttp:GroupMessage:100000001`` -> ``aiocqhttp``。"""
    return str(umo or "").split(":", 1)[0]


async def resolve_persona(
    persona_manager: Any,
    conversation_manager: Any,
    umo: str,
    *,
    platform_name: str = "",
    provider_settings: dict | None = None,
) -> tuple[str, str]:
    """解析与主 Agent 一致的人格；失败时回退默认人格。

    Args:
        persona_manager: ``context.persona_manager``。
        conversation_manager: ``context.conversation_manager``。
        umo: 统一消息来源。
        platform_name: 平台名；缺省从 umo 前缀推断。
        provider_settings: 用于 webchat 等特殊人格规则；可传空 dict。

    Returns:
        ``(name, prompt)``；解析不到时返回 ``("", "")``。
    """
    conversation_persona_id: str | None = None
    try:
        if conversation_manager is not None and umo:
            conversation_id = await conversation_manager.get_curr_conversation_id(umo)
            if conversation_id:
                conversation = await conversation_manager.get_conversation(
                    umo,
                    conversation_id,
                )
                conversation_persona_id = getattr(
                    conversation,
                    "persona_id",
                    None,
                )
    except Exception:
        conversation_persona_id = None

    resolver = getattr(persona_manager, "resolve_selected_persona", None)
    if callable(resolver):
        try:
            _selected_id, persona, _forced_id, _webchat = await resolver(
                umo=umo,
                conversation_persona_id=conversation_persona_id,
                platform_name=platform_name or _platform_from_umo(umo),
                provider_settings=provider_settings or {},
            )
            name, prompt = _persona_fields(persona)
            if prompt:
                return name, prompt
        except Exception:
            pass

    try:
        persona = await persona_manager.get_default_persona_v3(umo)
    except Exception:
        return "", ""
    name, prompt = _persona_fields(persona)
    if prompt:
        return name, prompt
    # 某些版本返回 Personality 对象而非 dict。
    return (
        str(getattr(persona, "name", "") or ""),
        str(getattr(persona, "prompt", "") or ""),
    )
