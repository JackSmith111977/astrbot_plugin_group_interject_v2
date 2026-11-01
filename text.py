# -*- coding: utf-8 -*-
"""Pure text helpers for ingesting and rendering group messages.

No AstrBot imports.  These functions are intentionally small and deterministic so
they can be tested without a bot instance.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import datetime, tzinfo

from .model import Msg, non_bot_count

PLACEHOLDERS = {
    "Image": "[图片]",
    "Face": "[表情]",
    "Record": "[语音]",
    "Video": "[视频]",
    "File": "[文件]",
    "Forward": "[合并转发]",
    "Nodes": "[合并转发]",
    "Node": "[合并转发]",
    "Json": "[卡片]",
    "Share": "[分享]",
    "Music": "[音乐]",
    "Poke": "[戳一戳]",
    "RPS": "[猜拳]",
    "Dice": "[骰子]",
    "Shake": "[窗口抖动]",
    "Contact": "[名片]",
    "Location": "[位置]",
    "Unknown": "[消息]",
}

MEDIA_PLACEHOLDERS = frozenset(PLACEHOLDERS.values()) | frozenset(
    f"[{kind}]" for kind in PLACEHOLDERS
)

_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
    ("《", "》"),
)

_LEADING_BRACKET_BLOCKS = re.compile(r"^(?:\[[^\]]*\]\s*)+")
_WAKE_SEPARATORS = frozenset("，,：:;；、.!！?？~ \u3000\t")

_NOISE_COMPLAINT = re.compile(
    r"别(?:吵|说|接|插|回|刷)|太吵|吵死|安静(?:点|些)?|闭嘴|少说|"
    r"话(?:真|太)多|别乱(?:回|接|说)|不要(?:回|接|插)|刷屏|烦不烦|别烦"
)

_QUIET_RANGE = re.compile(r"^(\d{1,2}):(\d{2})\s*[-~—到至]\s*(\d{1,2}):(\d{2})$")


def component_kind(component: object) -> str:
    """Return a stable component name for real and fake AstrBot components."""

    comp_type = getattr(component, "type", None)
    value = getattr(comp_type, "value", None)
    if isinstance(value, str) and value:
        return value
    return type(component).__name__


def _poke_target(component: object) -> str:
    """Return the target QQ id carried by a Poke component, if any."""

    getter = getattr(component, "target_id", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
        text = str(value or "").strip()
        if text and text != "0":
            return text
    for name in ("id", "qq"):
        text = str(getattr(component, name, "") or "").strip()
        if text and text != "0":
            return text
    return ""


def format_poke(component: object, self_id: str) -> str:
    """Render a Poke with its target instead of a bare ``[戳一戳]``."""

    target = _poke_target(component)
    self_value = str(self_id or "").strip()
    if not target:
        return "[戳一戳（目标未知）]"
    if target == "all":
        label = "@全体成员"
    else:
        name = str(getattr(component, "name", "") or "").strip()
        label = f"@{name}" if name else f"@{target}"
    if self_value and target == self_value:
        return "[戳了你]"
    if self_value:
        return f"[戳了 {label}（别人）]"
    return f"[戳了 {label}]"


def poke_event(components: Iterable[object] | None) -> tuple[bool, str]:
    """Return ``(is_poke, target_id)`` for the first Poke in a chain."""

    for component in components or ():
        if component_kind(component) == "Poke":
            return True, _poke_target(component)
    return False, ""


def raw_poke_event(raw: object) -> tuple[bool, str]:
    """Read a OneBot poke notice from ``message_obj.raw_message``."""

    if raw is None:
        return False, ""
    getter = getattr(raw, "get", None)
    try:
        if callable(getter):
            sub_type = getter("sub_type")
            target = getter("target_id")
        else:
            sub_type = getattr(raw, "sub_type", None)
            target = getattr(raw, "target_id", None)
    except Exception:
        return False, ""
    if str(sub_type or "").strip().lower() != "poke":
        return False, ""
    return True, str(target or "").strip()


def is_poke_at_other(is_poke: bool, target: str, self_id: str) -> bool:
    """Whether a poke is safely treated as not addressed to the bot."""

    if not is_poke:
        return False
    target_text = str(target or "").strip()
    self_text = str(self_id or "").strip()
    return not target_text or not self_text or target_text != self_text


def format_message_chain(
    components: Iterable[object] | None,
    self_id: str,
) -> tuple[str, bool]:
    """Format a message chain into one line and whether it addresses the bot.

    Args:
        components: The value of ``event.get_messages()``.
        self_id: The bot's own QQ number.

    Returns:
        ``(text, directed)``.  ``directed`` is true for an @ of the bot or a
        quote/reply to one of the bot's messages.
    """

    self_id = str(self_id or "")
    parts: list[str] = []
    directed = False
    only_at_self = False
    for component in components or ():
        kind = component_kind(component)
        if kind == "Plain":
            value = str(getattr(component, "text", "") or "")
            if value:
                parts.append(value)
        elif kind == "At":
            qq = str(getattr(component, "qq", "") or "")
            name = str(getattr(component, "name", "") or "").strip()
            if qq == "all":
                parts.append("[@全体成员]")
            elif qq == self_id:
                directed = True
                only_at_self = True
                parts.append(f"[@{name}]" if name else "[@你]")
            else:
                parts.append(f"[@{name or qq}(别人)]")
        elif kind == "Reply":
            sender_id = str(getattr(component, "sender_id", "") or "")
            nickname = str(getattr(component, "sender_nickname", "") or "").strip()
            quoted = str(getattr(component, "message_str", "") or "").strip()
            if sender_id and sender_id == self_id:
                directed = True
                nickname = "你"
            label = f"引用 {nickname or '群友'}"
            if quoted:
                label += f": {quoted}"
            parts.append(f"[{label}]")
        elif kind == "Poke":
            parts.append(format_poke(component, self_id))
        else:
            placeholder = PLACEHOLDERS.get(kind)
            if placeholder:
                parts.append(placeholder)
    if directed and only_at_self and len(parts) == 1:
        return "[只 @ 了你]", True
    return " ".join(part for part in parts if part).strip(), directed


def has_bot_reference(components: Iterable[object] | None, self_id: str) -> bool:
    """Return True only for an At of the bot or a quote/reply to it.

    This is deliberately narrower than "any At": @ing another group member must
    not turn a message into a directed message for this plugin.
    """

    self_id = str(self_id or "")
    if not self_id:
        return False
    for component in components or ():
        kind = component_kind(component)
        if kind == "At" and str(getattr(component, "qq", "") or "") == self_id:
            return True
        if kind == "Reply" and str(getattr(component, "sender_id", "") or "") == self_id:
            return True
    return False


def looks_like_noise(messages: Sequence[Msg], sample: int = 8) -> bool:
    """Deterministic flood/repeat detector (three identical non-media texts)."""

    meaningful: list[str] = []
    for msg in messages:
        if msg.is_bot:
            continue
        text = str(msg.text or "").strip()
        if not text or text in MEDIA_PLACEHOLDERS:
            continue
        meaningful.append(text)
    if len(meaningful) < 3:
        return False
    counts: dict[str, int] = {}
    for text in meaningful[-sample:]:
        counts[text] = counts.get(text, 0) + 1
        if counts[text] >= 3:
            return True
    return False


def format_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def format_datetime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def one_line(text: object) -> str:
    return " ".join(part.strip() for part in str(text).splitlines() if part.strip())


def message_body(msg: Msg) -> str:
    """Render one message, including an image caption when one is available."""

    body = str(msg.text or "")
    caption = one_line(msg.image_caption)
    if not caption:
        return body
    if not body.strip() or body.strip() in MEDIA_PLACEHOLDERS:
        return f"[图片: {caption}]"
    return f"{body}（图片：{caption}）"


def format_window(
    messages: Sequence[Msg],
    per_message_chars: int = 160,
    max_total_chars: int = 4000,
) -> str:
    """Render a window for the LLM, keeping the newest lines when over budget."""

    lines: list[tuple[str, Msg]] = []
    for msg in messages:
        body = one_line(message_body(msg))
        if len(body) > per_message_chars:
            body = body[:per_message_chars] + "…"
        if msg.is_bot:
            head = f"[{format_clock(msg.ts)}] 你:"
        else:
            name = msg.sender_name or "群友"
            if msg.sender_id:
                name = f"{name}#{msg.sender_id}"
            head = f"[{format_clock(msg.ts)}] {name}:"
        line = f"{head} {body}".rstrip()
        if msg.directed and not msg.is_bot:
            line += "（这条在跟你说话）"
        lines.append((line, msg))

    kept: list[str] = []
    used = 0
    for line, _msg in reversed(lines):
        extra = len(line) + 1
        if kept and used + extra > max_total_chars:
            break
        kept.append(line)
        used += extra
    kept.reverse()
    return "\n".join(kept)


def count_non_bot(messages: Sequence[Msg]) -> int:
    return non_bot_count(messages)


def match_wake_name(text: str, names: Sequence[str]) -> str | None:
    """Match a configured wake name only at the start of the message."""

    stripped = str(text or "").strip()
    if not stripped or not names:
        return None
    candidate = _LEADING_BRACKET_BLOCKS.sub("", stripped).strip()
    if not candidate:
        return None
    for name in sorted(
        (str(item).strip() for item in names if str(item).strip()),
        key=len,
        reverse=True,
    ):
        if candidate == name:
            return name
        if candidate.startswith(name):
            rest = candidate[len(name) : len(name) + 1]
            if rest and rest in _WAKE_SEPARATORS:
                return name
    return None


def looks_like_image_reference(text: str, keywords: Sequence[str]) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return any(
        keyword and keyword in stripped
        for keyword in (str(item).strip() for item in keywords)
    )


def is_noise_complaint(text: str) -> bool:
    """Whether a message is a "too noisy / stop replying" style complaint."""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return bool(_NOISE_COMPLAINT.search(normalized))


_ACK_MARKERS = (
    "哈哈",
    "笑死",
    "草",
    "确实",
    "真的",
    "可以",
    "还行",
    "厉害",
    "可爱",
    "奖励",
    "喜欢",
    "不要",
    "来",
    "吃",
)


def looks_like_continuation(
    text: str,
    *,
    bot_text: str = "",
    bot_names: Sequence[str] = (),
) -> bool:
    """Conservative heuristic for "this message is replying to the bot".

    The LLM still makes the final call.  This only unlocks the engaged
    follow-up path for messages that do not explicitly @ the bot.
    """

    if not isinstance(text, str) or not text.strip():
        return False
    normalized = unicodedata.normalize("NFKC", text).strip()
    if any(name and name in normalized for name in bot_names if name):
        return True
    if len(normalized) > 12 or "?" in normalized or "？" in normalized:
        return False
    for marker in _ACK_MARKERS:
        if marker in normalized:
            return True
    if not bot_text:
        return False
    bot_normalized = unicodedata.normalize("NFKC", bot_text)
    punctuation = set(" \t\r\n，,。.!！?？；;：:、~～…“”\"'‘’()（）[]【】")
    text_grams = {
        normalized[i : i + 2]
        for i in range(max(0, len(normalized) - 1))
        if not (set(normalized[i : i + 2]) & punctuation)
    }
    bot_grams = {
        bot_normalized[i : i + 2]
        for i in range(max(0, len(bot_normalized) - 1))
        if not (set(bot_normalized[i : i + 2]) & punctuation)
    }
    return bool(text_grams & bot_grams)


def build_bare_at_context(
    messages: Sequence[Msg],
    max_messages: int = 6,
    max_total_chars: int = 600,
) -> str:
    """Build context for a bare @ (an @ with no text)."""

    picked = [msg for msg in messages if message_body(msg).strip()]
    if not picked:
        return ""
    picked = picked[-max(1, int(max_messages)) :]
    lines: list[str] = []
    for msg in picked:
        body = one_line(message_body(msg))[:160]
        if not body:
            continue
        speaker = "你" if msg.is_bot else (msg.sender_name or "群友")
        lines.append(f"{speaker}：{body}")
    if not lines:
        return ""
    body = "；".join(lines)
    if len(body) > max_total_chars:
        body = body[-max_total_chars:]
    return (
        "（@了你但没有写具体内容。最近的群聊："
        f"{body}"
        "。请结合上下文判断对方是不是在找你/等你回应：如果是在问你，直接用工具查清楚再回答；"
        "如果是 @ 错了或没内容，就简短回应一句，不要机械追问。）"
    )


def parse_quiet_hours(spec: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ranges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if not isinstance(spec, str):
        return ranges
    for chunk in re.split(r"[,，;；\s]+", spec.strip()):
        match = _QUIET_RANGE.match(chunk)
        if not match:
            continue
        hour1, minute1, hour2, minute2 = (int(part) for part in match.groups())
        if not (
            0 <= hour1 < 24
            and 0 <= minute1 < 60
            and 0 <= hour2 < 24
            and 0 <= minute2 < 60
        ):
            continue
        if (hour1, minute1) == (hour2, minute2):
            continue
        ranges.append(((hour1, minute1), (hour2, minute2)))
    return ranges


def in_quiet_hours(spec: str, now: float, tz: tzinfo | None = None) -> bool:
    ranges = parse_quiet_hours(spec)
    if not ranges:
        return False
    local = datetime.fromtimestamp(now, tz)
    minute = local.hour * 60 + local.minute
    for (start_h, start_m), (end_h, end_m) in ranges:
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        if start < end:
            if start <= minute < end:
                return True
        elif minute >= start or minute < end:
            return True
    return False


__all__ = [
    "MEDIA_PLACEHOLDERS",
    "PLACEHOLDERS",
    "build_bare_at_context",
    "component_kind",
    "count_non_bot",
    "format_clock",
    "format_datetime",
    "format_message_chain",
    "format_poke",
    "format_window",
    "has_bot_reference",
    "in_quiet_hours",
    "is_noise_complaint",
    "is_poke_at_other",
    "looks_like_continuation",
    "looks_like_image_reference",
    "looks_like_noise",
    "match_wake_name",
    "message_body",
    "one_line",
    "parse_quiet_hours",
    "poke_event",
    "raw_poke_event",
]
