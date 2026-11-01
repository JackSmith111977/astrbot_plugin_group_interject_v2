# -*- coding: utf-8 -*-
"""Pure helpers for the plugin control view.

Validates the public schema, including the nested ``custom`` override object.
Network/persistence stays in ``main.py``; this module is safe to unit-test
without AstrBot.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .text import parse_quiet_hours

SCHEMA_FILE = Path(__file__).resolve().parent / "_conf_schema.json"
_TRUE = {"1", "true", "yes", "on", "开", "是"}
_FALSE = {"0", "false", "no", "off", "关", "否"}


def load_schema() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _field_meta(key: str, meta: dict[str, Any]) -> dict[str, Any]:
    kind = str(meta.get("type") or "string").strip().lower()
    field: dict[str, Any] = {
        "key": key,
        "type": kind,
        "description": str(meta.get("description") or ""),
        "hint": str(meta.get("hint") or ""),
        "options": list(meta.get("options") or []),
        "special": str(meta.get("_special") or ""),
        "default": meta.get("default"),
    }
    for name in ("min", "max", "step"):
        if name in meta:
            field[name] = meta[name]
    if kind == "object":
        field["items"] = schema_fields(meta.get("items") or {})
    return field


def schema_fields(
    schema: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schema = schema if schema is not None else load_schema()
    return [_field_meta(str(key), meta) for key, meta in schema.items()]


def merge_with_defaults(
    values: dict[str, Any],
    schema: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schema = schema if schema is not None else load_schema()
    values = values if isinstance(values, dict) else {}
    merged: dict[str, Any] = {}
    for key, meta in schema.items():
        kind = str(meta.get("type") or "string").strip().lower()
        if kind == "object":
            child = values.get(key)
            merged[key] = merge_with_defaults(
                child if isinstance(child, dict) else {},
                meta.get("items") or {},
            )
        else:
            merged[key] = values.get(key, meta.get("default"))
    for key, value in values.items():
        if key not in merged:
            merged[key] = value
    return merged


def _coerce_bool(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, bool):
        return value, None
    if isinstance(value, int | float):
        return bool(value), None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True, None
        if lowered in _FALSE:
            return False, None
    return False, "必须是布尔值"


def _coerce_string(value: Any) -> tuple[str, str | None]:
    if value is None:
        return "", None
    if isinstance(value, (dict, list, tuple, set)):
        return "", "必须是字符串"
    return str(value).strip(), None


def _coerce_list(value: Any) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw: list[Any] = [part for part in re.split(r"[,，;；\s]+", text) if part]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        return [], "必须是列表或逗号/换行分隔字符串"
    items: list[str] = []
    for item in raw:
        text = str(item).strip().strip("\"'“”「」")
        if text and text not in items:
            items.append(text)
    return items, None


def _coerce_number(
    value: Any,
    kind: str,
    meta: dict[str, Any],
) -> tuple[int | float | None, str | None]:
    if isinstance(value, bool):
        return None, "必须是数字"
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "必须是数字"
    if not math.isfinite(parsed):
        return None, "必须是有限数字"
    if kind == "int":
        parsed = int(round(parsed))
    low = meta.get("min")
    high = meta.get("max")
    if low is not None and parsed < float(low):
        return None, f"不能小于 {low}"
    if high is not None and parsed > float(high):
        return None, f"不能大于 {high}"
    return (int(parsed) if kind == "int" else float(parsed)), None


def _coerce_field(
    key: str,
    meta: dict[str, Any],
    value: Any,
    prefix: str = "",
) -> tuple[Any, list[str]]:
    kind = str(meta.get("type") or "string").strip().lower()
    path = f"{prefix}{key}"
    if kind == "object":
        if not isinstance(value, dict):
            return {}, [f"{path}: 必须是对象"]
        child_schema = meta.get("items") or {}
        unknown = [item for item in value if item not in child_schema]
        errors = [f"{path}: 未知参数 {item}" for item in unknown]
        result: dict[str, Any] = {}
        for child_key, child_meta in child_schema.items():
            if child_key not in value:
                continue
            coerced, child_errors = _coerce_field(
                str(child_key),
                child_meta,
                value[child_key],
                prefix=f"{path}.",
            )
            if child_errors:
                errors.extend(child_errors)
            else:
                result[str(child_key)] = coerced
        return result, errors

    if kind == "bool":
        coerced, error = _coerce_bool(value)
    elif kind == "list":
        coerced, error = _coerce_list(value)
    elif kind in ("int", "float"):
        coerced, error = _coerce_number(value, kind, meta)
    else:
        coerced, error = _coerce_string(value)
    if error:
        return None, [f"{path}: {error}"]
    if key == "mode":
        options = [str(item) for item in (meta.get("options") or [])]
        if options and coerced not in options:
            return None, [f"mode 必须是 {'/'.join(options)} 之一"]
    if key == "quiet_hours" and coerced:
        if not parse_quiet_hours(str(coerced)):
            return None, ["quiet_hours 格式应为 HH:MM-HH:MM（可逗号分隔多个）"]
    return coerced, []


def coerce_payload(
    payload: dict[str, Any],
    schema: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a WebUI config payload and return ``(values, errors)``."""

    schema = schema if schema is not None else load_schema()
    if not isinstance(payload, dict):
        return {}, ["请求体必须是 JSON 对象"]
    unknown = [key for key in payload if key not in schema]
    if unknown:
        return {}, ["未知参数：" + "、".join(map(str, unknown))]

    values: dict[str, Any] = {}
    errors: list[str] = []
    for key, meta in schema.items():
        if key not in payload:
            continue
        coerced, field_errors = _coerce_field(str(key), meta, payload[key])
        if field_errors:
            errors.extend(field_errors)
        else:
            values[str(key)] = coerced

    custom = values.get("custom")
    if isinstance(custom, dict):
        window_size = custom.get("window_size")
        min_messages = custom.get("min_messages")
        if (
            isinstance(window_size, int)
            and isinstance(min_messages, int)
            and min_messages > window_size
        ):
            errors.append("custom.min_messages 不能大于 custom.window_size")
    return values, errors


__all__ = [
    "coerce_payload",
    "load_schema",
    "merge_with_defaults",
    "schema_fields",
]
