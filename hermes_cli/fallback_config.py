"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _custom_provider_name(provider: str) -> str:
    prefix = "custom:"
    if not provider.lower().startswith(prefix):
        return ""
    return provider[len(prefix):].strip()


def _custom_provider_model_map(config: dict[str, Any]) -> dict[str, str]:
    """Return custom provider default models keyed by name/provider key."""

    try:
        from hermes_cli.config import get_compatible_custom_providers

        custom_providers = get_compatible_custom_providers(config)
    except Exception:
        custom_providers = config.get("custom_providers")

    if not isinstance(custom_providers, list):
        return {}

    models: dict[str, str] = {}
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("model") or entry.get("default_model") or "").strip()
        if not model:
            continue
        for key in (entry.get("name"), entry.get("provider_key")):
            key_str = str(key or "").strip().lower()
            if key_str:
                models.setdefault(key_str, model)
    return models


def _resolve_fallback_model(
    provider: str,
    raw_model: Any,
    custom_models: dict[str, str],
) -> str:
    model = str(raw_model or "").strip()
    custom_name = _custom_provider_name(provider)
    if custom_name and (not model or model.lower() == "auto"):
        return custom_models.get(custom_name.lower(), "")
    if model.lower() == "auto":
        return ""
    return model


def _iter_fallback_entries(
    raw: Any,
    custom_models: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    custom_models = custom_models or {}
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = _resolve_fallback_model(provider, entry.get("model"), custom_models)
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    custom_models = _custom_provider_model_map(config)

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key), custom_models):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
