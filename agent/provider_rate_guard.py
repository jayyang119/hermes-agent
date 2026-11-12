"""Cross-session rate limit guard for any provider.

Generalizes the nous_rate_guard pattern so that any provider's rate-limit
exhaustion state is persisted to a shared JSON file.  All sessions (CLI,
gateway, cron, subagents) can check whether a provider is currently
rate-limited before making requests, preventing subagents from burning a
request on an exhausted primary before reactive fallback kicks in.

File layout:
    ~/.hermes/rate_limits/{provider}.json

Data shape per file:
    {
        "provider": str,
        "exhausted_at": float,       # time.time() when exhaustion was recorded
        "reset_at": float,           # absolute time when the limit resets
        "reset_seconds": float,      # original reset duration (for diagnostics)
        "captured_at": float,        # when the rate-limit headers were captured
        "remaining_requests_min": int,
        "remaining_tokens_min": int,
        "limit_requests_min": int,
        "limit_tokens_min": int,
    }
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Optional

from utils import atomic_replace

logger = logging.getLogger(__name__)

_STATE_SUBDIR = "rate_limits"


def _state_path(provider: str) -> str:
    """Return the path to the provider rate limit state file."""
    # Sanitize provider name to avoid directory traversal
    safe_provider = provider.replace("/", "_").replace("\\", "_").replace("..", "_")
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(base, _STATE_SUBDIR, f"{safe_provider}.json")


def record_provider_rate_limit(provider: str, state) -> None:
    """Record that a provider is rate-limited.

    Called after capturing x-ratelimit-* headers showing exhaustion
    (remaining at/below floor).  Writes the JSON file only when at least
    one bucket has remaining <= floor.

    Args:
        provider: Provider identifier (e.g. "openai-codex", "nous").
        state: A RateLimitState dataclass from rate_limit_tracker.
    """
    from agent.rate_limit_tracker import (
        PREDICTIVE_RATE_LIMIT_MIN_RESET_SECONDS,
        PREDICTIVE_RATE_LIMIT_REQUEST_FLOOR,
        PREDICTIVE_RATE_LIMIT_TOKEN_FLOOR,
        should_predictively_skip,
    )

    # Only persist when we would actually predictively skip
    skip = should_predictively_skip(state)
    if not skip.should_skip:
        return

    # ``should_predictively_skip`` intentionally returns the first matching
    # bucket for a useful user-facing reason.  Persistence has a different
    # requirement: the provider remains blocked until *every* exhausted
    # bucket has reset.  Persist the latest blocking reset rather than the
    # first (often minute-window) reset, otherwise another session can resume
    # while an exhausted hourly window is still active.
    blocking_resets = []
    for bucket, floor in (
        (state.requests_min, PREDICTIVE_RATE_LIMIT_REQUEST_FLOOR),
        (state.requests_hour, PREDICTIVE_RATE_LIMIT_REQUEST_FLOOR),
        (state.tokens_min, PREDICTIVE_RATE_LIMIT_TOKEN_FLOOR),
        (state.tokens_hour, PREDICTIVE_RATE_LIMIT_TOKEN_FLOOR),
    ):
        if bucket.limit <= 0 or bucket.remaining > floor:
            continue
        reset_seconds = bucket.remaining_seconds_now
        if reset_seconds > PREDICTIVE_RATE_LIMIT_MIN_RESET_SECONDS:
            blocking_resets.append(reset_seconds)

    reset_seconds = max(blocking_resets, default=skip.reset_seconds)

    now = time.time()
    reset_at = now + reset_seconds

    path = _state_path(provider)
    try:
        state_dir = os.path.dirname(path)
        os.makedirs(state_dir, exist_ok=True)

        data = {
            "provider": provider,
            "exhausted_at": now,
            "reset_at": reset_at,
            "reset_seconds": reset_seconds,
            "captured_at": state.captured_at,
            "remaining_requests_min": state.requests_min.remaining,
            "remaining_tokens_min": state.tokens_min.remaining,
            "limit_requests_min": state.requests_min.limit,
            "limit_tokens_min": state.tokens_min.limit,
        }

        # Atomic write: write to temp file + rename
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            atomic_replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "Provider %s rate limit recorded: resets in %.0fs (at %.0f) — %s",
            provider, reset_seconds, reset_at, skip.reason,
        )
    except Exception as exc:
        logger.debug("Failed to write provider rate limit state for %s: %s", provider, exc)


def provider_rate_limit_remaining(provider: str) -> Optional[float]:
    """Check if a provider is currently rate-limited.

    Returns:
        Seconds remaining until reset, or None if not rate-limited.
        If the reset_at has passed, the file is deleted and None is returned.
    """
    path = _state_path(provider)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        reset_at = data.get("reset_at", 0)
        remaining = reset_at - time.time()
        if remaining > 0:
            return remaining
        # Expired — clean up
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def clear_provider_rate_limit(provider: str) -> None:
    """Clear persisted rate-limit state after healthy provider telemetry."""
    try:
        os.unlink(_state_path(provider))
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Failed to clear provider rate limit state for %s: %s", provider, exc)
