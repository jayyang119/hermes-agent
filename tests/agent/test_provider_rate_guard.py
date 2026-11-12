"""Tests for agent.provider_rate_guard — cross-session rate limit persistence."""

import json
import os
import time

import pytest

from agent.provider_rate_guard import (
    _state_path,
    record_provider_rate_limit,
    provider_rate_limit_remaining,
    clear_provider_rate_limit,
)
from agent.rate_limit_tracker import (
    RateLimitBucket,
    RateLimitState,
    PREDICTIVE_RATE_LIMIT_REQUEST_FLOOR,
    PREDICTIVE_RATE_LIMIT_TOKEN_FLOOR,
)


def _make_exhausted_state(
    *,
    remaining_requests: int = 0,
    remaining_tokens: int = 0,
    reset_requests: float = 60.0,
    reset_tokens: float = 60.0,
    limit_requests: int = 800,
    limit_tokens: int = 8000000,
) -> RateLimitState:
    """Build a RateLimitState with exhausted buckets."""
    now = time.time()
    return RateLimitState(
        requests_min=RateLimitBucket(
            limit=limit_requests,
            remaining=remaining_requests,
            reset_seconds=reset_requests,
            captured_at=now,
        ),
        requests_hour=RateLimitBucket(
            limit=33600, remaining=33590, reset_seconds=3500.0, captured_at=now
        ),
        tokens_min=RateLimitBucket(
            limit=limit_tokens,
            remaining=remaining_tokens,
            reset_seconds=reset_tokens,
            captured_at=now,
        ),
        tokens_hour=RateLimitBucket(
            limit=336000000, remaining=335999000, reset_seconds=3490.0, captured_at=now
        ),
        captured_at=now,
        provider="test-provider",
    )


def _make_healthy_state() -> RateLimitState:
    """Build a RateLimitState with plenty of headroom."""
    now = time.time()
    return RateLimitState(
        requests_min=RateLimitBucket(limit=800, remaining=795, reset_seconds=45.0, captured_at=now),
        requests_hour=RateLimitBucket(limit=33600, remaining=33590, reset_seconds=3500.0, captured_at=now),
        tokens_min=RateLimitBucket(limit=8000000, remaining=7999500, reset_seconds=42.0, captured_at=now),
        tokens_hour=RateLimitBucket(limit=336000000, remaining=335999000, reset_seconds=3490.0, captured_at=now),
        captured_at=now,
        provider="test-provider",
    )


class TestRecordAndRead:
    def test_record_exhaustion_and_read_remaining(self, tmp_path, monkeypatch):
        """Record exhaustion, verify remaining > 0."""
        monkeypatch.setenv("HOME", str(tmp_path))
        # Override _state_path to use tmp_path
        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda provider: os.path.join(str(tmp_path), "rate_limits", f"{provider}.json"),
        )

        provider = "test-provider"
        state = _make_exhausted_state(remaining_requests=0, reset_requests=60.0)

        record_provider_rate_limit(provider, state)

        remaining = provider_rate_limit_remaining(provider)
        assert remaining is not None
        assert remaining > 0
        # Should be roughly 60 seconds (allow some tolerance for test execution)
        assert 55 <= remaining <= 62


class TestExpiredCleanup:
    def test_expired_state_returns_none_and_deletes_file(self, tmp_path, monkeypatch):
        """Record exhaustion with reset_at in the past, verify returns None and file is deleted."""
        rate_dir = tmp_path / "rate_limits"
        rate_dir.mkdir()
        provider = "test-provider"
        path = rate_dir / f"{provider}.json"

        # Write a state file with reset_at in the past
        past = time.time() - 100
        data = {
            "provider": provider,
            "exhausted_at": past - 60,
            "reset_at": past,
            "reset_seconds": 60.0,
            "captured_at": past - 60,
            "remaining_requests_min": 0,
            "remaining_tokens_min": 0,
            "limit_requests_min": 800,
            "limit_tokens_min": 8000000,
        }
        path.write_text(json.dumps(data))

        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda p: str(rate_dir / f"{p}.json"),
        )

        remaining = provider_rate_limit_remaining(provider)
        assert remaining is None
        # File should have been deleted
        assert not path.exists()


class TestClear:
    def test_record_then_clear_deletes_file(self, tmp_path, monkeypatch):
        """Record then clear, verify file is gone."""
        rate_dir = tmp_path / "rate_limits"
        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda provider: os.path.join(str(rate_dir), f"{provider}.json"),
        )

        provider = "test-provider"
        state = _make_exhausted_state(remaining_requests=0, reset_requests=60.0)

        record_provider_rate_limit(provider, state)
        path = rate_dir / f"{provider}.json"
        assert path.exists()

        clear_provider_rate_limit(provider)
        assert not path.exists()

    def test_clear_nonexistent_is_noop(self, tmp_path, monkeypatch):
        """Clear on a provider with no file should not raise."""
        rate_dir = tmp_path / "rate_limits"
        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda provider: os.path.join(str(rate_dir), f"{provider}.json"),
        )

        # Should not raise
        clear_provider_rate_limit("nonexistent-provider")


class TestNoFile:
    def test_read_nonexistent_provider_returns_none(self, tmp_path, monkeypatch):
        """Read non-existent provider, verify returns None."""
        rate_dir = tmp_path / "rate_limits"
        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda provider: os.path.join(str(rate_dir), f"{provider}.json"),
        )

        remaining = provider_rate_limit_remaining("nonexistent-provider")
        assert remaining is None


class TestNotExhausted:
    def test_healthy_state_not_written(self, tmp_path, monkeypatch):
        """Record state with remaining > floor, verify file is NOT written."""
        rate_dir = tmp_path / "rate_limits"
        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda provider: os.path.join(str(rate_dir), f"{provider}.json"),
        )

        provider = "test-provider"
        state = _make_healthy_state()

        record_provider_rate_limit(provider, state)

        path = rate_dir / f"{provider}.json"
        assert not path.exists()

    def test_requests_at_floor_is_written(self, tmp_path, monkeypatch):
        """State with remaining == floor should be written."""
        rate_dir = tmp_path / "rate_limits"
        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda provider: os.path.join(str(rate_dir), f"{provider}.json"),
        )

        provider = "test-provider"
        # remaining == PREDICTIVE_RATE_LIMIT_REQUEST_FLOOR (1) triggers skip
        state = _make_exhausted_state(
            remaining_requests=PREDICTIVE_RATE_LIMIT_REQUEST_FLOOR,
            reset_requests=10.0,
        )

        record_provider_rate_limit(provider, state)

        path = rate_dir / f"{provider}.json"
        assert path.exists()

    def test_tokens_below_floor_is_written(self, tmp_path, monkeypatch):
        """State with remaining tokens below floor should be written."""
        rate_dir = tmp_path / "rate_limits"
        monkeypatch.setattr(
            "agent.provider_rate_guard._state_path",
            lambda provider: os.path.join(str(rate_dir), f"{provider}.json"),
        )

        provider = "test-provider"
        state = _make_exhausted_state(
            remaining_requests=795,
            remaining_tokens=PREDICTIVE_RATE_LIMIT_TOKEN_FLOOR - 1,
            reset_requests=45.0,
            reset_tokens=15.0,
        )

        record_provider_rate_limit(provider, state)

        path = rate_dir / f"{provider}.json"
        assert path.exists()
