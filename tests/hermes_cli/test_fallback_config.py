"""Tests for fallback config normalization."""

from hermes_cli.fallback_config import get_fallback_chain


def test_custom_provider_auto_model_resolves_from_custom_providers():
    chain = get_fallback_chain({
        "custom_providers": [
            {
                "name": "openrouter-cn",
                "base_url": "https://openrouter.example/v1",
                "model": "z-ai/glm-5.1",
            },
        ],
        "fallback_providers": [
            {"provider": "custom:openrouter-cn", "model": "auto"},
        ],
    })

    assert chain == [
        {"provider": "custom:openrouter-cn", "model": "z-ai/glm-5.1"},
    ]


def test_custom_provider_missing_model_resolves_from_custom_providers():
    chain = get_fallback_chain({
        "custom_providers": [
            {
                "name": "openrouter-cn",
                "base_url": "https://openrouter.example/v1",
                "model": "z-ai/glm-5.1",
            },
        ],
        "fallback_providers": [
            {"provider": "custom:openrouter-cn"},
        ],
    })

    assert chain == [
        {"provider": "custom:openrouter-cn", "model": "z-ai/glm-5.1"},
    ]


def test_custom_provider_auto_without_resolvable_model_is_dropped():
    chain = get_fallback_chain({
        "custom_providers": [
            {"name": "openrouter-cn", "base_url": "https://openrouter.example/v1"},
        ],
        "fallback_providers": [
            {"provider": "custom:openrouter-cn", "model": "auto"},
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        ],
    })

    assert chain == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
    ]


def test_non_custom_provider_still_requires_model():
    chain = get_fallback_chain({
        "fallback_providers": [
            {"provider": "openrouter"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ],
    })

    assert chain == [
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]


def test_non_custom_provider_auto_model_is_dropped():
    chain = get_fallback_chain({
        "fallback_providers": [
            {"provider": "openrouter", "model": "auto"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ],
    })

    assert chain == [
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]


def test_providers_dict_provider_key_auto_model_resolves():
    chain = get_fallback_chain({
        "providers": {
            "openrouter-cn": {
                "base_url": "https://openrouter.example/v1",
                "model": "z-ai/glm-5.1",
            },
        },
        "fallback_providers": [
            {"provider": "custom:openrouter-cn", "model": "auto"},
        ],
    })

    assert chain == [
        {"provider": "custom:openrouter-cn", "model": "z-ai/glm-5.1"},
    ]
