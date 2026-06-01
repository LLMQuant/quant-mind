"""Provider auto-detection: map ``cfg.model`` to the right SDK Model + cfg defaults.

User-facing API stays minimal — they just set ``cfg.model="deepseek-chat"``
(or any supported provider's model id). This module:

1. Resolves the provider from the model-name prefix.
2. Reads the right API key env var.
3. Builds the SDK ``Model`` (Chat Completions for non-OpenAI providers
   that lack the Responses API; Responses for OpenAI families).
4. Returns a cfg copy with ``tracing_disabled`` force-set when the
   provider cannot upload traces to ``platform.openai.com``.

**Adding a provider:** append one row to ``_PROVIDERS``. No other code
or docs change.

This intentionally is *not* a QuantMind facade over the SDK — it does
not wrap ``agents.Agent`` or ``openai.AsyncOpenAI``; it only assembles
the SDK's existing types with provider-correct defaults so that
end-users do not need to repeat the boilerplate themselves.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from agents import Model
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from quantmind.configs import BaseFlowCfg

_CfgT = TypeVar("_CfgT", bound=BaseFlowCfg)


@dataclass(frozen=True, slots=True)
class _ProviderConfig:
    """How to talk to a specific LLM provider."""

    name: str
    base_url: str | None  # None = OpenAI SDK default
    api_key_env: str
    use_chat_completions: (
        bool  # True = legacy ChatCompletions API; False = Responses
    )
    tracing_supported: (
        bool  # False = force-disable to avoid 4xx to platform.openai.com
    )


# Order matters: the first prefix that matches ``cfg.model.lower()`` wins.
# More specific prefixes (e.g., "o1-") therefore come before generic ones
# (e.g., "gpt-"). If you add a provider, place it at a sensible spot in
# the priority order.
_PROVIDERS: tuple[tuple[str, _ProviderConfig], ...] = (
    (
        "deepseek-",
        _ProviderConfig(
            name="deepseek",
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            use_chat_completions=True,
            tracing_supported=False,
        ),
    ),
    (
        "o1-",
        _ProviderConfig(
            name="openai",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            use_chat_completions=False,
            tracing_supported=True,
        ),
    ),
    (
        "o3-",
        _ProviderConfig(
            name="openai",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            use_chat_completions=False,
            tracing_supported=True,
        ),
    ),
    (
        "gpt-",
        _ProviderConfig(
            name="openai",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            use_chat_completions=False,
            tracing_supported=True,
        ),
    ),
)


_DEFAULT_PROVIDER: _ProviderConfig = _PROVIDERS[-1][1]


def _resolve(model: str) -> _ProviderConfig:
    """Match ``model`` to a known provider; fall back to OpenAI defaults."""
    model_lc = model.lower()
    for prefix, cfg in _PROVIDERS:
        if model_lc.startswith(prefix):
            return cfg
    return _DEFAULT_PROVIDER


@lru_cache(maxsize=16)
def _get_client(base_url: str | None, api_key: str) -> AsyncOpenAI:
    """Cache one ``AsyncOpenAI`` per ``(base_url, api_key)`` pair.

    Two paper_flow calls with the same provider should reuse a single
    client (fewer sockets, fewer TLS handshakes). The cache key includes
    the api_key so per-tenant clients in a multi-tenant deployment stay
    isolated.
    """
    if base_url is None:
        return AsyncOpenAI(api_key=api_key)
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def configure_provider(cfg: _CfgT) -> tuple[Model, _CfgT]:
    """Resolve ``cfg.model`` → ``(SDK Model, effective cfg)``.

    The returned ``Model`` is meant to be assigned directly to
    ``Agent(model=...)``. The returned cfg is a copy of the input with
    provider-required defaults applied (currently:
    ``tracing_disabled=True`` when the provider can't accept Responses /
    can't upload traces to OpenAI).

    Raises:
        RuntimeError: When the resolved provider's API key env var is
            unset. The message names the env var so users can fix it
            without digging through SDK code.
    """
    provider = _resolve(cfg.model)
    api_key = os.environ.get(provider.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Model {cfg.model!r} maps to provider {provider.name!r}; "
            f"please export {provider.api_key_env} in your environment "
            "before running the flow."
        )
    client = _get_client(provider.base_url, api_key)

    model_obj: Model
    if provider.use_chat_completions:
        model_obj = OpenAIChatCompletionsModel(
            model=cfg.model, openai_client=client
        )
    else:
        model_obj = OpenAIResponsesModel(model=cfg.model, openai_client=client)

    effective = cfg.model_copy(
        update={
            "tracing_disabled": (
                cfg.tracing_disabled or not provider.tracing_supported
            )
        }
    )
    return model_obj, effective
