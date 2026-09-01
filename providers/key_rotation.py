from __future__ import annotations

import os
from typing import Callable, List, Optional

from providers.base import LLMProvider, LLMResponse

_RATE_LIMIT_MARKERS = (
    "rate_limit",
    "rate limit",
    "429",
    "quota",
    "resource_exhausted",
    "too many requests",
)


def is_rate_limit_error(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def load_api_keys(env_var_name: str) -> List[str]:
    raw = os.environ.get(env_var_name, "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


class RotatingKeyProvider(LLMProvider):

    def __init__(
        self,
        model_id: str,
        model_name: str,
        api_keys: List[str],
        provider_cls: Callable[..., LLMProvider],
        base_url: str,
        rpd_limit: Optional[int] = None,
        **kwargs,
    ):
        if not api_keys:
            raise ValueError("api_keys must not be empty")
        super().__init__(model_id, model_name, api_keys[0], **kwargs)
        self.api_keys = api_keys
        self.provider_cls = provider_cls
        self.base_url = base_url
        self.rpd_limit = rpd_limit
        self.kwargs = kwargs

        self._key_index = 0
        self._requests_on_current_key = 0
        self._providers: List[Optional[LLMProvider]] = [None] * len(api_keys)

    @property
    def active_key_index(self) -> int:
        return self._key_index

    @property
    def n_keys(self) -> int:
        return len(self.api_keys)

    def _get_provider(self, idx: int) -> LLMProvider:
        if self._providers[idx] is None:
            self._providers[idx] = self.provider_cls(
                model_id=self.model_id,
                model_name=self.model_name,
                api_key=self.api_keys[idx],
                base_url=self.base_url,
                **self.kwargs,
            )
        return self._providers[idx]

    def _rotate(self) -> bool:
        if self._key_index + 1 >= len(self.api_keys):
            return False
        self._key_index += 1
        self._requests_on_current_key = 0
        print(
            f"    [KEY] {self.model_id}: key {self._key_index} exhausted, "
            f"switching to key {self._key_index + 1}/{len(self.api_keys)} "
            f"({_mask(self.api_keys[self._key_index])})"
        )
        return True

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self.rpd_limit and self._requests_on_current_key >= self.rpd_limit:
            self._rotate()

        tried_indices = set()
        while True:
            idx = self._key_index
            provider = self._get_provider(idx)
            response = provider.query(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._requests_on_current_key += 1

            if response.ok or not is_rate_limit_error(response.error):
                return response

            tried_indices.add(idx)
            if len(tried_indices) >= len(self.api_keys):
                return response
            if not self._rotate():
                return response
