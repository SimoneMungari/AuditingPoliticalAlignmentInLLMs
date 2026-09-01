from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    raw_text: str
    ok: bool
    error: Optional[str] = None
    latency_s: Optional[float] = None


class LLMProvider:

    def __init__(self, model_id: str, model_name: str, api_key: str, **kwargs):
        self.model_id = model_id
        self.model_name = model_name
        self.api_key = api_key
        self.extra = kwargs

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError
