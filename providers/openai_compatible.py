import time

from openai import OpenAI, APIError, RateLimitError

from providers.base import LLMProvider, LLMResponse


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model_id: str, model_name: str, api_key: str, base_url: str, **kwargs):
        super().__init__(model_id, model_name, api_key, **kwargs)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        start = time.monotonic()
        extra_body = self.extra.get("extra_body")
        try:
            kwargs = dict(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.extra.get("timeout", 60),
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = self.client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            return LLMResponse(raw_text=text, ok=True, latency_s=time.monotonic() - start)
        except RateLimitError as e:
            return LLMResponse(raw_text="", ok=False, error=f"rate_limit: {e}")
        except APIError as e:
            return LLMResponse(raw_text="", ok=False, error=f"api_error: {e}")
        except Exception as e:
            return LLMResponse(raw_text="", ok=False, error=f"unknown_error: {e}")
