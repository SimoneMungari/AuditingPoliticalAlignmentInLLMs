from __future__ import annotations

import re
import unicodedata


def slug(value) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "none"


def raw_filename(entity_name, prompt_variant, repetition) -> str:
    return f"{slug(entity_name)}_{slug(prompt_variant)}_{repetition}.json"


def raw_persona_filename(entity_name, prompt_variant, persona_id, repetition) -> str:
    return (
        f"{slug(entity_name)}_{slug(prompt_variant)}"
        f"_{slug(persona_id)}_{repetition}.json"
    )
