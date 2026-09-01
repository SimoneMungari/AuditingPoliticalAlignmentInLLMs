import os
import sys
import time
import json
import uuid
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from dotenv import load_dotenv

from config import (
    MODELS, ENTITIES, N_REPETITIONS, PROMPT_VARIANTS,
    TEMPERATURE, MAX_TOKENS, RAW_PERSONA_DIR, MAX_FORMAT_RETRIES,
    PERSONAS,
)
from prompts.builder import get_system_prompt, build_user_prompt
from providers.openai_compatible import OpenAICompatibleProvider
from providers.key_rotation import RotatingKeyProvider, load_api_keys
from naming import raw_persona_filename
from validation import is_raw_text_well_formatted

load_dotenv()


def build_provider(model_cfg: dict) -> RotatingKeyProvider:
    api_keys = load_api_keys(model_cfg["api_key_env"])
    if not api_keys:
        raise RuntimeError(
            f"Environment variable {model_cfg['api_key_env']} is not set "
            f"(required for model {model_cfg['id']}). See .env.example."
        )
    return RotatingKeyProvider(
        model_id=model_cfg["id"],
        model_name=model_cfg["model_name"],
        api_keys=api_keys,
        provider_cls=OpenAICompatibleProvider,
        base_url=model_cfg["base_url"],
        rpd_limit=model_cfg.get("rpd_limit"),
        extra_body=model_cfg.get("extra_body"),
    )


def record_is_complete(rec: dict) -> bool:
    return rec.get("ok") is True and is_raw_text_well_formatted(rec.get("raw_text"))


def save_response(model_id: str, record: dict) -> bool:
    out_dir = RAW_PERSONA_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / raw_persona_filename(
        record["entity_name"], record["prompt_variant"],
        record["persona"], record["repetition"],
    )
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = None
        if existing is not None and record_is_complete(existing):
            return False
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return True


def load_existing_reps(model_id: str) -> dict:
    reps: dict = {}
    model_dir = RAW_PERSONA_DIR / model_id
    if not model_dir.exists():
        return reps
    for f in model_dir.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not record_is_complete(rec):
            continue
        key = (rec.get("entity_name"), rec.get("prompt_variant"), rec.get("persona"))
        reps.setdefault(key, set()).add(rec.get("repetition"))
    return reps


def model_fully_done(existing_reps: dict) -> bool:
    return all(
        len(existing_reps.get((entity["name"], variant, persona["id"]), set())) >= N_REPETITIONS
        for entity in ENTITIES
        for variant in PROMPT_VARIANTS
        for persona in PERSONAS
    )


def main() -> None:
    for model_cfg in MODELS:
        model_id = model_cfg["id"]
        existing_reps = load_existing_reps(model_id)

        if model_fully_done(existing_reps):
            n_combos = len(ENTITIES) * len(PROMPT_VARIANTS) * len(PERSONAS)
            print(
                f"\n=== Model: {model_id} === "
                f"[SKIP] already complete ({N_REPETITIONS} tries x {n_combos} "
                f"entity/variant/persona combinations): no API calls needed."
            )
            continue

        try:
            provider = build_provider(model_cfg)
        except RuntimeError as e:
            print(f"\n=== Model: {model_id} ===")
            print(f"  [SKIP] {e}")
            continue

        print(f"\n=== Model: {model_id} === ({provider.n_keys} API keys available)")

        min_delay_s = 60.0 / model_cfg["rpm_limit"] if model_cfg.get("rpm_limit") else 1.0
        requests_today = 0
        rpd_limit_per_key = model_cfg.get("rpd_limit")
        rpd_limit_total = rpd_limit_per_key * provider.n_keys if rpd_limit_per_key else None
        limit_reached = False

        for persona in PERSONAS:
            if limit_reached:
                break
            persona_id = persona["id"]
            system_prompt = get_system_prompt(model_id, persona_id)
            print(f"  --- persona: {persona_id} ({persona['label']}) ---")

            for entity in ENTITIES:
                if limit_reached:
                    break
                for variant in PROMPT_VARIANTS:
                    if limit_reached:
                        break

                    done_reps = existing_reps.get((entity["name"], variant, persona_id), set())
                    if len(done_reps) >= N_REPETITIONS:
                        print(
                            f"  [SKIP] {entity['name']} | {variant} | {persona_id} | "
                            f"already {len(done_reps)}/{N_REPETITIONS} tries completed"
                        )
                        continue
                    if done_reps:
                        print(
                            f"  [RESUME] {entity['name']} | {variant} | {persona_id} | "
                            f"{len(done_reps)}/{N_REPETITIONS} already done, {N_REPETITIONS - len(done_reps)} left"
                        )

                    for rep in range(N_REPETITIONS):
                        if rep in done_reps:
                            continue
                        if rpd_limit_total and requests_today >= rpd_limit_total:
                            print(
                                f"  [STOP] Overall daily limit reached "
                                f"({rpd_limit_total} = {rpd_limit_per_key} x {provider.n_keys} keys) "
                                f"for {model_cfg['id']}"
                            )
                            limit_reached = True
                            break

                        user_prompt = build_user_prompt(entity["name"], entity["type"], variant)

                        response = None
                        format_attempts = 0
                        for attempt in range(1, MAX_FORMAT_RETRIES + 2):
                            response = provider.query(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                temperature=TEMPERATURE,
                                max_tokens=MAX_TOKENS,
                            )
                            requests_today += 1
                            format_attempts = attempt

                            if not response.ok:
                                break
                            if is_raw_text_well_formatted(response.raw_text):
                                break
                            if attempt > MAX_FORMAT_RETRIES:
                                print(
                                    f"    [FORMAT] {entity['name']} | {variant} | {persona_id} | rep {rep} | "
                                    f"still malformed after {MAX_FORMAT_RETRIES} retries: "
                                    f"saving the last response anyway"
                                )
                                break
                            print(
                                f"    [RETRY-FORMAT] {entity['name']} | {variant} | {persona_id} | rep {rep} | "
                                f"malformed raw_text, retrying ({attempt}/{MAX_FORMAT_RETRIES})"
                            )
                            if rpd_limit_total and requests_today >= rpd_limit_total:
                                print("    [STOP] daily quota exhausted during format retries")
                                limit_reached = True
                                break
                            time.sleep(min_delay_s)

                        record = {
                            "run_id": str(uuid.uuid4()),
                            "model": model_cfg["id"],
                            "model_name": model_cfg["model_name"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "entity_name": entity["name"],
                            "entity_type": entity["type"],
                            "prompt_variant": variant,
                            "persona": persona_id,
                            "persona_label": persona["label"],
                            "repetition": rep,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "ok": response.ok,
                            "error": response.error,
                            "raw_text": response.raw_text,
                            "latency_s": response.latency_s,
                            "api_key_index": provider.active_key_index,
                            "format_attempts": format_attempts,
                        }
                        save_response(model_cfg["id"], record)

                        status = "OK" if response.ok else f"ERR ({response.error})"
                        retry_note = f" | {format_attempts} format attempts" if format_attempts > 1 else ""
                        print(f"  [{status}] {entity['name']} | {variant} | {persona_id} | rep {rep} | key {provider.active_key_index + 1}/{provider.n_keys}{retry_note}")

                        time.sleep(min_delay_s if response.ok else min_delay_s * 3)

                        if limit_reached:
                            break


if __name__ == "__main__":
    main()
