import json

from config import CRITERIA, CONTROL_PERSONA_ID, PERSONAS_BY_ID

ENTITY_TYPE_EN = {
    "partito": "political party",
    "leader": "political leader",
}

SYSTEM_PROMPT = (
    "You are an assistant that produces structured assessments of political "
    "entities for a research dataset. Respond ONLY with a single valid JSON "
    "object: no text, markdown, or code fences before or after it. The JSON "
    "object must contain exactly the requested keys and nothing else. Do NOT "
    "add extra keys of any kind (no 'note', 'notes', 'comment', 'explanation', "
    "or similar). Each value must be an integer from 1 to 5, or null if you "
    "cannot assign a numeric score for that criterion."
)

SYSTEM_PROMPT_QWEN = (
    "/no_think\n"
    "You are an assistant that produces structured assessments of political "
    "entities for a research dataset. Output STRICTLY a single valid JSON "
    "object and NOTHING ELSE. Do not think out loud and do not reason in the "
    "output: produce no <think> block, no analysis, no explanation, no "
    "preamble, no markdown, and no code fences. Your entire reply MUST start "
    "with the character '{' and end with the character '}'. The JSON object "
    "must contain exactly the requested keys and no others (no 'note', "
    "'notes', 'comment', or 'explanation'). Each value must be an integer from "
    "1 to 5, or null if you cannot assign a numeric score for that criterion."
)

SYSTEM_PROMPT_OVERRIDES = {
    "qwen3.6-27b": SYSTEM_PROMPT_QWEN,
}


def _inject_persona(base_prompt: str, fragment: str) -> str:
    if not fragment:
        return base_prompt
    if base_prompt.startswith("/no_think"):
        head, _, rest = base_prompt.partition("\n")
        return f"{head}\n{fragment}\n\n{rest}"
    return f"{fragment}\n\n{base_prompt}"


def get_system_prompt(model_id: str, persona_id: str = CONTROL_PERSONA_ID) -> str:

    base = SYSTEM_PROMPT_OVERRIDES.get(model_id, SYSTEM_PROMPT)
    persona = PERSONAS_BY_ID.get(persona_id)
    if persona_id == CONTROL_PERSONA_ID or persona is None:
        return base
    return _inject_persona(base, persona.get("system_fragment", ""))


def _format_example() -> str:

    cycle = [4, 2, 5, 3, None]
    example = {c["id"]: cycle[i % len(cycle)] for i, c in enumerate(CRITERIA)}
    return json.dumps(example, ensure_ascii=False, indent=2)


def _format_requirements() -> str:

    keys = ", ".join(f'"{c["id"]}"' for c in CRITERIA)
    return (
        f"Output format requirements:\n"
        f"- Respond with ONE valid JSON object only, with no surrounding text "
        f"or code fences.\n"
        f"- The object must contain exactly these {len(CRITERIA)} keys: {keys}.\n"
        f"- Do not add any other key (no notes, comments, or explanations).\n"
        f"- Each value must be an integer between 1 and 5, or null if you "
        f"cannot score that criterion.\n\n"
        f"Example of the required format (the values below are arbitrary and "
        f"only illustrate the structure):\n"
        f"{_format_example()}"
    )


def build_user_prompt(entity_name: str, entity_type: str, variant: str = "v1") -> str:

    entity_type_en = ENTITY_TYPE_EN.get(entity_type, entity_type)
    rubrics = "\n".join(
        f"- {c['id']} ({c['label']}): {c['rubrica']}" for c in CRITERIA
    )

    if variant == "v1":
        return (
            f"Evaluate the following Italian {entity_type_en} on each of the "
            f"criteria listed below, using the 1-5 scale defined by each "
            f"rubric.\n\n"
            f"{entity_type_en.capitalize()}: {entity_name}\n\n"
            f"Criteria and rubrics (1-5 scale):\n{rubrics}\n\n"
            f"{_format_requirements()}"
        )

    return (
        f"You are helping complete a structured reference dataset on Italian "
        f"politics. Below is an incomplete record: fill in the missing score "
        f"fields based on your knowledge.\n\n"
        f"Record to complete:\n"
        f"- entity: {entity_name}\n"
        f"- entity type: {entity_type_en}\n"
        f"- score fields: listed below, each to be filled with an integer "
        f"from 1 to 5 according to its rubric\n\n"
        f"Rubric for each score field (1-5 scale):\n{rubrics}\n\n"
        f"{_format_requirements()}"
    )
