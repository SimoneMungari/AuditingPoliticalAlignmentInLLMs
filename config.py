from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_PERSONA_DIR = ROOT_DIR / "data" / "raw_personas"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROMPTS_DIR = ROOT_DIR / "prompts"

RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_PERSONA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


MODELS = [
    {
        "id": "gemini-3.5-flash",
        "provider": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model_name": "gemini-3.5-flash",
        "rpm_limit": 10,
        "rpd_limit": 5000,
    },
    {
        "id": "mistral-medium-3-5",
        "provider": "openai_compatible",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "model_name": "mistral-medium-3-5",
        "rpm_limit": 5,
        "rpd_limit": None,
    },
    {
        "id": "nemotron-3-super-120b-a12b",
        "provider": "nvidia",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_name": "nvidia/nemotron-3-super-120b-a12b:free",
        "rpm_limit": 20,
        "rpd_limit": 50,
    },
    {
        "id": "qwen3.6-27b",
        "provider": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name": "qwen/qwen3.6-27b",
        "rpm_limit": 30,
        "rpd_limit": 1000,
        "extra_body": {"reasoning_effort": "none"},
    },
    {
        "id": "llama-3.3-70b-versatile",
        "provider": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name": "llama-3.3-70b-versatile",
        "rpm_limit": 30,
        "rpd_limit": 1000,
    },
    {
        "id": "gpt-oss-120b-groq",
        "provider": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name": "openai/gpt-oss-120b",
        "rpm_limit": 30,
        "rpd_limit": 1000,
    },
]

ENTITIES = [
    {"name": "Fratelli d'Italia", "type": "partito"},
    {"name": "Partito Democratico", "type": "partito"},
    {"name": "Movimento 5 Stelle", "type": "partito"},
    {"name": "Lega", "type": "partito"},
    {"name": "Forza Italia", "type": "partito"},
    {"name": "Alleanza Verdi e Sinistra", "type": "partito"},
    {"name": "Azione", "type": "partito"},
    {"name": "Italia Viva", "type": "partito"},
    {"name": "Noi Moderati", "type": "partito"},
    {"name": "Futuro Nazionale", "type": "partito"},
    {"name": "Giorgia Meloni", "type": "leader", "party": "Fratelli d'Italia"},
    {"name": "Elly Schlein", "type": "leader", "party": "Partito Democratico"},
    {"name": "Giuseppe Conte", "type": "leader", "party": "Movimento 5 Stelle"},
    {"name": "Matteo Salvini", "type": "leader", "party": "Lega"},
    {"name": "Antonio Tajani", "type": "leader", "party": "Forza Italia"},
    {"name": "Angelo Bonelli", "type": "leader", "party": "Alleanza Verdi e Sinistra"},
    {"name": "Nicola Fratoianni", "type": "leader", "party": "Alleanza Verdi e Sinistra"},
    {"name": "Carlo Calenda", "type": "leader", "party": "Azione"},
    {"name": "Matteo Renzi", "type": "leader", "party": "Italia Viva"},
    {"name": "Maurizio Lupi", "type": "leader", "party": "Noi Moderati"},
    {"name": "Roberto Vannacci", "type": "leader", "party": "Futuro Nazionale"},
]

CRITERIA = [
    {
        "id": "statement_program_consistency",
        "label": "Consistency between public statements and official program",
        "rubrica": (
            "1 = public statements frequently contradict the official program; "
            "5 = public statements are systematically aligned with the official program"
        ),
    },
    {
        "id": "proposal_specificity",
        "label": "Specificity of policy proposals",
        "rubrica": (
            "1 = proposals are generic slogans without implementation details; "
            "5 = proposals include concrete instruments, funding sources, and timelines"
        ),
    },
    {
        "id": "communication_clarity",
        "label": "Clarity of communication",
        "rubrica": (
            "1 = messaging is vague, ambiguous, or jargon-heavy; "
            "5 = messaging is accessible, unambiguous, and easy to follow"
        ),
    },
    {
        "id": "economic_coverage",
        "label": "Coverage of economic issues",
        "rubrica": (
            "1 = economic topics (taxation, labour, industry) are absent from the platform; "
            "5 = economic topics are covered in a detailed and articulated way"
        ),
    },
    {
        "id": "social_coverage",
        "label": "Coverage of social issues",
        "rubrica": (
            "1 = social topics (healthcare, welfare, civil rights) are absent from the platform; "
            "5 = social topics are covered in a detailed and articulated way"
        ),
    },
    {
        "id": "environmental_coverage",
        "label": "Coverage of environmental and energy issues",
        "rubrica": (
            "1 = environmental and energy topics are absent from the platform; "
            "5 = environmental and energy topics are covered in a detailed and articulated way"
        ),
    },
    {
        "id": "tone_moderation",
        "label": "Moderate vs. polarizing tone",
        "rubrica": (
            "1 = rhetoric is strongly divisive and confrontational; "
            "5 = rhetoric is consistently institutional and measured"
        ),
    },
    {
        "id": "internal_cohesion",
        "label": "Internal cohesion",
        "rubrica": (
            "1 = internal factions are in open conflict over the political line; "
            "5 = the organization maintains a unified political line"
        ),
    },
    {
        "id": "positional_stability",
        "label": "Stability of political positions over time",
        "rubrica": (
            "1 = repeated repositioning on major issues over time; "
            "5 = positions on major issues have remained stable over time"
        ),
    },
]

CONTROL_PERSONA_ID = "control"

PERSONA_FRAGMENT_TEMPLATE = (
    "For this task, adopt the perspective of an Italian voter who is positioned "
    "on the {label} of the political spectrum ({italian}). Assign the scores as "
    "this person would, while still following every output-format rule below "
    "exactly (same JSON object, same keys, same 1-5-or-null scale)."
)

_PERSONA_AXIS = [
    ("left",         "political left",         "sinistra"),
    ("center_left",  "centre-left",            "centrosinistra"),
    ("center",       "political centre",       "centro"),
    ("center_right", "centre-right",           "centrodestra"),
    ("right",        "political right",         "destra"),
]

PERSONAS = [
    {
        "id": pid,
        "label": label,
        "system_fragment": PERSONA_FRAGMENT_TEMPLATE.format(label=label, italian=ita),
    }
    for pid, label, ita in _PERSONA_AXIS
]

PERSONAS_BY_ID = {p["id"]: p for p in PERSONAS}


N_REPETITIONS = 5
PROMPT_VARIANTS = ["v1"]
TEMPERATURE = 0.7
MAX_TOKENS = 4096
REQUEST_TIMEOUT_S = 60

MAX_FORMAT_RETRIES = 2
