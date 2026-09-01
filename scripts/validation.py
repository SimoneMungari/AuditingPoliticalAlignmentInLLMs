import json
import re

from config import CRITERIA

CRITERIA_IDS = [c["id"] for c in CRITERIA]


def extract_json(raw_text: str):
    if not raw_text:
        return None
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def is_raw_text_well_formatted(raw_text: str) -> bool:
    parsed = extract_json(raw_text)
    if not isinstance(parsed, dict):
        return False
    for cid in CRITERIA_IDS:
        if cid not in parsed:
            return False
        val = parsed[cid]
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, int) or not (1 <= val <= 5):
            return False
    return True
