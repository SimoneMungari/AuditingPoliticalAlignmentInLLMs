import os
import sys
import json
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from config import RAW_DIR, RAW_PERSONA_DIR, PROCESSED_DIR, CRITERIA, CONTROL_PERSONA_ID

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


def rows_from_root(root, rows: list) -> int:
    if not root.exists():
        return 0
    n_files = 0
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        for file in model_dir.glob("*.json"):
            with open(file, encoding="utf-8") as f:
                record = json.load(f)
            n_files += 1

            parsed = extract_json(record["raw_text"]) if record["ok"] else None
            call_ok_but_unparsable = record["ok"] and parsed is None
            persona = record.get("persona", CONTROL_PERSONA_ID)

            for criterio in CRITERIA_IDS:
                punteggio = parsed.get(criterio) if parsed else None
                rows.append({
                    "run_id": record["run_id"],
                    "model": record["model"],
                    "timestamp": record["timestamp"],
                    "entity_name": record["entity_name"],
                    "entity_type": record["entity_type"],
                    "prompt_variant": record["prompt_variant"],
                    "persona": persona,
                    "repetition": record["repetition"],
                    "criterio": criterio,
                    "punteggio": punteggio,
                    "api_error": not record["ok"],
                    "rifiuto_o_non_parsabile": call_ok_but_unparsable or (parsed is not None and punteggio is None),
                    "raw_text": record["raw_text"],
                })
    return n_files


def main() -> None:
    rows = []
    n_baseline = rows_from_root(RAW_DIR, rows)
    n_persona = rows_from_root(RAW_PERSONA_DIR, rows)
    print(f"Files read: {n_baseline} baseline (control) + {n_persona} personas.")

    if not rows:
        print(
            f"No raw responses to process in {RAW_DIR} or "
            f"{RAW_PERSONA_DIR}. Run collect_data.py first."
        )
        return

    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "scores.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset with {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
