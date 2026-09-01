import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from config import PROCESSED_DIR, CONTROL_PERSONA_ID


def main() -> None:
    scores_path = PROCESSED_DIR / "scores.csv"
    if not scores_path.exists():
        print(f"{scores_path} not found. Run parse_responses.py.")
        return

    df = pd.read_csv(scores_path)
    if "persona" in df.columns:
        n_before = len(df)
        df = df[df["persona"] == CONTROL_PERSONA_ID].copy()

    summary = (
        df.groupby(["model", "entity_name", "criterio"])["punteggio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(PROCESSED_DIR / "summary_mean_std.csv", index=False)

    refusal_rate = (
        df.groupby("model")
        .agg(
            tasso_rifiuto=("rifiuto_o_non_parsabile", "mean"),
            tasso_errore_api=("api_error", "mean"),
            n_osservazioni=("run_id", "count"),
        )
        .reset_index()
    )
    refusal_rate.to_csv(PROCESSED_DIR / "refusal_rates.csv", index=False)

    pivot = (
        df.dropna(subset=["punteggio"])
        .groupby(["model", "entity_name", "criterio"])["punteggio"]
        .mean()
        .unstack("model")
    )
    correlation = pivot.corr()
    correlation.to_csv(PROCESSED_DIR / "model_correlation.csv")


if __name__ == "__main__":
    main()
