import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import PROCESSED_DIR, CRITERIA

CRITERIA_IDS = [c["id"] for c in CRITERIA]

AXIS_LABELS = {
    "statement_program_consistency": "Stmt.–program\nconsistency",
    "proposal_specificity": "Proposal\nspecificity",
    "communication_clarity": "Communication\nclarity",
    "economic_coverage": "Economic\ncoverage",
    "social_coverage": "Social\ncoverage",
    "environmental_coverage": "Environmental\ncoverage",
    "tone_moderation": "Tone\nmoderation",
    "internal_cohesion": "Internal\ncohesion",
    "positional_stability": "Positional\nstability",
}

MODEL_ORDER = [
    "gemini-3.5-flash", "mistral-medium-3-5", "qwen3.6",
    "llama-3.3", "gpt-oss",
    "nemotron-3-super",
]
MODEL_SHORT = {
    "gemini-3.5-flash": "gemini", "mistral-medium-3-5": "mistral",
    "qwen3.6-27b": "qwen", "llama-3.3-70b-versatile": "llama",
    "gpt-oss-120b-groq": "gpt-oss", "nemotron-3-super-120b-a12b": "nemotron",
}
MODEL_COLOR = {
    "gemini-3.5-flash": "#4477aa",
    "mistral-medium-3-5": "#ee6677",
    "qwen3.6": "#228833",
    "llama-3.3": "#ccbb44",
    "gpt-oss": "#aa3377",
    "nemotron-3-super": "#66ccee",
}


def _axis_labels() -> list[str]:
    return [AXIS_LABELS.get(cid, cid.replace("_", "\n")) for cid in CRITERIA_IDS]


def _align_polar_labels(ax, angles) -> None:
    for angle, label in zip(angles, ax.get_xticklabels()):
        cos, sin = np.cos(angle), np.sin(angle)
        if cos > 0.15:
            label.set_horizontalalignment("left")
        elif cos < -0.15:
            label.set_horizontalalignment("right")
        else:
            label.set_horizontalalignment("center")
        if sin > 0.7:
            label.set_verticalalignment("bottom")
        elif sin < -0.7:
            label.set_verticalalignment("top")
        else:
            label.set_verticalalignment("center")


def radar_chart_for_entity(df: pd.DataFrame, entity_name: str, out_dir) -> None:
    subset = df[df["entity_name"] == entity_name]
    models = [m for m in MODEL_ORDER if m in set(subset["model"])]

    angles = np.linspace(0, 2 * np.pi, len(CRITERIA_IDS), endpoint=False).tolist()
    closed = angles + angles[:1]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4.6, 4.6))
    for model in models:
        vals = [
            subset.loc[(subset["model"] == model) & (subset["criterio"] == cid), "mean"].mean()
            for cid in CRITERIA_IDS
        ]
        vals += vals[:1]
        ax.plot(closed, vals, color=MODEL_COLOR[model], lw=1.7, zorder=3,
                solid_joinstyle="round")

    ax.set_ylim(1, 5)
    ax.set_yticks([2, 3, 4])
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(_axis_labels(), fontsize=8.5)
    ax.tick_params(axis="x", pad=9)
    _align_polar_labels(ax, angles)
    ax.grid(color="0.75", lw=0.6, alpha=0.9)
    ax.spines["polar"].set_color("0.6")
    ax.spines["polar"].set_linewidth(0.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = entity_name.replace(" ", "_").replace("'", "")
    fig.savefig(out_dir / f"radar_{safe_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_legend(out_dir, present_models: list[str] | None = None) -> None:
    models = present_models or MODEL_ORDER
    handles = [
        Line2D([], [], color=MODEL_COLOR[m], lw=2.2, label=MODEL_SHORT.get(m, m))
        for m in models
    ]
    fig = plt.figure(figsize=(7.4, 0.42))
    fig.legend(handles=handles, loc="center", ncol=len(handles),
               frameon=False, fontsize=9, handlelength=1.8,
               columnspacing=1.9, handletextpad=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "radar_legend.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary_path = PROCESSED_DIR / "summary_mean_std.csv"
    if not summary_path.exists():
        print(f"{summary_path} not found. Run aggregate_analysis.py first.")
        return

    summary = pd.read_csv(summary_path)
    out_dir = PROCESSED_DIR / "plots"
    for entity_name in summary["entity_name"].unique():
        radar_chart_for_entity(summary, entity_name, out_dir)
    save_legend(out_dir)
    print(f"Plots saved to {out_dir} (radar_*.pdf + radar_legend.pdf)")


if __name__ == "__main__":
    main()
