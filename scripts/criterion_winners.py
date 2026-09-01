from __future__ import annotations

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, CONTROL_PERSONA_ID

CRIT_SHORT = {
    "statement_program_consistency": "Stmt.-program consistency",
    "proposal_specificity": "Proposal specificity",
    "communication_clarity": "Communication clarity",
    "economic_coverage": "Economic coverage",
    "social_coverage": "Social coverage",
    "environmental_coverage": "Environmental coverage",
    "tone_moderation": "Tone moderation",
    "internal_cohesion": "Internal cohesion",
    "positional_stability": "Positional stability",
}
CRIT_ORDER = list(CRIT_SHORT.keys())

MODEL_ORDER = [
    "gemini-3.5-flash", "mistral-medium-3-5", "qwen3.6-27b",
    "llama-3.3-70b-versatile", "gpt-oss-120b-groq",
    "nemotron-3-super-120b-a12b",
]
MODEL_SHORT = {
    "gemini-3.5-flash": "gemini", "mistral-medium-3-5": "mistral",
    "qwen3.6-27b": "qwen", "llama-3.3-70b-versatile": "llama",
    "gpt-oss-120b-groq": "gpt-oss", "nemotron-3-super-120b-a12b": "nemotron",
}
TYPE_LABEL = {"partito": "Parties", "leader": "Leaders"}

TOL = 1e-9

SHORT_ENTITY = {
    "Alleanza Verdi e Sinistra": "AVS", "Partito Democratico": "PD",
    "Movimento 5 Stelle": "M5S", "Fratelli d'Italia": "FdI",
    "Forza Italia": "FI", "Italia Viva": "IV", "Noi Moderati": "NM",
    "Futuro Nazionale": "FN", "Azione": "Azione", "Lega": "Lega",
}


def short_entity(name: str) -> str:
    if name in SHORT_ENTITY:
        return SHORT_ENTITY[name]
    parts = str(name).split()
    return parts[-1] if parts else str(name)


def cell_label(winners: list[str]) -> str:
    shorts = [short_entity(w) for w in winners]
    if len(shorts) == 1:
        return shorts[0]
    if len(shorts) == 2:
        return "=".join(shorts)
    return f"{shorts[0]}+{len(shorts) - 1}"


def _cell_winner(g: pd.DataFrame) -> dict | None:
    means = g.groupby("entity_name")["punteggio"].mean().sort_values(ascending=False)
    if len(means) < 2:
        return None

    top = float(means.iloc[0])
    winners = sorted(means.index[means >= top - TOL].tolist())
    lower = means[means < top - TOL]

    return {
        "winners": "|".join(winners),
        "n_winners": len(winners),
        "winner_mean": top,
        "runner_up": str(lower.index[0]) if len(lower) else None,
        "runner_up_mean": float(lower.iloc[0]) if len(lower) else np.nan,
        "margin": float(top - lower.iloc[0]) if len(lower) else np.nan,
        "tied": len(winners) > 1,
        "n_candidates": int(len(means)),
    }


def compute_winners(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, criterio, etype), g in df.groupby(["model", "criterio", "entity_type"]):
        res = _cell_winner(g)
        if res is None:
            continue
        rows.append({"model": model, "criterio": criterio, "entity_type": etype, **res})
    out = pd.DataFrame(rows)
    out["criterio"] = pd.Categorical(out["criterio"], CRIT_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    return out.sort_values(["entity_type", "criterio", "model"]).reset_index(drop=True)


def compute_agreement(win: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (criterio, etype), g in win.groupby(["criterio", "entity_type"], observed=True):
        firsts: dict[str, int] = {}
        alone: dict[str, int] = {}
        for _, r in g.iterrows():
            ents = str(r["winners"]).split("|")
            for e in ents:
                firsts[e] = firsts.get(e, 0) + 1
                if len(ents) == 1:
                    alone[e] = alone.get(e, 0) + 1
        tally = pd.Series(firsts).sort_values(ascending=False)
        n_models = int(len(g))
        top = int(tally.iloc[0])
        modal = sorted(tally.index[tally == top].tolist())
        rows.append({
            "criterio": criterio,
            "entity_type": etype,
            "n_models": n_models,
            "n_models_tied_cells": int(g["tied"].sum()),
            "max_tie_width": int(g["n_winners"].max()),
            "n_entities_ranked_first": int(len(tally)),
            "top_entity": modal[0] if len(modal) == 1 else "|".join(modal),
            "top_is_ambiguous": len(modal) > 1,
            "n_models_first": top,
            "n_models_first_alone": int(max(alone.get(e, 0) for e in modal)),
            "share_models_first": top / n_models if n_models else np.nan,
            "lead_over_second": int(top - tally.iloc[len(modal)]) if len(tally) > len(modal) else top,
            "unanimous": bool(len(modal) == 1 and top == n_models),
        })
    out = pd.DataFrame(rows)
    out["criterio"] = pd.Categorical(out["criterio"], CRIT_ORDER, ordered=True)
    return out.sort_values(["entity_type", "criterio"]).reset_index(drop=True)


def compute_positional(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, criterio, etype), g in df.groupby(["model", "criterio", "entity_type"]):
        means = g.groupby("entity_name")["punteggio"].mean().sort_values(ascending=False)
        n = len(means)
        if n < 2:
            continue
        w = 1.0 / np.log2(np.arange(1, n + 1) + 1)
        vals = means.to_numpy()
        pos = np.arange(1, n + 1, dtype=float)
        w_tied, rank_tied = w.copy(), pos.copy()
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(vals[j + 1] - vals[i]) < TOL:
                j += 1
            if j > i:
                w_tied[i:j + 1] = w[i:j + 1].mean()
                rank_tied[i:j + 1] = pos[i:j + 1].mean()
            i = j + 1
        for e, m, wt, rk in zip(means.index, vals, w_tied, rank_tied):
            rows.append({
                "model": model, "criterio": criterio, "entity_type": etype,
                "entity_name": e, "mean_score": float(m),
                "rank": float(rk), "dcg_weight": float(wt),
                "n_entities": int(n),
            })
    out = pd.DataFrame(rows)
    out["criterio"] = pd.Categorical(out["criterio"], CRIT_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    return out.sort_values(["entity_type", "criterio", "model", "rank"]).reset_index(drop=True)


def aggregate_positional(pos: pd.DataFrame) -> pd.DataFrame:
    agg = (pos.groupby(["criterio", "entity_type", "entity_name"], observed=True)
              .agg(pos_score=("dcg_weight", "mean"),
                   pos_score_sd=("dcg_weight", "std"),
                   mean_rank=("rank", "mean"),
                   mean_score=("mean_score", "mean"),
                   n_models=("dcg_weight", "size"))
              .reset_index())
    agg["rank_in_race"] = (agg.groupby(["criterio", "entity_type"], observed=True)["pos_score"]
                              .rank(ascending=False, method="min").astype(int))
    return agg.sort_values(["entity_type", "criterio", "rank_in_race"]).reset_index(drop=True)


def positional_winners(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (criterio, etype), g in agg.groupby(["criterio", "entity_type"], observed=True):
        g = g.sort_values("pos_score", ascending=False)
        top = float(g["pos_score"].iloc[0])
        tied = sorted(g.loc[g["pos_score"] >= top - TOL, "entity_name"].tolist())
        rest = g[g["pos_score"] < top - TOL]
        rows.append({
            "criterio": criterio,
            "entity_type": etype,
            "top_entity": tied[0] if len(tied) == 1 else "|".join(tied),
            "top_is_ambiguous": len(tied) > 1,
            "pos_score": top,
            "mean_rank": float(g["mean_rank"].iloc[0]),
            "runner_up": str(rest["entity_name"].iloc[0]) if len(rest) else None,
            "runner_up_score": float(rest["pos_score"].iloc[0]) if len(rest) else np.nan,
            "gap": float(top - rest["pos_score"].iloc[0]) if len(rest) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["criterio"] = pd.Categorical(out["criterio"], CRIT_ORDER, ordered=True)
    return out.sort_values(["entity_type", "criterio"]).reset_index(drop=True)


def save_heatmap(win: pd.DataFrame, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    types = ["partito", "leader"]
    models = [m for m in MODEL_ORDER if m in win["model"].astype(str).unique()]
    solid = win.loc[~win["tied"], "margin"]
    vmax = float(solid.max()) if len(solid) else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")

    fig, axes = plt.subplots(
        1, 2, figsize=(2.2 + 1.55 * len(models) * 2, 0.52 * len(CRIT_ORDER) + 1.6),
        sharey=True,
    )
    for ax, etype in zip(axes, types):
        sub = win[win["entity_type"] == etype]
        grid = np.full((len(CRIT_ORDER), len(models)), np.nan)
        labels = np.empty(grid.shape, dtype=object)
        labels[:] = ""
        for _, r in sub.iterrows():
            i = CRIT_ORDER.index(str(r["criterio"]))
            j = models.index(str(r["model"]))
            labels[i, j] = cell_label(str(r["winners"]).split("|"))
            grid[i, j] = np.nan if r["tied"] else r["margin"]

        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isnan(grid[i, j]) and labels[i, j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               facecolor="0.88", edgecolor="none"))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not labels[i, j]:
                    continue
                val = grid[i, j]
                dark = (not np.isnan(val)) and norm(val) > 0.55
                ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=6.6,
                        color="white" if dark else "black")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models],
                           rotation=35, ha="right", fontsize=7.5)
        ax.set_title(TYPE_LABEL[etype], fontsize=9)
        ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(CRIT_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)

    axes[0].set_yticks(range(len(CRIT_ORDER)))
    axes[0].set_yticklabels([CRIT_SHORT[c] for c in CRIT_ORDER], fontsize=7.5)
    axes[1].tick_params(axis="y", length=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.030, pad=0.02)
    cb.set_label("Margin over next lower entity (1-5 points); grey $=$ tie at the top",
                 fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    path = PROCESSED_DIR / "scores.csv"
    if not path.exists():
        raise SystemExit("scores.csv not found. Run parse_responses.py first.")

    df = pd.read_csv(path)
    n_tot = len(df)
    if "persona" in df.columns:
        df = df[df["persona"] == CONTROL_PERSONA_ID]
    df = df.dropna(subset=["punteggio"])
    print(f"Control condition, valid scores: {len(df)}/{n_tot} rows.")

    win = compute_winners(df)
    win.to_csv(PROCESSED_DIR / "criterion_winners.csv", index=False)

    agree = compute_agreement(win)
    agree.to_csv(PROCESSED_DIR / "criterion_winner_agreement.csv", index=False)

    pos = compute_positional(df)
    pos.to_csv(PROCESSED_DIR / "criterion_positional_cells.csv", index=False)
    agg = aggregate_positional(pos)
    agg.to_csv(PROCESSED_DIR / "criterion_positional_scores.csv", index=False)
    pwin = positional_winners(agg)
    pwin.to_csv(PROCESSED_DIR / "criterion_positional_winners.csv", index=False)

    save_heatmap(win, PROCESSED_DIR / "plots" / "criterion_winners_heatmap.pdf")

    firsts: dict[str, int] = {}
    firsts_by_type: dict[str, dict[str, int]] = {"partito": {}, "leader": {}}
    for _, r in win.iterrows():
        ents = str(r["winners"]).split("|")
        for e in ents:
            firsts[e] = firsts.get(e, 0) + 1
            d = firsts_by_type[str(r["entity_type"])]
            d[e] = d.get(e, 0) + 1
    tally = pd.Series(firsts).sort_values(ascending=False)

    solid = win[~win["tied"]]
    stats = {
        "method": "count of first places over per-model entity means (no bootstrap); "
                  "every entity sharing the top mean in a cell counts one full first place",
        "n_cells": int(len(win)),
        "n_tied_cells": int(win["tied"].sum()),
        "tied_cell_share": float(win["tied"].mean()),
        "median_margin_untied": float(solid["margin"].median()) if len(solid) else None,
        "max_margin_untied": float(solid["margin"].max()) if len(solid) else None,
        "n_unanimous": int(agree["unanimous"].sum()),
        "n_races": int(len(agree)),
        "n_ambiguous_races": int(agree["top_is_ambiguous"].sum()),
        "firsts_overall": tally.to_dict(),
        "top_entity": tally.index[0] if len(tally) else None,
        "top_entity_firsts": int(tally.iloc[0]) if len(tally) else 0,
        "n_entities_ranked_first": int(tally.size),
        "by_type": {},
        "agreement": agree.assign(criterio=agree["criterio"].astype(str))
                          .to_dict(orient="records"),
        "positional": {
            "weighting": "1/log2(rank+1), ties get the mean weight of the "
                         "positions they share; averaged over models",
            "n_races_same_top_as_first_place_count": int(sum(
                str(p["top_entity"]) == str(a["top_entity"])
                for p, a in zip(
                    pwin.sort_values(["entity_type", "criterio"]).to_dict("records"),
                    agree.sort_values(["entity_type", "criterio"]).to_dict("records"))
            )),
            "winners": pwin.assign(criterio=pwin["criterio"].astype(str))
                           .to_dict(orient="records"),
        },
    }
    for etype in ["partito", "leader"]:
        d = pd.Series(firsts_by_type[etype]).sort_values(ascending=False)
        stats["by_type"][etype] = {
            "n_entities_ranked_first": int(d.size),
            "top": d.index[0] if len(d) else None,
            "top_firsts": int(d.iloc[0]) if len(d) else 0,
            "firsts": {k: int(v) for k, v in d.to_dict().items()},
        }
    with open(PROCESSED_DIR / "criterion_winners_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\ncriterion_winners.csv ({len(win)} cells, "
          f"{stats['n_tied_cells']} with a tie at the top)")
    print(f"criterion_winner_agreement.csv ({len(agree)} races)")
    print("plots/criterion_winners_heatmap.pdf")
    print(f"\nEntities ranked first at least once: {stats['n_entities_ranked_first']}; "
          f"most frequent: {stats['top_entity']} "
          f"({stats['top_entity_firsts']} first places)")
    print(f"Unanimous races: {stats['n_unanimous']}/{stats['n_races']} "
          f"(ambiguous winner in {stats['n_ambiguous_races']})")
    print("\nAgreement by criterion:")
    show = agree.copy()
    show["criterio"] = show["criterio"].astype(str).map(CRIT_SHORT)
    print(show[["entity_type", "criterio", "top_entity", "n_models_first",
                "n_models", "n_entities_ranked_first", "n_models_tied_cells"]]
          .to_string(index=False))

    print("\nPositional score (DCG, averaged over models):")
    showp = pwin.copy()
    showp["criterio"] = showp["criterio"].astype(str).map(CRIT_SHORT)
    print(showp[["entity_type", "criterio", "top_entity", "pos_score",
                 "mean_rank", "runner_up", "gap"]].round(3).to_string(index=False))
    print(f"\nSame winner as the first-place count in "
          f"{stats['positional']['n_races_same_top_as_first_place_count']}/"
          f"{len(pwin)} races.")


if __name__ == "__main__":
    main()
