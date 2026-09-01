import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED_DIR, CONTROL_PERSONA_ID

ANALYSIS_VARIANT = "v1"

PERSONA_POSITION = {
    "left": -2,
    "center_left": -1,
    "center": 0,
    "center_right": 1,
    "right": 2,
}
PERSONA_ORDER = ["left", "center_left", "center", "center_right", "right"]
PERSONA_LABEL = {
    "left": "Left",
    "center_left": "Centre-left",
    "center": "Centre",
    "center_right": "Centre-right",
    "right": "Right",
}

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = m / (rank + 1) * p[idx]
        running = min(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "scores.csv")
    if "persona" not in df.columns:
        raise SystemExit(""
        )
    n_tot = len(df)
    df = df[df["prompt_variant"] == ANALYSIS_VARIANT].copy()

    return df


def _cell_stat(df: pd.DataFrame, how: str) -> pd.DataFrame:
    return (
        df.dropna(subset=["punteggio"])
        .groupby(["model", "entity_name", "criterio", "persona"])["punteggio"]
        .agg(how)
        .reset_index(name=f"{how}_punteggio")
    )


def score_shift(df: pd.DataFrame) -> pd.DataFrame:
    means = _cell_stat(df, "mean")
    control = (
        means[means["persona"] == CONTROL_PERSONA_ID]
        .drop(columns="persona")
        .rename(columns={"mean_punteggio": "mean_control"})
    )
    treat = means[means["persona"] != CONTROL_PERSONA_ID]
    merged = treat.merge(control, on=["model", "entity_name", "criterio"], how="inner")
    merged["shift"] = merged["mean_punteggio"] - merged["mean_control"]
    merged["persona_position"] = merged["persona"].map(PERSONA_POSITION)
    return merged.sort_values(["persona", "entity_name", "model", "criterio"])


def variance_shift(df: pd.DataFrame) -> pd.DataFrame:
    stds = _cell_stat(df, "std")
    control = (
        stds[stds["persona"] == CONTROL_PERSONA_ID]
        .drop(columns="persona")
        .rename(columns={"std_punteggio": "std_control"})
    )
    treat = stds[stds["persona"] != CONTROL_PERSONA_ID]
    merged = treat.merge(control, on=["model", "entity_name", "criterio"], how="inner")
    merged["delta_std"] = merged["std_punteggio"] - merged["std_control"]
    return merged.sort_values(["persona", "entity_name", "model", "criterio"])


def refusal_rates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "persona"])
        .agg(
            tasso_rifiuto=("rifiuto_o_non_parsabile", "mean"),
            tasso_errore_api=("api_error", "mean"),
            n_osservazioni=("run_id", "count"),
        )
        .reset_index()
        .sort_values(["persona", "model"])
    )


def shift_summary(shift_df: pd.DataFrame) -> pd.DataFrame:
    per_persona = (
        shift_df.groupby("persona")["shift"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .assign(entity_name="__ALL__")
    )
    per_persona_entity = (
        shift_df.groupby(["persona", "entity_name"])["shift"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out = pd.concat([per_persona, per_persona_entity], ignore_index=True)
    return out[["persona", "entity_name", "mean", "std", "count"]].sort_values(
        ["persona", "entity_name"]
    )


def persona_tests(df: pd.DataFrame, shift_df: pd.DataFrame,
                  var_df: pd.DataFrame) -> pd.DataFrame:

    ctrl_rows = df[df["persona"] == CONTROL_PERSONA_ID]
    n_ctrl = len(ctrl_rows)
    ref_ctrl = int(ctrl_rows["rifiuto_o_non_parsabile"].sum())

    rows = []
    for persona in PERSONA_ORDER:
        s = shift_df[shift_df["persona"] == persona]
        unit = s.groupby(["model", "entity_name"])["shift"].mean()
        w_stat, w_p = stats.wilcoxon(unit.values)

        pr = df[df["persona"] == persona]
        n_p, ref_p = len(pr), int(pr["rifiuto_o_non_parsabile"].sum())
        table = [[ref_p, n_p - ref_p], [ref_ctrl, n_ctrl - ref_ctrl]]
        chi2, chi_p, _, _ = stats.chi2_contingency(table)

        v = var_df[var_df["persona"] == persona]["delta_std"].dropna()
        v_stat, v_p = stats.wilcoxon(v.values)

        rows.append({
            "persona": persona,
            "n_cells": int(len(s)),
            "n_units": int(len(unit)),
            "shift_mean": float(s["shift"].mean()),
            "shift_median_unit": float(unit.median()),
            "shift_p": float(w_p),
            "refusal_persona": n_p and ref_p / n_p,
            "refusal_control": ref_ctrl / n_ctrl,
            "refusal_chi2": float(chi2),
            "refusal_p": float(chi_p),
            "delta_std_mean": float(v.mean()),
            "delta_std_p": float(v_p),
        })

    out = pd.DataFrame(rows)
    out["shift_p_holm"] = holm(out["shift_p"])
    out["refusal_p_holm"] = holm(out["refusal_p"])
    out["delta_std_p_holm"] = holm(out["delta_std_p"])
    return out


def entity_slopes(shift_df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for entity, g in shift_df.groupby("entity_name"):
        g = g.dropna(subset=["shift"])
        res = stats.linregress(g["persona_position"], g["shift"])
        n = len(g)
        tcrit = stats.t.ppf(0.975, n - 2)
        rows.append({
            "entity_name": entity,
            "n_points": n,
            "slope": float(res.slope),
            "stderr": float(res.stderr),
            "ci_low": float(res.slope - tcrit * res.stderr),
            "ci_high": float(res.slope + tcrit * res.stderr),
            "r2": float(res.rvalue ** 2),
            "p": float(res.pvalue),
        })
    out = pd.DataFrame(rows).sort_values("slope").reset_index(drop=True)
    out["p_bh"] = benjamini_hochberg(out["p"])
    return out


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_heatmap(shift_df: pd.DataFrame, slopes: pd.DataFrame, out_path,
                 center_rows: bool = False) -> None:

    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm

    order = list(slopes["entity_name"])
    pivot = (
        shift_df.groupby(["persona", "entity_name"])["shift"]
        .mean()
        .unstack("entity_name")
        .reindex(index=PERSONA_ORDER, columns=order)
    )
    vals = pivot.to_numpy(dtype=float)
    label = "Mean shift vs control (1-5 points)"
    if center_rows:
        vals = vals - np.nanmean(vals, axis=1, keepdims=True)
        label = "Shift vs control, persona mean removed"

    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))

    norm = TwoSlopeNorm(vmin=min(lo, -1e-6), vcenter=0.0, vmax=max(hi, 1e-6))
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(max(9, 0.62 * pivot.shape[1]), 3.6))
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            shade = abs(norm(v) - 0.5) * 2
            ax.text(j, i, f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "-."),
                    ha="center", va="center", fontsize=5.4,
                    color="white" if shade > 0.62 else "0.15")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([PERSONA_LABEL[p] for p in pivot.index], fontsize=8)
    ax.set_xlabel("Entity (sorted by affinity slope, left to right)", fontsize=8)
    ax.set_ylabel("Assigned persona", fontsize=8)
    ax.set_xticks(np.arange(-.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_affinity_fan(shift_df: pd.DataFrame, slopes: pd.DataFrame, out_path) -> None:

    plt = _plt()
    from matplotlib.colors import Normalize, TwoSlopeNorm

    sl = slopes.set_index("entity_name")["slope"]
    pivot = (
        shift_df.groupby(["entity_name", "persona_position"])["shift"]
        .mean()
        .unstack("persona_position")
        .reindex(index=sl.sort_values().index)
    )
    xs = np.array(sorted(pivot.columns))
    vmax = float(np.abs(sl).max())
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0, color="0.35", lw=0.9, zorder=1)
    for ent, row in pivot.iterrows():
        ax.plot(xs, row[xs].to_numpy(), color=cmap(norm(sl[ent])),
                lw=1.4, marker="o", ms=3, alpha=0.9, zorder=2)

    for ent, side in [("Alleanza Verdi e Sinistra", "left"),
                      ("Fratelli d'Italia", "left"),
                      ("Giorgia Meloni", "right"),
                      ("Matteo Renzi", "right")]:
        if ent not in pivot.index:
            continue
        if side == "left":
            ax.annotate(ent, (xs[0], pivot.loc[ent, xs[0]]), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=7, color=cmap(norm(sl[ent])))
        else:
            ax.annotate(ent, (xs[-1], pivot.loc[ent, xs[-1]]), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7, color=cmap(norm(sl[ent])))

    ax.set_xticks(xs)
    ax.set_xticklabels([PERSONA_LABEL[p] for p in PERSONA_ORDER], fontsize=8)
    ax.set_xlabel("Assigned persona", fontsize=9)
    ax.set_ylabel("Mean shift vs no-persona control", fontsize=9)
    ax.set_xlim(xs[0] - 1.15, xs[-1] + 0.95)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.14)
    cb.set_label("Affinity slope", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _persona_composite(shift_df: pd.DataFrame) -> pd.DataFrame:

    return (shift_df.groupby(["entity_name", "persona"])["mean_punteggio"]
                    .mean().unstack("persona")[PERSONA_ORDER])


def save_rank_bump(shift_df: pd.DataFrame, out_path) -> None:

    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm

    comp = _persona_composite(shift_df)
    rk = comp.rank(ascending=False, method="average")
    pref = rk["left"] - rk["right"]
    norm = TwoSlopeNorm(vmin=min(pref.min(), -1e-6), vcenter=0.0,
                        vmax=max(pref.max(), 1e-6))
    cmap = plt.get_cmap("RdBu_r")

    x = np.arange(len(PERSONA_ORDER))
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for e in rk.index:
        col = cmap(norm(pref[e]))
        ax.plot(x, rk.loc[e].to_numpy(), color=col, lw=1.5, marker="o", ms=3.4, alpha=0.9)
        ax.annotate(e, (x[0], rk.loc[e, "left"]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=6.2, color=col)
        ax.annotate(e, (x[-1], rk.loc[e, "right"]), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.2, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABEL[p] for p in PERSONA_ORDER], fontsize=8)
    ax.set_yticks(range(1, len(rk) + 1))
    ax.set_yticklabels(range(1, len(rk) + 1), fontsize=7)
    ax.invert_yaxis()
    ax.set_ylabel("Rank among the 21 entities (1 = highest score)", fontsize=9)
    ax.set_xlabel("Assigned persona", fontsize=9)
    ax.set_xlim(-2.7, len(x) + 1.7)
    ax.grid(axis="y", alpha=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_extremes_scatter(shift_df: pd.DataFrame, out_path) -> None:

    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm

    comp = _persona_composite(shift_df)
    gap = comp["right"] - comp["left"]
    norm = TwoSlopeNorm(vmin=min(gap.min(), -1e-6), vcenter=0.0,
                        vmax=max(gap.max(), 1e-6))
    cmap = plt.get_cmap("RdBu_r")

    lo = float(comp[["left", "right"]].to_numpy().min()) - 0.15
    hi = float(comp[["left", "right"]].to_numpy().max()) + 0.15

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.fill_between([lo, hi], [lo, hi], hi, color="#2c6fbb", alpha=0.06, zorder=0)
    ax.fill_between([lo, hi], lo, [lo, hi], color="#c0392b", alpha=0.06, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls="--", zorder=1)
    ax.scatter(comp["right"], comp["left"], s=38, zorder=3, edgecolor="white", lw=0.5,
               c=[cmap(norm(gap[e])) for e in comp.index])
    for e in comp.index:
        ax.annotate(e, (comp.loc[e, "right"], comp.loc[e, "left"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=6, color="0.25")

    ax.text(0.03, 0.97, "favoured by the left persona", transform=ax.transAxes,
            fontsize=7.5, color="#2c6fbb", va="top")
    ax.text(0.97, 0.04, "favoured by the right persona", transform=ax.transAxes,
            fontsize=7.5, color="#c0392b", ha="right")

    ax.set_xlabel("Mean score under the right persona", fontsize=9)
    ax.set_ylabel("Mean score under the left persona", fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_persona_splom(shift_df: pd.DataFrame, out_path) -> None:

    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm

    order = ["left", "center_left", "center_right", "right"]
    comp = _persona_composite(shift_df)[order]
    gap = comp["right"] - comp["left"]
    norm = TwoSlopeNorm(vmin=min(gap.min(), -1e-6), vcenter=0.0,
                        vmax=max(gap.max(), 1e-6))
    cmap = plt.get_cmap("RdBu_r")
    colors = [cmap(norm(gap[e])) for e in comp.index]

    lo = float(comp.to_numpy().min()) - 0.2
    hi = float(comp.to_numpy().max()) + 0.2
    n = len(order)

    fig, axes = plt.subplots(n, n, figsize=(7.6, 7.6), sharex=True, sharey=True)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

            if i == j:
                ax.set_facecolor("0.96")
                ax.text(0.5, 0.5, PERSONA_LABEL[order[i]].replace(" ", "\n"),
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="0.25")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            if i < j:
                d = float((comp[order[j]] - comp[order[i]]).abs().mean())
                ax.set_facecolor("0.98")
                ax.text(0.5, 0.58, f"{d:.2f}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=13, color="0.2")
                ax.text(0.5, 0.30, "mean |gap|", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6.5, color="0.45")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            ax.fill_between([lo, hi], [lo, hi], hi, color="#c0392b", alpha=0.05, zorder=0)
            ax.fill_between([lo, hi], lo, [lo, hi], color="#2c6fbb", alpha=0.05, zorder=0)
            ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.8, ls="--", zorder=1)
            ax.scatter(comp[order[j]], comp[order[i]], c=colors, s=17,
                       edgecolor="white", lw=0.3, zorder=3)
            ax.grid(alpha=0.18)

    ax = axes[n - 1, 0]
    for e in ["Alleanza Verdi e Sinistra", "Elly Schlein", "Movimento 5 Stelle",
              "Matteo Renzi", "Carlo Calenda", "Fratelli d'Italia",
              "Giorgia Meloni", "Futuro Nazionale"]:
        if e not in comp.index:
            continue
        ax.annotate(short_label(e), (comp.loc[e, "left"], comp.loc[e, "right"]),
                    xytext=(3, 2), textcoords="offset points",
                    fontsize=5.4, color="0.25")

    for k, lab in enumerate(order):
        axes[n - 1, k].set_xlabel(PERSONA_LABEL[lab], fontsize=8)
        axes[k, 0].set_ylabel(PERSONA_LABEL[lab], fontsize=8)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6)
    axes[0, 0].set_xlim(lo, hi); axes[0, 0].set_ylim(lo, hi)

    fig.suptitle("Mean score under each persona, all pairwise comparisons",
                 fontsize=9.5, y=0.945)
    fig.text(0.5, 0.917,
             "In each panel the row is the more right-wing persona: above the "
             "diagonal $=$ favoured by it, below $=$ favoured by the column persona",
             ha="center", fontsize=6.8, color="0.35")
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_gap_bars(shift_df: pd.DataFrame, out_path) -> None:

    plt = _plt()

    comp = _persona_composite(shift_df)
    ext = (comp["right"] - comp["left"]).sort_values()
    mod = (comp["center_right"] - comp["center_left"]).reindex(ext.index)

    y = np.arange(len(ext))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(ext) + 1.3))
    ax.barh(y - h / 2, ext.to_numpy(), height=h, zorder=3,
            color=["#c0392b" if v > 0 else "#2c6fbb" for v in ext],
            label="Right $-$ Left")
    ax.barh(y + h / 2, mod.to_numpy(), height=h, zorder=3,
            color=["#e8a49c" if v > 0 else "#9dc0e8" for v in mod],
            label="Centre-right $-$ Centre-left")
    ax.axvline(0, color="0.25", lw=0.9, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(ext.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Score difference between the two opposite personas "
                  "(1--5 points)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    lim = float(max(ext.abs().max(), mod.abs().max())) * 1.12
    ax.set_xlim(-lim, lim)
    ax.text(-lim * 0.97, -0.9, "favoured by the left-wing persona",
            fontsize=7.5, color="#2c6fbb", va="center")
    ax.text(lim * 0.97, -0.9, "favoured by the right-wing persona",
            fontsize=7.5, color="#c0392b", va="center", ha="right")
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_persona_level_grid(shift_df: pd.DataFrame, out_path,
                            full_scale: bool = True,
                            cmap_name: str = "inferno") -> None:

    plt = _plt()
    from matplotlib.colors import Normalize

    comp = _persona_composite(shift_df)
    order = (comp["right"] - comp["left"]).sort_values().index
    comp = comp.loc[order]
    vals = comp.to_numpy(dtype=float)

    vmin, vmax = (1.0, 5.0) if full_scale else (float(vals.min()), float(vals.max()))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(5.2, 0.34 * len(comp) + 1.4))
    ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.2,
                    color="white" if norm(vals[i, j]) < 0.62 else "0.12")

    ax.set_xticks(range(len(PERSONA_ORDER)))
    ax.set_xticklabels([PERSONA_LABEL[p] for p in PERSONA_ORDER],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels(comp.index, fontsize=7)
    ax.set_xlabel("Assigned persona", fontsize=8.5)
    ax.set_xticks(np.arange(-.5, len(PERSONA_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(comp), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.03)
    cb.set_label("Mean score (1--5)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def short_label(name: str) -> str:
    short = {
        "Alleanza Verdi e Sinistra": "AVS", "Partito Democratico": "PD",
        "Movimento 5 Stelle": "M5S", "Fratelli d'Italia": "FdI",
        "Forza Italia": "FI", "Italia Viva": "IV", "Noi Moderati": "NM",
        "Futuro Nazionale": "FN",
    }
    if name in short:
        return short[name]
    parts = str(name).split()
    return parts[-1] if parts else str(name)


def save_slope_plot(slopes: pd.DataFrame, out_path) -> None:
    plt = _plt()
    s = slopes.sort_values("slope")
    y = np.arange(len(s))
    colors = ["#c0392b" if v > 0 else "#2c6fbb" for v in s["slope"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.30 * len(s) + 1.1))
    ax.errorbar(
        s["slope"], y,
        xerr=[s["slope"] - s["ci_low"], s["ci_high"] - s["slope"]],
        fmt="none", ecolor="0.55", elinewidth=1, capsize=2,
    )
    ax.scatter(s["slope"], y, c=colors, s=22, zorder=3)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(s["entity_name"], fontsize=7)
    ax.set_xlabel("Affinity slope: change in score per step rightward on the persona axis",
                  fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not (PROCESSED_DIR / "scores.csv").exists():
        raise SystemExit("scores.csv not found. Run parse_responses.py.")

    df = load_scores()
    personas_present = sorted(p for p in df["persona"].unique() if p != CONTROL_PERSONA_ID)
    if not personas_present:
        raise SystemExit(""
        )
    print(f"Personas found: {', '.join(personas_present)}")

    shift = score_shift(df)
    shift.to_csv(PROCESSED_DIR / "persona_score_shift.csv", index=False)

    summary = shift_summary(shift)
    summary.to_csv(PROCESSED_DIR / "persona_shift_summary.csv", index=False)

    refusals = refusal_rates(df)
    refusals.to_csv(PROCESSED_DIR / "persona_refusal_rates.csv", index=False)

    variance = variance_shift(df)
    variance.to_csv(PROCESSED_DIR / "persona_variance.csv", index=False)

    tests = persona_tests(df, shift, variance)
    tests.to_csv(PROCESSED_DIR / "persona_tests.csv", index=False)

    slopes = entity_slopes(shift)
    slopes.to_csv(PROCESSED_DIR / "persona_entity_slope.csv", index=False)

    save_heatmap(shift, slopes, PROCESSED_DIR / "plots" / "persona_shift_heatmap.pdf")
    save_heatmap(shift, slopes,
                 PROCESSED_DIR / "plots" / "persona_shift_heatmap_centered.pdf",
                 center_rows=True)
    save_slope_plot(slopes, PROCESSED_DIR / "plots" / "persona_entity_slope.pdf")
    save_affinity_fan(shift, slopes, PROCESSED_DIR / "plots" / "persona_affinity_fan.pdf")
    save_rank_bump(shift, PROCESSED_DIR / "plots" / "persona_rank_bump.pdf")
    save_extremes_scatter(shift, PROCESSED_DIR / "plots" / "persona_extremes_scatter.pdf")
    save_persona_splom(shift, PROCESSED_DIR / "plots" / "persona_splom.pdf")
    save_gap_bars(shift, PROCESSED_DIR / "plots" / "persona_gap_bars.pdf")
    save_persona_level_grid(shift, PROCESSED_DIR / "plots" / "persona_level_grid.pdf")

    ctrl = df[df["persona"] == CONTROL_PERSONA_ID]
    any_persona = df[df["persona"] != CONTROL_PERSONA_ID]
    sig = slopes[slopes["p_bh"] < 0.05]
    stat = {
        "variant": ANALYSIS_VARIANT,
        "n_cells_paired_per_persona": int(len(shift) / len(PERSONA_ORDER)),
        "shift_overall_mean": float(shift["shift"].mean()),
        "refusal_control": float(ctrl["rifiuto_o_non_parsabile"].mean()),
        "refusal_any_persona": float(any_persona["rifiuto_o_non_parsabile"].mean()),
        "per_persona": tests.to_dict(orient="records"),
        "slope_min": float(slopes["slope"].min()),
        "slope_max": float(slopes["slope"].max()),
        "slope_n_significant": int(len(sig)),
        "slope_n_total": int(len(slopes)),
        "slope_most_left": slopes.iloc[0]["entity_name"],
        "slope_most_right": slopes.iloc[-1]["entity_name"],
        "slopes": slopes.to_dict(orient="records"),
    }

    summ_path = PROCESSED_DIR / "summary_mean_std.csv"
    if summ_path.exists():
        summ = pd.read_csv(summ_path)
        col = "mean" if "mean" in summ.columns else "media"
        comp = summ.groupby("entity_name")[col].mean().rename("composite")
        j = slopes.set_index("entity_name").join(comp).dropna(subset=["composite"])
        rho, p = stats.spearmanr(j["slope"], j["composite"])
        stat["slope_vs_composite_rho"] = float(rho)
        stat["slope_vs_composite_p"] = float(p)
        stat["slope_vs_composite_n"] = int(len(j))

    with open(PROCESSED_DIR / "persona_stats.json", "w", encoding="utf-8") as f:
        json.dump(stat, f, indent=2, ensure_ascii=False)

    for name, obj in [
        ("persona_score_shift.csv", shift), ("persona_shift_summary.csv", summary),
        ("persona_refusal_rates.csv", refusals), ("persona_variance.csv", variance),
        ("persona_tests.csv", tests), ("persona_entity_slope.csv", slopes),
    ]:
        print(f"  - {name:<28} ({len(obj)} rows)")

    show = tests[["persona", "shift_mean", "shift_p_holm",
                  "refusal_persona", "refusal_p_holm", "delta_std_mean"]].copy()
    show.columns = ["persona", "shift", "p(shift)", "refusals", "p(refusals)", "d.std"]


if __name__ == "__main__":
    main()
