from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

OUT = Path("evaluation") / "dr_overview2.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "DisProt": Path("protspace_results/Uniprot_DisProt/eval/prott5"),
    "Shuffled": Path("protspace_results/Uniprot_Shuffled/eval/prott5"),
    "polyA":   Path("protspace_results/Uniprot_polyA/eval/prott5"),
    "Random":  Path("protspace_results/Uniprot_Random/eval/prott5"),
}

C_UNSUP = "#0077BB"
C_CAT   = "#EE7733"
C_CONT  = "#009988"

SCOPE_COLORS = {"local": "#4477AA", "mixed": "#BBAA22", "global": "#AA3333"}

METRIC_CONFIG = [
    # Unsupervised
    ("trust_mean",           "Trust.",        "local",  C_UNSUP),
    ("recall_mean",          "kNN\nRecall",    "local",  C_UNSUP),
    ("cont_mean",            "Cont.",          "local",  C_UNSUP),
    # Categorical─
    ("knn_acc_mean",         "kNN\nAcc.",      "local",  C_CAT),
    ("concordex",            "CONCORDEX",      "local",  C_CAT),
    ("silhouette",           "Silhouette",     "mixed",  C_CAT),
    # Continuous──
    ("knn_r2",               "kNN\nR²",        "local",  C_CONT),
    ("linear_r2",            "Linear\nR²",     "global", C_CONT),
    ("spearman_dcorr",       "Spearman\nDist.","global", C_CONT),
    ("distance_correlation", "Dist.\nCorr.",   "global", C_CONT),
]


def load_eval_dir(eval_dir: Path) -> pd.DataFrame:
    """Merge unsupervised + categorical + continuous TSVs into one wide table."""

    # Unsupervised
    uns = pd.read_csv(eval_dir / "summary.tsv", sep="\t")
    uns["space"] = (
        uns["space"].str.replace(r"\s*-\s*Full$", "", regex=True).str.strip()
    )
    uns_cols = [c for c in ["recall_mean", "trust_mean", "cont_mean"] if c in uns.columns]
    uns = uns[["space"] + uns_cols].drop_duplicates("space").set_index("space")
    proj_names = set(uns.index)

    # Categorical
    cat_path = eval_dir / "protein_families" / "summary.tsv"
    cat = pd.DataFrame()
    if cat_path.exists():
        df = pd.read_csv(cat_path, sep="\t")
        df = df[df["space"].isin(proj_names)]
        cols = [c for c in ["knn_acc_mean", "silhouette", "concordex"] if c in df.columns]
        if cols:
            cat = df[["space"] + cols].drop_duplicates("space").set_index("space")

    # Continuous
    cont_path = eval_dir / "length" / "summary.tsv"
    cont = pd.DataFrame()
    if cont_path.exists():
        df = pd.read_csv(cont_path, sep="\t")
        df = df[df["space"].isin(proj_names)]
        cols = [c for c in ["knn_r2", "linear_r2", "spearman_dcorr", "distance_correlation"]
                if c in df.columns]
        if cols:
            cont = df[["space"] + cols].drop_duplicates("space").set_index("space")

    result = uns
    if not cat.empty:
        result = result.join(cat, how="left")
    if not cont.empty:
        result = result.join(cont, how="left")
    return result


def active_metrics(df: pd.DataFrame):
    """Return only METRIC_CONFIG entries present in df."""
    return [(col, disp, scope, color)
            for col, disp, scope, color in METRIC_CONFIG
            if col in df.columns]


def make_heatmap_df(df: pd.DataFrame, active):
    data = df[[col for col, *_ in active]].copy()
    data.columns = [disp for _, disp, _, _ in active]
    return data


def draw_heatmap(ax, heatmap_df, active, title,
                 show_ylabels=True, show_scope=True):
    n_rows, n_cols = heatmap_df.shape
    norm_df = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())
    cm = plt.get_cmap("YlGnBu")
    vn = Normalize(vmin=0, vmax=1)

    for ri in range(n_rows):
        for ci in range(n_cols):
            raw = heatmap_df.iloc[ri, ci]
            nrm = norm_df.iloc[ri, ci]
            if pd.isna(nrm):
                fc, tc, txt = "#D8D8D8", "#555555", "N/A"
            else:
                fc = cm(vn(nrm))
                r, g, b, _ = fc
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                tc = "white" if lum < 0.45 else "#1a1a1a"
                txt = f"{raw:.2f}"
            ax.add_patch(plt.Rectangle(
                (ci, n_rows - ri - 1), 1, 1,
                facecolor=fc, edgecolor="white", linewidth=1.2,
            ))
            ax.text(ci + 0.5, n_rows - ri - 0.5, txt,
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=tc)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(heatmap_df.columns, fontsize=8, ha="center", va="top")
    if show_ylabels:
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(heatmap_df.index[::-1], fontsize=8,
                           fontweight="bold", ha="right")
    else:
        ax.set_yticks([])
    ax.tick_params(axis="both", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=22, color="#222222")

    if show_scope:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(np.arange(n_cols) + 0.5)
        ax_top.set_xticklabels(
            [scope for _, _, scope, _ in active], fontsize=7.5, ha="center"
        )
        for tick, (_, _, scope, _) in zip(ax_top.get_xticklabels(), active):
            tick.set_color(SCOPE_COLORS[scope])
            tick.set_fontweight("bold")
        ax_top.tick_params(axis="x", length=0, pad=1)
        for spine in ax_top.spines.values():
            spine.set_visible(False)

    for ci, (_, _, _, color) in enumerate(active):
        ax.add_patch(plt.Rectangle(
            (ci, -0.28), 1, 0.16,
            facecolor=color, edgecolor="white", linewidth=1,
            transform=ax.transData, clip_on=False,
        ))

    unsup_n = sum(1 for *_, c in active if c == C_UNSUP)
    cat_n   = sum(1 for *_, c in active if c == C_CAT)
    for x_div in [unsup_n, unsup_n + cat_n]:
        if 0 < x_div < n_cols:
            ax.axvline(x_div, color="#888888", linewidth=1.5, zorder=10)



datasets = {name: load_eval_dir(path) for name, path in DATASETS.items()}

all_projs = list(dict.fromkeys(
    p for df in datasets.values() for p in df.index
))

datasets = {name: df.reindex(all_projs) for name, df in datasets.items()}

all_cols = set(col for df in datasets.values() for col in df.columns)
active = [(col, disp, scope, color)
          for col, disp, scope, color in METRIC_CONFIG
          if col in all_cols]

n_rows = len(all_projs)

fig_h = max(10, 2.8 + n_rows * 0.54)
fig, axes = plt.subplots(
    2, 2, figsize=(18, fig_h),
    facecolor="white",
    gridspec_kw={"wspace": 0.06, "hspace": 0.42},
)
fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.07)
fig.suptitle(
    "DR Evaluation Metrics — Local vs Global Scope",
    fontsize=14, fontweight="bold", y=0.975, color="#111111",
)

dataset_items = list(datasets.items())
for idx, (ax, (name, df)) in enumerate(zip(axes.flat, dataset_items)):
    hdf = make_heatmap_df(df, active)
    draw_heatmap(ax, hdf, active, name,
                 show_ylabels=(idx % 2 == 0),
                 show_scope=True)

legend_handles = [
    mpatches.Patch(facecolor=C_UNSUP, label="Unsupervised"),
    mpatches.Patch(facecolor=C_CAT,   label="Categorical (protein families)"),
    mpatches.Patch(facecolor=C_CONT,  label="Continuous (sequence length)"),
    mpatches.Patch(facecolor=SCOPE_COLORS["local"],  label="Local scope"),
    mpatches.Patch(facecolor=SCOPE_COLORS["mixed"],  label="Mixed scope"),
    mpatches.Patch(facecolor=SCOPE_COLORS["global"], label="Global scope"),
]
fig.legend(
    handles=legend_handles, loc="lower center",
    ncol=6, fontsize=8.5, framealpha=0.9,
    edgecolor="#cccccc", bbox_to_anchor=(0.55, -0.01),
)

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
