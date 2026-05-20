import io
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import silhouette_score

# --- Config ---
DATASET         = "3FTx"
RESULTS_DIR      = Path("protspace_results") / DATASET
BUNDLE           = RESULTS_DIR / "data.parquetbundle"
ANNOT_PARQUET    = RESULTS_DIR / "annotated_embeddings.parquet"
H5               = Path("embeddings") / f"{DATASET}.h5"
LABEL_COL        = "length" 
K_VALUES         = [1, 2, 5, 10, 15, 20, 30, 50]
N_PCA_COMPONENTS = 50
DELIMITER        = b"---PARQUET_DELIMITER---"
OUT_DIR          = Path("evaluation") / DATASET
OUT_DIR.mkdir(exist_ok=True)

# =============================================================
# 1. Read parquetbundle
# =============================================================
print("Reading parquetbundle...")
with open(BUNDLE, "rb") as f:
    parts = f.read().split(DELIMITER)

annot_df    = pq.read_table(io.BytesIO(parts[0])).to_pandas()
proj_df     = pq.read_table(io.BytesIO(parts[2])).to_pandas()
parquet_ids = set(annot_df["protein_id"])

# =============================================================
# 2. Read labels
# =============================================================
print("Reading labels...")
label_df = pd.read_parquet(ANNOT_PARQUET)
label_df[LABEL_COL] = (
    label_df[LABEL_COL]
    .str.split("|").str[0]
    .str.strip()
)
label_df[LABEL_COL] = label_df[LABEL_COL].replace("", np.nan)
label_df = label_df.dropna(subset=[LABEL_COL])
labelled_ids = set(label_df["identifier"].astype(str))

print(f"  Proteins with labels : {len(labelled_ids)}")
print(f"  Dropped (no label)   : {6436 - len(labelled_ids)}")

# =============================================================
# 3. Read H5 embeddings
# =============================================================
print("Reading H5 embeddings...")
with h5py.File(H5, "r") as f:
    h5_keys = set(f.keys())
    shared  = sorted(parquet_ids & h5_keys & labelled_ids)

    if not shared:
        raise ValueError("No overlapping IDs across H5, parquetbundle and labels.")

    print(f"  Proteins used : {len(shared)}")
    embeddings = np.stack([f[pid][:] for pid in shared])

N        = len(shared)
id_index = {pid: i for i, pid in enumerate(shared)}

label_df   = label_df.set_index("identifier").loc[shared]
labels_raw = label_df[LABEL_COL].values
le         = LabelEncoder()
labels_int = le.fit_transform(labels_raw)
print(f"  Unique families : {len(le.classes_)}")

# =============================================================
# 4. Helpers
# =============================================================
def ranked_neighbours(dist_matrix):
    return np.argsort(dist_matrix, axis=1)[:, 1:]

def build_rank_matrix(dist_matrix):
    n           = dist_matrix.shape[0]
    sorted_idx  = np.argsort(dist_matrix, axis=1)
    rank_matrix = np.empty((n, n), dtype=np.int32)
    rows        = np.arange(n)[:, None]
    rank_matrix[rows, sorted_idx] = np.arange(n)
    rank_matrix -= 1
    return rank_matrix

def short_name(proj_name):
    """Strip model prefix: 'ProtT5 — UMAP 2' → 'UMAP 2'."""
    return proj_name.split("—")[-1].strip()

# =============================================================
# 5. Metric functions
# =============================================================
def knn_recall_at_k(ranked_highd, ranked_proj, k):
    recalls = []
    for i in range(N):
        gt   = set(ranked_highd[i, :k])
        pred = set(ranked_proj[i,  :k])
        recalls.append(len(gt & pred) / k)
    return float(np.mean(recalls))

def trustworthiness_at_k(rank_highd, ranked_proj, k):
    penalty = 0.0
    for i in range(N):
        for j in ranked_proj[i, :k]:
            r = rank_highd[i, j]
            if r >= k:
                penalty += r - k
    normalisation = k * N * (2 * N - 3 * k - 1) / 2
    return float(1 - (2 / normalisation) * penalty)

def continuity_at_k(rank_proj, ranked_highd, k):
    penalty = 0.0
    for i in range(N):
        for j in ranked_highd[i, :k]:
            r = rank_proj[i, j]
            if r >= k:
                penalty += r - k
    normalisation = k * N * (2 * N - 3 * k - 1) / 2
    return float(1 - (2 / normalisation) * penalty)

def knn_accuracy_at_k(ranked, labels, k):
    correct = 0
    for i in range(N):
        neighbour_labels = labels[ranked[i, :k]]
        counts           = np.bincount(neighbour_labels)
        majority         = np.argmax(counts)
        if majority == labels[i]:
            correct += 1
    return correct / N


def compute_silhouette(coords_or_dist, labels, metric="euclidean"):
    return float(silhouette_score(
        coords_or_dist, labels,
        metric=metric,
        sample_size=min(5000, N),
        random_state=42,
    ))

# =============================================================
# 6. Color palette — consistent across all plots
#    Base color per projection, linestyle for ground truth
# =============================================================
BASE_COLORS = {
    "Full 1024-d":               "#1F77B4",   # blue
    f"PCA-{N_PCA_COMPONENTS}":   "#2CA02C",   # green
}
# projection-specific colors assigned dynamically below
PROJ_PALETTE  = ["#FF7F0E", "#9467BD", "#8C564B", "#E377C2", "#17BECF"]
LINESTYLES_GT = {"Full": "-", "PCA50": "--"}

# =============================================================
# 7. Precompute ground truth spaces
# =============================================================
print("\nComputing ground truth (full 1024-dim, cosine)...")
emb_norm    = normalize(embeddings, norm="l2")
dist_full   = cosine_distances(emb_norm)
ranked_full = ranked_neighbours(dist_full)
rank_full   = build_rank_matrix(dist_full)

print(f"Computing ground truth (top-{N_PCA_COMPONENTS} PCA, cosine)...")
pca             = PCA(n_components=N_PCA_COMPONENTS, svd_solver="arpack", random_state=42)
emb_pca50       = pca.fit_transform(embeddings)
emb_pca50_norm  = normalize(emb_pca50, norm="l2")
dist_pca50      = cosine_distances(emb_pca50_norm)
ranked_pca50    = ranked_neighbours(dist_pca50)
rank_pca50      = build_rank_matrix(dist_pca50)

# =============================================================
# 8. Results containers
#
#   unsupervised : dict keyed by display label (e.g. "UMAP 2 - Full")
#                  → {recall, trust, cont} each a list over K_VALUES
#   knn_acc      : dict keyed by space name → list over K_VALUES
#   silhouette   : dict keyed by space name → scalar
# =============================================================
unsupervised = {}
knn_acc      = {}
silhouette   = {}

# Ground truth spaces
print("\nComputing metrics for ground truth spaces...")
knn_acc["Full 1024-d"]             = [knn_accuracy_at_k(ranked_full,  labels_int, k) for k in K_VALUES]
knn_acc[f"PCA-{N_PCA_COMPONENTS}"] = [knn_accuracy_at_k(ranked_pca50, labels_int, k) for k in K_VALUES]

silhouette["Full 1024-d"]             = compute_silhouette(dist_full,  labels_int, metric="precomputed")
silhouette[f"PCA-{N_PCA_COMPONENTS}"] = compute_silhouette(dist_pca50, labels_int, metric="precomputed")

# =============================================================
# 9. Compute metrics for each projection
# =============================================================
projection_names = proj_df["projection_name"].unique()
print(f"\nProjection methods found: {list(projection_names)}")

proj_colors = {}
for idx, proj_name in enumerate(projection_names):
    proj_colors[proj_name] = PROJ_PALETTE[idx % len(PROJ_PALETTE)]

for proj_name in projection_names:
    sname = short_name(proj_name)
    print(f"\nProcessing {proj_name}...")

    subset = proj_df[proj_df["projection_name"] == proj_name].copy()
    subset = subset[subset["identifier"].isin(id_index)]
    subset = subset.set_index("identifier").loc[shared]

    dims   = ["x", "y"] if subset["z"].isna().all() else ["x", "y", "z"]
    coords = subset[dims].values.astype(float)

    dist_proj   = euclidean_distances(coords)
    ranked_proj = ranked_neighbours(dist_proj)
    rank_proj   = build_rank_matrix(dist_proj)

    knn_acc[sname]    = [knn_accuracy_at_k(ranked_proj, labels_int, k) for k in K_VALUES]
    silhouette[sname] = compute_silhouette(coords, labels_int, metric="euclidean")

    for gt_label, (ranked_gt, rank_gt) in [
        ("Full",  (ranked_full,  rank_full)),
        ("PCA50", (ranked_pca50, rank_pca50)),
    ]:
        line_label = f"{sname} - {gt_label}"
        recalls, trusts, conts = [], [], []
        for k in K_VALUES:
            print(f"  [{gt_label}] k={k}...", end=" ", flush=True)
            recalls.append(knn_recall_at_k(ranked_gt,  ranked_proj, k))
            trusts.append( trustworthiness_at_k(rank_gt,   ranked_proj, k))
            conts.append(  continuity_at_k(rank_proj,  ranked_gt,   k))
        print()
        unsupervised[line_label] = {
            "recall":     recalls,
            "trust":      trusts,
            "cont":       conts,
            "proj_name":  proj_name,
            "gt_label":   gt_label,
        }

# =============================================================
# 10. Summary TSV
# =============================================================
rows = []
for line_label, m in unsupervised.items():
    rows.append({
        "space":        line_label,
        "type":         "unsupervised",
        "recall_mean":  round(float(np.mean(m["recall"])), 4),
        "trust_mean":   round(float(np.mean(m["trust"])),  4),
        "cont_mean":    round(float(np.mean(m["cont"])),   4),
        "knn_acc_mean": "",
        "silhouette":   "",
    })
for space, acc_vals in knn_acc.items():
    rows.append({
        "space":        space,
        "type":         "supervised",
        "recall_mean":  "",
        "trust_mean":   "",
        "cont_mean":    "",
        "knn_acc_mean": round(float(np.mean(acc_vals)), 4),
        "silhouette":   round(silhouette.get(space, float("nan")), 4),
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "summary.tsv", sep="\t", index=False)
print("\n--- Summary ---")
print(summary_df.to_string(index=False))

# =============================================================
# 11. Plotting helpers
# =============================================================
def get_line_color(line_label, proj_name):
    return proj_colors[proj_name]

def get_linestyle(gt_label):
    return LINESTYLES_GT[gt_label]

def get_space_color(space):
    if space in BASE_COLORS:
        return BASE_COLORS[space]
    # match to projection by short name
    for proj_name in projection_names:
        if short_name(proj_name) == space:
            return proj_colors[proj_name]
    return "#333333"

def style_ax(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9, framealpha=0.7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

# =============================================================
# 12. Plot 1 — kNN Recall
# =============================================================
fig, ax = plt.subplots(figsize=(10, 5))
for line_label, m in unsupervised.items():
    ax.plot(K_VALUES, m["recall"],
            color=get_line_color(line_label, m["proj_name"]),
            linestyle=get_linestyle(m["gt_label"]),
            marker="o", markersize=4, linewidth=1.8, label=line_label)
style_ax(ax, "kNN Recall")
fig.tight_layout()
fig.savefig(OUT_DIR / "recall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: recall.png")

# =============================================================
# 13. Plot 2 — Trustworthiness
# =============================================================
fig, ax = plt.subplots(figsize=(10, 5))
for line_label, m in unsupervised.items():
    ax.plot(K_VALUES, m["trust"],
            color=get_line_color(line_label, m["proj_name"]),
            linestyle=get_linestyle(m["gt_label"]),
            marker="o", markersize=4, linewidth=1.8, label=line_label)
style_ax(ax, "Trustworthiness")
fig.tight_layout()
fig.savefig(OUT_DIR / "trustworthiness.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: trustworthiness.png")

# =============================================================
# 14. Plot 3 — Continuity
# =============================================================
fig, ax = plt.subplots(figsize=(10, 5))
for line_label, m in unsupervised.items():
    ax.plot(K_VALUES, m["cont"],
            color=get_line_color(line_label, m["proj_name"]),
            linestyle=get_linestyle(m["gt_label"]),
            marker="o", markersize=4, linewidth=1.8, label=line_label)
style_ax(ax, "Continuity")
fig.tight_layout()
fig.savefig(OUT_DIR / "continuity.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: continuity.png")

# =============================================================
# 15. Plot 4 — kNN Accuracy
# =============================================================
fig, ax = plt.subplots(figsize=(10, 5))
for space, acc_vals in knn_acc.items():
    ax.plot(K_VALUES, acc_vals,
            color=get_space_color(space),
            linestyle="-",
            marker="o", markersize=4, linewidth=1.8, label=space)
ax.set_title("kNN Accuracy (protein_families)", fontsize=13, fontweight="bold")
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_xticks(K_VALUES)
ax.legend(fontsize=9, framealpha=0.7, loc="best")
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "knn_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: knn_accuracy.png")

# =============================================================
# 16. Plot 5 — Silhouette (vertical bar)
# =============================================================
fig, ax = plt.subplots(figsize=(8, 5))
sil_spaces = list(silhouette.keys())
sil_values = list(silhouette.values())
sil_colors = [get_space_color(s) for s in sil_spaces]
bars = ax.bar(sil_spaces, sil_values, color=sil_colors, edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
ax.set_title("Silhouette Score (protein_families)", fontsize=13, fontweight="bold")
ax.set_ylabel("Score", fontsize=11)
ax.set_xlabel("Space", fontsize=11)
ax.tick_params(axis="x", rotation=15)
ax.grid(True, axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "silhouette.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: silhouette.png")

print(f"\nDone. All outputs in: {OUT_DIR}")