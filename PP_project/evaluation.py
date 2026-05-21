import gc
import io
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import silhouette_score

# --- Config ---
DATASET         = "toxins"
RESULTS_DIR      = Path("protspace_results") / DATASET
BUNDLE           = RESULTS_DIR / "data.parquetbundle"
#ANNOT_PARQUET    = RESULTS_DIR / "annotated_embeddings.parquet"
H5               = Path("embeddings") / f"{DATASET}.h5"
LABEL_COL        = "protein_families"   # column in parquet labels to use for evaluation
K_VALUES         = [1, 2, 5, 10, 15, 20, 30, 50]
N_PCA_COMPONENTS = 50
DELIMITER        = b"---PARQUET_DELIMITER---"
OUT_DIR          = Path("evaluation") / DATASET
OUT_DIR.mkdir(exist_ok=True)

K_STORED = 300

# =============================================================
# 1. Read parquetbundle
# =============================================================
print("Reading parquetbundle...")
with open(BUNDLE, "rb") as f:
    parts = f.read().split(DELIMITER)

annot_df    = pq.read_table(io.BytesIO(parts[0])).to_pandas()
proj_df     = pq.read_table(io.BytesIO(parts[2])).to_pandas()
parquet_ids = set(annot_df["protein_id"])
del annot_df          # free immediately

# =============================================================
# 2. Read labels
# =============================================================
print("Reading labels...")
annot_df = pq.read_table(io.BytesIO(parts[0])).to_pandas()
parquet_ids = set(annot_df["protein_id"])

label_df = annot_df.rename(columns={"protein_id": "identifier"})
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
# 3. Read H5 embeddings  (load as float16 to halve footprint,
#    cast back to float32 only when math requires it)
# =============================================================
print("Reading H5 embeddings...")
with h5py.File(H5, "r") as f:
    h5_keys = set(f.keys())
    shared  = sorted(parquet_ids & h5_keys & labelled_ids)

    if not shared:
        raise ValueError("No overlapping IDs across H5, parquetbundle and labels.")

    print(f"  Proteins used : {len(shared)}")
    # float16 halves the memory vs float32 at negligible precision cost
    embeddings = np.stack([f[pid][:] for pid in shared]).astype(np.float16)

N        = len(shared)
id_index = {pid: i for i, pid in enumerate(shared)}

label_df   = label_df.set_index("identifier").loc[shared]
labels_raw = label_df[LABEL_COL].values
le         = LabelEncoder()
labels_int = le.fit_transform(labels_raw)
print(f"  Unique families : {len(le.classes_)}")

# =============================================================
# 4. Core helper — builds a NearestNeighbors index and returns
#    the (N, K_STORED) ranked-neighbor index array.
#    Never materialises an N×N matrix.
# =============================================================
def get_neighbors(X, k_stored=K_STORED, metric="cosine"):
    """Return (N, k_stored) array of neighbor indices (closest first, self excluded)."""
    nn = NearestNeighbors(
        n_neighbors=k_stored + 1,   # +1 because self is included in results
        metric=metric,
        algorithm="brute",          # exact; use 'auto' for lower-d projections
        n_jobs=-1,
    )
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return indices[:, 1:].astype(np.int32)   # drop self → (N, k_stored)

def short_name(proj_name):
    return proj_name.split("—")[-1].strip()

# =============================================================
# 5. Metric functions — all work on (N, K_STORED) neighbor
#    arrays; no N×N rank matrix is ever built.
# =============================================================

def knn_recall_at_k(neighbors_gt, neighbors_proj, k):
    gt_k   = neighbors_gt[:, :k]
    pr_k   = neighbors_proj[:, :k]
    hits   = np.array([
        len(np.intersect1d(gt_k[i], pr_k[i])) for i in range(N)
    ])
    return float(hits.mean() / k)

def trustworthiness_at_k(neighbors_highd, neighbors_proj, k):
    """Approximate trustworthiness.

    For neighbors returned by the projection that are *not* in the
    stored high-d list we assign rank = K_STORED as a conservative
    upper bound (i.e. they are treated as if they were just outside
    the stored window, which underestimates the penalty slightly).
    """
    K_stored = neighbors_highd.shape[1]
    penalty  = 0.0
    for i in range(N):
        hd_rank = {j: r for r, j in enumerate(neighbors_highd[i])}
        for j in neighbors_proj[i, :k]:
            r = hd_rank.get(j, K_stored)
            if r >= k:
                penalty += (r - k)
    norm = k * N * (2 * N - 3 * k - 1) / 2
    return float(1 - (2 / norm) * penalty)

def continuity_at_k(neighbors_proj, neighbors_highd, k):
    """Approximate continuity (symmetric counterpart of trustworthiness)."""
    K_stored = neighbors_proj.shape[1]
    penalty  = 0.0
    for i in range(N):
        pr_rank = {j: r for r, j in enumerate(neighbors_proj[i])}
        for j in neighbors_highd[i, :k]:
            r = pr_rank.get(j, K_stored)
            if r >= k:
                penalty += (r - k)
    norm = k * N * (2 * N - 3 * k - 1) / 2
    return float(1 - (2 / norm) * penalty)

def knn_accuracy_at_k(neighbors, labels, k):
    correct = 0
    for i in range(N):
        nbr_labels = labels[neighbors[i, :k]]
        majority   = np.bincount(nbr_labels).argmax()
        correct   += int(majority == labels[i])
    return correct / N

def compute_silhouette(X, labels, metric="euclidean"):
    return float(silhouette_score(
        X, labels,
        metric=metric,
        sample_size=min(3000, N),   # keep under 3k for Colab RAM
        random_state=42,
    ))

# =============================================================
# 6. Color palette
# =============================================================
BASE_COLORS = {
    "Full 1024-d":             "#1F77B4",
    f"PCA-{N_PCA_COMPONENTS}": "#2CA02C",
}
PROJ_PALETTE  = ["#FF7F0E", "#9467BD", "#8C564B", "#E377C2", "#17BECF"]
LINESTYLES_GT = {"Full": "-", "PCA50": "--"}

# =============================================================
# 7. Ground truth spaces  — neighbors only, no N×N matrices
# =============================================================
print("\nComputing ground truth (full 1024-dim, cosine)...")
emb_norm = normalize(embeddings.astype(np.float32), norm="l2")
neighbors_full = get_neighbors(emb_norm, metric="cosine")
del emb_norm; gc.collect()

print(f"Computing ground truth (top-{N_PCA_COMPONENTS} PCA, cosine)...")
pca      = PCA(n_components=N_PCA_COMPONENTS, svd_solver="arpack", random_state=42)
emb_pca  = pca.fit_transform(embeddings.astype(np.float32))
emb_pca  = normalize(emb_pca, norm="l2")
neighbors_pca50 = get_neighbors(emb_pca, metric="cosine")
# keep emb_pca for silhouette; free normalized copy after knn
gc.collect()

# =============================================================
# 8. Result containers
# =============================================================
unsupervised = {}
knn_acc      = {}
silhouette   = {}

print("\nComputing metrics for ground truth spaces...")
knn_acc["Full 1024-d"]             = [knn_accuracy_at_k(neighbors_full,  labels_int, k) for k in K_VALUES]
knn_acc[f"PCA-{N_PCA_COMPONENTS}"] = [knn_accuracy_at_k(neighbors_pca50, labels_int, k) for k in K_VALUES]

# For silhouette we pass the PCA coords (no N×N matrix needed with metric='euclidean')
# For full-dim, sample-based silhouette on the float16 embeddings
silhouette["Full 1024-d"]             = compute_silhouette(embeddings.astype(np.float32), labels_int, metric="cosine")
silhouette[f"PCA-{N_PCA_COMPONENTS}"] = compute_silhouette(emb_pca, labels_int, metric="euclidean")
del emb_pca; gc.collect()

# =============================================================
# 9. Per-projection metrics
# =============================================================
projection_names = proj_df["projection_name"].unique()
print(f"\nProjection methods found: {list(projection_names)}")

proj_colors = {pn: PROJ_PALETTE[i % len(PROJ_PALETTE)] for i, pn in enumerate(projection_names)}

for proj_name in projection_names:
    sname = short_name(proj_name)
    print(f"\nProcessing {proj_name}...")

    subset = proj_df[proj_df["projection_name"] == proj_name].copy()
    subset = subset[subset["identifier"].isin(id_index)]
    subset = subset.set_index("identifier").loc[shared]

    dims   = ["x", "y"] if subset["z"].isna().all() else ["x", "y", "z"]
    coords = subset[dims].values.astype(np.float32)

    # NearestNeighbors on low-d coords — very cheap
    neighbors_proj = get_neighbors(coords, metric="euclidean")

    knn_acc[sname]    = [knn_accuracy_at_k(neighbors_proj, labels_int, k) for k in K_VALUES]
    silhouette[sname] = compute_silhouette(coords, labels_int, metric="euclidean")

    for gt_label, neighbors_gt in [("Full", neighbors_full), ("PCA50", neighbors_pca50)]:
        line_label = f"{sname} - {gt_label}"
        recalls, trusts, conts = [], [], []
        for k in K_VALUES:
            print(f"  [{gt_label}] k={k}...", end=" ", flush=True)
            recalls.append(knn_recall_at_k(neighbors_gt,   neighbors_proj, k))
            trusts.append( trustworthiness_at_k(neighbors_gt, neighbors_proj, k))
            conts.append(  continuity_at_k(neighbors_proj, neighbors_gt,    k))
        print()
        unsupervised[line_label] = {
            "recall": recalls, "trust": trusts, "cont": conts,
            "proj_name": proj_name, "gt_label": gt_label,
        }

    del neighbors_proj, coords; gc.collect()

# Free the large neighbor arrays now that all projections are done
del neighbors_full, neighbors_pca50, embeddings; gc.collect()

# =============================================================
# 10. Summary TSV
# =============================================================
rows = []
for line_label, m in unsupervised.items():
    rows.append({
        "space": line_label, "type": "unsupervised",
        "recall_mean": round(float(np.mean(m["recall"])), 4),
        "trust_mean":  round(float(np.mean(m["trust"])),  4),
        "cont_mean":   round(float(np.mean(m["cont"])),   4),
        "knn_acc_mean": "", "silhouette": "",
    })
for space, acc_vals in knn_acc.items():
    rows.append({
        "space": space, "type": "supervised",
        "recall_mean": "", "trust_mean": "", "cont_mean": "",
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
def get_line_color(line_label, proj_name): return proj_colors[proj_name]
def get_linestyle(gt_label): return LINESTYLES_GT[gt_label]
def get_space_color(space):
    if space in BASE_COLORS: return BASE_COLORS[space]
    for pn in projection_names:
        if short_name(pn) == space: return proj_colors[pn]
    return "#333333"

def style_ax(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("k", fontsize=11); ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9, framealpha=0.7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

def save_line_plot(metric_key, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    for line_label, m in unsupervised.items():
        ax.plot(K_VALUES, m[metric_key],
                color=get_line_color(line_label, m["proj_name"]),
                linestyle=get_linestyle(m["gt_label"]),
                marker="o", markersize=4, linewidth=1.8, label=line_label)
    style_ax(ax, title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

# =============================================================
# 12-14. Line plots
# =============================================================
save_line_plot("recall", "kNN Recall",       "recall.png")
save_line_plot("trust",  "Trustworthiness",  "trustworthiness.png")
save_line_plot("cont",   "Continuity",       "continuity.png")

# =============================================================
# 15. kNN Accuracy
# =============================================================
fig, ax = plt.subplots(figsize=(10, 5))
for space, acc_vals in knn_acc.items():
    ax.plot(K_VALUES, acc_vals, color=get_space_color(space),
            marker="o", markersize=4, linewidth=1.8, label=space)
ax.set_title(f"kNN Accuracy {LABEL_COL}", fontsize=13, fontweight="bold")
ax.set_xlabel("k", fontsize=11); ax.set_ylabel("Accuracy", fontsize=11)
ax.set_xticks(K_VALUES)
ax.legend(fontsize=9, framealpha=0.7, loc="best")
ax.grid(True, alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "knn_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: knn_accuracy.png")

# =============================================================
# 16. Silhouette bar chart
# =============================================================
fig, ax = plt.subplots(figsize=(8, 5))
sil_spaces = list(silhouette.keys())
sil_values = list(silhouette.values())
bars = ax.bar(sil_spaces, sil_values,
              color=[get_space_color(s) for s in sil_spaces],
              edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
ax.set_title(f"Silhouette Score ({LABEL_COL})", fontsize=13, fontweight="bold")
ax.set_ylabel("Score", fontsize=11); ax.set_xlabel("Space", fontsize=11)
ax.tick_params(axis="x", rotation=15)
ax.grid(True, axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "silhouette.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: silhouette.png")

print(f"\nDone. All outputs in: {OUT_DIR}")