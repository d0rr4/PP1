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
DATASET                = "toxins"
RESULTS_DIR            = Path("protspace_results") / DATASET
BUNDLE                 = RESULTS_DIR / "data.parquetbundle"
H5                     = Path("embeddings") / f"{DATASET}.h5"
LABEL_COL              = "protein_families"
K_VALUES               = [1, 2, 5, 10, 15, 20, 30, 50]
N_PCA_COMPONENTS       = 50
DELIMITER              = b"---PARQUET_DELIMITER---"
OUT_DIR                = Path("evaluation") / DATASET
OUT_DIR.mkdir(exist_ok=True)

K_STORED               = 300
MIN_PROTEINS_PER_CLASS = 0
MAX_EVAL_SAMPLE        = 20_000  # cap applied to ALL metrics when N > this value; None = use all

# =============================================================
# 1. Read parquetbundle  — single read, no duplicate
# =============================================================
print("Reading parquetbundle...")
with open(BUNDLE, "rb") as f:
    parts = f.read().split(DELIMITER)

annot_df    = pq.read_table(io.BytesIO(parts[0])).to_pandas()   # read ONCE
proj_df     = pq.read_table(io.BytesIO(parts[2])).to_pandas()
parquet_ids = set(annot_df["protein_id"])

# =============================================================
# 2. Labels — derived from the single annot_df read above
# =============================================================
print("Reading labels...")
label_df = annot_df.rename(columns={"protein_id": "identifier"})
del annot_df; gc.collect()                                       # free now we're done

label_df[LABEL_COL] = (
    label_df[LABEL_COL]
    .str.split("|").str[0]
    .str.strip()
)
label_df[LABEL_COL] = label_df[LABEL_COL].replace("", np.nan)
label_df = label_df.dropna(subset=[LABEL_COL])

initial_protein_count = len(label_df)
initial_family_count  = label_df[LABEL_COL].nunique()

class_counts  = label_df[LABEL_COL].value_counts()
valid_classes = class_counts[class_counts >= MIN_PROTEINS_PER_CLASS].index
label_df      = label_df[label_df[LABEL_COL].isin(valid_classes)]

dropped_proteins = initial_protein_count - len(label_df)
dropped_families = initial_family_count - len(valid_classes)
labelled_ids     = set(label_df["identifier"].astype(str))

print(f"  Proteins with labels : {initial_protein_count}")
print(f"  Dropped (no label)   : {6436 - initial_protein_count}")
print(f"  Dropped rare classes (<{MIN_PROTEINS_PER_CLASS}) : {dropped_families} families ({dropped_proteins} proteins)")
print(f"  Final active proteins: {len(labelled_ids)}")

# =============================================================
# 3. H5 embeddings
# =============================================================
print("Reading H5 embeddings...")
with h5py.File(H5, "r") as f:
    h5_keys = set(f.keys())
    shared  = sorted(parquet_ids & h5_keys & labelled_ids)

    if not shared:
        raise ValueError("No overlapping IDs across H5, parquetbundle and labels.")

    print(f"  Proteins used : {len(shared)}")
    embeddings = np.stack([f[pid][:] for pid in shared]).astype(np.float16)

N = len(shared)

label_df   = label_df.set_index("identifier").loc[shared]
labels_raw = label_df[LABEL_COL].values
le         = LabelEncoder()
labels_int = le.fit_transform(labels_raw)
print(f"  Unique families : {len(le.classes_)}")

# --- Global sample for all metrics -------------------------------------------
# A single RNG draw so every metric sees the exact same proteins.
# When N <= MAX_EVAL_SAMPLE the sample IS the full dataset (no information lost).
rng = np.random.default_rng(42)
if MAX_EVAL_SAMPLE and N > MAX_EVAL_SAMPLE:
    _sample_idx = np.sort(rng.choice(N, size=MAX_EVAL_SAMPLE, replace=False))
    print(f"  Sampling {MAX_EVAL_SAMPLE} / {N} proteins for all metrics (seed=42)")
else:
    _sample_idx = np.arange(N)   # full dataset

N_EVAL        = len(_sample_idx)
labels_eval   = labels_int[_sample_idx]        # (N_EVAL,) — used by supervised metrics
shared_eval   = [shared[i] for i in _sample_idx]  # for coord alignment later
# -----------------------------------------------------------------------------

# Pre-split proj_df by projection name to avoid repeated full-dataframe scans
proj_groups = {
    name: grp.copy()
    for name, grp in proj_df.groupby("projection_name")
}
del proj_df; gc.collect()

# =============================================================
# 4. NearestNeighbors helper
# =============================================================
def get_neighbors(X, k_stored=K_STORED, metric="cosine"):
    """Return (N, k_stored) array of neighbor indices (closest first, self excluded)."""
    nn = NearestNeighbors(
        n_neighbors=k_stored + 1,
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return indices[:, 1:].astype(np.int32)

def short_name(proj_name):
    return proj_name.split("—")[-1].strip()

# =============================================================
# 5. Vectorised metric functions
#    All neighbour-based metrics operate on (N_EVAL, K_STORED)
#    arrays that have already been sub-sampled.  Silhouette also
#    receives the sub-sampled coordinate / embedding matrix.
# =============================================================

def _sample_neighbors(neighbors_full_n):
    """
    Slice a (N, K_STORED) neighbor array down to the eval sample,
    then remap the stored neighbor indices so they refer to positions
    within the sample rather than the full dataset.

    Points whose neighbors fall outside the sample are remapped to a
    sentinel (N_EVAL) — they won't appear in any top-k lookup since
    k <= K_STORED and the sentinel is out-of-range.
    """
    sub = neighbors_full_n[_sample_idx]          # (N_EVAL, K_STORED)
    # Build a lookup: original index → position in sample (-1 if absent)
    inv = np.full(N, -1, dtype=np.int32)
    inv[_sample_idx] = np.arange(N_EVAL, dtype=np.int32)
    remapped = inv[sub]                           # (N_EVAL, K_STORED), -1 for outsiders
    # Replace -1 (not in sample) with sentinel N_EVAL so they're harmlessly out-of-range
    remapped[remapped == -1] = N_EVAL
    return remapped.astype(np.int32)


def knn_recall_at_k(neighbors_gt, neighbors_proj, k):
    """Fully vectorised recall on the eval sample."""
    gt_k = neighbors_gt[:, :k]    # (N_EVAL, k)
    pr_k = neighbors_proj[:, :k]  # (N_EVAL, k)
    hits = np.count_nonzero(
        (gt_k[:, :, None] == pr_k[:, None, :]).any(axis=2),
        axis=1,
    )
    return float(hits.mean() / k)


def _build_rank_matrix(neighbors):
    """
    Build a dense (N_EVAL, N_EVAL) int32 rank matrix.
    Entries for neighbors outside the sample already carry sentinel N_EVAL
    (set by _sample_neighbors), so they naturally sort as 'far away'.
    """
    n, K_ = neighbors.shape
    rank_matrix = np.full((n, n), K_, dtype=np.int32)
    row_idx = np.repeat(np.arange(n), K_)
    col_idx = neighbors.ravel()
    ranks   = np.tile(np.arange(K_), n)
    # Only write entries that are valid (< n); sentinels are out-of-bounds for columns
    mask = col_idx < n
    rank_matrix[row_idx[mask], col_idx[mask]] = ranks[mask]
    return rank_matrix


def trustworthiness_at_k(neighbors_highd, neighbors_proj, k):
    """Vectorised trustworthiness on the eval sample."""
    K_stored = neighbors_highd.shape[1]
    rank_hd  = _build_rank_matrix(neighbors_highd)

    pr_k     = neighbors_proj[:, :k]
    # Clip to valid column range before indexing
    pr_k_clipped = np.clip(pr_k, 0, N_EVAL - 1)
    row_idx  = np.repeat(np.arange(N_EVAL), k)
    hd_ranks = rank_hd[row_idx, pr_k_clipped.ravel()].reshape(N_EVAL, k)

    penalty_mask = hd_ranks >= k
    penalties    = np.where(penalty_mask, hd_ranks - k, 0).sum()

    norm = k * N_EVAL * (2 * N_EVAL - 3 * k - 1) / 2
    del rank_hd
    return float(1 - (2 / norm) * penalties)


def continuity_at_k(neighbors_proj, neighbors_highd, k):
    """Vectorised continuity on the eval sample."""
    rank_proj = _build_rank_matrix(neighbors_proj)

    hd_k     = neighbors_highd[:, :k]
    hd_k_clipped = np.clip(hd_k, 0, N_EVAL - 1)
    row_idx  = np.repeat(np.arange(N_EVAL), k)
    pr_ranks = rank_proj[row_idx, hd_k_clipped.ravel()].reshape(N_EVAL, k)

    penalty_mask = pr_ranks >= k
    penalties    = np.where(penalty_mask, pr_ranks - k, 0).sum()

    norm = k * N_EVAL * (2 * N_EVAL - 3 * k - 1) / 2
    del rank_proj
    return float(1 - (2 / norm) * penalties)


def knn_accuracy_at_k(neighbors, labels, k):
    """Vectorised kNN accuracy on the eval sample."""
    n          = len(labels)
    nbr_labels = labels[np.clip(neighbors[:, :k], 0, n - 1)]   # (n, k)
    n_classes  = int(labels.max()) + 1
    counts     = np.zeros((n, n_classes), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(n), k), nbr_labels.ravel()), 1)
    predicted  = counts.argmax(axis=1)
    return float((predicted == labels).mean())


def compute_silhouette(X, labels, metric="euclidean"):
    """Silhouette on whatever matrix is passed in (already sub-sampled if needed)."""
    return float(silhouette_score(X, labels, metric=metric))

# =============================================================
# 6. Color palette
# =============================================================
BASE_COLORS = {
    "Full 1024-d":             "#1F77B4",
    f"PCA-{N_PCA_COMPONENTS}": "#2CA02C",
}
PROJ_PALETTE  = ["#FF7F0E", "#9467BD", "#8C564B", "#E377C2", "#17BECF"]
LINESTYLES_GT = {"Full": "-", "PCA50": "--"}

projection_names = list(proj_groups.keys())
proj_colors      = {pn: PROJ_PALETTE[i % len(PROJ_PALETTE)] for i, pn in enumerate(projection_names)}

# =============================================================
# 7. Ground-truth neighbor spaces
# =============================================================
print("\nComputing ground truth (full 1024-dim, cosine)...")
emb_f32        = normalize(embeddings.astype(np.float32), norm="l2")
neighbors_full = get_neighbors(emb_f32, metric="cosine")
# Sub-sample for metrics (no-op when N <= MAX_EVAL_SAMPLE)
neighbors_full_eval = _sample_neighbors(neighbors_full)
del emb_f32; gc.collect()

print(f"Computing ground truth (top-{N_PCA_COMPONENTS} PCA, cosine)...")
pca     = PCA(n_components=N_PCA_COMPONENTS, svd_solver="arpack", random_state=42)
emb_pca = pca.fit_transform(embeddings.astype(np.float32))
emb_pca = normalize(emb_pca, norm="l2")
neighbors_pca50      = get_neighbors(emb_pca, metric="cosine")
neighbors_pca50_eval = _sample_neighbors(neighbors_pca50)
gc.collect()

# =============================================================
# 8. Result containers + ground-truth supervised metrics
# =============================================================
unsupervised = {}
knn_acc      = {}
silhouette   = {}

print("\nComputing metrics for ground truth spaces...")
knn_acc["Full 1024-d"]             = [knn_accuracy_at_k(neighbors_full_eval,  labels_eval, k) for k in K_VALUES]
knn_acc[f"PCA-{N_PCA_COMPONENTS}"] = [knn_accuracy_at_k(neighbors_pca50_eval, labels_eval, k) for k in K_VALUES]

# Silhouette — pass only the sampled rows of each matrix
emb_f32_sil = embeddings.astype(np.float32)[_sample_idx]
silhouette["Full 1024-d"] = compute_silhouette(emb_f32_sil, labels_eval, metric="cosine")
del emb_f32_sil; gc.collect()

emb_pca_eval = emb_pca[_sample_idx]
silhouette[f"PCA-{N_PCA_COMPONENTS}"] = compute_silhouette(emb_pca_eval, labels_eval, metric="euclidean")
del emb_pca_eval, emb_pca; gc.collect()

# =============================================================
# 9. Per-projection metrics
# =============================================================
print(f"\nProjection methods found: {projection_names}")

for proj_name, subset in proj_groups.items():
    sname = short_name(proj_name)
    print(f"\nProcessing {proj_name}...")

    # Align to `shared` order — only proteins in our eval set
    subset = subset[subset["identifier"].isin(labelled_ids)].set_index("identifier").loc[shared]

    dims   = ["x", "y"] if subset["z"].isna().all() else ["x", "y", "z"]
    coords = subset[dims].values.astype(np.float32)

    neighbors_proj      = get_neighbors(coords, metric="euclidean")
    neighbors_proj_eval = _sample_neighbors(neighbors_proj)

    coords_eval = coords[_sample_idx]

    knn_acc[sname]    = [knn_accuracy_at_k(neighbors_proj_eval, labels_eval, k) for k in K_VALUES]
    silhouette[sname] = compute_silhouette(coords_eval, labels_eval, metric="euclidean")

    for gt_label, nbrs_gt_eval in [("Full", neighbors_full_eval), ("PCA50", neighbors_pca50_eval)]:
        line_label = f"{sname} - {gt_label}"
        recalls, trusts, conts = [], [], []
        for k in K_VALUES:
            print(f"  [{gt_label}] k={k}...", end=" ", flush=True)
            recalls.append(knn_recall_at_k(nbrs_gt_eval,        neighbors_proj_eval, k))
            trusts.append( trustworthiness_at_k(nbrs_gt_eval,   neighbors_proj_eval, k))
            conts.append(  continuity_at_k(neighbors_proj_eval, nbrs_gt_eval,        k))
        print()
        unsupervised[line_label] = {
            "recall": recalls, "trust": trusts, "cont": conts,
            "proj_name": proj_name, "gt_label": gt_label,
        }

    del neighbors_proj, neighbors_proj_eval, coords, coords_eval; gc.collect()

del neighbors_full, neighbors_full_eval, neighbors_pca50, neighbors_pca50_eval, embeddings; gc.collect()

# =============================================================
# 10. Summary TSV
# =============================================================
rows = []
for line_label, m in unsupervised.items():
    rows.append({
        "space": line_label, "type": "unsupervised",
        "recall_mean":  round(float(np.mean(m["recall"])), 4),
        "trust_mean":   round(float(np.mean(m["trust"])),  4),
        "cont_mean":    round(float(np.mean(m["cont"])),   4),
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
def get_linestyle(gt_label):               return LINESTYLES_GT[gt_label]
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
# 12–14. Line plots
# =============================================================
save_line_plot("recall", "kNN Recall",      "recall.png")
save_line_plot("trust",  "Trustworthiness", "trustworthiness.png")
save_line_plot("cont",   "Continuity",      "continuity.png")

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
ax.grid(True, axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "silhouette.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: silhouette.png")

print(f"\nDone. All outputs in: {OUT_DIR}")