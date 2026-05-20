import io
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize

# --- Config ---
BUNDLE           = Path("/mnt/c/Users/janni/OneDrive/repos/PP1/PP/protspace/data.parquetbundle")
H5               = Path("embeds/toxins.h5")
K_VALUES         = [1, 2, 5, 10, 15, 20, 30, 50]
N_PCA_COMPONENTS = 50
DELIMITER        = b"---PARQUET_DELIMITER---"
OUT_DIR          = Path("evaluation")
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
# 2. Read H5 embeddings
# =============================================================
print("Reading H5 embeddings...")
with h5py.File(H5, "r") as f:
    h5_keys = set(f.keys())
    shared  = sorted(parquet_ids & h5_keys)

    if not shared:
        raise ValueError(
            "No overlapping IDs between H5 and parquetbundle. "
            "Check identifier format."
        )

    print(f"  Shared proteins : {len(shared)}")
    print(f"  H5 only         : {len(h5_keys - parquet_ids)}")
    print(f"  Parquet only    : {len(parquet_ids - h5_keys)}")

    embeddings = np.stack([f[pid][:] for pid in shared])  # (N, 1024)

id_index = {pid: i for i, pid in enumerate(shared)}
N = len(shared)

# =============================================================
# 3. Helper: ranked neighbour array and rank lookup matrix
# =============================================================
def ranked_neighbours(dist_matrix):
    """
    Returns (N, N-1) array of neighbour indices sorted by distance,
    self excluded. Used for iterating over neighbours in order.
    """
    return np.argsort(dist_matrix, axis=1)[:, 1:]


def build_rank_matrix(dist_matrix):
    """
    Returns (N, N) matrix where rank_matrix[i, j] = 0-based rank of
    protein j as seen from protein i, with self excluded (self = -1).
    Allows direct lookup by protein index.
    """
    n           = dist_matrix.shape[0]
    sorted_idx  = np.argsort(dist_matrix, axis=1)
    rank_matrix = np.empty((n, n), dtype=np.int32)
    rows        = np.arange(n)[:, None]
    rank_matrix[rows, sorted_idx] = np.arange(n)
    rank_matrix -= 1  # self: 0 → -1; closest neighbour: 1 → 0
    return rank_matrix

# =============================================================
# 4. Metric functions
# =============================================================
def knn_recall_at_k(ranked_highd, ranked_proj, k):
    """
    Mean fraction of high-d k-neighbours recovered in projection
    k-neighbours.
    """
    recalls = []
    for i in range(N):
        gt   = set(ranked_highd[i, :k])
        pred = set(ranked_proj[i,  :k])
        recalls.append(len(gt & pred) / k)
    return float(np.mean(recalls))


def trustworthiness_at_k(rank_highd, ranked_proj, k):
    """
    Penalises false positives: points that appear as neighbours in the
    projection but were NOT neighbours in high-d space.
    Range [0, 1], 1 = perfect.
    """
    penalty = 0.0
    for i in range(N):
        for j in ranked_proj[i, :k]:
            r = rank_highd[i, j]
            if r >= k:
                penalty += r - k
    normalisation = k * N * (2 * N - 3 * k - 1) / 2
    return float(1 - (2 / normalisation) * penalty)


def continuity_at_k(rank_proj, ranked_highd, k):
    """
    Penalises false negatives: high-d neighbours that went missing
    in the projection.
    Range [0, 1], 1 = perfect.
    """
    penalty = 0.0
    for i in range(N):
        for j in ranked_highd[i, :k]:
            r = rank_proj[i, j]
            if r >= k:
                penalty += r - k
    normalisation = k * N * (2 * N - 3 * k - 1) / 2
    return float(1 - (2 / normalisation) * penalty)

# =============================================================
# 5. Precompute ground truths
# =============================================================
print("Computing ground truth distances (full 1024-dim, cosine)...")
emb_norm     = normalize(embeddings, norm="l2")
dist_full    = cosine_distances(emb_norm)
ranked_full  = ranked_neighbours(dist_full)
rank_full    = build_rank_matrix(dist_full)

print(f"Computing ground truth distances (top-{N_PCA_COMPONENTS} PCA, cosine)...")
pca            = PCA(n_components=N_PCA_COMPONENTS, svd_solver="arpack", random_state=42)
emb_pca50      = pca.fit_transform(embeddings)
emb_pca50_norm = normalize(emb_pca50, norm="l2")
dist_pca50     = cosine_distances(emb_pca50_norm)
ranked_pca50   = ranked_neighbours(dist_pca50)
rank_pca50     = build_rank_matrix(dist_pca50)

# =============================================================
# 6. Compute all metrics across k for each projection method
# =============================================================
projection_names = proj_df["projection_name"].unique()
print(f"\nProjection methods found: {list(projection_names)}")

results = {}

for proj_name in projection_names:
    print(f"\nProcessing {proj_name}...")
    subset = proj_df[proj_df["projection_name"] == proj_name].copy()
    subset = subset[subset["identifier"].isin(id_index)]
    subset = subset.set_index("identifier").loc[shared]

    dims   = ["x", "y"] if subset["z"].isna().all() else ["x", "y", "z"]
    coords = subset[dims].values.astype(float)

    dist_proj   = euclidean_distances(coords)
    ranked_proj = ranked_neighbours(dist_proj)
    rank_proj   = build_rank_matrix(dist_proj)

    results[proj_name] = {"full": {}, "pca50": {}}

    for gt_label, (ranked_gt, rank_gt) in [
        ("full",  (ranked_full,  rank_full)),
        ("pca50", (ranked_pca50, rank_pca50)),
    ]:
        recalls, trusts, conts = [], [], []
        for k in K_VALUES:
            print(f"  [{gt_label}] k={k}...", end=" ", flush=True)
            recalls.append(knn_recall_at_k(ranked_gt,   ranked_proj, k))
            trusts.append( trustworthiness_at_k(rank_gt,    ranked_proj, k))
            conts.append(  continuity_at_k(rank_proj,   ranked_gt,   k))
        print()
        results[proj_name][gt_label] = {
            "recall": recalls,
            "trust":  trusts,
            "cont":   conts,
        }

# =============================================================
# 7. Save summary CSV (mean across k values)
# =============================================================
rows = []
for proj_name, gt_dict in results.items():
    for gt_label, metrics in gt_dict.items():
        rows.append({
            "projection":   proj_name,
            "ground_truth": gt_label,
            "recall_mean":  round(float(np.mean(metrics["recall"])), 4),
            "trust_mean":   round(float(np.mean(metrics["trust"])),  4),
            "cont_mean":    round(float(np.mean(metrics["cont"])),   4),
        })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
print("\n--- Summary (mean across k values) ---")
print(summary_df.to_string(index=False))

# =============================================================
# 8. One plot per projection method
# =============================================================
METRIC_LABELS = {
    "recall": "kNN Recall",
    "trust":  "Trustworthiness",
    "cont":   "Continuity",
}
COLORS = {
    "full":  {"recall": "#2196F3", "trust": "#4CAF50", "cont": "#FF5722"},
    "pca50": {"recall": "#90CAF9", "trust": "#A5D6A7", "cont": "#FFCCBC"},
}
LINESTYLES = {"full": "-", "pca50": "--"}

for proj_name, gt_dict in results.items():
    fig, (ax_recall, ax_tc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(proj_name, fontsize=14, fontweight="bold", y=1.02)

    for gt_label, metrics in gt_dict.items():
        ls         = LINESTYLES[gt_label]
        gt_display = "Full 1024-d" if gt_label == "full" else f"PCA-{N_PCA_COMPONENTS}"

        # --- left subplot: recall only ---
        ax_recall.plot(
            K_VALUES,
            metrics["recall"],
            color=COLORS[gt_label]["recall"],
            linestyle=ls,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=gt_display,
        )

        # --- right subplot: trustworthiness + continuity ---
        for metric_key, metric_label in [("trust", "Trustworthiness"), ("cont", "Continuity")]:
            ax_tc.plot(
                K_VALUES,
                metrics[metric_key],
                color=COLORS[gt_label][metric_key],
                linestyle=ls,
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=f"{metric_label} ({gt_display})",
            )

    for ax, title in [(ax_recall, "kNN Recall"), (ax_tc, "Trustworthiness & Continuity")]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        #ax.set_ylim(0, 1.05)
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fname = OUT_DIR / f"{proj_name.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

print("\nDone. All outputs in:", OUT_DIR)