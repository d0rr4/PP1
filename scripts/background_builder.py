"""
run_all_backgrounds.py
======================
Generates rhoPCA projections for all background strategies and saves HTML plots.

Strategies:
  1a. length_matched_random  - synthetic embeddings (average of same-length real proteins)
  1b. length_matched         - real proteins matched ±10 aa from background pool
  1c. poly-alanine           - synthetic low-complexity baseline
  2a. signal_peptide         - only proteins WITH signal peptide
  2b. signal_peptide_mixed   - half with, half without signal peptide

Usage:
    cd ~/PP1
    python3 scripts/run_all_backgrounds.py

Outputs HTML plots to output/rhopca_backgrounds/
"""

from __future__ import annotations
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as sp
from rhopca.core import rhoPCA
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FG_H5      = Path("data/backgrounds/epw9NceGu9")     # 6444 foreground toxins
LM_H5      = Path("data/backgrounds/length_matched")     # 101k length-matched pool
SP_H5      = Path("data/backgrounds/q2GpWSr29d")     # 19k signal peptide pool
ANN_CSV    = Path("data/backgrounds/annotations_foreground.csv")
FG_FASTA   = Path("data/uniprotkb_reviewed_true_AND_keyword_KW_2026_06_11.fasta.gz")
OUT_DIR    = Path("output/rhopca_backgrounds")
RANDOM_H5  = Path("data/backgrounds/background_1a_random_prot_t5.h5")
POLYALA_H5 = Path("data/backgrounds/background_1c_polyala_prot_t5.h5")
SEED       = 42
PCA_DIMS   = 50
LM_WINDOW  = 10
SP_BATCH   = 20
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)


# =============================================================================
# Helpers
# =============================================================================

def parse_fasta_gz(path: Path) -> dict[str, str]:
    import gzip
    seqs, cur_id, cur_seq = {}, None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur_seq)
                # store as plain accession to match H5 keys
                pid = line[1:].split()[0]
                cur_id = pid.split("|")[1] if "|" in pid else pid
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id:
        seqs[cur_id] = "".join(cur_seq)
    return seqs


def run_rhopca(fg: np.ndarray, bg: np.ndarray, fg_ids: list[str],
               ann: pd.DataFrame, title: str, outname: str) -> None:
    n_comp = min(PCA_DIMS, len(fg) - 1, len(bg) - 1)
    logger.info("%s | fg=%d bg=%d pca_dims=%d", title, len(fg), len(bg), n_comp)

    if n_comp < 2:
        logger.warning("Skipping %s — not enough samples for PCA (n_comp=%d)", title, n_comp)
        return

    all_data = np.vstack([fg, bg])
    reduced  = PCA(n_components=n_comp, random_state=SEED).fit_transform(all_data)
    fg_r     = reduced[:len(fg)]
    bg_r     = reduced[len(fg):]

    X      = np.vstack([fg_r, bg_r])
    labels = ["target"] * len(fg_r) + ["background"] * len(bg_r)
    adata  = ad.AnnData(sp.csr_matrix(X.astype(np.float32)))
    adata.obs["group"] = pd.Categorical(labels)

    model = rhoPCA(adata, contrast_column="group",
                   target="target", background="background", n_GEs=2)
    model.fit()
    coords = np.array(model.target_proj)[:, :2]

    df = pd.DataFrame({
        "identifier": fg_ids,
        "rhopca_x":   coords[:, 0],
        "rhopca_y":   coords[:, 1],
    }).merge(ann, on="identifier", how="left")

    fig = px.scatter(
        df, x="rhopca_x", y="rhopca_y", color="organism",
        hover_data=["identifier", "protein_name"],
        title=title, width=1100, height=750,
    )
    out = OUT_DIR / f"{outname}.html"
    fig.write_html(out)
    logger.info("Saved → %s", out)


def synthetic_embedding_for_length(
    target_len: int,
    pool_embs: np.ndarray,
    pool_lengths: np.ndarray,
    n_avg: int = 5,
) -> np.ndarray:
    for window in [0, 5, 10, 20, 50, 100]:
        mask = (pool_lengths >= target_len - window) & \
               (pool_lengths <= target_len + window)
        idxs = np.where(mask)[0]
        if len(idxs) >= 1:
            break
    if len(idxs) == 0:
        logger.warning("No pool protein for length=%d — using random", target_len)
        return rng.standard_normal(pool_embs.shape[1]).astype(np.float32)
    chosen = rng.choice(idxs, size=min(n_avg, len(idxs)), replace=False)
    return pool_embs[chosen].mean(axis=0)


def fetch_signal_peptide_annotations(
    accessions: list[str],
    batch_size: int = SP_BATCH,
    sleep: float = 0.5,
) -> set[str]:
    sp_set: set[str] = set()
    batches = [accessions[i:i + batch_size]
               for i in range(0, len(accessions), batch_size)]
    logger.info("Fetching SP annotations for %d accessions (%d batches)...",
                len(accessions), len(batches))
    for i, batch in enumerate(batches):
        query = " OR ".join(f"accession:{a}" for a in batch)
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({
            "query": query,
            "fields": "accession,ft_signal",
            "format": "json",
            "size":   batch_size,
        })
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            for entry in data.get("results", []):
                acc      = entry.get("primaryAccession", "")
                features = entry.get("features", [])
                if any(f.get("type") == "Signal" for f in features):
                    sp_set.add(acc)
        except Exception as e:
            logger.warning("UniProt API error (batch %d): %s", i, e)
        if i < len(batches) - 1:
            time.sleep(sleep)
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d batches, %d with SP so far", i+1, len(batches), len(sp_set))
    logger.info("SP annotations done: %d/%d have SP", len(sp_set), len(accessions))
    return sp_set


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    # ── Load foreground ───────────────────────────────────────────────────────
    with h5py.File(FG_H5, "r") as f:
        fg_ids = list(f.keys())
        fg     = np.vstack([f[k][:] for k in fg_ids])
    logger.info("Foreground: %d proteins, dim=%d", len(fg), fg.shape[1])

    # ── Load annotations ──────────────────────────────────────────────────────
    ann = pd.read_csv(ANN_CSV)
    logger.info("Annotations: %d rows, %d unique organisms",
                len(ann), ann["organism"].nunique())

    # ── Load foreground sequences for lengths ─────────────────────────────────
    fg_seqs    = parse_fasta_gz(FG_FASTA)
    fg_lengths = np.array([len(fg_seqs.get(pid, "")) for pid in fg_ids])
    logger.info("Foreground lengths: min=%d max=%d mean=%.1f",
                fg_lengths.min(), fg_lengths.max(), fg_lengths.mean())

    # ── Load length-matched pool ──────────────────────────────────────────────
    logger.info("Loading length-matched pool...")
    with h5py.File(LM_H5, "r") as f:
        lm_ids  = list(f.keys())
        lm_pool = np.vstack([f[k][:] for k in lm_ids])
    logger.info("Length-matched pool: %d proteins", len(lm_pool))

    # Get lengths from FASTA directly
    logger.info("Reading lengths from FASTA...")
    lm_fasta_seqs = parse_fasta_gz(Path("data/uniprotkb_reviewed_true_AND_taxonomy_na_2026_06_11.fasta.gz"))
    # parse_fasta_gz already extracts plain accession (splits on | )
    lm_lengths = np.array([len(lm_fasta_seqs.get(pid, "")) for pid in lm_ids])
    missing = (lm_lengths == 0).sum()
    logger.info("Pool lengths: min=%d max=%d mean=%.1f missing=%d",
                lm_lengths.min(), lm_lengths.max(), lm_lengths.mean(), missing)

    # =========================================================================
    # 1a. Synthetic — average of same-length real proteins from LM pool
    # =========================================================================
    logger.info("=== 1a. Random sequences (real ProtT5 embeddings) ===")
    with h5py.File(RANDOM_H5, "r") as f:
        rand_ids = list(f.keys())
        bg_1a    = np.vstack([f[k][:] for k in rand_ids])
    logger.info("Random background: %d proteins", len(bg_1a))
    run_rhopca(fg, bg_1a, fg_ids, ann,
            "rhoPCA — 1a: random sequences (real ProtT5 embeddings)",
            "1a_length_matched_random")

    # =========================================================================
    # 1b. Real proteins matched ±10 aa from LM pool
    # =========================================================================
    logger.info("=== 1b. Length-matched ±%d aa (real proteins) ===", LM_WINDOW)
    lm_idx, used = [], set()
    for l in fg_lengths:
        mask  = (lm_lengths >= l - LM_WINDOW) & (lm_lengths <= l + LM_WINDOW)
        cands = [i for i in np.where(mask)[0] if i not in used]
        if cands:
            chosen = int(rng.choice(cands))
            lm_idx.append(chosen)
            used.add(chosen)
        else:
            logger.warning("No ±%d match for length=%d", LM_WINDOW, l)

    if lm_idx:
        run_rhopca(fg, lm_pool[lm_idx], fg_ids, ann,
                   f"rhoPCA — 1b: length-matched ±{LM_WINDOW} aa ({len(lm_idx)} proteins)",
                   "1b_length_matched_real")
    else:
        logger.warning("1b skipped — no length matches found")

    # =========================================================================
    # 1c. Poly-alanine approximation
    # =========================================================================
    logger.info("=== 1c. Poly-alanine sequences (real ProtT5 embeddings) ===")
    with h5py.File(POLYALA_H5, "r") as f:
        poly_ids = list(f.keys())
        bg_1c    = np.vstack([f[k][:] for k in poly_ids])
    logger.info("Poly-alanine background: %d proteins", len(bg_1c))
    run_rhopca(fg, bg_1c, fg_ids, ann,
            "rhoPCA — 1c: poly-alanine sequences (real ProtT5 embeddings)",
            "1c_polyala_approx")

    # =========================================================================
    # 2. Signal peptide strategies
    # =========================================================================
    logger.info("Loading signal peptide pool...")
    with h5py.File(SP_H5, "r") as f:
        sp_ids  = list(f.keys())
        sp_pool = np.vstack([f[k][:] for k in sp_ids])
    logger.info("Signal peptide pool: %d proteins", len(sp_pool))

    logger.info("=== Fetching SP annotations from UniProt ===")
    sp_annotations = fetch_signal_peptide_annotations(sp_ids)

    with_sp    = [i for i, pid in enumerate(sp_ids) if pid in sp_annotations]
    without_sp = [i for i, pid in enumerate(sp_ids) if pid not in sp_annotations]
    logger.info("SP pool: %d with SP, %d without SP", len(with_sp), len(without_sp))

    # 2a. Background = only proteins WITH signal peptide
    logger.info("=== 2a. Signal peptide only ===")
    if len(with_sp) >= 10:
        run_rhopca(fg, sp_pool[with_sp], fg_ids, ann,
                   f"rhoPCA — 2a: signal peptide background ({len(with_sp)} proteins with SP)",
                   "2a_signal_peptide_only")
    else:
        logger.warning("2a skipped — only %d proteins with SP", len(with_sp))

    # 2b. Mixed: half with SP, half without
    logger.info("=== 2b. Signal peptide mixed ===")
    n_each = min(len(with_sp), len(without_sp), len(fg))
    if n_each >= 10:
        sample_with    = list(rng.choice(with_sp,    size=n_each, replace=False))
        sample_without = list(rng.choice(without_sp, size=n_each, replace=False))
        mixed_idx      = sample_with + sample_without
        run_rhopca(fg, sp_pool[mixed_idx], fg_ids, ann,
                   f"rhoPCA — 2b: signal peptide mixed ({n_each} with + {n_each} without)",
                   "2b_signal_peptide_mixed")
    else:
        logger.warning("2b skipped — too few proteins (n_each=%d)", n_each)

    logger.info("=== All done. Plots saved to %s ===", OUT_DIR)


def _fetch_lengths(accessions: list[str], batch_size: int = 500) -> np.ndarray:
    """Fetch sequence lengths from UniProt for a list of accessions."""
    length_map: dict[str, int] = {}
    batches = [accessions[i:i + batch_size]
               for i in range(0, len(accessions), batch_size)]
    logger.info("Fetching lengths for %d proteins (%d batches)...",
                len(accessions), len(batches))
    for i, batch in enumerate(batches):
        query = " OR ".join(f"accession:{a}" for a in batch)
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({
            "query": query,
            "fields": "accession,length",
            "format": "json",
            "size":   batch_size,
        })
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            for entry in data.get("results", []):
                acc = entry.get("primaryAccession", "")
                l   = entry.get("sequence", {}).get("length", 0)
                length_map[acc] = l
        except Exception as e:
            logger.warning("UniProt length fetch error (batch %d): %s", i, e)
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d batches done", i+1, len(batches))
        time.sleep(0.2)

    lengths = np.array([length_map.get(pid, 0) for pid in accessions])
    missing = (lengths == 0).sum()
    if missing:
        logger.warning("%d/%d proteins have no length from UniProt", missing, len(accessions))
    return lengths


if __name__ == "__main__":
    main()