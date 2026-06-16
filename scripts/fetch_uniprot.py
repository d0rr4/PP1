"""
run_all_backgrounds.py
======================
Generates rhoPCA projections for all background strategies and saves HTML plots.

Strategies:
  1a. length_matched_random  - synthetic embeddings (average of same-length real proteins)
  1b. length_matched         - real proteins matched ±10 aa
  1c. length_exact_aaaa      - synthetic embeddings approximating poly-A sequences
  2a. signal_peptide         - only proteins WITH signal peptide
  2b. signal_peptide_mixed   - half with, half without signal peptide

Usage:
    python scripts/run_all_backgrounds.py

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
# CONFIG — edit these paths if needed
# ─────────────────────────────────────────────────────────────────────────────
POOL_H5   = Path("scripts/1P1FTWE2Ua_clean.h5")   # 19k toxin pool
FG_H5     = Path("notebooks/foreground_3ftx.h5")   # 87 foreground proteins
FASTA     = Path("notebooks/merged_processed.fasta")
ANN_CSV   = Path("scripts/annotations_3ftx.csv")
OUT_DIR   = Path("output/rhopca_backgrounds")
SEED      = 42
PCA_DIMS  = 50      # pre-reduction dimensionality
LM_WINDOW = 10      # ±aa for length matching
SP_BATCH  = 200     # accessions per UniProt API request
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)


# =============================================================================
# Helpers
# =============================================================================

def parse_fasta(path: Path) -> dict[str, str]:
    seqs, cur_id, cur_seq = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id, cur_seq = line[1:].split()[0], []
            else:
                cur_seq.append(line)
    if cur_id:
        seqs[cur_id] = "".join(cur_seq)
    return seqs


def run_rhopca(fg: np.ndarray, bg: np.ndarray, fg_ids: list[str],
               ann: pd.DataFrame, title: str, outname: str) -> None:
    """Run PCA pre-reduction + rhoPCA and save HTML plot."""
    n_comp = min(PCA_DIMS, len(fg) - 1, len(bg) - 1)
    logger.info("%s | fg=%d bg=%d pca_dims=%d", title, len(fg), len(bg), n_comp)

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
        title=title, width=1000, height=700,
    )
    out = OUT_DIR / f"{outname}.html"
    fig.write_html(out)
    logger.info("Saved → %s", out)


def synthetic_embedding_for_length(
    target_len: int,
    pool_embs: np.ndarray,
    pool_lengths: np.ndarray,
    n_avg: int = 5,
    window_steps: tuple = (0, 5, 10, 20, 50),
) -> np.ndarray:
    """
    Approximate embedding for a protein of target_len by averaging
    n_avg real embeddings of the same (or similar) length.
    """
    for window in window_steps:
        mask = (pool_lengths >= target_len - window) & \
               (pool_lengths <= target_len + window)
        idxs = np.where(mask)[0]
        if len(idxs) >= 1:
            break
    if len(idxs) == 0:
        logger.warning("No pool protein found for length=%d — using random", target_len)
        return rng.standard_normal(pool_embs.shape[1]).astype(np.float32)
    chosen = rng.choice(idxs, size=min(n_avg, len(idxs)), replace=False)
    return pool_embs[chosen].mean(axis=0)


def fetch_signal_peptide_annotations(
    accessions: list[str],
    batch_size: int = SP_BATCH,
    sleep: float = 0.5,
) -> set[str]:
    """
    Query UniProt REST API and return the set of accessions that have
    a signal peptide annotation.
    """
    sp_set: set[str] = set()
    batches = [accessions[i:i + batch_size]
               for i in range(0, len(accessions), batch_size)]

    logger.info("Fetching signal peptide annotations for %d accessions "
                "(%d batches)...", len(accessions), len(batches))

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

        if (i + 1) % 5 == 0:
            logger.info("  ... %d/%d batches done, %d with SP so far",
                        i + 1, len(batches), len(sp_set))

    logger.info("Signal peptide annotations: %d/%d have SP",
                len(sp_set), len(accessions))
    return sp_set


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    # ── Load foreground ───────────────────────────────────────────────────────
    with h5py.File(FG_H5, "r") as f:
        fg_ids = list(f.keys())
        fg     = np.vstack([f[k][:] for k in fg_ids])

    fg_set = set(fg_ids)
    logger.info("Foreground: %d proteins, dim=%d", len(fg), fg.shape[1])

    # ── Load pool ─────────────────────────────────────────────────────────────
    with h5py.File(POOL_H5, "r") as f:
        pool_ids = [k for k in f.keys() if k not in fg_set]
        pool     = np.vstack([f[k][:] for k in pool_ids])

    logger.info("Pool (background candidates): %d proteins", len(pool))

    # ── Load sequences + annotations ─────────────────────────────────────────
    seqs = parse_fasta(FASTA) if FASTA.exists() else {}
    ann  = pd.read_csv(ANN_CSV)

    # Lengths from FASTA for foreground
    fg_lengths = np.array([len(seqs.get(pid, "")) for pid in fg_ids])

    # Lengths for pool — try FASTA first, fallback to 0
    pool_lengths = np.array([len(seqs.get(pid, "")) for pid in pool_ids])

    # If FASTA doesn't cover pool IDs (different format), warn
    n_missing = (pool_lengths == 0).sum()
    if n_missing > 0:
        logger.warning(
            "%d/%d pool proteins have no FASTA sequence — "
            "length-based strategies will have reduced coverage",
            n_missing, len(pool_ids)
        )

    # =========================================================================
    # 1a. Length-matched random (synthetic, average of same-length real proteins)
    # =========================================================================
    logger.info("=== 1a. Length-matched random (synthetic) ===")
    bg_1a = np.vstack([
        synthetic_embedding_for_length(l, pool, pool_lengths)
        for l in fg_lengths
    ])
    run_rhopca(fg, bg_1a, fg_ids, ann,
               "rhoPCA — 1a: length-matched random (synthetic, averaged)",
               "1a_length_matched_random")

    # =========================================================================
    # 1b. Length-matched ±10 aa (real proteins from pool)
    # =========================================================================
    logger.info("=== 1b. Length-matched ±10 aa (real proteins) ===")
    lm_idx, used = [], set()
    for l in fg_lengths:
        mask  = (pool_lengths >= l - LM_WINDOW) & (pool_lengths <= l + LM_WINDOW)
        cands = [i for i in np.where(mask)[0] if i not in used]
        if cands:
            chosen = int(rng.choice(cands))
            lm_idx.append(chosen)
            used.add(chosen)
        else:
            logger.warning("No ±%d match for length=%d", LM_WINDOW, l)

    if lm_idx:
        run_rhopca(fg, pool[lm_idx], fg_ids, ann,
                   f"rhoPCA — 1b: length-matched ±{LM_WINDOW} aa (real proteins)",
                   "1b_length_matched_real")
    else:
        logger.warning("1b skipped — no length matches found (pool may lack FASTA lengths)")

    # =========================================================================
    # 1c. Poly-alanine approximation (synthetic, same length, biologically neutral)
    # =========================================================================
    logger.info("=== 1c. Poly-alanine approximation (synthetic) ===")
    # Approximate poly-A embedding: average embeddings of the 10 shortest
    # proteins in the pool (minimal sequence complexity baseline).
    # For a true poly-A embedding you'd need to run ProtT5 on "AAA...A" sequences.
    sorted_by_len = np.argsort(pool_lengths)
    short_idx = sorted_by_len[:min(50, len(sorted_by_len))]
    short_mean = pool[short_idx].mean(axis=0, keepdims=True)

    # Each background protein is the short-sequence mean + small noise
    # so the background has the right shape but minimal biological signal
    noise  = rng.standard_normal((len(fg), fg.shape[1])).astype(np.float32) * 0.01
    bg_1c  = np.tile(short_mean, (len(fg), 1)) + noise
    run_rhopca(fg, bg_1c, fg_ids, ann,
               "rhoPCA — 1c: poly-alanine approximation (synthetic, low-complexity baseline)",
               "1c_polyala_approx")

    # =========================================================================
    # 2. Signal peptide strategies — fetch annotations from UniProt
    # =========================================================================
    logger.info("=== Fetching signal peptide annotations from UniProt ===")
    sp_annotations = fetch_signal_peptide_annotations(pool_ids)

    with_sp    = [i for i, pid in enumerate(pool_ids) if pid in sp_annotations]
    without_sp = [i for i, pid in enumerate(pool_ids) if pid not in sp_annotations]

    logger.info("Pool: %d with SP, %d without SP",
                len(with_sp), len(without_sp))

    # ── 2a. Background = only proteins WITH signal peptide ───────────────────
    logger.info("=== 2a. Signal peptide background (with SP only) ===")
    if len(with_sp) >= 10:
        run_rhopca(fg, pool[with_sp], fg_ids, ann,
                   f"rhoPCA — 2a: signal peptide background ({len(with_sp)} proteins with SP)",
                   "2a_signal_peptide_only")
    else:
        logger.warning("2a skipped — only %d proteins with SP (need ≥10)", len(with_sp))

    # ── 2b. Mixed: half with SP, half without ────────────────────────────────
    logger.info("=== 2b. Signal peptide mixed (half/half) ===")
    n_each = min(len(with_sp), len(without_sp), len(fg))
    if n_each >= 10:
        sample_with    = list(rng.choice(with_sp,    size=n_each, replace=False))
        sample_without = list(rng.choice(without_sp, size=n_each, replace=False))
        mixed_idx      = sample_with + sample_without
        run_rhopca(fg, pool[mixed_idx], fg_ids, ann,
                   f"rhoPCA — 2b: signal peptide mixed "
                   f"({n_each} with SP + {n_each} without SP)",
                   "2b_signal_peptide_mixed")
    else:
        logger.warning("2b skipped — too few proteins on one side (n_each=%d)", n_each)

    logger.info("=== All done. Plots saved to %s ===", OUT_DIR)


if __name__ == "__main__":
    main()