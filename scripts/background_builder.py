from __future__ import annotations
 
import argparse
import logging
from pathlib import Path
from typing import Literal
 
import h5py
import numpy as np
import pandas as pd
 
logger = logging.getLogger(__name__)
 
BackgroundStrategy = Literal["complement", "length_matched", "mixed"]
 
 
# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
 
def build_background(
    pool_h5: Path | str,
    foreground_ids: list[str],
    metadata: pd.DataFrame,
    strategy: BackgroundStrategy = "complement",
    output_path: Path | str = Path("background.h5"),
    length_window: int = 10,
    complement_multiplier: int = 2,
    seed: int = 42,
) -> Path:
    """Build and write a background .h5 file for rhoPCA.
 
    Parameters
    ----------
    pool_h5 : Path | str
        Path to the .h5 file containing ALL protein embeddings (foreground +
        everything you want to draw background proteins from).  Each dataset
        key must be a protein ID matching the ``id`` column of *metadata*.
    foreground_ids : list[str]
        Protein IDs that form the *target* (foreground) set.  These are
        excluded from the background regardless of strategy.
    metadata : pd.DataFrame
        Must contain at least the columns ``id`` (str) and ``length`` (int).
        Optional columns like ``family`` or ``taxonomy`` can be used for
        future extensions.
    strategy : {"complement", "length_matched", "mixed"}
        Background-selection strategy (see module docstring).
    output_path : Path | str
        Where to write the resulting background .h5 file.
    length_window : int
        Half-width of the length-matching window (default ±10 residues).
    complement_multiplier : int
        For the "mixed" strategy: how many times the foreground size to sample
        from the complement pool (default 2×).
    seed : int
        Random seed for reproducibility.
 
    Returns
    -------
    Path
        Absolute path to the written background .h5 file.
    """
    pool_h5 = Path(pool_h5)
    output_path = Path(output_path)
    rng = np.random.default_rng(seed)
 
    if not pool_h5.exists():
        raise FileNotFoundError(f"Pool H5 not found: {pool_h5.resolve()}")
 
    required_cols = {"id", "length"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
 
    # ------------------------------------------------------------------ #
    # Load pool IDs and build background candidate pool
    # ------------------------------------------------------------------ #
    logger.info("Reading pool H5: %s", pool_h5)
    with h5py.File(pool_h5, "r") as f:
        all_pool_ids: list[str] = list(f.keys())
 
    fg_set = set(foreground_ids)
    bg_pool_ids = [pid for pid in all_pool_ids if pid not in fg_set]
 
    if not bg_pool_ids:
        raise ValueError(
            "Background pool is empty after excluding foreground IDs. "
            "Make sure pool_h5 contains proteins beyond the foreground set."
        )
 
    bg_pool_meta = (
        metadata[metadata["id"].isin(bg_pool_ids)]
        .drop_duplicates("id")
        .copy()
        .reset_index(drop=True)
    )
 
    logger.info(
        "Pool size: %d | Foreground: %d | Background candidates: %d",
        len(all_pool_ids),
        len(fg_set),
        len(bg_pool_meta),
    )
 
    # ------------------------------------------------------------------ #
    # Strategy dispatch
    # ------------------------------------------------------------------ #
    if strategy == "complement":
        selected_ids = _strategy_complement(bg_pool_ids)
 
    elif strategy == "length_matched":
        fg_meta = (
            metadata[metadata["id"].isin(fg_set)]
            .drop_duplicates("id")
            .copy()
        )
        selected_ids = _strategy_length_matched(
            fg_meta, bg_pool_meta, length_window, rng
        )
 
    elif strategy == "mixed":
        fg_meta = (
            metadata[metadata["id"].isin(fg_set)]
            .drop_duplicates("id")
            .copy()
        )
        selected_ids = _strategy_mixed(
            fg_meta,
            bg_pool_meta,
            bg_pool_ids,
            length_window,
            complement_multiplier,
            rng,
        )
 
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            "Choose from: complement, length_matched, mixed"
        )
 
    if not selected_ids:
        raise RuntimeError(
            f"Strategy '{strategy}' produced an empty background. "
            "Check your metadata coverage and length_window."
        )
 
    logger.info("Background size (%s): %d proteins", strategy, len(selected_ids))
 
    # ------------------------------------------------------------------ #
    # Write background H5
    # ------------------------------------------------------------------ #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected_ids)
 
    written = 0
    with h5py.File(pool_h5, "r") as f_in, h5py.File(output_path, "w") as f_out:
        for pid in selected_set:
            if pid in f_in:
                f_out.create_dataset(pid, data=f_in[pid][:].astype(np.float32))
                written += 1
            else:
                logger.warning("ID %r selected but not found in pool H5 — skipping", pid)
 
    logger.info("Wrote %d embeddings → %s", written, output_path.resolve())
    return output_path.resolve()
 
 
# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------
 
def _strategy_complement(bg_pool_ids: list[str]) -> list[str]:
    """Return all non-foreground proteins as background."""
    return list(bg_pool_ids)
 
 
def _strategy_length_matched(
    fg_meta: pd.DataFrame,
    bg_pool_meta: pd.DataFrame,
    length_window: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    For each foreground protein, sample one background protein whose length
    falls within [fg_length - window, fg_length + window].
 
    Already-selected IDs are excluded from future draws to avoid duplicates.
    Foreground proteins with no length-compatible candidate are skipped with
    a warning.
    """
    selected_ids: list[str] = []
    used: set[str] = set()
 
    for _, row in fg_meta.iterrows():
        lo = row["length"] - length_window
        hi = row["length"] + length_window
        candidates = bg_pool_meta[
            bg_pool_meta["length"].between(lo, hi) &
            ~bg_pool_meta["id"].isin(used)
        ]
        if candidates.empty:
            logger.warning(
                "No length-matched candidate for %s (length=%d, window=±%d)",
                row["id"], row["length"], length_window,
            )
            continue
        chosen_id = candidates.sample(1, random_state=int(rng.integers(1_000_000)))["id"].values[0]
        selected_ids.append(chosen_id)
        used.add(chosen_id)
 
    return selected_ids
 
 
def _strategy_mixed(
    fg_meta: pd.DataFrame,
    bg_pool_meta: pd.DataFrame,
    bg_pool_ids: list[str],
    length_window: int,
    complement_multiplier: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    Union of:
      - A random subsample of the complement (complement_multiplier × |fg|)
      - All length-matched proteins
 
    De-duplicated so each protein appears at most once.
    """
    n_fg = len(fg_meta)
 
    # Subsample complement
    n_sample = min(complement_multiplier * n_fg, len(bg_pool_ids))
    complement_sample = list(
        rng.choice(bg_pool_ids, size=n_sample, replace=False)
    )
 
    # Length-matched subset
    length_matched = _strategy_length_matched(
        fg_meta, bg_pool_meta, length_window, rng
    )
 
    combined = list(dict.fromkeys(complement_sample + length_matched))  # ordered dedup
    return combined
 
 
# ---------------------------------------------------------------------------
# Convenience: load foreground IDs from a plain-text file (one ID per line)
# ---------------------------------------------------------------------------
 
def load_foreground_ids(path: Path | str) -> list[str]:
    """Read a text file with one protein ID per line."""
    path = Path(path)
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    return ids
 
 
def load_metadata(path: Path | str, sep: str = "\t") -> pd.DataFrame:
    """
    Load a TSV/CSV metadata file.
 
    Expected columns (at minimum):
        id      – protein identifier matching H5 keys
        length  – sequence length (integer)
 
    Optional useful columns:
        family, taxonomy, organism
    """
    df = pd.read_csv(path, sep=sep)
    if "id" not in df.columns:
        raise ValueError("metadata file must have an 'id' column")
    if "length" not in df.columns:
        raise ValueError("metadata file must have a 'length' column")
    df["length"] = df["length"].astype(int)
    return df
 
 
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
 
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a background .h5 file for rhoPCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pool", required=True, type=Path,
        help="Path to .h5 file with ALL protein embeddings.",
    )
    p.add_argument(
        "--fg-ids", required=True, type=Path,
        help="Text file with one foreground protein ID per line.",
    )
    p.add_argument(
        "--metadata", required=True, type=Path,
        help="TSV file with columns: id, length (and optionally family, taxonomy).",
    )
    p.add_argument(
        "--strategy", default="complement",
        choices=["complement", "length_matched", "mixed"],
        help="Background-selection strategy.",
    )
    p.add_argument(
        "--output", default=Path("background.h5"), type=Path,
        help="Output path for the background .h5 file.",
    )
    p.add_argument(
        "--length-window", default=10, type=int,
        help="Half-width (±residues) for length matching.",
    )
    p.add_argument(
        "--complement-multiplier", default=2, type=int,
        help="For 'mixed': sample this many × |foreground| from complement.",
    )
    p.add_argument(
        "--seed", default=42, type=int,
        help="Random seed.",
    )
    p.add_argument(
        "--metadata-sep", default="\t",
        help="Column separator for the metadata file (default: tab).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()
 
 
def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
 
    fg_ids = load_foreground_ids(args.fg_ids)
    metadata = load_metadata(args.metadata, sep=args.metadata_sep)
 
    out = build_background(
        pool_h5=args.pool,
        foreground_ids=fg_ids,
        metadata=metadata,
        strategy=args.strategy,
        output_path=args.output,
        length_window=args.length_window,
        complement_multiplier=args.complement_multiplier,
        seed=args.seed,
    )
    print(f"Done. Background written to: {out}")
 
 
if __name__ == "__main__":
    main()
