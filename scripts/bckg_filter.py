"""
create_filtered_background.py
==============================
Creates a background embedding set by filtering a large embedding pool
(per_protein.h5) to exclude any accession that is already present in
another embedding file (Prot_T5.h5) — e.g. your foreground/target set.
This prevents background <-> foreground overlap (data leakage) when the
background is later used in rhoPCA runs (see run_all_backgrounds.py).

If the roles are reversed for your files (i.e. Prot_T5.h5 is the pool and
per_protein.h5 is the exclude set), just swap ALL_POOL_H5 and EXCLUDE_H5
below — nothing else needs to change.

Usage:
    cd ~/PP1
    python3 scripts/create_filtered_background.py

Output:
    A new .h5 background file with the same key->embedding structure as
    your other background files, ready to be used as a new BG_H5 path in
    run_all_backgrounds.py.
"""

from __future__ import annotations
import logging
from pathlib import Path

import h5py

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust paths as needed
# ─────────────────────────────────────────────────────────────────────────────
ALL_POOL_H5 = Path("../data/per-protein.h5")     # large pool to draw background from
EXCLUDE_H5  = Path("../data/Prot-T5.h5")         # accessions to remove (e.g. foreground)
OUT_H5      = Path("../data/backgrounds/background_filtered_prot_t5.h5")
# ─────────────────────────────────────────────────────────────────────────────


def load_keys(path: Path) -> set[str]:
    with h5py.File(path, "r") as f:
        return set(f.keys())


def main() -> None:
    OUT_H5.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading exclusion accessions from %s ...", EXCLUDE_H5)
    exclude_ids = load_keys(EXCLUDE_H5)
    logger.info("Exclude set: %d accessions", len(exclude_ids))

    logger.info("Opening pool %s ...", ALL_POOL_H5)
    with h5py.File(ALL_POOL_H5, "r") as f_in, h5py.File(OUT_H5, "w") as f_out:
        pool_ids = list(f_in.keys())
        logger.info("Pool: %d total accessions", len(pool_ids))

        kept, skipped = 0, 0
        for pid in pool_ids:
            if pid in exclude_ids:
                skipped += 1
                continue
            f_out.create_dataset(pid, data=f_in[pid][:])
            kept += 1
            if kept % 5000 == 0:
                logger.info("  ... %d kept so far", kept)

    logger.info(
        "Done. Kept %d / %d accessions (skipped %d overlapping with exclude set)",
        kept, len(pool_ids), skipped,
    )
    logger.info("Background written → %s", OUT_H5)


if __name__ == "__main__":
    main()