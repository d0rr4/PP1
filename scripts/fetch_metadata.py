from __future__ import annotations

import argparse
import gzip
import io
import logging
import time
from pathlib import Path

import h5py
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,length,organism_name,keyword,protein_families,cc_family"
RETRY_WAIT = 5
MAX_RETRIES = 3

def read_h5_ids(path: Path) -> list[str]:
    """Return all top-level keys from an H5 file (plain or .gz)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as gz:
            data = gz.read()
        with h5py.File(io.BytesIO(data), "r") as f:
            return list(f.keys())
    else:
        with h5py.File(path, "r") as f:
            return list(f.keys())


def _parse_keywords(raw: str) -> tuple[str, str]:
    """
    UniProt returns keywords as 'Label [KW-XXXX]' entries separated by '; '.
    Returns (labels_str, ids_str) both semicolon-separated.
    """
    if not raw or pd.isna(raw):
        return "", ""
    labels, kw_ids = [], []
    for entry in raw.split("; "):
        entry = entry.strip()
        if "[" in entry and entry.endswith("]"):
            label, kw_id = entry.rsplit(" [", 1)
            labels.append(label.strip())
            kw_ids.append(kw_id.rstrip("]").strip())
        else:
            labels.append(entry)
    return ";".join(labels), ";".join(kw_ids)


def fetch_batch(accessions: list[str], session: requests.Session) -> pd.DataFrame:
    """Fetch metadata for a batch of accessions from UniProt."""
    query = " OR ".join(f"accession:{acc}" for acc in accessions)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(
                UNIPROT_SEARCH,
                params={
                    "query": query,
                    "fields": FIELDS,
                    "format": "tsv",
                    "size": len(accessions) + 10,  # small buffer
                },
                timeout=60,
            )
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), sep="\t")
            return df
        except Exception as e:
            logger.warning("Batch attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
    logger.error("Batch permanently failed for %d accessions", len(accessions))
    return pd.DataFrame()


def fetch_all(
    id_map: dict[str, str],   # accession -> source_file
    batch_size: int = 500,
    requests_per_second: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch metadata for all IDs in id_map in batches.
    id_map maps accession -> source_file label.
    """
    all_ids = list(id_map.keys())
    batches = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]
    min_interval = 1.0 / requests_per_second

    session = requests.Session()
    session.headers.update({"User-Agent": "fetch_metadata.py/1.0 (python-requests)"})

    rows = []
    last_request = 0.0

    for batch in tqdm(batches, desc="Fetching UniProt metadata", unit="batch"):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        df = fetch_batch(batch, session)
        last_request = time.monotonic()

        if df.empty:
            continue

        rows.append(df)

    if not rows:
        raise RuntimeError("No data returned from UniProt — check your accession IDs.")

    result = pd.concat(rows, ignore_index=True)
    return result

def process(raw: pd.DataFrame, id_map: dict[str, str]) -> pd.DataFrame:
    """Clean and normalise the raw UniProt TSV response."""

    raw.columns = [c.strip() for c in raw.columns]

    col_rename = {
        "Entry": "id",
        "Length": "length",
        "Organism": "organism",
        "Keywords": "keywords_raw",
        "Protein families": "protein_families",
        "Family & Domains": "cc_family",  # fallback name
    }
    for col in raw.columns:
        if "family" in col.lower() and "protein" not in col.lower():
            col_rename[col] = "cc_family"
            break

    raw = raw.rename(columns={k: v for k, v in col_rename.items() if k in raw.columns})

    if "keywords_raw" in raw.columns:
        parsed = raw["keywords_raw"].apply(_parse_keywords)
        raw["keywords"] = parsed.apply(lambda x: x[0])
        raw["keyword_ids"] = parsed.apply(lambda x: x[1])
        raw = raw.drop(columns=["keywords_raw"])
    else:
        raw["keywords"] = ""
        raw["keyword_ids"] = ""

    for col in ("protein_families", "cc_family", "organism"):
        if col not in raw.columns:
            raw[col] = ""

    raw["length"] = pd.to_numeric(raw.get("length", pd.Series(dtype=int)), errors="coerce").astype("Int64")

    # Add source_file
    raw["source_file"] = raw["id"].map(id_map).fillna("unknown")

    # Select and order columns
    cols = ["id", "length", "organism", "keywords", "keyword_ids",
            "protein_families", "cc_family", "source_file"]
    cols = [c for c in cols if c in raw.columns]
    return raw[cols].drop_duplicates("id").reset_index(drop=True)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch UniProt metadata for proteins in H5 embedding files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--h5", nargs="+", required=True, type=Path,
        help="One or more .h5 or .h5.gz embedding files.",
    )
    p.add_argument(
        "--output", default=Path("metadata.tsv"), type=Path,
        help="Output TSV path.",
    )
    p.add_argument(
        "--batch-size", default=500, type=int,
        help="Number of accessions per UniProt API request.",
    )
    p.add_argument(
        "--requests-per-second", default=5.0, type=float,
        help="Rate limit for UniProt API calls.",
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

    # Collect IDs from all H5 files, track source
    id_map: dict[str, str] = {}
    for h5_path in args.h5:
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path.resolve()}")
        ids = read_h5_ids(h5_path)
        logger.info("%s: %d proteins", h5_path.name, len(ids))
        for acc in ids:
            if acc not in id_map:   # first file wins if overlap
                id_map[acc] = h5_path.name

    logger.info("Total unique accessions: %d", len(id_map))

    raw = fetch_all(id_map, batch_size=args.batch_size,
                    requests_per_second=args.requests_per_second)

    metadata = process(raw, id_map)

    # Warn about any IDs that came back empty
    fetched = set(metadata["id"])
    missing = set(id_map) - fetched
    if missing:
        logger.warning(
            "%d accessions not returned by UniProt (obsolete / merged): %s%s",
            len(missing),
            ", ".join(sorted(missing)[:10]),
            "..." if len(missing) > 10 else "",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, sep="\t", index=False)
    logger.info("Wrote %d rows → %s", len(metadata), args.output.resolve())
    print(f"Done. Metadata written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()