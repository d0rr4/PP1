"""Run 80% subsampling robustness analyses for UniRef50 ProtT5 embeddings.

The script creates seeded HDF5 subsets in:

    PP_project/embeddings/robustness/

and runs ``protspace prepare --eval`` for each subset in:

    PP_project/protspace_results/robustness/

It intentionally does not use rhoPCA or backgrounds. Each run uses the same
method string and hyperparameters; only the subset seed and reducer run seed
change. Existing completed runs are skipped by default, so this is safe to
resume.

Run from the repository root:

    uv run PP_project/robustness_uniref50.py

If ``uv`` is not available in the active environment, the script automatically
falls back to launching ProtSpace with the current Python interpreter.

For a quick smoke test without launching ProtSpace:

    uv run PP_project/robustness_uniref50.py --runs 2 --create-subsets-only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py

PP_PROJECT = Path(__file__).resolve().parent
ROOT = PP_PROJECT.parent

DEFAULT_INPUT_H5 = PP_PROJECT / "embeddings" / "uniref50_under_2k_subset.h5"
DEFAULT_SUBSET_DIR = PP_PROJECT / "embeddings" / "robustness"
DEFAULT_RESULTS_DIR = PP_PROJECT / "protspace_results" / "robustness_subsets"
DEFAULT_METHODS = "pca2,umap2,tsne2,pacmap2,densmap2,trimap2,phate2"
DEFAULT_LABELS = ["protein_families"]

MANIFEST_COLUMNS = [
    "run_id",
    "subset_seed",
    "run_seed",
    "fraction",
    "n_total",
    "n_selected",
    "subset_h5",
    "subset_entries_tsv",
    "output_dir",
    "summary_path",
    "methods",
    "labels",
    "filter",
    "status",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=0.80)
    parser.add_argument("--subset-seed-start", type=int, default=1000)
    parser.add_argument("--run-seed-start", type=int, default=2000)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help=(
            "Evaluation label passed to protspace prepare --label. Repeatable. "
            "Defaults to protein_families."
        ),
    )
    parser.add_argument(
        "--filter",
        type=int,
        default=50,
        help="Minimum proteins per categorical class for supervised evaluation.",
    )
    parser.add_argument(
        "--launcher",
        choices=["auto", "uv", "current-python", "protspace"],
        default="auto",
        help=(
            "How to launch ProtSpace runs. 'auto' uses uv when available, "
            "otherwise the current Python interpreter."
        ),
    )
    parser.add_argument(
        "--uv-executable",
        default="uv",
        help="Executable used when --launcher is uv or auto finds uv.",
    )
    parser.add_argument(
        "--protspace-executable",
        default="protspace",
        help="Executable used when --launcher is protspace.",
    )
    parser.add_argument(
        "--overwrite-subsets",
        action="store_true",
        help="Rewrite subset HDF5/entry TSV files even if they already exist.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Launch ProtSpace even when a run summary.tsv already exists.",
    )
    parser.add_argument(
        "--create-subsets-only",
        action="store_true",
        help="Create subset HDF5 files and manifest without launching ProtSpace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write manifest without launching ProtSpace.",
    )
    parser.add_argument(
        "--prepare-arg",
        action="append",
        default=[],
        help=(
            "Extra argument appended to every protspace prepare command. "
            "Use repeatedly, e.g. --prepare-arg=-v."
        ),
    )
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def embedding_group(handle: h5py.File) -> h5py.Group | h5py.File:
    """Return the dataset-containing group for flat or ``prot_t5`` HDF5 files."""
    if "prot_t5" in handle and isinstance(handle["prot_t5"], h5py.Group):
        return handle["prot_t5"]
    return handle


def list_embedding_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        group = embedding_group(handle)
        keys = [key for key in group.keys() if isinstance(group[key], h5py.Dataset)]
    if not keys:
        raise ValueError(f"{path} contains no embedding datasets")
    return sorted(keys)


def select_subset_keys(keys: Sequence[str], fraction: float, seed: int) -> list[str]:
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be in the interval (0, 1]")
    n_selected = max(1, math.floor(len(keys) * fraction))
    rng = random.Random(seed)
    selected = set(rng.sample(list(keys), n_selected))
    return [key for key in keys if key in selected]


def write_entries_tsv(path: Path, selected_keys: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Entry"])
        for key in selected_keys:
            writer.writerow([key])


def write_subset_h5(
    *,
    source_h5: Path,
    output_h5: Path,
    selected_keys: Sequence[str],
    overwrite: bool,
) -> None:
    if output_h5.exists() and not overwrite:
        return

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(source_h5, "r") as source, h5py.File(output_h5, mode) as output:
        group = embedding_group(source)

        for attr_name, attr_value in source.attrs.items():
            output.attrs[attr_name] = attr_value
        for attr_name, attr_value in getattr(group, "attrs", {}).items():
            if attr_name not in output.attrs:
                output.attrs[attr_name] = attr_value
        if "model_name" not in output.attrs:
            output.attrs["model_name"] = "prot_t5"

        missing = [key for key in selected_keys if key not in group]
        if missing:
            raise ValueError(
                f"{source_h5} is missing {len(missing)} selected keys "
                f"(first: {missing[:10]})"
            )
        for key in selected_keys:
            group.copy(key, output, name=key)


def find_summary(output_dir: Path) -> Path | None:
    summaries = sorted(output_dir.rglob("summary.tsv"))
    if not summaries:
        return None
    preferred = output_dir / "eval" / "prott5" / "summary.tsv"
    if preferred in summaries:
        return preferred
    return summaries[0]


def resolve_launcher(args: argparse.Namespace) -> str:
    """Resolve the requested launcher, falling back cleanly when uv is absent."""
    if args.launcher != "auto":
        return args.launcher
    if shutil.which(args.uv_executable):
        return "uv"
    return "current-python"


def subprocess_environment() -> dict[str, str]:
    """Expose the local src-layout package to subprocesses."""
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    )
    return env


def build_prepare_command(
    *,
    launcher: str,
    uv_executable: str,
    protspace_executable: str,
    subset_h5: Path,
    output_dir: Path,
    methods: str,
    run_seed: int,
    labels: Sequence[str],
    min_class_size: int,
    extra_args: Sequence[str],
) -> list[str]:
    prepare_args = [
        "prepare",
        "-i",
        f"{subset_h5}:prot_t5",
        "-m",
        methods,
        "-o",
        str(output_dir),
        "--eval",
        "--random-state",
        str(run_seed),
        "--filter",
        str(min_class_size),
        "--no-scores",
    ]
    for label in labels:
        prepare_args.extend(["--label", label])
    prepare_args.extend(extra_args)

    if launcher == "uv":
        return [uv_executable, "run", "protspace", *prepare_args]
    if launcher == "protspace":
        return [protspace_executable, *prepare_args]
    if launcher == "current-python":
        return [
            sys.executable,
            "-c",
            "from protspace.cli.app import app; app()",
            *prepare_args,
        ]
    raise ValueError(f"Unknown launcher: {launcher}")


def write_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.filter < 0:
        raise ValueError("--filter must be non-negative")
    if not args.input_h5.is_file():
        raise FileNotFoundError(args.input_h5)

    labels = args.labels or DEFAULT_LABELS
    launcher = resolve_launcher(args)
    if launcher == "uv" and shutil.which(args.uv_executable) is None:
        raise FileNotFoundError(
            f"Could not find uv executable {args.uv_executable!r}. "
            "Use --launcher current-python, install uv, or leave --launcher auto."
        )
    if launcher == "protspace" and shutil.which(args.protspace_executable) is None:
        raise FileNotFoundError(
            f"Could not find protspace executable {args.protspace_executable!r}. "
            "Use --launcher current-python or leave --launcher auto."
        )
    args.subset_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / "manifest.tsv"

    all_keys = list_embedding_keys(args.input_h5)
    print(f"Input: {rel(args.input_h5)}")
    print(f"Total embeddings: {len(all_keys):,}")
    print(f"Runs: {args.runs:,}; subset fraction: {args.fraction:.2%}")
    print(f"Methods: {args.methods}")
    print(f"Labels: {', '.join(labels)}")
    print(f"Subset output: {rel(args.subset_dir)}")
    print(f"Run output: {rel(args.results_dir)}")
    if args.launcher == "auto" and launcher == "current-python":
        print("Launcher: current-python (uv was not found on PATH)")
    else:
        print(f"Launcher: {launcher}")

    manifest_rows: list[dict[str, object]] = []
    for run_index in range(args.runs):
        run_id = f"run_{run_index:03d}"
        subset_seed = args.subset_seed_start + run_index
        run_seed = args.run_seed_start + run_index
        subset_h5 = args.subset_dir / f"uniref50_under_2k_80pct_{run_id}.h5"
        entries_tsv = args.subset_dir / f"uniref50_under_2k_80pct_{run_id}_entries.tsv"
        output_dir = args.results_dir / run_id

        selected_keys = select_subset_keys(all_keys, args.fraction, subset_seed)
        if args.overwrite_subsets or not subset_h5.exists():
            print(
                f"[{run_id}] Writing subset {rel(subset_h5)} "
                f"({len(selected_keys):,}/{len(all_keys):,}; subset_seed={subset_seed})"
            )
            write_subset_h5(
                source_h5=args.input_h5,
                output_h5=subset_h5,
                selected_keys=selected_keys,
                overwrite=args.overwrite_subsets,
            )
            write_entries_tsv(entries_tsv, selected_keys)
        else:
            print(f"[{run_id}] Reusing subset {rel(subset_h5)}")
            if not entries_tsv.exists():
                write_entries_tsv(entries_tsv, selected_keys)

        command = build_prepare_command(
            launcher=launcher,
            uv_executable=args.uv_executable,
            protspace_executable=args.protspace_executable,
            subset_h5=subset_h5,
            output_dir=output_dir,
            methods=args.methods,
            run_seed=run_seed,
            labels=labels,
            min_class_size=args.filter,
            extra_args=args.prepare_arg,
        )
        summary_path = find_summary(output_dir)
        status = "subset_created"

        if args.create_subsets_only:
            status = "subset_only"
        elif args.dry_run:
            status = "dry_run"
            print(f"[{run_id}] DRY RUN: {shlex.join(command)}")
        elif summary_path is not None and not args.rerun_existing:
            status = "skipped_existing_summary"
            print(f"[{run_id}] Skipping existing summary {rel(summary_path)}")
        else:
            print(
                f"[{run_id}] Running ProtSpace "
                f"(run_seed={run_seed}): {shlex.join(command)}"
            )
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                env=subprocess_environment(),
            )
            if completed.returncode != 0:
                status = f"failed_returncode_{completed.returncode}"
                write_manifest(manifest_path, manifest_rows)
                raise subprocess.CalledProcessError(completed.returncode, command)
            summary_path = find_summary(output_dir)
            status = "completed" if summary_path is not None else "completed_no_summary"

        manifest_rows.append(
            {
                "run_id": run_id,
                "subset_seed": subset_seed,
                "run_seed": run_seed,
                "fraction": args.fraction,
                "n_total": len(all_keys),
                "n_selected": len(selected_keys),
                "subset_h5": rel(subset_h5),
                "subset_entries_tsv": rel(entries_tsv),
                "output_dir": rel(output_dir),
                "summary_path": rel(summary_path) if summary_path else "",
                "methods": args.methods,
                "labels": ",".join(labels),
                "filter": args.filter,
                "status": status,
                "command": shlex.join(command),
            }
        )
        write_manifest(manifest_path, manifest_rows)

    print(f"\nDone. Manifest: {rel(manifest_path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise
