"""
Write TS guesses from a samples pkl into per-reaction XYZ files.

Reads a pickle containing data objects (each with rxn_index, mol_name,
atom_type, pos_gen) and writes one XYZ file per sample into the local
results directory expected by the validator.

Optionally the pkl is fetched from a source cluster first, and/or the
resulting files are pushed to a destination cluster afterwards.
"""

import sys
import pickle
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

from motsart.common import PathHandler, get_root_dir
from motsart.complex_finder.utils import write_xyz_from_tensor


# ===================== Default Configuration =====================
DEFAULT_FETCH_PKL_PATH = 'data/samples_all.pkl'      # relative to project root (local mode)
DEFAULT_LOCAL_RESULTS_FOLDER = 'results_fetched'

# Known cluster hostnames — must match SSH config / rsync aliases
KNOWN_CLUSTERS = ['cluster1', 'cluster2', 'cluster3']


# ===================== Helper Functions =====================

def fetch_pkl_from_cluster(fetch_cluster: str, fetch_remote_path: str, local_fetch_dir: Path) -> Path:
    """Rsync a single pkl file from a source cluster to a local directory.

    Args:
        fetch_cluster:     Rsync hostname (one of KNOWN_CLUSTERS).
        fetch_remote_path: Full absolute path to the file on the cluster.
        local_fetch_dir:   Local directory to save the fetched file into.

    Returns:
        Path to the fetched local copy.
    """
    remote_src = f"{fetch_cluster}:{fetch_remote_path}"
    local_fetch_dst = local_fetch_dir / Path(fetch_remote_path).name
    local_fetch_dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching  {remote_src}")
    print(f"       -> {local_fetch_dst}")
    subprocess.run(
        ['rsync', '-avh', '--progress', remote_src, str(local_fetch_dst)],
        check=True,
    )
    return local_fetch_dst


def push_results_to_cluster(local_results_dir: Path, dest_cluster: str, dest_dir: str):
    """Rsync the local results directory to a destination cluster.

    Args:
        local_results_dir: Local results directory to push.
        dest_cluster:      Rsync hostname (one of KNOWN_CLUSTERS).
        dest_dir:          Full absolute destination path on the cluster.
    """
    remote_dst = f"{dest_cluster}:{dest_dir}"
    print(f"Pushing   {local_results_dir}/")
    print(f"       -> {remote_dst}")
    subprocess.run(
        ['rsync', '-avh', '--progress', str(local_results_dir) + '/', remote_dst],
        check=True,
    )


def resolve_fetch_pkl_path(args) -> Path:
    """Return the local path to the pkl, fetching from a cluster if needed."""
    if args.fetch_cluster:
        return fetch_pkl_from_cluster(
            args.fetch_cluster, args.fetch_pkl_path, Path(args.local_fetch_dir)
        )

    local_pkl_path = Path(args.fetch_pkl_path)
    if not local_pkl_path.is_absolute():
        local_pkl_path = get_root_dir() / local_pkl_path
    return local_pkl_path


def write_samples_to_xyz(data_L, local_results_folder: str):
    """Iterate over data objects and write each TS guess as an XYZ file."""
    for data in tqdm(data_L, desc="Writing XYZ files"):
        path_handler = PathHandler(
            rxn_id=data.rxn_index,
            r_or_p='r',
            ts_method='learning',
            results_folder=local_results_folder,
        )
        path_handler.create_dirs()
        local_al_path = path_handler.ts_to_validate / data.mol_name
        write_xyz_from_tensor(data.atom_type, data.pos_gen, local_al_path)
        print(f"  Saved rxn_id {data.rxn_index} -> {local_al_path}")


# ===================== Main =====================

def main(args):
    # --- Validate required co-dependencies ---
    if args.fetch_cluster and not args.local_fetch_dir:
        print("Error: --local_fetch_dir is required when --fetch_cluster is set")
        sys.exit(1)
    if args.dest_cluster and not args.dest_dir:
        print("Error: --dest_dir is required when --dest_cluster is set")
        sys.exit(1)

    print("Configuration:")
    print(f"  fetch_cluster:        {args.fetch_cluster or 'local'}")
    print(f"  fetch_pkl_path:       {args.fetch_pkl_path}")
    print(f"  local_fetch_dir:      {args.local_fetch_dir or 'N/A'}")
    print(f"  local_results_folder: {args.local_results_folder}")
    print(f"  dest_cluster:         {args.dest_cluster or 'N/A'}")
    print(f"  dest_dir:             {args.dest_dir or 'N/A'}")
    print()

    local_pkl_path = resolve_fetch_pkl_path(args)

    print(f"Loading {local_pkl_path} ...")
    with open(local_pkl_path, 'rb') as f:
        data_L = pickle.load(f)
    print(f"Loaded {len(data_L)} samples.\n")

    local_results_dir = PathHandler(results_folder=args.local_results_folder).results_dir
    print(f"Output directory: {local_results_dir}")
    if args.dest_cluster:
        print(f"Will push to:     {args.dest_cluster}:{args.dest_dir}")
    response = input("Press Enter to proceed, or type anything to abort: ")
    if response != "":
        print("Aborted.")
        return

    write_samples_to_xyz(data_L, args.local_results_folder)

    if args.dest_cluster:
        print()
        push_results_to_cluster(local_results_dir, args.dest_cluster, args.dest_dir)

    print(f"\nDone. {len(data_L)} samples written.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write TS guesses from a data pkl into per-reaction XYZ files."
    )

    # --- Fetch (source) ---
    parser.add_argument(
        "--fetch_cluster", type=str, default=None, choices=KNOWN_CLUSTERS,
        help="Cluster to fetch the pkl from. Omit to use a local pkl.",
    )
    parser.add_argument(
        "--fetch_pkl_path", type=str, default=DEFAULT_FETCH_PKL_PATH,
        help=(
            f"Path to the samples pkl. Local: relative paths are resolved from "
            f"the project root (default: {DEFAULT_FETCH_PKL_PATH}). With "
            f"--fetch_cluster: must be the full absolute path on that machine."
        ),
    )
    parser.add_argument(
        "--local_fetch_dir", type=str, default=None,
        help="Local directory to save the fetched pkl into. Required when --fetch_cluster is set.",
    )

    # --- Local (output) ---
    parser.add_argument(
        "--local_results_folder", type=str, default=DEFAULT_LOCAL_RESULTS_FOLDER,
        help=f"Local results folder name (default: {DEFAULT_LOCAL_RESULTS_FOLDER})",
    )

    # --- Destination (push) ---
    parser.add_argument(
        "--dest_cluster", type=str, default=None, choices=KNOWN_CLUSTERS,
        help="Cluster to push the results to after writing locally. Omit to skip.",
    )
    parser.add_argument(
        "--dest_dir", type=str, default=None,
        help="Full absolute destination path on the dest cluster. Required when --dest_cluster is set.",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
