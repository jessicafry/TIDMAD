"""
Created on Mon March 27 2026
@author: TIDMAD Team

TIDMAD download script allows you to download TIDMAD data. Use -h flag for usage.
"""

import argparse
import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download


# -----------------------------
# Utility: argparse range check
# -----------------------------
def check_range(min_val, max_val):
    class Range:
        def __call__(self, value):
            value_int = int(value)
            if value_int < min_val or value_int > max_val:
                raise argparse.ArgumentTypeError(
                    f"Value must be between {min_val} and {max_val}"
                )
            return value_int
    return Range()


# -----------------------------
# Hugging Face download
# -----------------------------
def download_hf(hf_filename, dest_file):
    try:
        cached_file = hf_hub_download(
            repo_id="jessicafry/TIDMAD",
            filename=hf_filename,
            repo_type="dataset"
        )
        shutil.copy(cached_file, dest_file)
        return True
    except Exception as e:
        print(f"[HF ERROR] {hf_filename}: {e}")
        return False


# -----------------------------
# SDSC download
# -----------------------------
def download_sdsc(folder, fname, dest_file):
    """
    Download using Pelican object store.
    """

    # Map folders to Pelican paths
    BASE_PATHS = {
        "training": "pelican://osg-htc.org/ucsd/physics/ABRACADABRA/ABRA_aires_training_data",
        "validation": "pelican://osg-htc.org/ucsd/physics/ABRACADABRA/ABRA_aires_validation_data",
        "science": "pelican://osg-htc.org/ucsd/physics/ABRACADABRA/ABRA_aires_science_data"
    }

    if folder not in BASE_PATHS:
        raise ValueError(f"Unknown folder: {folder}")

    source = f"{BASE_PATHS[folder]}/{fname}"

    try:
        subprocess.run(
            ["pelican", "object", "get", source, dest_file],
            check=True
        )
        return True
    except Exception as e:
        print(f"[SDSC PELICAN ERROR] {source}: {e}")
        return False

# -----------------------------
# Core download logic
# -----------------------------
def download_file(folder, fname, args):
    dest_file = os.path.join(args.output_dir, fname)

    if args.skip_downloaded and os.path.exists(dest_file):
        print(f"Skipping {fname} (already exists)")
        return

    hf_filename = f"data/{folder}/{fname}"

    if args.print:
        BASE_PATHS = {
            "training": "pelican://osg-htc.org/ucsd/physics/ABRACADABRA/ABRA_aires_training_data",
            "validation": "pelican://osg-htc.org/ucsd/physics/ABRACADABRA/ABRA_aires_validation_data",
            "science": "pelican://osg-htc.org/ucsd/physics/ABRACADABRA/ABRA_aires_science_data"
        }
        if args.source in ("auto", "hf"):
            hf_url = f"https://huggingface.co/datasets/jessicafry/TIDMAD/resolve/main/{hf_filename}"
            print(f"wget -O {dest_file} {hf_url}")
        if args.source in ("sdsc"):
            pelican_path = f"{BASE_PATHS[folder]}/{fname}"
            print(f"pelican object get {pelican_path} {dest_file}")
        return

    success = False

    # --- AUTO MODE (HF → SDSC fallback)
    if args.source == "auto":
        print(f"Trying Hugging Face: {fname}")
        success = download_hf(hf_filename, dest_file)

        if not success:
            print(f"Falling back to SDSC: {fname}")
            success = download_sdsc(folder, fname, dest_file)

    # --- HF ONLY
    elif args.source == "hf":
        success = download_hf(hf_filename, dest_file)

    # --- SDSC ONLY
    elif args.source == "sdsc":
        success = download_sdsc(folder, fname, dest_file)

    if success:
        print(f"✅ Downloaded {fname}")
    else:
        print(f"❌ Failed to download {fname}")


def download_type(folder, prefix, num_files, args):
    start_i = 20 if (args.weak and folder in ["training", "validation"]) else 0

    for i in range(start_i, start_i + num_files):
        fname = f"{prefix}{i:04d}.h5"
        download_file(folder, fname, args)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download the TIDMAD dataset with HF + SDSC fallback"
    )

    parser.add_argument('--output_dir', '-o', type=str, default=os.getcwd(), help='Directory where the files will be saved (default: current directory).')
    parser.add_argument('--train_files', '-t', type=check_range(0, 20), default=20, help='Number of training files to download, must be an  integer between 0 and 20, default 20.')
    parser.add_argument('--validation_files', '-v', type=check_range(0, 20), default=20, help='Number of validation files to download, must be an  integer between 0 and 20, default 20.')
    parser.add_argument('--science_files', '-s', type=check_range(0, 208), default=208, help='Number of science files to download, must be an  integer between 0 and 208, default 208.')

    parser.add_argument('--source', choices=["auto", "hf", "sdsc"], default="auto",
                        help="Download source: hf, sdsc, or auto (default: hf → sdsc fallback)")

    parser.add_argument('-f', '--force', action='store_true', help='Directly proceed to download without showing the file size and confirmation.')
    parser.add_argument('-sk', '--skip_downloaded', action='store_true', help='Skip the file that has already been downloaded at --output_dir')
    parser.add_argument('-w', '--weak', action='store_true', help='Download the "weak signal" version of training and validation file. In this version, the injected signal is 1/5 of the normal version.')
    parser.add_argument('-p', '--print', action='store_true', help='Print out all downloading scripts instead of actually downloading the file.')

    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")
    print(f"Download source: {args.source}")
    print(f"Estimated total download size: "
          f"{args.train_files*2.06 + args.validation_files*2.06 + args.science_files*2.8:.2f} GB")
    if not args.force and not args.print:
        response = input("Proceed with download? [y/n] ").strip().lower()
        if response not in ["y", "yes"]:
            return

    if not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory does not exist: {args.output_dir}")

    download_type("training", "abra_training_", args.train_files, args)
    download_type("validation", "abra_validation_", args.validation_files, args)
    download_type("science", "abra_science_", args.science_files, args)


if __name__ == "__main__":
    main()
