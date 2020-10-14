"""
Script for running the archr gene activity calculation in batch
We group the files by their folder name
"""

import os
import sys
import glob
import logging
import argparse
import collections
import subprocess
import shlex

logging.basicConfig(level=logging.INFO)

ARCHR_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "archr_gene_activity.R"
)
assert os.path.isfile(
    ARCHR_SCRIPT_PATH
), f"Cannot find archr gene activity script {ARCHR_SCRIPT_PATH}"


def get_file_samplename(fname: str, strip_rep: bool = False) -> str:
    """Return the sample name associated with file"""
    with_rep = os.path.basename(os.path.dirname(os.path.abspath(fname)))
    without_rep = with_rep.strip("_rep1").strip("_rep2")
    return without_rep if strip_rep else with_rep


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", "-f", type=str, nargs="*", help="Fragment files")
    parser.add_argument(
        "--genome",
        "-g",
        type=str,
        default="hg38",
        choices=["hg38", "hg19"],
        help="Genome to use",
    )
    parser.add_argument("--dry", action="store_true", help="Dry run")
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    files_by_sample = collections.defaultdict(list)
    for fname in args.files:
        samplename = get_file_samplename(fname, strip_rep=True)
        files_by_sample[samplename].append(os.path.abspath(fname))

    for samplename, files in files_by_sample.items():
        samples_with_rep = [get_file_samplename(f, strip_rep=False) for f in files]
        cmd = f"Rscript {ARCHR_SCRIPT_PATH} -f {','.join(files)} -n {','.join(samples_with_rep)} -o {samplename} -g {args.genome}"
        logging.info(f"Command to run: {cmd}")
        if not args.dry:
            tokens = shlex.split(cmd)
            retval = subprocess.run(tokens, check=True)


if __name__ == "__main__":
    main()
