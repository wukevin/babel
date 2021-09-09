"""
Script to convert files to adata objects for feeding into BABEL

Formatting for all files is autodetected
"""

import os, sys
import argparse
from typing import *

import pandas as pd

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import utils


def auto_read_matrix_values(fname: str) -> pd.DataFrame:
    """Read the given counts file"""
    assert os.path.isfile(fname)
    ext = utils.get_file_extension_no_gz(fname)

    if ext == "csv":
        df = pd.read_csv(fname, sep=",")
    elif ext == "txt" or ext == "tsv":
        df = pd.read_csv(fname, sep="\t")
    else:
        raise ValueError(f"Cannot recognize file extension for {fname}")
    return df


def auto_read_metadata(fname: str) -> pd.DataFrame:
    """Read the given metadata file"""
    assert os.path.isfile(fname)
    ext = utils.get_file_extension_no_gz(fname)
    if ext == "csv":
        df = pd.read_csv(fname, sep=",")
    elif ext == "tsv" or ext == "txt":
        df = pd.read_csv(fname, sep="\t")
    else:
        raise ValueError(f"Cannot recognize file extension for {fname}")
    return df


def build_parser():
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "matfile",
        type=str,
        help="File containing matrix of vlaues. Expects (cell x feature), use transpose otherwise.",
    )
    parser.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        help="Apply transpose to matrix values",
    )
    parser.add_argument(
        "--obsinfo", type=str, help="Optional file for cell annotations"
    )
    parser.add_argument(
        "--varinfo", type=str, help="Optional file for feature annotations"
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    # Read in the counts file
    counts_df = auto_read_matrix_values(args.matfile)
    print(counts_df)


if __name__ == "__main__":
    main()

