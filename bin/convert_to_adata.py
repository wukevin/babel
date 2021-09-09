"""
Script to convert files to adata objects for feeding into BABEL

Formatting for all files is autodetected
"""

import os, sys
import argparse
import logging
from typing import *

import pandas as pd
import anndata as ad

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import utils

logging.basicConfig(level=logging.INFO)


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


def auto_read_metadata(fname: str, index_col: Optional[int] = None) -> pd.DataFrame:
    """Read the given metadata file"""
    assert os.path.isfile(fname)
    ext = utils.get_file_extension_no_gz(fname)
    if ext == "csv":
        df = pd.read_csv(fname, sep=",")
    elif ext == "tsv" or ext == "txt":
        df = pd.read_csv(fname, sep="\t")
    else:
        raise ValueError(f"Cannot recognize file extension for {fname}")
    if index_col:
        df.set_index(df.columns[index_col], inplace=True)
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
    parser.add_argument("out_h5ad", type=str, help="Output file (should end in .h5ad)")
    parser.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        help="Apply transpose to matrix values to match expected (cell x feature) format",
    )
    parser.add_argument(
        "--obsinfo", type=str, help="Optional file for cell (observation) annotations"
    )
    parser.add_argument(
        "--obscol",
        type=int,
        help="Column of obs table to use as obs names",
    )
    parser.add_argument(
        "--varinfo",
        type=str,
        help="Optional file for feature (gene or peaks) annotations",
    )
    parser.add_argument(
        "--varcol",
        type=int,
        help="Column of var table to use as var names",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    # Read in the counts file
    counts_df = auto_read_matrix_values(args.matfile)
    if args.transpose:
        counts_df = counts_df.T
    logging.info(f"Read input matrix of (cell x feature) {counts_df.shape}")

    # Read in metadata if given
    obs_df = None
    if args.obsinfo:
        obs_df = auto_read_metadata(args.obsinfo, index_col=args.obscol)
    var_df = None
    if args.varinfo:
        var_df = auto_read_metadata(args.varinfo, index_col=args.varcol)

    adata = ad.AnnData(counts_df, obs=obs_df, var=var_df)
    logging.info(f"Created anndata object: {adata}")

    logging.info(f"Writing adata object to {args.out_h5ad}")
    adata.write(args.out_h5ad)


if __name__ == "__main__":
    main()
