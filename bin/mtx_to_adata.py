"""
Script for combining mtx files and its two corresponding metadata files into a anndata object.
Does not do any normalization, but can handle some basic things like reindexing
"""

import os
import sys
from typing import *
import argparse
import logging

import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import adata_utils
import utils

logging.basicConfig(level=logging.INFO)


def ensure_sane_interval(s: str, f: Callable = lambda x: x.split("_")) -> str:
    """
    Ensure that the atac interval follows format chr1:100-200
    f is a function that maps input to a tuple of 3 parts
    """
    chrom, start, stop = f(s)
    start = int(start)
    stop = int(stop)
    assert chrom.startswith("chr")
    return f"{chrom}:{start}-{stop}"


def build_parser():
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("cell_info", type=str, help="File with cell metadata")
    parser.add_argument(
        "var_info", type=str, help="File with var (peak or gene) metadata"
    )
    parser.add_argument("mat_file", type=str, help="File with actual matrix of values")
    parser.add_argument("out_h5ad", type=str, help="File to write output")
    parser.add_argument(
        "--cellindexcol",
        type=int,
        default=None,
        help="Index column for cells. For ArchR output, this should be 0",
    )
    parser.add_argument(
        "--varindexcol",
        type=int,
        default=None,
        help="Index col for var. For ArchR output, this should be 5",
    )
    parser.add_argument(
        "--reindexvar",
        type=str,
        default="",
        help="File containing a list of variables that the outputted h5ad must adhere to",
    )
    parser.add_argument(
        "--noheader",
        action="store_true",
        help="Do not consider the first line a header",
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    cell_df = pd.read_csv(
        args.cell_info,
        delimiter=","
        if utils.get_file_extension_no_gz(args.cell_info) == "csv"
        else "\t",
        index_col=args.cellindexcol,
        header=None if args.noheader else "infer",  # 'infer' is default
    )
    if "Barcodes" in cell_df.columns and args.cellindexcol is not None:
        cell_df.index = cell_df["Barcodes"]
    cell_df.index = cell_df.index.rename("barcode")
    cell_df.columns = cell_df.columns.map(str)

    logging.info(f"Read cell metadata from {args.cell_info} {cell_df.shape}")
    logging.info(f"Cell metadata cols: {cell_df.columns}")
    logging.info(cell_df)

    var_df = pd.read_csv(
        args.var_info,
        delimiter=","
        if utils.get_file_extension_no_gz(args.var_info) == "csv"
        else "\t",
        index_col=args.varindexcol,
        header=None if args.noheader else "infer",  # 'infer' is default
    )
    if "Feature" in var_df.columns and args.varindexcol is not None:
        var_df.index = [ensure_sane_interval(s) for s in var_df["Feature"]]
    var_df.index = var_df.index.rename("ft")
    var_df.columns = var_df.columns.map(str)
    # var_df.index = var_df.index.map(str)
    logging.info(f"Read variable metadata from {args.var_info} {var_df.shape}")
    logging.info(f"Var metadata cols: {var_df.columns}")
    logging.info(var_df)

    # Transpose because bio considers rows to be features
    adata = ad.read_mtx(args.mat_file).T
    logging.info(f"Read matrix {args.mat_file} {adata.shape}")
    adata.obs = cell_df
    adata.var = var_df
    logging.info(f"Created AnnData object: {adata}")
    logging.info(f"Obs names: {adata.obs_names}")
    logging.info(f"Var names: {adata.var_names}")

    if args.reindexvar:
        assert args.varindexcol is not None, "Must provide var index col to reindex var"
        target_vars = utils.read_delimited_file(args.reindexvar)
        logging.info(f"Read {args.reindexvar} for {len(target_vars)} vars to reindex")
        adata = adata_utils.reindex_adata_vars(adata, target_vars)

    adata.X = csr_matrix(adata.X)
    logging.info(f"Writing to {args.out_h5ad}")
    adata.write_h5ad(args.out_h5ad, compression=None)


if __name__ == "__main__":
    main()
