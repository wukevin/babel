"""
Code for merging two h5 objects
"""

import os
import sys
import argparse
import logging

import anndata as ad
import scanpy as sc

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import sc_data_loaders
import adata_utils
import utils


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input1", type=str)
    parser.add_argument("input2", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="ATAC",
        choices=["ATAC", "RNA"],
        help="Merging RNA or ATAC data",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "RNA":
        raise NotImplemented
    else:
        x = utils.sc_read_10x_h5_ft_type(args.input1, ft_type="Peaks")
        y = utils.sc_read_10x_h5_ft_type(args.input2, ft_type="Peaks")

        atac_bins = sc_data_loaders.harmonize_atac_intervals(x.var_names, y.var_names)
        logging.info(f"Harmonized to {len(atac_bins)} intervals")

        x_repool = sc_data_loaders.repool_atac_bins(x, atac_bins)
        y_repool = sc_data_loaders.repool_atac_bins(y, atac_bins)

        result = x_repool.concatenate(y_repool)

    logging.info(f"Shape after concat: {result.shape}")
    if "." in os.path.basename(args.output):
        _root, ext = os.path.splitext(args.output)
        if ext == ".h5ad":
            result.write_h5ad(args.output)
        else:
            raise NotImplementedError(f"Unrecognized file extension: {ext}")
    else:
        adata_utils.write_adata_as_10x_dir(result, args.output)


if __name__ == "__main__":
    main()
