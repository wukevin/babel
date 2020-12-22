"""
Convert file to input format BIRD expects
BIRD requires that genes are in the format ENSG00000236743.1
"""
# Example file: https://raw.githubusercontent.com/WeiqiangZhou/BIRD/master/example/FPKM_data_matrix.txt

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import adata_utils
import utils

logging.basicConfig(level=logging.INFO)


def build_parser():
    """Build CLI parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Input file to convert. Should be of gene expression",
    )
    parser.add_argument("output_table_txt", type=str, help="File to write")
    parser.add_argument(
        "--transforms", default="", required=False, type=str, choices=["log1p", "exp"]
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    _, input_ext = os.path.splitext(args.input)
    if input_ext == ".h5ad":
        x = ad.read_h5ad(args.input)
    elif input_ext == ".h5":
        x = sc.read_10x_h5(args.input)
    else:
        raise ValueError(f"Unrecognized file extension: {args.input}")
    logging.info(f"Read input: {x}")

    logging.info("Reading gtf for gene name map")
    gene_name_map = utils.read_gtf_gene_symbol_to_id()

    # Tranpose because BIRD wants features x obs
    x_df = pd.DataFrame(utils.ensure_arr(x.X), index=x.obs_names, columns=x.var_names).T
    assert np.all(x_df.values >= 0.0)
    x_df.index = [gene_name_map[g] for g in x_df.index]

    # Write output (tab-separated table
    logging.info(f"Writing output to {args.output_table_txt}")
    x_df.to_csv(args.output_table_txt, sep="\t")


if __name__ == "__main__":
    main()
