"""
Script for calculating gene activity scores
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import sc_data_loaders
import adata_utils
import metrics
import atac_utils
import utils

logging.basicConfig(level=logging.INFO)


ANNOTATION_DICT = {
    "hg19": sc_data_loaders.HG19_GTF,
    "hg38": sc_data_loaders.HG38_GTF,
}


def build_parser():
    """Build parser for running as a script"""
    parser = argparse.ArgumentParser(
        usage="Convert ATAC h5ad to infered gene activity scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_h5ad",
        type=str,
        nargs="*",
        help="ATAC h5ad to infer gene activity scores for",
    )
    parser.add_argument(
        "output_h5ad", type=str, help="h5ad file to write gene activity scores to"
    )
    parser.add_argument(
        "--genome",
        "-g",
        choices=ANNOTATION_DICT.keys(),
        default="hg38",
        help="Genome annotation to use",
    )
    parser.add_argument("--raw", action="store_true", help="Use raw atribaute of input")
    parser.add_argument(
        "--sizenorm",
        action="store_true",
        help="Normalize gene activity scores by span of gene",
    )
    parser.add_argument(
        "--naive",
        action="store_true",
        help="Use naive method instead of archr-derived method",
    )
    return parser


def main():
    """On the fly testing"""
    parser = build_parser()
    args = parser.parse_args()
    logging.info(f"Inputs: {args.input_h5ad}")

    # Note that all inputs should have the same suffix
    for input_h5ad in args.input_h5ad:
        assert input_h5ad.endswith(".h5ad") or input_h5ad.endswith(".h5")
    assert args.output_h5ad.endswith(".h5ad")
    logging.info(f"Calculating gene activity scores for: {args.input_h5ad}")

    reader_func = (
        ad.read_h5ad
        if args.input_h5ad[0].endswith(".h5ad")
        else lambda x: utils.sc_read_10x_h5_ft_type(x, "Peaks")
    )
    # adata = reader_func(args.input_h5ad)
    adata = utils.sc_read_multi_files(args.input_h5ad, reader=reader_func)
    logging.info(f"Received input of {adata.shape}")
    if args.naive:
        gene_act_adata = atac_utils.gene_activity_matrix_from_adata(
            adata if not args.raw else adata.raw,
            size_norm=args.sizenorm,
            annotation=ANNOTATION_DICT[args.genome],
        )
    else:
        logging.warning("Using ArchR based method, size norm is always on")
        gene_act_adata = atac_utils.archr_gene_activity_matrix_from_adata(
            adata if not args.raw else adata.raw,
            annotation=ANNOTATION_DICT[args.genome],
        )
    logging.info(f"Writing gene activity scores to {args.output_h5ad}")
    gene_act_adata.write(args.output_h5ad)


if __name__ == "__main__":
    main()
