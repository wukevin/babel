"""
Convert input file to 10x data dir
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
from scipy import io
import scanpy as sc
import anndata as ad

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
from genomic_interval import GenomicInterval as GI
import adata_utils
import utils

logging.basicConfig(level=logging.INFO)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file to convert")
    parser.add_argument("outdir", type=str, help="Output directory to create")
    parser.add_argument(
        "--mode", "-m", type=str, default="ATAC", choices=["ATAC", "RNA"]
    )
    parser.add_argument(
        "--transpose", "-t", action="store_true", help="Transpose before saving"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    _, input_ext = os.path.splitext(args.input)
    if input_ext == ".h5ad":
        x = ad.read_h5ad(args.input)
    elif input_ext == ".h5":
        if args.mode == "RNA":
            x = sc.read_10x_h5(args.input)
        else:
            x = utils.sc_read_10x_h5_ft_type(args.input, "Peaks")
    else:
        raise ValueError(f"Unrecongized file extension: {input_ext}")

    adata_utils.write_adata_as_10x_dir(x, args.outdir, transpose=args.transpose)


if __name__ == "__main__":
    main()
