"""
Script for evaluating how similar aggregated RNA signatures are
"""

import os
import sys
from typing import *
import random
import functools
import logging
import argparse
import copy

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

import torch
import skorch

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import sc_data_loaders
import loss_functions
import plot_utils
import adata_utils
import utils
from models import autoencoders

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
assert os.path.isdir(DATA_DIR)

logging.basicConfig(level=logging.INFO)


def load_file_flex_format(fname: str) -> sc.AnnData:
    """
    Intelligently load file based on file suffix
    """
    ext = os.path.splitext(fname)[-1]

    if ext == ".h5":
        retval = sc.read_10x_h5(fname, gex_only=True)
    elif ext == ".h5ad":
        retval = sc.read_h5ad(fname)
    else:
        raise ValueError(f"Unrecognized file extension: {ext}")
    retval.var_names_make_unique()
    return retval


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("rna_x", type=str, help="RNA truth")
    parser.add_argument("rna_y", type=str, help="RNA inferred")
    parser.add_argument("plotname", type=str, help="File to store scatterplot")
    parser.add_argument(
        "--normalize",
        nargs="*",
        choices=["x", "y"],
        default=["x", "y"],
        help="Apply normalization to x and/or y inputs",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="all",
        choices=["all", "random"],
        help="Mode for analysis",
    )
    parser.add_argument("--log", "-l", action="store_true", help="Plot in logspace")
    parser.add_argument("--density", "-d", action="store_true", help="Desntiy plot")
    parser.add_argument(
        "--title", "-t", type=str, default="Aggregated bulk scatterplot"
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()
    logging.info(f"Truth RNA file: {args.rna_x}")
    logging.info(f"Preds RNA file: {args.rna_y}")

    truth = load_file_flex_format(args.rna_x)
    preds = load_file_flex_format(args.rna_y)

    logging.info(f"Truth shape: {truth.shape}")
    logging.info(f"Preds shape: {preds.shape}")

    if "y" in args.normalize:
        logging.info("Normalizing y inferred input")
        preds = adata_utils.normalize_count_table(
            preds, size_factors=True, normalize=False, log_trans=False
        )
    if "x" in args.normalize:
        logging.info("Normalizing x inferred input")
        truth = adata_utils.normalize_count_table(
            truth, size_factors=True, normalize=False, log_trans=False
        )

    truth_bulk = pd.Series(
        np.array(truth.X.sum(axis=0)).flatten(), index=truth.var_names
    )
    preds_bulk = pd.Series(
        np.array(preds.X.sum(axis=0)).flatten(), index=preds.var_names
    )

    common_genes = sorted(list(set(truth_bulk.index).intersection(preds_bulk.index)))
    assert common_genes
    logging.info(f"{len(common_genes)} genes in common")

    plot_genes = common_genes

    if args.mode == "random":
        random.seed(1234)
        random.shuffle(plot_genes)
        plot_genes = plot_genes[:5000]
    elif args.mode == "all":
        pass
    else:
        raise ValueError(f"Unrecognized value for mode: {args.mode}")

    truth_bulk = truth_bulk[plot_genes]
    preds_bulk = preds_bulk[plot_genes]

    plot_utils.plot_scatter_with_r(
        truth_bulk,
        preds_bulk,
        subset=0,
        logscale=args.log,
        density_heatmap=args.density,
        xlabel="Reference",
        ylabel="Predicted",
        one_to_one=True,
        title=args.title + f" ({args.mode}, n={len(truth_bulk)})",
        fname=args.plotname,
    )


if __name__ == "__main__":
    main()
