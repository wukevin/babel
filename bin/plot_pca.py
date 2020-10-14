"""
Basic script for pltoting pca of a dataset(s)
"""

import os
import sys
import argparse
import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import interpretation
import adata_utils
import plot_utils
import utils

from evaluate_bulk_rna_concordance import load_file_flex_format


def var_barplot(var: np.ndarray, fname: str = ""):
    """Basic barplot of explained variance"""
    fig, ax = plt.subplots(dpi=300)
    ax.bar(np.arange(len(var)), var)
    ax.set(
        xlabel=f"Principal component",
        ylabel="Explained variance",
        title=f"Top {len(var)} PCs ({np.sum(var):.4f} explained variance)",
    )
    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def build_parser():
    """Build a basic CLI parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("adata_fname", type=str, help="Adata object to plot PCA for")
    parser.add_argument("plot_prefix", type=str, help="Prefix to save plots to")
    parser.add_argument(
        "--numdims", "-n", type=int, default=16, help="Number of top PCs to consider"
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    adata = load_file_flex_format(args.adata_fname)

    var = adata.uns["pca"]["variance_ratio"][: args.numdims]
    var_barplot(var, fname=args.plot_prefix + "_explained_var.pdf")


if __name__ == "__main__":
    main()
