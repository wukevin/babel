"""
Short script to plot RNA scatterplots
"""

import os
import sys
import re
import logging
import argparse
from typing import *

import numpy as np
import scipy
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import plot_utils
import utils


logging.basicConfig(level=logging.INFO)


def sanitize_obs_names(names: List[str]) -> List[str]:
    """
    Sanitize the obs names
    >>> sanitize_obs_names(['a', 'b'])
    ['a', 'b']
    >>> sanitize_obs_names(['foo#a', 'bar#b'])
    ['a', 'b']
    >>> sanitize_obs_names(['10xPBMC#TAAGTGCAGCGCACAA-1', '10xPBMC#AGCTATGTCTATCTTG-1'])
    ['TAAGTGCAGCGCACAA-1', 'AGCTATGTCTATCTTG-1']
    """
    # Strips out the prefix that archr inserts
    def relocate_rep_num(s: str) -> str:
        """
        Use the replicate as a suffix instead of prefix
        """
        if "#" not in s:
            return s
        prefix, samplename = s.split("#")
        rep_matches = re.findall(f"_rep[0-9]+$", prefix)
        if rep_matches:
            rep_match = rep_matches.pop()
            # Reps are 1 indexed, names are 0 indexed
            num = int(rep_match.strip("_rep")) - 1
            assert num >= 0, f"Error when processing {s}"
            return samplename + f"-{num}"
        else:
            return samplename

    def drop_extra_dash(s: str) -> str:
        """This may cause issues but it seems to be fine for now"""
        tokens = s.split("-")
        return "-".join(tokens[:2])

    retval = [relocate_rep_num(n) for n in names]
    retval = [drop_extra_dash(n) for n in retval]
    if not utils.is_all_unique(retval):
        logging.warning("Got duplicated names after sanitization")
    return retval


def build_parser():
    """Build a simple commandline parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("x_rna", type=str, help="X axis RNA data")
    parser.add_argument("y_rna", type=str, help="Y axis RNA data")
    parser.add_argument(
        "--outfname", type=str, default="", required=False, help="Filename to save plot"
    )
    parser.add_argument(
        "--subset", "-s", type=int, default=100000, help="Subset amount (0 to disable)"
    )
    parser.add_argument(
        "-g", "--genelist", type=str, default="", help="File containing list to plot"
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Plot in linear space instead of log space",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot density scatterplot instead of individual points",
    )
    parser.add_argument(
        "--densitylogstretch",
        type=int,
        default=1000,
        help="Density logstretch for image normalization",
    )
    parser.add_argument("--title", "-t", type=str, default="")
    parser.add_argument("--xlabel", type=str, default="Original norm counts")
    parser.add_argument("--ylabel", type=str, default="Inferred norm counts")
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(7, 5), help="Figure size"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.x_rna.endswith(".h5ad"):
        x_rna = ad.read_h5ad(args.x_rna)
    elif args.x_rna.endswith(".h5"):
        x_rna = sc.read_10x_h5(args.x_rna, gex_only=False)
    else:
        raise ValueError(f"Unrecognized file extension: {args.x_rna}")
    x_rna.X = utils.ensure_arr(x_rna.X)
    x_rna.obs_names = sanitize_obs_names(x_rna.obs_names)
    x_rna.obs_names_make_unique()
    logging.info(f"Read in {args.x_rna} for {x_rna.shape}")

    if args.y_rna.endswith(".h5ad"):
        y_rna = ad.read_h5ad(args.y_rna)
    elif args.y_rna.endswith(".h5"):
        y_rna = sc.read_10x_h5(args.y_rna, gex_only=False)
    else:
        raise ValueError(f"Unrecognized file extension: {args.y_rna}")
    y_rna.X = utils.ensure_arr(y_rna.X)
    y_rna.obs_names = sanitize_obs_names(y_rna.obs_names)
    y_rna.obs_names_make_unique()
    logging.info(f"Read in {args.y_rna} for {y_rna.shape}")

    if not (
        len(x_rna.obs_names) == len(y_rna.obs_names)
        and np.all(x_rna.obs_names == y_rna.obs_names)
    ):
        logging.warning("Rematching obs axis")
        shared_obs_names = sorted(
            list(set(x_rna.obs_names).intersection(y_rna.obs_names))
        )
        logging.info(f"Found {len(shared_obs_names)} shared obs")
        assert shared_obs_names, (
            "Got empty list of shared obs"
            + "\n"
            + str(x_rna.obs_names)
            + "\n"
            + str(y_rna.obs_names)
        )
        x_rna = x_rna[shared_obs_names]
        y_rna = y_rna[shared_obs_names]
    assert np.all(x_rna.obs_names == y_rna.obs_names)
    if not (
        len(x_rna.var_names) == len(y_rna.var_names)
        and np.all(x_rna.var_names == y_rna.var_names)
    ):
        logging.warning("Rematching variable axis")
        shared_var_names = sorted(
            list(set(x_rna.var_names).intersection(y_rna.var_names))
        )
        logging.info(f"Found {len(shared_var_names)} shared variables")
        assert shared_var_names, (
            "Got empty list of shared vars"
            + "\n"
            + str(x_rna.var_names)
            + "\n"
            + str(y_rna.var_names)
        )
        x_rna = x_rna[:, shared_var_names]
        y_rna = y_rna[:, shared_var_names]
    assert np.all(x_rna.var_names == y_rna.var_names)

    # Subset by gene list if given
    if args.genelist:
        gene_list = utils.read_delimited_file(args.genelist)
        logging.info(f"Read {len(gene_list)} genes from {args.genelist}")
        x_rna = x_rna[:, gene_list]
        y_rna = y_rna[:, gene_list]

    assert x_rna.shape == y_rna.shape, f"Mismatched shapes {x_rna.shape} {y_rna.shape}"

    fig = plot_utils.plot_scatter_with_r(
        x_rna.X,
        y_rna.X,
        subset=args.subset,
        one_to_one=True,
        logscale=not args.linear,
        density_heatmap=args.density,
        density_logstretch=args.densitylogstretch,
        fname=args.outfname,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=args.figsize,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
