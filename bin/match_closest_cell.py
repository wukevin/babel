"""
Script for calculating pairwise distances between 2 adatas
"""
import os
import sys
import logging
import argparse
import json

import numpy as np

from evaluate_bulk_rna_concordance import load_file_flex_format

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import adata_utils
import metrics

logging.basicConfig(level=logging.INFO)


def build_parser():
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("adata1", type=str, help="First adata object to compare")
    parser.add_argument("adata2", type=str, help="Second adata object to compare")
    parser.add_argument(
        "--output", "-o", type=str, default="", help="Json file to write matches to"
    )
    parser.add_argument(
        "--method", "-m", type=str, choices=["euclidean", "cosine"], default="euclidean"
    )
    parser.add_argument(
        "--log",
        "-l",
        action="store_true",
        help="Log transform before computing distance",
    )
    parser.add_argument(
        "--numtop",
        "-n",
        type=int,
        default=0,
        help="Number of top matches to report. 0 indicates reporting all in descending order",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    # Compute distances
    x = load_file_flex_format(args.adata1)
    logging.info(f"Loaded in {args.adata1} for {x.shape}")
    y = load_file_flex_format(args.adata2)
    logging.info(f"Loaded in {args.adata2} for {y.shape}")

    # Log, because often times in this project the output is actually linear space
    # and comparing expression is typically done in log space
    if args.log:
        logging.info("Log transforming inputs")
        x.X = np.log1p(x.X)
        y.X = np.log1p(y.X)

    pairwise_dist = adata_utils.evaluate_pairwise_cell_distance(
        x, y, method=args.method
    )
    if args.numtop == 0:
        args.numtop = y.n_obs

    # Figure out the top few matches per cell
    matches = {}
    for i, row in pairwise_dist.iterrows():
        sorted_idx = np.argsort(row.values)
        cell_matches = pairwise_dist.columns[sorted_idx[: args.numtop]]
        matches[i] = list(cell_matches)
    if args.output:
        assert args.output.endswith(".json")
        with open(args.output, "w") as sink:
            logging.info(f"Writing matches to {args.output}")
            json.dump(matches, sink, indent=4)

    # Report Top N accuracy if relevant
    if x.n_obs == y.n_obs and np.all(x.obs_names == y.obs_names):
        n = args.numtop if args.numtop < y.n_obs else 10
        acc = metrics.top_n_accuracy(matches.values(), matches.keys(), n=n)
        logging.info(f"Top {n} accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
