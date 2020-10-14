"""
Plot the variance of a gene versus how accurately we can predict it
"""

import os
import sys
import argparse
import logging
import itertools

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

REF_MARKER_GENES = {
    "PBMC": set(
        itertools.chain.from_iterable(interpretation.PBMC_MARKER_GENES.values())
    ),
    "PBMC_Seurat": set(
        itertools.chain.from_iterable(interpretation.SEURAT_PBMC_MARKER_GENES.values())
    ),
    "Housekeeper": utils.read_delimited_file(
        os.path.join(os.path.dirname(SRC_DIR), "data", "housekeeper_genes.txt")
    ),
}


def build_parser():
    """Build basic CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("preds", type=str, help="File with predicted expression")
    parser.add_argument("truth", type=str, help="File with ground truth expression")
    parser.add_argument("plotname", type=str, help="File to write plot to")
    parser.add_argument(
        "--genelist",
        "-g",
        type=str,
        default="",
        help="File to write outliers in expained variance",
    )
    parser.add_argument(
        "--highlight",
        choices=REF_MARKER_GENES.keys(),
        nargs="*",
        default=["Housekeeper"],
        help="HGenes to highlight",
    )
    parser.add_argument(
        "--linear", action="store_true", help="Plot in linear space instead of log"
    )
    parser.add_argument(
        "--unconstriained", action="store_true", help="Do not constrain axes"
    )
    parser.add_argument("--outliers", action="store_true", help="Label outliers")
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    truth = load_file_flex_format(args.truth)
    truth.X = utils.ensure_arr(truth.X)
    logging.info(f"Loaded truth {args.truth}: {truth.shape}")
    preds = load_file_flex_format(args.preds)
    preds.X = utils.ensure_arr(preds.X)
    logging.info(f"Loaded preds {args.preds}: {preds.shape}")

    common_genes = sorted(list(set(truth.var_names).intersection(preds.var_names)))
    logging.info(f"Shared genes: {len(common_genes)}")

    common_obs = sorted(list(set(truth.obs_names).intersection(preds.obs_names)))
    # All obs naames should intersect between preds and truth
    assert len(common_obs) == len(truth.obs_names) == len(preds.obs_names)

    plot_utils.plot_var_vs_explained_var(
        truth,
        preds,
        highlight_genes={k: REF_MARKER_GENES[k] for k in args.highlight},
        logscale=not args.linear,
        constrain_y_axis=not args.unconstriained,
        label_outliers=args.outliers,
        fname=args.plotname,
        fname_gene_list=args.genelist,
    )


if __name__ == "__main__":
    main()
