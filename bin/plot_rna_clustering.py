"""
Script for plotting clustering, while also evaluating distance between clusters
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
    "PBMC": interpretation.PBMC_MARKER_GENES,
    "PBMC_Seurat": interpretation.SEURAT_PBMC_MARKER_GENES,
}


def build_parser():
    """Build commandline argument parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fname", type=str, help="File to plot clustering for")
    parser.add_argument("plotprefix", type=str, help="File prefix to save plots to")
    parser.add_argument(
        "--resolution", "-r", type=float, default=1.0, help="Clustering resolution"
    )
    parser.add_argument(
        "--markers",
        "-m",
        type=str,
        choices=REF_MARKER_GENES.keys(),
        default="PBMC",
        help="Marker genes to evaluate with",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Skip pairwise comparisons for speed"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    adata = load_file_flex_format(args.fname)
    logging.info(f"Read in {os.path.abspath(args.fname)} for a matrix of {adata.shape}")

    # Do clustering
    logging.info(f"Clustering with resolution {args.resolution}")
    plot_utils.preprocess_anndata(
        adata,
        louvain_resolution=args.resolution,
        leiden_resolution=args.resolution,
        seed=1234,
    )

    plot_utils.plot_clustering_anndata(
        adata, label_counter=True, fname=args.plotprefix + "_leiden_clustering.pdf"
    )
    if not args.fast:
        logging.info(f"Computing pairwise distances between clusters")
        (
            clustering_distance_means,
            clustering_distance_sds,
        ) = adata_utils.evaluate_pairwise_cluster_distance(adata, stratify="leiden")
        clustering_distance_means.to_csv(args.plotprefix + "_cluster_dist_means.csv")
        clustering_distance_sds.to_csv(args.plotprefix + "_cluster_dist_sds.csv")

    # Find marker genes and label clusters
    logging.info(f"Finding marker genes")
    adata_utils.find_marker_genes(adata, n_genes=25)
    logging.info(
        f"Labelling clusters using {args.markers} marker genes (n={len(set(itertools.chain.from_iterable(REF_MARKER_GENES[args.markers])))})"
    )
    marker_matches = interpretation.annotate_clusters_to_celltypes(
        adata, REF_MARKER_GENES[args.markers],
    )
    plot_utils.plot_clustering_anndata(
        adata,
        color="leiden_celltypes",
        label_counter=True,
        fname=args.plotprefix + "_celltype_clustering.pdf",
    )
    if not args.fast:
        logging.info("Computing pairwise distances between labelled clusters")
        (
            labelled_cluster_distance_means,
            labelled_cluster_distance_sds,
        ) = adata_utils.evaluate_pairwise_cluster_distance(
            adata, stratify="leiden_celltypes"
        )
        labelled_cluster_distance_means.to_csv(
            args.plotprefix + "_labelled_cluster_dist_means.csv"
        )
        labelled_cluster_distance_sds.to_csv(
            args.plotprefix + "_labelled_cluster_dist_sds.csv"
        )


if __name__ == "__main__":
    main()
