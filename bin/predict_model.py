"""
Code for evaluating a model's ability to generalize to cells that it wasn't trained on.
Can only be used to evalute within a species.
Generates raw predictions of data modality transfer, and optionally, plots.
"""

import os
import sys
from typing import *
import functools
import logging
import argparse
import copy

import scipy

import anndata as ad
import scanpy as sc

import torch
import skorch

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import sc_data_loaders
import loss_functions
import model_utils
import plot_utils
import adata_utils
import utils
from models import autoencoders

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
assert os.path.isdir(DATA_DIR)

logging.basicConfig(level=logging.INFO)

DATASET_NAME = ""


def do_evaluation_rna_from_rna(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    """
    Evaluate the given network on the dataset
    """
    # Do inference and plotting
    ### RNA > RNA
    logging.info("Inferring RNA from RNA...")
    sc_rna_full_preds = spliced_net.translate_1_to_1(sc_dual_full_dataset)
    sc_rna_full_preds_anndata = sc.AnnData(
        sc_rna_full_preds,
        obs=sc_dual_full_dataset.dataset_x.data_raw.obs,
    )
    sc_rna_full_preds_anndata.var_names = gene_names

    logging.info("Writing RNA from RNA")
    sc_rna_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_rna_rna_adata.h5ad".strip("_"))
    )
    if hasattr(sc_dual_full_dataset.dataset_x, "size_norm_counts") and ext is not None:
        logging.info("Plotting RNA from RNA")
        plot_utils.plot_scatter_with_r(
            sc_dual_full_dataset.dataset_x.size_norm_counts.X,
            sc_rna_full_preds,
            one_to_one=True,
            logscale=True,
            density_heatmap=True,
            title=f"{DATASET_NAME} RNA > RNA".strip(),
            fname=os.path.join(outdir, f"{prefix}_rna_rna_log.{ext}".strip("_")),
        )


def do_evaluation_atac_from_rna(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    ### RNA > ATAC
    logging.info("Inferring ATAC from RNA")
    sc_rna_atac_full_preds = spliced_net.translate_1_to_2(sc_dual_full_dataset)
    sc_rna_atac_full_preds_anndata = sc.AnnData(
        scipy.sparse.csr_matrix(sc_rna_atac_full_preds),
        obs=sc_dual_full_dataset.dataset_x.data_raw.obs,
    )
    sc_rna_atac_full_preds_anndata.var_names = atac_names
    logging.info("Writing ATAC from RNA")
    sc_rna_atac_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_rna_atac_adata.h5ad".strip("_"))
    )

    if hasattr(sc_dual_full_dataset.dataset_y, "data_raw") and ext is not None:
        logging.info("Plotting ATAC from RNA")
        plot_utils.plot_auroc(
            utils.ensure_arr(sc_dual_full_dataset.dataset_y.data_raw.X).flatten(),
            utils.ensure_arr(sc_rna_atac_full_preds).flatten(),
            title_prefix=f"{DATASET_NAME} RNA > ATAC".strip(),
            fname=os.path.join(outdir, f"{prefix}_rna_atac_auroc.{ext}".strip("_")),
        )
        # plot_utils.plot_auprc(
        #     utils.ensure_arr(sc_dual_full_dataset.dataset_y.data_raw.X).flatten(),
        #     utils.ensure_arr(sc_rna_atac_full_preds),
        #     title_prefix=f"{DATASET_NAME} RNA > ATAC".strip(),
        #     fname=os.path.join(outdir, f"{prefix}_rna_atac_auprc.{ext}".strip("_")),
        # )


def do_evaluation_atac_from_atac(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    ### ATAC > ATAC
    logging.info("Inferring ATAC from ATAC")
    sc_atac_full_preds = spliced_net.translate_2_to_2(sc_dual_full_dataset)
    sc_atac_full_preds_anndata = sc.AnnData(
        sc_atac_full_preds,
        obs=sc_dual_full_dataset.dataset_y.data_raw.obs.copy(deep=True),
    )
    sc_atac_full_preds_anndata.var_names = atac_names
    logging.info("Writing ATAC from ATAC")

    # Infer marker bins
    # logging.info("Getting marker bins for ATAC from ATAC")
    # plot_utils.preprocess_anndata(sc_atac_full_preds_anndata)
    # adata_utils.find_marker_genes(sc_atac_full_preds_anndata)
    # inferred_marker_bins = adata_utils.flatten_marker_genes(
    #     sc_atac_full_preds_anndata.uns["rank_genes_leiden"]
    # )
    # logging.info(f"Found {len(inferred_marker_bins)} marker bins for ATAC from ATAC")
    # with open(
    #     os.path.join(outdir, f"{prefix}_atac_atac_marker_bins.txt".strip("_")), "w"
    # ) as sink:
    #     sink.write("\n".join(inferred_marker_bins) + "\n")

    sc_atac_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_atac_atac_adata.h5ad".strip("_"))
    )
    if hasattr(sc_dual_full_dataset.dataset_y, "data_raw") and ext is not None:
        logging.info("Plotting ATAC from ATAC")
        plot_utils.plot_auroc(
            utils.ensure_arr(sc_dual_full_dataset.dataset_y.data_raw.X).flatten(),
            utils.ensure_arr(sc_atac_full_preds).flatten(),
            title_prefix=f"{DATASET_NAME} ATAC > ATAC".strip(),
            fname=os.path.join(outdir, f"{prefix}_atac_atac_auroc.{ext}".strip("_")),
        )
        # plot_utils.plot_auprc(
        #     utils.ensure_arr(sc_dual_full_dataset.dataset_y.data_raw.X).flatten(),
        #     utils.ensure_arr(sc_atac_full_preds).flatten(),
        #     title_prefix=f"{DATASET_NAME} ATAC > ATAC".strip(),
        #     fname=os.path.join(outdir, f"{prefix}_atac_atac_auprc.{ext}".strip("_")),
        # )

    # Remove some objects to free memory
    del sc_atac_full_preds
    del sc_atac_full_preds_anndata


def do_evaluation_rna_from_atac(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    ### ATAC > RNA
    logging.info("Inferring RNA from ATAC")
    sc_atac_rna_full_preds = spliced_net.translate_2_to_1(sc_dual_full_dataset)
    # Seurat expects everything to be sparse
    # https://github.com/satijalab/seurat/issues/2228
    sc_atac_rna_full_preds_anndata = sc.AnnData(
        scipy.sparse.csr_matrix(sc_atac_rna_full_preds),
        obs=sc_dual_full_dataset.dataset_y.data_raw.obs.copy(deep=True),
    )
    sc_atac_rna_full_preds_anndata.var_names = gene_names
    logging.info("Writing RNA from ATAC")

    # Seurat also expects the raw attribute to be populated
    sc_atac_rna_full_preds_anndata.raw = sc_atac_rna_full_preds_anndata.copy()
    sc_atac_rna_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_atac_rna_adata.h5ad".strip("_"))
    )
    # sc_atac_rna_full_preds_anndata.write_csvs(
    #     os.path.join(outdir, f"{prefix}_atac_rna_constituent_csv".strip("_")),
    #     skip_data=False,
    # )
    # sc_atac_rna_full_preds_anndata.to_df().to_csv(
    #     os.path.join(outdir, f"{prefix}_atac_rna_table.csv".strip("_"))
    # )

    # If there eixsts a ground truth RNA, do RNA plotting
    if hasattr(sc_dual_full_dataset.dataset_x, "size_norm_counts") and ext is not None:
        logging.info("Plotting RNA from ATAC")
        plot_utils.plot_scatter_with_r(
            sc_dual_full_dataset.dataset_x.size_norm_counts.X,
            sc_atac_rna_full_preds,
            one_to_one=True,
            logscale=True,
            density_heatmap=True,
            title=f"{DATASET_NAME} ATAC > RNA".strip(),
            fname=os.path.join(outdir, f"{prefix}_atac_rna_log.{ext}".strip("_")),
        )

    # Remove objects to free memory
    del sc_atac_rna_full_preds
    del sc_atac_rna_full_preds_anndata


def do_latent_evaluation(
    spliced_net, sc_dual_full_dataset, outdir: str, prefix: str = ""
):
    """
    Pull out latent space and write to file
    """
    logging.info("Inferring latent representations")
    encoded_from_rna, encoded_from_atac = spliced_net.get_encoded_layer(
        sc_dual_full_dataset
    )

    if hasattr(sc_dual_full_dataset.dataset_x, "data_raw"):
        encoded_from_rna_adata = sc.AnnData(
            encoded_from_rna,
            obs=sc_dual_full_dataset.dataset_x.data_raw.obs.copy(deep=True),
        )
        encoded_from_rna_adata.write(
            os.path.join(outdir, f"{prefix}_rna_encoded_adata.h5ad".strip("_"))
        )
    if hasattr(sc_dual_full_dataset.dataset_y, "data_raw"):
        encoded_from_atac_adata = sc.AnnData(
            encoded_from_atac,
            obs=sc_dual_full_dataset.dataset_y.data_raw.obs.copy(deep=True),
        )
        encoded_from_atac_adata.write(
            os.path.join(outdir, f"{prefix}_atac_encoded_adata.h5ad".strip("_"))
        )


def infer_reader(fname: str, mode: str = "atac") -> Callable:
    """Given a filename, infer the correct reader to use"""
    assert mode in ["atac", "rna"], f"Unrecognized mode: {mode}"
    if fname.endswith(".h5"):
        if mode == "atac":
            return functools.partial(utils.sc_read_10x_h5_ft_type, ft_type="Peaks")
        else:
            return utils.sc_read_10x_h5_ft_type
    elif fname.endswith(".h5ad"):
        return ad.read_h5ad
    else:
        raise ValueError(f"Unrecognized extension: {fname}")


def build_parser():
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="*",
        required=False,
        default=[None],
        help="Checkpoint directory to load model from. If not given, automatically download and use a human pretrained model",
    )
    parser.add_argument("--prefix", type=str, default="net_", help="Checkpoint prefix")
    parser.add_argument("--data", required=True, nargs="*", help="Data files")
    parser.add_argument(
        "--dataname", default="", help="Name of dataset to include in plot titles"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory for files and plots"
    )
    parser.add_argument(
        "--genes",
        type=str,
        default="",
        help="Genes that the model uses (inferred based on checkpoint dir if not given)",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="",
        help="ATAC bins that the model uses (inferred based on checkpoint dir if not given)",
    )
    parser.add_argument(
        "--liftHg19toHg38",
        action="store_true",
        help="Liftover input ATAC bins from hg19 to hg38",
    )
    parser.add_argument("--device", type=str, default="0", help="Device to use")
    parser.add_argument(
        "--ext",
        type=str,
        default="pdf",
        choices=["pdf", "png", "jpg"],
        help="File format to use for plotting",
    )
    parser.add_argument(
        "--noplot", action="store_true", help="Disable plotting, writing output only"
    )
    parser.add_argument(
        "--transonly",
        action="store_true",
        help="Disable doing same-modality inference",
    )
    parser.add_argument(
        "--skiprnasource", action="store_true", help="Skip analysis starting from RNA"
    )
    parser.add_argument(
        "--skipatacsource", action="store_true", help="Skip analysis starting from ATAC"
    )
    parser.add_argument(
        "--nofilter",
        action="store_true",
        help="Whether or not to perform filtering (note that we always discard cells with no expressed genes)",
    )
    return parser


def load_rna_files_for_eval(
    data, checkpoint: str, rna_genes_list_fname: str = "", no_filter: bool = False
):
    """ """
    if not rna_genes_list_fname:
        rna_genes_list_fname = os.path.join(checkpoint, "rna_genes.txt")
    assert os.path.isfile(
        rna_genes_list_fname
    ), f"Cannot find RNA genes file: {rna_genes_list_fname}"
    rna_genes = utils.read_delimited_file(rna_genes_list_fname)
    rna_data_kwargs = copy.copy(sc_data_loaders.TENX_PBMC_RNA_DATA_KWARGS)
    if no_filter:
        rna_data_kwargs = {
            k: v for k, v in rna_data_kwargs.items() if not k.startswith("filt_")
        }
        # Always discard cells with no expressed genes
        rna_data_kwargs["filt_cell_min_genes"] = 1
    rna_data_kwargs["fname"] = data
    reader_func = functools.partial(
        utils.sc_read_multi_files,
        reader=lambda x: sc_data_loaders.repool_genes(
            utils.get_ad_reader(x, ft_type="Gene Expression")(x), rna_genes
        ),
    )
    rna_data_kwargs["reader"] = reader_func
    try:
        logging.info(f"Building RNA dataset with parameters: {rna_data_kwargs}")
        sc_rna_full_dataset = sc_data_loaders.SingleCellDataset(
            mode="skip",
            **rna_data_kwargs,
        )
        assert all(
            [x == y for x, y in zip(rna_genes, sc_rna_full_dataset.data_raw.var_names)]
        ), "Mismatched genes"
        _temp = sc_rna_full_dataset[0]  # Try that query works
        # adata_utils.find_marker_genes(sc_rna_full_dataset.data_raw, n_genes=25)
        # marker_genes = adata_utils.flatten_marker_genes(
        #     sc_rna_full_dataset.data_raw.uns["rank_genes_leiden"]
        # )
        marker_genes = []
        # Write out the truth
    except (AssertionError, IndexError) as e:
        logging.warning(f"Error when reading RNA gene expression data from {data}: {e}")
        logging.warning("Ignoring RNA data")
        # Update length later
        sc_rna_full_dataset = sc_data_loaders.DummyDataset(
            shape=len(rna_genes), length=-1
        )
        marker_genes = []
    return sc_rna_full_dataset, rna_genes, marker_genes


def load_atac_files_for_eval(
    data: List[str],
    checkpoint: str,
    atac_bins_list_fname: str = "",
    lift_hg19_to_hg39: bool = False,
    predefined_split=None,
):
    """Load the ATAC files for evaluation"""
    if not atac_bins_list_fname:
        atac_bins_list_fname = os.path.join(checkpoint, "atac_bins.txt")
        logging.info(f"Auto-set atac bins fname to {atac_bins_list_fname}")
    assert os.path.isfile(
        atac_bins_list_fname
    ), f"Cannot find ATAC bins file: {atac_bins_list_fname}"
    atac_bins = utils.read_delimited_file(
        atac_bins_list_fname
    )  # These are the bins we are using (i.e. the bins the model was trained on)
    atac_data_kwargs = copy.copy(sc_data_loaders.TENX_PBMC_ATAC_DATA_KWARGS)
    atac_data_kwargs["fname"] = data
    atac_data_kwargs["cluster_res"] = 0  # Disable clustering
    filt_atac_keys = [k for k in atac_data_kwargs.keys() if k.startswith("filt")]
    for k in filt_atac_keys:  # Reset filtering
        atac_data_kwargs[k] = None
    atac_data_kwargs["pool_genomic_interval"] = atac_bins
    if not lift_hg19_to_hg39:
        atac_data_kwargs["reader"] = functools.partial(
            utils.sc_read_multi_files,
            reader=lambda x: sc_data_loaders.repool_atac_bins(
                infer_reader(data[0], mode="atac")(x),
                atac_bins,
            ),
        )
    else:  # Requires liftover
        # Read, liftover, then repool
        atac_data_kwargs["reader"] = functools.partial(
            utils.sc_read_multi_files,
            reader=lambda x: sc_data_loaders.repool_atac_bins(
                sc_data_loaders.liftover_atac_adata(
                    # utils.sc_read_10x_h5_ft_type(x, "Peaks")
                    infer_reader(data[0], mode="atac")(x)
                ),
                atac_bins,
            ),
        )

    try:
        sc_atac_full_dataset = sc_data_loaders.SingleCellDataset(
            mode="skip",
            predefined_split=predefined_split if predefined_split else None,
            **atac_data_kwargs,
        )
        _temp = sc_atac_full_dataset[0]  # Try that query works
        assert all(
            [x == y for x, y in zip(atac_bins, sc_atac_full_dataset.data_raw.var_names)]
        )
    except AssertionError as err:
        logging.warning(f"Error when reading ATAC data from {data}: {err}")
        logging.warning("Ignoring ATAC data, returning dummy dataset instead")
        sc_atac_full_dataset = sc_data_loaders.DummyDataset(
            shape=len(atac_bins), length=-1
        )
    return sc_atac_full_dataset, atac_bins


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.info(f"Evaluating: {' '.join(args.data)}")

    global DATASET_NAME
    DATASET_NAME = args.dataname

    # Create output directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Set up logging
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.outdir, "logging.log"), "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    (sc_rna_full_dataset, rna_genes, marker_genes,) = load_rna_files_for_eval(
        args.data, args.checkpoint[0], args.genes, no_filter=args.nofilter
    )

    if hasattr(sc_rna_full_dataset, "size_norm_counts"):
        logging.info("Writing truth RNA size normalized counts")
        sc_rna_full_dataset.size_norm_counts.write_h5ad(
            os.path.join(args.outdir, "truth_rna.h5ad")
        )

    sc_atac_full_dataset, atac_bins = load_atac_files_for_eval(
        args.data,
        args.checkpoint[0],
        args.bins,
        args.liftHg19toHg38,
        sc_rna_full_dataset if hasattr(sc_rna_full_dataset, "data_raw") else None,
    )
    # Write out the truth
    if hasattr(sc_atac_full_dataset, "data_raw"):
        logging.info("Writing truth ATAC binary counts")
        sc_atac_full_dataset.data_raw.write_h5ad(
            os.path.join(args.outdir, "truth_atac.h5ad")
        )

    if isinstance(sc_rna_full_dataset, sc_data_loaders.DummyDataset) and isinstance(
        sc_atac_full_dataset, sc_data_loaders.DummyDataset
    ):
        raise ValueError("Cannot proceed with two dummy datasets for both RNA and ATAC")
    # Update the RNA counts if we do not actually have RNA data
    if isinstance(sc_rna_full_dataset, sc_data_loaders.DummyDataset) and not isinstance(
        sc_atac_full_dataset, sc_data_loaders.DummyDataset
    ):
        sc_rna_full_dataset.length = len(sc_atac_full_dataset)
    elif isinstance(
        sc_atac_full_dataset, sc_data_loaders.DummyDataset
    ) and not isinstance(sc_rna_full_dataset, sc_data_loaders.DummyDataset):
        sc_atac_full_dataset.length = len(sc_rna_full_dataset)

    # Build the dual combined dataset
    sc_dual_full_dataset = sc_data_loaders.PairedDataset(
        sc_rna_full_dataset,
        sc_atac_full_dataset,
        flat_mode=True,
    )

    # Write some basic outputs related to variable and obs names
    with open(os.path.join(args.outdir, "rna_genes.txt"), "w") as sink:
        sink.write("\n".join(rna_genes) + "\n")
    with open(os.path.join(args.outdir, "atac_bins.txt"), "w") as sink:
        sink.write("\n".join(atac_bins) + "\n")
    with open(os.path.join(args.outdir, "obs_names.txt"), "w") as sink:
        sink.write("\n".join(sc_dual_full_dataset.obs_names))

    for i, ckpt in enumerate(args.checkpoint):
        # Dynamically determine the model we are looking at based on name
        if ckpt is not None:
            checkpoint_basename = os.path.basename(ckpt)
            if checkpoint_basename.startswith("naive"):
                logging.info(f"Inferred model to be naive")
                model_class = autoencoders.NaiveSplicedAutoEncoder
            else:
                logging.info(f"Inferred model to be normal (non-naive)")
                model_class = autoencoders.AssymSplicedAutoEncoder

        prefix = "" if len(args.checkpoint) == 1 else f"model_{checkpoint_basename}"
        spliced_net = model_utils.load_model(
            ckpt,
            prefix=args.prefix,
            device=args.device,
        )

        do_latent_evaluation(
            spliced_net=spliced_net,
            sc_dual_full_dataset=sc_dual_full_dataset,
            outdir=args.outdir,
            prefix=prefix,
        )

        if (
            isinstance(sc_rna_full_dataset, sc_data_loaders.SingleCellDataset)
            and not args.skiprnasource
        ):
            if not args.transonly:
                do_evaluation_rna_from_rna(
                    spliced_net,
                    sc_dual_full_dataset,
                    rna_genes,
                    atac_bins,
                    args.outdir,
                    None if args.noplot else args.ext,
                    marker_genes,
                    prefix=prefix,
                )
            do_evaluation_atac_from_rna(
                spliced_net,
                sc_dual_full_dataset,
                rna_genes,
                atac_bins,
                args.outdir,
                None if args.noplot else args.ext,
                marker_genes,
                prefix=prefix,
            )
        if (
            isinstance(sc_atac_full_dataset, sc_data_loaders.SingleCellDataset)
            and not args.skipatacsource
        ):
            do_evaluation_rna_from_atac(
                spliced_net,
                sc_dual_full_dataset,
                rna_genes,
                atac_bins,
                args.outdir,
                None if args.noplot else args.ext,
                marker_genes,
                prefix=prefix,
            )
            if not args.transonly:
                do_evaluation_atac_from_atac(
                    spliced_net,
                    sc_dual_full_dataset,
                    rna_genes,
                    atac_bins,
                    args.outdir,
                    None if args.noplot else args.ext,
                    marker_genes,
                    prefix=prefix,
                )
        del spliced_net


if __name__ == "__main__":
    main()
