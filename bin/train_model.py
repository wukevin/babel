"""
Code to train a model
"""

import os
import sys
import logging
import argparse
import copy
import functools
import itertools

import numpy as np
import pandas as pd
import scipy.spatial
import scanpy as sc

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import skorch
import skorch.helper

torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

MODELS_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)

import sc_data_loaders
import adata_utils
import model_utils
import autoencoders
import loss_functions
import layers
import activations
import plot_utils
import utils
import metrics
import interpretation

MODELS_SAVE_DIR = os.path.join(os.path.dirname(SRC_DIR), "models")
assert os.path.isdir(MODELS_SAVE_DIR), f"Cannot find: {MODELS_SAVE_DIR}"

logging.basicConfig(level=logging.INFO)

OPTIMIZER_DICT = {
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
}


def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data", "-d", type=str, nargs="*", help="Data files to train on",
    )
    input_group.add_argument(
        "--snareseq",
        action="store_true",
        help="Data in SNAREseq format, use custom data loading logic for separated RNA ATC files",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Do clustering data splitting in linear instead of log space",
    )
    parser.add_argument(
        "--clustermethod",
        type=str,
        choices=["leiden", "louvain"],
        default="leiden",
        help="Clustering method to determine data splits",
    )
    parser.add_argument(
        "--validcluster", type=int, default=0, help="Cluster ID to use as valid cluster"
    )
    parser.add_argument(
        "--testcluster", type=int, default=1, help="Cluster ID to use as test cluster"
    )
    parser.add_argument(
        "--outdir", "-o", required=True, type=str, help="Directory to output to"
    )
    parser.add_argument(
        "--naive",
        "-n",
        action="store_true",
        help="Use a naive model instead of lego model",
    )
    parser.add_argument(
        "--hidden", type=int, nargs="*", default=[16], help="Hidden dimensions"
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="",
        help="params.pt file to use to warm initialize the model (instead of starting from scratch)",
    )
    parser.add_argument(
        "--lossweight",
        type=float,
        nargs="*",
        default=[1.33],
        help="Relative loss weight",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=OPTIMIZER_DICT.keys(),
        help="Optimizer to use",
    )
    parser.add_argument(
        "--lr", "-l", type=float, default=[0.01], nargs="*", help="Learning rate"
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, nargs="*", default=[512], help="Batch size"
    )
    parser.add_argument(
        "--earlystop", type=int, default=25, help="Early stopping after N epochs"
    )
    parser.add_argument(
        "--seed", type=int, nargs="*", default=[182822], help="Random seed to use"
    )
    parser.add_argument("--device", default=0, type=int, help="Device to train on")
    parser.add_argument(
        "--ext",
        type=str,
        choices=["png", "pdf", "jpg"],
        default="pdf",
        help="Output format for plots",
    )
    return parser


def plot_loss_history(history, fname: str):
    """Create a plot of train valid loss"""
    fig, ax = plt.subplots(dpi=300)
    ax.plot(
        np.arange(len(history)), history[:, "train_loss"], label="Train",
    )
    ax.plot(
        np.arange(len(history)), history[:, "valid_loss"], label="Valid",
    )
    ax.legend()
    ax.set(
        xlabel="Epoch", ylabel="Loss",
    )
    fig.savefig(fname)
    return fig


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}_training.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Borrow parameters
    logging.info("Reading RNA data")
    if not args.snareseq:
        rna_data_kwargs = copy.copy(sc_data_loaders.TENX_PBMC_RNA_DATA_KWARGS)
        rna_data_kwargs["fname"] = args.data
    else:
        rna_data_kwargs = copy.copy(sc_data_loaders.SNARESEQ_RNA_DATA_KWARGS)
    rna_data_kwargs["data_split_by_cluster_log"] = not args.linear
    rna_data_kwargs["data_split_by_cluster"] = args.clustermethod

    sc_rna_train_dataset = sc_data_loaders.SingleCellDataset(
        mode="train",
        valid_cluster_id=args.validcluster,
        test_cluster_id=args.testcluster,
        **rna_data_kwargs,
    )
    sc_rna_valid_dataset = sc_data_loaders.SingleCellDataset(
        mode="valid",
        valid_cluster_id=args.validcluster,
        test_cluster_id=args.testcluster,
        **rna_data_kwargs,
    )
    sc_rna_test_dataset = sc_data_loaders.SingleCellDataset(
        mode="test",
        valid_cluster_id=args.validcluster,
        test_cluster_id=args.testcluster,
        **rna_data_kwargs,
    )
    assert (
        sc_rna_test_dataset.data_raw.shape[1]
        == sc_rna_valid_dataset.data_raw.shape[1]
        == sc_rna_train_dataset.data_raw.shape[1]
    ), "Mismatched shapes for RNA data loaders"

    # logging.info(f"Identifying marker genes")
    # adata_utils.find_marker_genes(sc_rna_full_dataset.data_raw, n_genes=25)
    # marker_genes = adata_utils.flatten_marker_genes(
    #     sc_rna_full_dataset.data_raw.uns["rank_genes_leiden"]
    # )

    # ATAC
    logging.info("Aggregating ATAC clusters")
    if not args.snareseq:
        atac_parsed = [
            utils.sc_read_10x_h5_ft_type(fname, "Peaks") for fname in args.data
        ]
        atac_bins = sc_data_loaders.harmonize_atac_intervals(
            atac_parsed[0].var_names, atac_parsed[1].var_names
        )
        for bins in atac_parsed[2:]:
            atac_bins = sc_data_loaders.harmonize_atac_intervals(
                atac_bins, bins.var_names
            )
        logging.info(f"Aggregated {len(atac_bins)} bins")

        atac_data_kwargs = copy.copy(sc_data_loaders.TENX_PBMC_ATAC_DATA_KWARGS)
        atac_data_kwargs["fname"] = rna_data_kwargs["fname"]
        atac_data_kwargs["pool_genomic_interval"] = 0  # Do not pool
        atac_data_kwargs["reader"] = functools.partial(
            utils.sc_read_multi_files,
            reader=lambda x: sc_data_loaders.repool_atac_bins(
                utils.sc_read_10x_h5_ft_type(x, "Peaks"), atac_bins,
            ),
        )
    else:
        atac_data_kwargs = copy.copy(sc_data_loaders.SNARESEQ_ATAC_DATA_KWARGS)
    atac_data_kwargs["cluster_res"] = 0  # Do not bother clustering ATAC data

    sc_atac_train_dataset = sc_data_loaders.SingleCellDataset(
        predefined_split=sc_rna_train_dataset.data_raw.obs_names, **atac_data_kwargs,
    )
    sc_atac_valid_dataset = sc_data_loaders.SingleCellDataset(
        predefined_split=sc_rna_valid_dataset.data_raw.obs_names, **atac_data_kwargs,
    )
    sc_atac_test_dataset = sc_data_loaders.SingleCellDataset(
        predefined_split=sc_rna_test_dataset.data_raw.obs_names, **atac_data_kwargs,
    )
    assert (
        sc_atac_train_dataset.data_raw.shape[1]
        == sc_atac_valid_dataset.data_raw.shape[1]
        == sc_atac_test_dataset.data_raw.shape[1]
    )

    sc_dual_train_dataset = sc_data_loaders.PairedDataset(
        sc_rna_train_dataset, sc_atac_train_dataset, flat_mode=True,
    )
    sc_dual_valid_dataset = sc_data_loaders.PairedDataset(
        sc_rna_valid_dataset, sc_atac_valid_dataset, flat_mode=True,
    )
    sc_dual_test_dataset = sc_data_loaders.PairedDataset(
        sc_rna_test_dataset, sc_atac_test_dataset, flat_mode=True,
    )

    # Model
    param_combos = list(
        itertools.product(
            args.hidden, args.lossweight, args.lr, args.batchsize, args.seed
        )
    )
    for h_dim, lw, lr, bs, rand_seed in param_combos:
        outdir_name = (
            f"{args.outdir}_hidden_{h_dim}_lossweight_{lw}_lr_{lr}_batchsize_{bs}_seed_{rand_seed}"
            if len(param_combos) > 1
            else args.outdir
        )
        if not os.path.isdir(outdir_name):
            assert not os.path.exists(outdir_name)
            os.makedirs(outdir_name)
        assert os.path.isdir(outdir_name)
        with open(os.path.join(outdir_name, "rna_genes.txt"), "w") as sink:
            for gene in sc_rna_train_dataset.data_raw.var_names:
                sink.write(gene + "\n")
        with open(os.path.join(outdir_name, "atac_bins.txt"), "w") as sink:
            for atac_bin in sc_atac_train_dataset.data_raw.var_names:
                sink.write(atac_bin + "\n")

        # Write train and validation data
        sc_rna_train_dataset.size_norm_counts.write_h5ad(
            os.path.join(outdir_name, "train_rna.h5ad")
        )
        sc_rna_valid_dataset.size_norm_counts.write_h5ad(
            os.path.join(outdir_name, "valid_rna.h5ad")
        )
        sc_atac_train_dataset.data_raw.write_h5ad(
            os.path.join(outdir_name, "train_atac.h5ad")
        )
        sc_atac_valid_dataset.data_raw.write_h5ad(
            os.path.join(outdir_name, "valid_atac.h5ad")
        )
        # Write test set ground truth
        sc_rna_test_dataset.size_norm_counts.write_h5ad(
            os.path.join(outdir_name, "truth_rna.h5ad")
        )
        sc_atac_test_dataset.data_raw.write_h5ad(
            os.path.join(outdir_name, "truth_atac.h5ad")
        )

        model_class = (
            autoencoders.NaiveSplicedAutoEncoder
            if args.naive
            else autoencoders.AssymSplicedAutoEncoder
        )
        spliced_net = autoencoders.SplicedAutoEncoderSkorchNet(
            module=model_class,
            module__hidden_dim=h_dim,  # Based on hyperparam tuning
            module__input_dim1=sc_rna_train_dataset.data_raw.shape[1],
            module__input_dim2=sc_atac_train_dataset.get_per_chrom_feature_count(),
            module__final_activations1=[
                activations.Exp(),
                activations.ClippedSoftplus(),
            ],
            module__final_activations2=nn.Sigmoid(),
            module__flat_mode=True,
            module__seed=rand_seed,
            lr=lr,  # Based on hyperparam tuning
            criterion=loss_functions.QuadLoss,
            criterion__loss2=loss_functions.BCELoss,  # handle output of encoded layer
            criterion__loss2_weight=lw,  # numerically balance the two losses with different magnitudes
            criterion__record_history=True,
            optimizer=OPTIMIZER_DICT[args.optim],
            iterator_train__shuffle=True,
            device=utils.get_device(args.device),
            batch_size=bs,  # Based on  hyperparam tuning
            max_epochs=500,
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=args.earlystop),
                skorch.callbacks.LRScheduler(
                    policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    **model_utils.REDUCE_LR_ON_PLATEAU_PARAMS,
                ),
                skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
                skorch.callbacks.Checkpoint(
                    dirname=outdir_name, fn_prefix="net_", monitor="valid_loss_best",
                ),
            ],
            train_split=skorch.helper.predefined_split(sc_dual_valid_dataset),
            iterator_train__num_workers=8,
            iterator_valid__num_workers=8,
        )
        if args.pretrain:
            # Load in the warm start parameters
            spliced_net.load_params(f_params=args.pretrain)
            spliced_net.partial_fit(sc_dual_train_dataset, y=None)
        else:
            spliced_net.fit(sc_dual_train_dataset, y=None)

        fig = plot_loss_history(
            spliced_net.history, os.path.join(outdir_name, f"loss.{args.ext}")
        )
        plt.close(fig)

        # Evaluate on test set
        ### RNA > RNA
        sc_rna_test_preds = spliced_net.translate_1_to_1(sc_dual_test_dataset)
        sc_rna_test_preds_anndata = sc.AnnData(
            sc_rna_test_preds,
            var=sc_rna_test_dataset.data_raw.var,
            obs=sc_rna_test_dataset.data_raw.obs,
        )
        sc_rna_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "rna_rna_test_preds.h5ad")
        )
        fig = plot_utils.plot_scatter_with_r(
            sc_rna_test_dataset.size_norm_counts.X,
            sc_rna_test_preds,
            subset=100000,
            one_to_one=True,
            title="RNA > RNA (test set)",
            fname=os.path.join(outdir_name, f"rna_rna_scatter.{args.ext}"),
        )
        plt.close(fig)
        fig = plot_utils.plot_scatter_with_r(
            sc_rna_test_dataset.size_norm_counts.X,
            sc_rna_test_preds,
            subset=100000,
            one_to_one=True,
            logscale=True,
            title="RNA > RNA (test set)",
            fname=os.path.join(outdir_name, f"rna_rna_scatter_log.{args.ext}"),
        )
        plt.close(fig)
        # fig = plot_utils.plot_scatter_with_r(
        #     sc_rna_test_dataset.size_norm_counts[:, marker_genes].X.flatten(),
        #     sc_rna_test_preds_anndata[:, marker_genes].X.flatten(),
        #     subset=100000,
        #     one_to_one=True,
        #     title="RNA > RNA, marker genes",
        #     fname=os.path.join(outdir_name, f"rna_rna_scatter_marker.{args.ext}"),
        # )
        # plt.close(fig)

        ### ATAC > ATAC
        sc_atac_test_preds = spliced_net.translate_2_to_2(sc_dual_test_dataset)
        sc_atac_test_preds_anndata = sc.AnnData(
            sc_atac_test_preds,
            var=sc_atac_test_dataset.data_raw.var,
            obs=sc_atac_test_dataset.data_raw.obs,
        )
        sc_atac_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "atac_atac_test_preds.h5ad")
        )
        fig = plot_utils.plot_auroc(
            sc_atac_test_dataset.data_raw.X,
            sc_atac_test_preds,
            title_prefix="ATAC > ATAC",
            fname=os.path.join(outdir_name, f"atac_atac_auroc.{args.ext}"),
        )
        plt.close(fig)
        # fig = plot_utils.plot_auprc(
        #     utils.ensure_arr(sc_atac_test_dataset.data_raw.X).flatten(),
        #     utils.ensure_arr(sc_atac_test_preds).flatten(),
        #     title_prefix="ATAC > ATAC",
        #     fname=os.path.join(outdir_name, f"atac_atac_auprc.{args.ext}"),
        # )
        # plt.close(fig)

        ### ATAC > RNA
        sc_atac_rna_test_preds = spliced_net.translate_2_to_1(sc_dual_test_dataset)
        sc_atac_rna_test_preds_anndata = sc.AnnData(
            sc_atac_rna_test_preds,
            var=sc_rna_test_dataset.data_raw.var,
            obs=sc_rna_test_dataset.data_raw.obs,
        )
        sc_atac_rna_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "atac_rna_test_preds.h5ad")
        )

        fig = plot_utils.plot_scatter_with_r(
            sc_rna_test_dataset.size_norm_counts.X,
            sc_atac_rna_test_preds,
            subset=100000,
            one_to_one=True,
            title="ATAC > RNA (test set)",
            fname=os.path.join(outdir_name, f"atac_rna_scatter.{args.ext}"),
        )
        plt.close(fig)

        fig = plot_utils.plot_scatter_with_r(
            sc_rna_test_dataset.size_norm_counts.X,
            sc_atac_rna_test_preds,
            subset=100000,
            one_to_one=True,
            logscale=True,
            title="ATAC > RNA (test set)",
            fname=os.path.join(outdir_name, f"atac_rna_scatter_log.{args.ext}"),
        )
        plt.close(fig)
        # fig = plot_utils.plot_scatter_with_r(
        #     sc_rna_test_dataset.size_norm_counts[:, marker_genes].X.flatten(),
        #     sc_atac_rna_test_preds_anndata[:, marker_genes].X.flatten(),
        #     one_to_one=True,
        #     title="ATAC > RNA, marker genes",
        #     fname=os.path.join(outdir_name, f"atac_rna_scatter_marker.{args.ext}"),
        # )
        # plt.close(fig)

        ### RNA > ATAC
        sc_rna_atac_test_preds = spliced_net.translate_1_to_2(sc_dual_test_dataset)
        sc_rna_atac_test_preds_anndata = sc.AnnData(
            sc_rna_atac_test_preds,
            var=sc_atac_test_dataset.data_raw.var,
            obs=sc_atac_test_dataset.data_raw.obs,
        )
        sc_rna_atac_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "rna_atac_test_preds.h5ad")
        )
        fig = plot_utils.plot_auroc(
            sc_atac_test_dataset.data_raw.X,
            sc_rna_atac_test_preds,
            title_prefix="RNA > ATAC",
            fname=os.path.join(outdir_name, f"rna_atac_auroc.{args.ext}"),
        )
        plt.close(fig)
        # fig = plot_utils.plot_auprc(
        #     sc_atac_test_dataset.data_raw.X.toarray().flatten(),
        #     sc_rna_atac_test_preds.flatten(),
        #     title_prefix="RNA > ATAC",
        #     fname=os.path.join(outdir_name, f"rna_atac_auprc.{args.ext}"),
        # )
        # plt.close(fig)

        # Evaluate latent space
        rna_encoded, atac_encoded = spliced_net.get_encoded_layer(sc_dual_test_dataset)
        # Plot each latent dimension
        fig, axes = plt.subplots(
            dpi=300, ncols=4, nrows=4, sharex=True, sharey=True, figsize=(7, 4)
        )
        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                idx = i * 4 + j
                _n, bins, _patches = ax.hist(
                    rna_encoded[:, idx], alpha=0.7, label="RNA"
                )
                ax.hist(atac_encoded[:, idx], alpha=0.7, bins=bins, label="ATAC")
                v = np.var(
                    np.hstack(
                        (rna_encoded[:, idx].flatten(), atac_encoded[:, idx].flatten())
                    )
                )
                ax.text(
                    0.9,
                    0.9,
                    f"var: {v:.3f}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                )
        fig.suptitle(f"{idx + 1} dimensional latent space")
        fig.savefig(os.path.join(outdir_name, f"latent_dimensions.{args.ext}"))
        plt.close(fig)

        matched_distance = np.array(
            [
                scipy.spatial.distance.euclidean(i, j)
                for i, j in zip(rna_encoded, atac_encoded)
            ]
        )
        shuffled_distance = np.array(
            [
                scipy.spatial.distance.euclidean(rna_encoded[i, :], atac_encoded[j, :])
                for i, j in zip(
                    np.random.permutation(rna_encoded.shape[0]),
                    np.random.permutation(atac_encoded.shape[0]),
                )
            ]
        )
        assert matched_distance.shape == shuffled_distance.shape

        fig, ax = plt.subplots(dpi=300)
        ax.hist(shuffled_distance, label="Mismatched pairs", alpha=0.7)
        ax.hist(matched_distance, label="Matched pairs", alpha=0.7)
        ax.legend()
        ax.set(
            xlabel="Euclidean distance in latent space",
            title=f"Test set latent space representations (p={scipy.stats.ttest_ind(matched_distance, shuffled_distance)[1]:.4g})",
        )
        fig.savefig(os.path.join(outdir_name, f"latent_distance.{args.ext}"))
        plt.close(fig)

        all_encoded = sc.AnnData(
            np.vstack([rna_encoded, atac_encoded]),
            obs=pd.DataFrame(
                index=[l + "-rna" for l in sc_dual_test_dataset.get_obs_labels()]
                + [l + "-atac" for l in sc_dual_test_dataset.get_obs_labels()]
            ),
        )
        all_encoded.obs["data_platform"] = [
            l.split("-")[-1] for l in all_encoded.obs_names
        ]

        plot_utils.preprocess_anndata(all_encoded)
        fig = plot_utils.plot_clustering_anndata(
            all_encoded, color="data_platform"
        ).savefig(os.path.join(outdir_name, f"latent_clustering.{args.ext}"))
        plt.close(fig)
        del spliced_net


if __name__ == "__main__":
    main()
