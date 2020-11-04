"""
Script for training a protein predictor
"""

import os
import sys
import logging
import argparse
import copy
import functools
import itertools
import collections
from typing import *

import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc
import anndata as ad

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
import autoencoders
import loss_functions
import model_utils
import utils

from train_model import plot_loss_history

logging.basicConfig(level=logging.INFO)


def load_rna_files(
    rna_counts_fnames: List[str], model_dir: str, transpose: bool = True
) -> ad.AnnData:
    """Load the RNA files in, filling in unmeasured genes as necessary"""
    # Find the genes that the model understands
    rna_genes_list_fname = os.path.join(model_dir, "rna_genes.txt")
    assert os.path.isfile(
        rna_genes_list_fname
    ), f"Cannot find RNA genes file: {rna_genes_list_fname}"
    learned_rna_genes = utils.read_delimited_file(rna_genes_list_fname)
    assert isinstance(learned_rna_genes, list)
    assert utils.is_all_unique(
        learned_rna_genes
    ), "Learned genes list contains duplicates"

    temp_ad = utils.sc_read_multi_files(
        rna_counts_fnames,
        feature_type="Gene Expression",
        transpose=transpose,
        join="outer",
    )
    logging.info(f"Read input RNA files for {temp_ad.shape}")
    temp_ad.X = utils.ensure_arr(temp_ad.X)

    # Filter for mouse genes and remove human/mouse prefix
    temp_ad.var_names_make_unique()
    kept_var_names = [
        vname for vname in temp_ad.var_names if not vname.startswith("MOUSE_")
    ]
    if len(kept_var_names) != temp_ad.n_vars:
        temp_ad = temp_ad[:, kept_var_names]
    temp_ad.var = pd.DataFrame(index=[v.strip("HUMAN_") for v in kept_var_names])

    # Expand adata to span all genes
    # Initiating as a sparse matrix doesn't allow vectorized building
    intersected_genes = set(temp_ad.var_names).intersection(learned_rna_genes)
    assert intersected_genes, "No overlap between learned and input genes!"
    expanded_mat = np.zeros((temp_ad.n_obs, len(learned_rna_genes)))
    skip_count = 0
    for gene in intersected_genes:
        dest_idx = learned_rna_genes.index(gene)
        src_idx = temp_ad.var_names.get_loc(gene)
        if not isinstance(src_idx, int):
            logging.warn(f"Got multiple source matches for {gene}, skipping")
            skip_count += 1
            continue
        v = utils.ensure_arr(temp_ad.X[:, src_idx]).flatten()
        expanded_mat[:, dest_idx] = v
    if skip_count:
        logging.warning(
            f"Skipped {skip_count}/{len(intersected_genes)} genes due to multiple matches"
        )
    expanded_mat = sparse.csr_matrix(expanded_mat)  # Compress
    retval = ad.AnnData(
        expanded_mat, obs=temp_ad.obs, var=pd.DataFrame(index=learned_rna_genes)
    )
    return retval


def build_parser():
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--rnaCounts",
        type=str,
        nargs="*",
        required=True,
        help="file containing raw RNA counts",
    )
    parser.add_argument(
        "--proteinCounts",
        type=str,
        nargs="*",
        required=True,
        help="file containing raw protein counts",
    )
    parser.add_argument(
        "--encoder", required=True, type=str, help="Model folder to find encoder"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.getcwd(),
        help="Output directory for model, defaults to current dir",
    )
    parser.add_argument(
        "--preprocessonly",
        action="store_true",
        help="Preprocess data only, do not train model",
    )
    parser.add_argument(
        "--epochs", type=int, default=600, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--notrans",
        action="store_true",
        help="Do not transpose (already in row obs form)",
    )
    parser.add_argument("--device", default=0, type=int, help="Device for training")
    return parser


def main():
    """"""
    parser = build_parser()
    args = parser.parse_args()

    # Create output directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.outdir, "training.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Load the model
    pretrained_net = model_utils.load_model(args.encoder, device=args.device)

    # Load in some files
    rna_genes = utils.read_delimited_file(os.path.join(args.encoder, "rna_genes.txt"))
    atac_bins = utils.read_delimited_file(os.path.join(args.encoder, "atac_bins.txt"))

    # Read in the RNA
    rna_data_kwargs = copy.copy(sc_data_loaders.TENX_PBMC_RNA_DATA_KWARGS)
    rna_data_kwargs["fname"] = args.rnaCounts
    rna_data_kwargs["reader"] = lambda x: load_rna_files(
        x, args.encoder, transpose=not args.notrans
    )

    # Construct data folds
    train_valid_test_dsets = []
    for mode in ["all", "train", "valid", "test"]:
        # for mode in ["test"]:
        logging.info(f"Constructing {mode} dataset")
        sc_rna_dataset = sc_data_loaders.SingleCellDataset(mode=mode, **rna_data_kwargs)
        sc_rna_dataset.data_raw.write_h5ad(
            os.path.join(args.outdir, f"{mode}_rna.h5ad")
        )  # Write RNA input
        sc_atac_dummy_dataset = sc_data_loaders.DummyDataset(
            shape=len(atac_bins), length=len(sc_rna_dataset)
        )
        # RNA and fake ATAC
        sc_dual_dataset = sc_data_loaders.PairedDataset(
            sc_rna_dataset, sc_atac_dummy_dataset, flat_mode=True,
        )
        # encoded(RNA) as "x" and RNA + fake ATAC as "y"
        sc_rna_encoded_dataset = sc_data_loaders.EncodedDataset(
            sc_dual_dataset, model=pretrained_net, input_mode="RNA"
        )
        np.savetxt(
            os.path.join(args.outdir, f"{mode}_encoded.txt.gz"),
            sc_rna_encoded_dataset.encoded,
        )  # Write encoded
        sc_protein_dataset = sc_data_loaders.SingleCellProteinDataset(
            args.proteinCounts,
            obs_names=sc_rna_dataset.data_raw.obs_names,
            transpose=not args.notrans,
        )
        sc_protein_dataset.data_raw.write_h5ad(
            os.path.join(args.outdir, f"{mode}_protein.h5ad")
        )  # Write protein
        # x = 16 dimensional encoded layer, y = 25 dimensional protein array
        sc_rna_protein_dataset = sc_data_loaders.SplicedDataset(
            sc_rna_encoded_dataset, sc_protein_dataset
        )
        _temp = sc_rna_protein_dataset[0]  # ensure calling works
        train_valid_test_dsets.append(sc_rna_protein_dataset)
    _, sc_rna_prot_train, sc_rna_prot_valid, sc_rna_prot_test = train_valid_test_dsets
    x, y, z = sc_rna_prot_train[0], sc_rna_prot_valid[0], sc_rna_prot_test[0]
    assert (
        x[0].shape == y[0].shape == z[0].shape
    ), f"Got mismatched shapes: {x[0].shape} {y[0].shape} {z[0].shape}"
    assert (
        x[1].shape == y[1].shape == z[1].shape
    ), f"Got mismatched shapes: {x[1].shape} {y[1].shape} {z[1].shape}"

    protein_markers = list(sc_protein_dataset.data_raw.var_names)
    with open(os.path.join(args.outdir, "protein_proteins.txt"), "w") as sink:
        sink.write("\n".join(protein_markers) + "\n")
    assert len(
        utils.read_delimited_file(os.path.join(args.outdir, "protein_proteins.txt"))
    ) == len(protein_markers)
    logging.info(f"Predicting on {len(protein_markers)} proteins")

    if args.preprocessonly:
        return

    protein_decoder_skorch = skorch.NeuralNet(
        module=autoencoders.Decoder,
        module__num_units=16,
        module__num_outputs=len(protein_markers),
        module__final_activation=nn.Linear(
            len(protein_markers), len(protein_markers), bias=True
        ),  # Paper uses identity activation instead
        lr=1e-3,
        criterion=loss_functions.L1Loss,  # Other works use L1 loss
        optimizer=torch.optim.Adam,
        batch_size=512,
        max_epochs=args.epochs,
        callbacks=[
            skorch.callbacks.EarlyStopping(patience=25),
            skorch.callbacks.LRScheduler(
                policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                **model_utils.REDUCE_LR_ON_PLATEAU_PARAMS,
            ),
            skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
            skorch.callbacks.Checkpoint(
                dirname=args.outdir, fn_prefix="net_", monitor="valid_loss_best",
            ),
        ],
        train_split=skorch.helper.predefined_split(sc_rna_prot_valid),
        iterator_train__num_workers=8,
        iterator_valid__num_workers=8,
        device=utils.get_device(args.device),
    )
    protein_decoder_skorch.fit(sc_rna_prot_train, y=None)

    # Plot the loss history
    fig = plot_loss_history(
        protein_decoder_skorch.history, os.path.join(args.outdir, "loss.pdf")
    )


if __name__ == "__main__":
    main()
