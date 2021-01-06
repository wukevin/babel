"""
Script for predicting protein expression
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import skorch

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
import protein_utils
import utils

from predict_model import (
    load_atac_files_for_eval,
    load_rna_files_for_eval,
)


def build_parser():
    """Build commandline parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--babel", type=str, required=True, help="Path to babel model")
    parser.add_argument(
        "--protmodel", type=str, required=True, help="Path to latent-to-protein model"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--atac", type=str, nargs="*", help="Input ATAC")
    input_group.add_argument("--rna", type=str, nargs="*", help="Input RNA")
    parser.add_argument(
        "--liftHg19toHg38",
        action="store_true",
        help="Liftover input ATAC bins from hg19 to hg38 (only used for ATAC input)",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="csv file to output"
    )
    parser.add_argument("--device", default=1, type=int, help="Device for training")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    assert args.output.endswith(".csv")

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(args.output + ".log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Load the model
    babel = model_utils.load_model(args.babel, device=args.device)
    # Load in some related files
    rna_genes = utils.read_delimited_file(os.path.join(args.babel, "rna_genes.txt"))
    atac_bins = utils.read_delimited_file(os.path.join(args.babel, "atac_bins.txt"))

    # Load in the protein accesory model
    babel_prot_acc_model = protein_utils.load_protein_accessory_model(args.protmodel)
    proteins = utils.read_delimited_file(
        os.path.join(args.protmodel, "protein_proteins.txt")
    )

    # Get the encoded layer based on input
    if args.rna:
        (
            sc_rna_dset,
            _rna_genes,
            _marker_genes,
            _housekeeper_genes,
        ) = load_rna_files_for_eval(args.rna, checkpoint=args.babel, no_filter=True)
        sc_atac_dummy_dset = sc_data_loaders.DummyDataset(
            shape=len(atac_bins), length=len(sc_rna_dset)
        )
        sc_dual_dataset = sc_data_loaders.PairedDataset(
            sc_rna_dset,
            sc_atac_dummy_dset,
            flat_mode=True,
        )
        sc_dual_encoded_dataset = sc_data_loaders.EncodedDataset(
            sc_dual_dataset, model=babel, input_mode="RNA"
        )
        cell_barcodes = list(sc_rna_dset.data_raw.obs_names)
        encoded = sc_dual_encoded_dataset.encoded
    else:
        sc_atac_dset, _loaded_atac_bins = load_atac_files_for_eval(
            args.atac, checkpoint=args.babel, lift_hg19_to_hg39=args.liftHg19toHg38
        )
        sc_rna_dummy_dset = sc_data_loaders.DummyDataset(
            shape=len(rna_genes), length=len(sc_atac_dset)
        )
        sc_dual_dataset = sc_data_loaders.PairedDataset(
            sc_rna_dummy_dset, sc_atac_dset, flat_mode=True
        )
        sc_dual_encoded_dataset = sc_data_loaders.EncodedDataset(
            sc_dual_dataset, model=babel, input_mode="ATAC"
        )
        cell_barcodes = list(sc_atac_dset.data_raw.obs_names)
        encoded = sc_dual_encoded_dataset.encoded

    # Array of preds
    prot_preds = babel_prot_acc_model.predict(encoded.X)
    prot_preds_df = pd.DataFrame(
        prot_preds,
        index=cell_barcodes,
        columns=proteins,
    )
    prot_preds_df.to_csv(args.output)


if __name__ == "__main__":
    main()
