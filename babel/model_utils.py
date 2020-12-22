"""
Utility functions for working with models, including some callbacks
"""

import os
import sys
import logging
import argparse
import warnings
import inspect
import itertools
import copy
import collections

from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import skorch

import tqdm

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)
import loss_functions
import metrics
import utils
import autoencoders

DATA_LOADER_PARAMS = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": 6,
}

OPTIMIZER_PARAMS = {
    "lr": 1e-3,
}

REDUCE_LR_ON_PLATEAU_PARAMS = {
    "mode": "min",
    "factor": 0.1,
    "patience": 10,
    "min_lr": 1e-6,
}

DEVICE = utils.get_device()

ClassificationModelPerf = collections.namedtuple(
    "ModelPerf",
    [
        "auroc",
        "auroc_curve",
        "auprc",
        "accuracy",
        "recall",
        "precision",
        "f1",
        "ce_loss",
    ],
)
ReconstructionModelPerf = collections.namedtuple(
    "ReconstructionModelPerf", ["mse_loss"]
)


def recursive_to_device(t, device="cpu"):
    """Recursively transfer t to the given device"""
    if isinstance(t, tuple) or isinstance(t, list):
        return tuple(recursive_to_device(x, device=device) for x in t)
    return t.to(device)


def state_dict_to_cpu(d):
    """Transfer the state dict to CPU"""
    retval = collections.OrderedDict()
    for k, v in d.items():
        retval[k] = v.cpu()
    return retval


def load_model(
    checkpoint: str,
    input_dim1: int = -1,
    input_dim2: int = -1,
    prefix: str = "net_",
    device: str = "cpu",
):
    """Load the primary model, flexible to hidden dim, for evaluation only"""
    # Load the model
    device_parsed = device
    try:
        device_parsed = utils.get_device(int(device))
    except (TypeError, ValueError):
        device_parsed = "cpu"

    # Infer input dim sizes if they aren't given
    if input_dim1 is None or input_dim1 <= 0:
        rna_genes = utils.read_delimited_file(os.path.join(checkpoint, "rna_genes.txt"))
        input_dim1 = len(rna_genes)
        logging.info(f"Inferred RNA input dimension: {input_dim1}")
    if input_dim2 is None or (isinstance(input_dim2, int) and input_dim2 <= 0):
        atac_bins = utils.read_delimited_file(os.path.join(checkpoint, "atac_bins.txt"))
        chrom_counter = collections.defaultdict(int)
        for b in atac_bins:
            chrom = b.split(":")[0]
            chrom_counter[chrom] += 1
        # input_dim2 = list(chrom_counter.values())
        input_dim2 = [chrom_counter[c] for c in sorted(chrom_counter.keys())]
        logging.info(
            f"Inferred ATAC input dimension: {input_dim2} (sum={np.sum(input_dim2)})"
        )

    # Dynamically determine the model we are looking at based on name
    checkpoint_basename = os.path.basename(checkpoint)
    if checkpoint_basename.startswith("naive"):
        logging.info(f"Inferred model with basename {checkpoint_basename} to be naive")
        model_class = autoencoders.NaiveSplicedAutoEncoder
    else:
        logging.info(
            f"Inferred model with basename {checkpoint_basename} be normal (non-naive)"
        )
        model_class = autoencoders.AssymSplicedAutoEncoder

    spliced_net = None
    for hidden_dim_size in [16, 32]:
        try:
            spliced_net_ = autoencoders.SplicedAutoEncoderSkorchNet(
                module=model_class,
                module__input_dim1=input_dim1,
                module__input_dim2=input_dim2,
                module__hidden_dim=hidden_dim_size,
                # These don't matter because we're not training
                lr=0.01,
                criterion=loss_functions.QuadLoss,
                optimizer=torch.optim.Adam,
                batch_size=128,  # Reduced for memory saving
                max_epochs=500,
                # iterator_train__num_workers=8,
                # iterator_valid__num_workers=8,
                device=device_parsed,
            )
            spliced_net_.initialize()
            if checkpoint:
                cp = skorch.callbacks.Checkpoint(dirname=checkpoint, fn_prefix=prefix)
                spliced_net_.load_params(checkpoint=cp)
            else:
                logging.warn("Using untrained model")
            # Upon successfully finding correct hiden size, break out of loop
            logging.info(f"Loaded model with hidden size {hidden_dim_size}")
            spliced_net = spliced_net_
            break
        except RuntimeError as e:
            logging.info(f"Failed to load with hidden size {hidden_dim_size}")
    if spliced_net is None:
        raise RuntimeError("Could not infer hidden size")

    spliced_net.module_.eval()
    return spliced_net


def generate_classification_perf(truths, pred_probs, multiclass=False):
    """Given truths, and predicted probabilities, generate ModelPerf object"""
    pred_classes = np.round(pred_probs).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retval = ClassificationModelPerf(
            auroc=metrics.roc_auc_score(truths, pred_probs),
            auroc_curve=metrics.roc_curve(truths, pred_probs)
            if not multiclass
            else None,
            auprc=metrics.average_precision_score(truths, pred_probs),
            accuracy=metrics.accuracy_score(truths, pred_classes)
            if not multiclass
            else None,
            recall=metrics.recall_score(truths, pred_classes)
            if not multiclass
            else None,
            precision=metrics.precision_score(truths, pred_classes)
            if not multiclass
            else None,
            f1=metrics.f1_score(truths, pred_classes) if not multiclass else None,
            ce_loss=metrics.log_loss(truths, pred_probs, normalize=False)
            / np.prod(truths.shape),
        )
    return retval


def generate_reconstruction_perf(truths, preds):
    """Given truths and probs, generate appropriate perf object"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retval = ReconstructionModelPerf(
            mse_loss=metrics.mean_squared_error(truths, preds),
        )
        return retval


def skorch_grid_search(skorch_net, fixed_params: dict, search_params: dict, train_dset):
    """
    Perform parameter grid search using skorch API
    Note there is no valid dataset because that should be given in fixed_params
    """
    # Get the valid loss using .history[-1]['valid_loss']
    retval = {}
    for param_combo in itertools.product(*search_params.values()):
        param_combo_dict = dict(zip(search_params.keys(), param_combo))
        arg_dict = fixed_params.copy()
        arg_dict.update(param_combo_dict)

        net = skorch_net(**arg_dict)
        net.fit(train_dset)
        valid_loss = net.history[-1]["valid_loss"]
        retval[param_combo] = valid_loss
    return retval


def main():
    """On the fly debugging"""
    load_model(sys.argv[1])


if __name__ == "__main__":
    main()
