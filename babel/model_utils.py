"""
Utility functions for working with models, including some callbacks
"""

import os
import sys
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

import metrics
import utils

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


def build_parser():
    """Build commandline parser"""
    parser = argparse.ArgumentParser(
        description="Commandline model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_py_file", help="py file specifying architecture of model"
    )
    parser.add_argument("")
    return parser


def main():
    """For training models from commandline"""
    parser = build_parser()
    args = parser.parse_args()


if __name__ == "__main__":
    main()
