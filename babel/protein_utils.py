"""
Utility functions for working with protein model
"""
import os, sys
import json

import torch
import torch.nn as nn
import skorch

import loss_functions
import model_utils
import utils

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)
import autoencoders

LOSS_DICT = {"L1": loss_functions.L1Loss, "MSE": loss_functions.MSELoss}
OPTIM_DICT = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
ACT_DICT = {"prelu": nn.PReLU, "relu": nn.ReLU, "relu6": nn.ReLU6, "tanh": nn.Tanh}


def load_protein_accessory_model(dirname: str) -> skorch.NeuralNet:
    """Loads the protein accessory model"""
    predicted_proteins = utils.read_delimited_file(
        os.path.join(dirname, "protein_proteins.txt")
    )
    with open(os.path.join(dirname, "params.json")) as source:
        model_params = json.load(source)

    encoded_to_protein_skorch = skorch.NeuralNet(
        module=autoencoders.Decoder,
        module__num_units=16,
        module__intermediate_dim=model_params["interdim"],
        module__num_outputs=len(predicted_proteins),
        module__final_activation=nn.Identity(),
        module__activation=ACT_DICT[model_params["act"]],
        # module__final_activation=nn.Linear(
        #     len(predicted_proteins), len(predicted_proteins), bias=True
        # ),  # Paper uses identity activation instead
        lr=model_params["lr"],
        criterion=LOSS_DICT[model_params["loss"]],  # Other works use L1 loss
        optimizer=OPTIM_DICT[model_params["optim"]],
        batch_size=model_params["bs"],
        max_epochs=500,
        callbacks=[
            skorch.callbacks.EarlyStopping(patience=25),
            skorch.callbacks.LRScheduler(
                policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                **model_utils.REDUCE_LR_ON_PLATEAU_PARAMS,
            ),
            skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
        ],
        iterator_train__num_workers=8,
        iterator_valid__num_workers=8,
        device="cpu",
    )
    encoded_to_protein_skorch_cp = skorch.callbacks.Checkpoint(
        dirname=dirname, fn_prefix="net_"
    )
    encoded_to_protein_skorch.load_params(checkpoint=encoded_to_protein_skorch_cp)
    return encoded_to_protein_skorch


if __name__ == "__main__":
    print(load_protein_accessory_model(sys.argv[1]))
