# BABEL

BABEL is a deep learning model designed to translate between mutliple single cell modalities. Currently, it is designed to translate between scATAC-seq and scRNA-seq profiles. It does so by learning encoder networks that can project these two modalities into a shared latent representation, and decoder networks that can take this representation and reconstruct expression or chromatin accessibility profiles.

## Installation

We do not yet have a mechanism for "installing" BABEL for the time being. Currently, BABEL is obtained by simply cloning the repository.

After cloning the repository, the necessary software dependencies (i.e. the environment) to run BABEL can be installed using:

```bash
conda create -f environment.yml
```

## Usage

Before using BABEL, make sure to activate the environment that includes its dependencies by running:

```bash
conda activate babel
```

An example command to train BABEL from scratch using .h5 data files `FILE1.h5` `FILE2.h5` containing joint ATAC/RNA profiles would then be:

```bash
python bin/train.py --data FILE1.h5 FILE2.h5 --outdir mymodel
```
