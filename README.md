# BABEL

BABEL is a deep learning model written in Python designed to translate between mutliple single cell modalities. Currently, it is designed to translate between scATAC-seq and scRNA-seq profiles. It does so by learning encoder networks that can project these two modalities into a shared latent representation, and decoder networks that can take this representation and reconstruct expression or chromatin accessibility profiles.

For more information, please see our preprint: https://doi.org/10.1101/2020.11.09.375550

## Installation

We do not yet have a mechanism for "installing" BABEL for the time being. Currently, BABEL is obtained by simply cloning the repository.

After cloning the repository, the necessary software dependencies (i.e. the environment) to run BABEL can be installed using `conda`:

```bash
conda env create -f environment.yml
```
This will create a new environment named `babel`.

## Pre-trained model
We provide a pre-trained BABEL model at the following [link](https://office365stanford-my.sharepoint.com/:u:/g/personal/wukevin_stanford_edu/EeiPjchAkxVOrkJp109HKakB6MigU-VcxTLzwr0J8QEqrA?e=VxWF6s) (md5sum `5e2f68466a1460a36e39a45229b21b1b`). To use this model, extract it into a folder and supply the path to `bin/predict_model.py` using the `--checkpoint` parameter (see "Making predictions on new data" section below). This model is trained on a set of peripheral blood mononuclear cells (PBMCs), colon adenocarcinoma COLO-320DM (DM) cells, colorectal adenocarcinoma COLO-320HSR (HSR) cells; as we discuss in the manuscript, BABEL performs best for cells that are related to these training cell types. Metrics such as psuedo-bulk concordance can be a litmus test for whether or not BABEL generalizes to a particular sample.

## Usage

Before using BABEL, make sure to activate the environment that includes its dependencies by running:

```bash
conda activate babel
```

### Training
BABEL is trained using paired scRNA-seq/scATAC-seq measurements. An example command to train BABEL from scratch using .h5 data files `FILE1.h5` `FILE2.h5` containing joint ATAC/RNA profiles would then be:

```bash
python bin/train.py --data FILE1.h5 FILE2.h5 --outdir mymodel
```

Note that each input `h5` file must contain **both** RNA and ATAC paired modalities. In addition, these files should contain raw data (without preprocessing like size normalization), as these steps are performed automatically. This will create a new directory `mymodel` that contains:

* `net_*` files, which contain the trained model parameters. Note that these, as well as the two txt files disussed below, are the only files that are required to run BABEL once it's been trained (see section below), so other files can be deleted/archived to save disk space.
* `rna_genes.txt` and `atac_bins.txt` describing the genes and peaks that BABEL has learned to predict.
* Various `*.h5ad` files containing the training, validation, and test data. These have the prefixes train/valid/truth, respectively.
* Various `*.h5ad` files containing the model's predictions on test data. These are named with the convention `inputMode_outputMode_testpreds.h5ad`. For example the file `atac_rna_test_preds.h5ad` contains the test set predictions when inferring RNA from ATAC.
	* ATAC predictions are probabilities that each peak is accessible, and are thus bound between 0 and 1. Note, however, that these probablities are not guaranteed to be well-calibrated.
	* RNA predictions continuous estimates of the expression of each gene in each cell, in linear (**not** log) space.
* Various `*.pdf` files that contain summary test set metrics such as correlation and AUROC.

This command will also generate a log file `mymodel_training.log` (outside of the output directory).

### Making predictions on new data
Once trained, BABEL can be used to generate new predictions using the following example command. This assumes that `mymodel` is the directory containing the trained BABEL model, and will create an output folder `myoutput`.

```bash
python bin/predict_model.py --checkpoint mymodel --data data1.h5 data2.h5 --outdir myoutput
```
BABEL will try determine whether the input files contain ATAC or RNA (or both) input modalities, and will create its outputs in the folder `myoutput` accordingly:

* Various `*.h5ad` files containing the predictions. These are named with the convention `inputMode_outputMode_adata.h5ad`. For example the file `atac_rna_adata.h5ad` contains the RNA predictions from ATAC input.
* If given paired data, this script will also generate concordance metrics in `*.pdf` files with a similar naming convention. For example, `atac_rna_log.pdf` will contain a log-scaled scatterplot of 

## Misc.
### What are `h5ad` files?
These files contain `AnnData` objects, a Python object/container designed to store single-cell data (https://anndata.readthedocs.io/en/latest/index.html). For those familiar with Python, these can be colloqially described as "Pandas DataFrames on steroids." For those more familiar with R, these are similar to Seurat objects. A brief Python code snippet to load in an AnnData object `mydata.h5ad` is shown below:

```python
import anndata as ad
x = ad.read_h5ad()
```

### Additional commandline options
Both scripts for training and evaluation described above have many more options designed for advanced users, exposing functionality like exposing batch size, learning rate, etc. These options can be accessed by using the `-h` commandline flag; for example: `python bin/train.py -h`.

### Example downstream analyses
Under the `jupyter` folder, we have included an example notebook that takes BABEL's PBMC ATAC to RNA predictions, and performs downstream analysis and visualization. This notebook generates BABEL visualizations shown in Figure 3 of our manuscript.
