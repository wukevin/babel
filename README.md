# BABEL

BABEL is a deep learning model written in Python designed to translate between mutliple single cell modalities. Currently, it is designed to translate between scATAC-seq and scRNA-seq profiles, though we show proof-of-concept of BABEL integrating additional modalities like proteomics. BABEL does this by learning encoder networks that can project these modalities into a shared latent representation, and decoder networks that can take this representation and reconstruct expression or chromatin accessibility profiles.

For more information, please see our peer-reviewed manuscript:

*[Wu, Kevin E., Kathryn E. Yost, Howard Y. Chang, and James Zou. "BABEL enables cross-modality translation between multiomic profiles at single-cell resolution." Proceedings of the National Academy of Sciences 118, no. 15 (2021).](https://doi.org/10.1073/pnas.2023070118)*

## Installation

We do not yet have a mechanism for "installing" BABEL directly from sources like `pip` or `conda` for the time being. Currently, BABEL is obtained by simply cloning the repository.

After cloning the repository, the necessary software dependencies (i.e. the environment) to run BABEL can be installed using `conda`:

```bash
conda env create -f environment.yml
```
This will create a new environment named `babel`. This environment needs to be activated via `conda activate babel` before running any of the code in this repository.

## Pre-trained model
We provide a human pre-trained BABEL model at the following [link](https://drive.google.com/file/d/1uJDbiDrBb5M0d9I5hjj2Ext-N08CXESS/view?usp=sharing) (md5sum `5e2f68466a1460a36e39a45229b21b1b`). Running `predict_model.py` (see below) will automatically donwload this pre-trained model (or use a cached copy) and use it to make predictions. You can also manually download this model, extract it, and supply the path to `bin/predict_model.py` using the `--checkpoint` parameter (see "Making predictions on new data" section below).

This provided model is trained on a set of peripheral blood mononuclear cells (PBMCs), colon adenocarcinoma COLO-320DM (DM) cells, colorectal adenocarcinoma COLO-320HSR (HSR) cells; as we discuss in the manuscript, BABEL performs best for cells that are related to these training cell types. Metrics such as psuedo-bulk concordance can be a litmus test for whether or not BABEL generalizes to a particular sample.

### Reproducing pre-trained model
To reproduce the pre-trained model, download the relevant training data at the following [link](https://office365stanford-my.sharepoint.com/:u:/g/personal/wukevin_stanford_edu/Edq1Cr6qejpOgzjZGa4bkvwB-LyH5MLbkLD6wGQCL4jvwA?e=T8IO54). There should be 5 h5 files in the tarball. Simply pass these 5 files to the training script:

```bash
python bin/train.py --data DM_rep4.h5 DM_rep8.h5 HSR_rep7.h5 HSR_rep8.h5 pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 --outdir my_model
```
See below for additional information regarding the training script.

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

Note that each input `h5` file must contain **both** RNA and ATAC paired modalities. In addition, these files should contain raw data (without preprocessing like size normalization), as these steps are performed automatically. For additional reference on formatting of these h5 inputs, please see the multi-omic h5 files available from 10x's website, or the h5 files included in the tarball under the "Reproducing pre-trained model" section above.

This training script will create a new directory `mymodel` that contains:

* `net_*` files, which contain the trained model parameters. Note that these, as well as the two txt files disussed below, are the only files that are required to run BABEL once it's been trained (see section below), so other files can be deleted/archived to save disk space.
* `rna_genes.txt` and `atac_bins.txt` describing the genes and peaks that BABEL has learned to predict.
* Various `*.h5ad` files containing the training, validation, and test data. These have the prefixes train/valid/truth, respectively.
* Various `*.h5ad` files containing the model's predictions on test data. These are named with the convention `inputMode_outputMode_testpreds.h5ad`. For example the file `atac_rna_test_preds.h5ad` contains the test set predictions when inferring RNA from ATAC.
	* ATAC predictions are probabilities that each peak is accessible, and are thus bound between 0 and 1. Note, however, that these probablities are not guaranteed to be well-calibrated.
	* RNA predictions continuous estimates of the expression of each gene in each cell, in linear (**not** log) space.
* Various `*.pdf` files that contain summary test set metrics such as correlation and AUROC.

This command will also generate a log file `mymodel_training.log` (outside of the output directory).

#### Training on SHARE-seq/SNARE-seq
Due to differences in file formats, the training code contains special logic for loading in these two experiments' data and training BABEL accordingly. 

For SNARE-seq use the `--snareseq` flag, for example:

```bash
python ~/projects/babel/bin/train_model.py --snareseq --outdir snareseq_model
```

For SHARE-seq, use the `--shareseq` flag along with keyword arguments to specify which SHARE-seq datasets to use, for example:

```bash
python ~/projects/babel/bin/train_model.py --shareseq skin --outdir shareseq_model
```

### Making predictions on new data
Once trained, BABEL can be used to generate new predictions using the following example command. This assumes that `mymodel` is the directory containing the trained BABEL model, and will create an output folder `myoutput`. Alternatively, you can also omit the `--checkpoint` parameter to automatically download and use the pre-trained human BABEL model described above.

```bash
python bin/predict_model.py --checkpoint mymodel --data data1.h5 data2.h5 --outdir myoutput
```
BABEL will try determine whether the input files contain ATAC or RNA (or both) input modalities, and will create its outputs in the folder `myoutput` accordingly:

* Various `*.h5ad` files containing the predictions. These are named with the convention `inputMode_outputMode_adata.h5ad`. For example the file `atac_rna_adata.h5ad` contains the RNA predictions from ATAC input.
* If given paired data, this script will also generate concordance metrics in `*.pdf` files with a similar naming convention. For example, `atac_rna_log.pdf` will contain a log-scaled scatterplot comparing measured and imputed expression values per gene per cell.

## Misc.
### What are `h5ad` files?
These files contain `AnnData` objects, a Python object/container designed to store single-cell data (https://anndata.readthedocs.io/en/latest/index.html). For those familiar with Python, these can be colloqially described as "Pandas DataFrames on steroids." For those more familiar with R, these are similar to Seurat objects. A brief Python code snippet to load in an AnnData object `mydata.h5ad` is shown below:

```python
import anndata as ad
x = ad.read_h5ad()
```

#### Converting to `h5ad` files
These `h5ad` files are convenient as they are self-contained datasets with metadata, and can thus be concisely given to BABEL as input (BABEL cannot handle giving, for example, 3 files that specify a datasets's counts, cell metadata, and gene metadata). If you have data that is in these separate formats, we provide a script to help convert them into `h5ad` files that are compatible with BABEL.

```bash
python bin/convert_to_adata.py foobar_genematrix.tsv.gz foobar.h5ad -t --obsinfo foobar_cell_annotations.csv --obscol 1
```
This command takes two positional arguments, the first being the counts matrix and the second being the `h5ad` file to write. The `-t` parameters toggles transposing the given input (BABEL expects input of cell x feature). Additional annotations can optionally be given by the `--obsinfo` and the `--varinfo` arguments, which specify files containing cell and feature metadata annotations, respectively. The `--obscol` and `--varcol` arguments specify the columns within the respective metadata files to use as the "names" of the respective files (e.g. cell names, gene/peak names).

### Additional commandline options
Both scripts for training and evaluation described above have many more options designed for advanced users, exposing functionality like exposing batch size, learning rate, etc. These options can be accessed by using the `-h` commandline flag; for example: `python bin/train.py -h`.

### Example usage and downstream analyses
Under the `jupyter` folder, we have included an example notebook that describes how to infer expression from scATAC-seq using BABEL. We then take BABEL's PBMC ATAC to RNA predictions and perform downstream analysis and visualization. This notebook generates BABEL visualizations shown in Figure 3 of our manuscript.
