"""
Code to help with interpreting models

Some ideas
- The ATAC regions most influential for a gene should be mostly proximal
- Most naive solution might be to average out the inputs giving the output of interest Use SHAP GradientExplainer
"""

import os
import sys
import logging
from typing import *
import collections
import json

import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import sparse
import scanpy as sc
from anndata import AnnData

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

import tqdm

from utils import get_device, isnotebook, ensure_arr, is_numeric
import metrics

DEVICE = get_device()

# Combination of
# https://www.nature.com/articles/s41598-020-59827-1#MOESM1
#  - this is generally a superset of what Seurat uses
# https://satijalab.org/seurat/v3.1/pbmc3k_tutorial.html
PBMC_MARKER_GENES = {
    "CD4+ T cells": ["IL7R", "CD3D", "CD4", "CTLA4"],
    "IL7RCD4+ T Cells": ["CD8A", "IL7R", "CD3D"],  # Nature paper only
    "CD8+ T cells": ["CD8A", "GZMB", "CD3D", "CD8B"],  # Common both
    "B cells": ["CD19", "MS4A1", "CD79A", "CD79B", "BLNK"],  # Common both
    "Natural Killer cells": [
        "FCGR3A",
        "NCAM1",
        "KLRB1",
        "KLRC1",
        "KLRD1",
        "KLRF1",
        "GNLY",
        "NKG7",
    ],  # Common across both
    "CD14+ monocytes": ["CD14", "LYZ"],  # Common across both
    "Dendratic cells": [
        "IL3RA",
        "CLEC4C",
        "NRP1",
        "FCER1A",
        "CST3",
    ],  # Common across both
    "FCGR3A monocytes": ["FCGR3A", "MS4A7"],  # Common across both
    "Platelet": ["PPBP"],  # Only from Seurat
}

SEURAT_PBMC_MARKER_GENES = {
    "Naive CD4+ T": ["IL7R", "CCR7"],
    "Memory CD4+": ["IL7R", "S100A4"],
    "CD14+ Mono": ["CD14", "LYZ"],
    "B": ["MS4A1"],
    "CD8+ T": ["CD8A"],
    "FCGR3A+ Mono": ["FCGR3A", "MS4A7"],
    "NK cells": ["GNLY", "NKG7"],
    "Dendratic cells": ["FCER1A", "CST3"],
    "Platelet": ["PPBP"],
}

# https://www.nature.com/articles/s41591-019-0522-3
YOST_BCC_MARKER_GENES = {
    "T Cell": ["CD3G", "CD3D", "CD3E", "CD2"],
    "CD8+": ["CD8A", "GZMA"],
    "CD4+": ["CD4", "FOXP3"],
    "NK": ["KLRC1", "KLRC3"],
    "B Cell": ["CD19", "CD79A"],
    "Plasma": ["SLAMF7", "IGKC"],
    "Macrophage": ["FCGR2A", "CSF1R"],
    "Dendritic": ["FLT3"],
    "Plasmacytoid dendritic": ["CLEC4C"],
    "Fibroblast": ["COL1A2"],
    "Myofibroblast": ["MCAM", "MYLK"],
    "Cancer-related fibroblast": ["FAP", "PDPN"],
    "Malignant": ["EPCAM", "TP63"],
    "Endothelial": ["PECAM1", "VWF"],
    "Melanocytes": ["PMEL", "MLANA"],
}


def annotate_clusters_to_celltypes(
    adata: AnnData,
    marker_genes: Union[Dict[str, List[str]], str],
    groupby: str = "leiden",
    method: str = "overlap_coef",
    min_score: float = 0.0,
    store_key_suffix: str = "",
    simplify_output: bool = False,
) -> pd.DataFrame:
    """
    Annotate each cell with a celltype, modify adata.obs in place, and return coef matrix
    Modification is adding the key <GROUPBY>_celltypes(_<STOREKEYSUFFIX>)
    """
    # These function mappings are useful for building a random baseline
    method_str_to_func = {
        "overlap_coef": metrics.overlap_coef,
        "overlap_count": metrics.overlap_count,
        "jaccard": metrics.jaccard_index,
    }
    assert method in method_str_to_func.keys(), f"Unrecognized method: {method}"

    uns_key = f"rank_genes_{groupby}"
    if isinstance(marker_genes, str):
        assert os.path.isfile(marker_genes), f"Is not a file: {marker_genes}"
        assert marker_genes.endswith(".json")
        logging.info(f"Reading marker genes from {marker_genes}")
        with open(marker_genes) as source:
            marker_genes = json.load(source)

    # Build a baseline of what overlap a random sampling would produce
    np.random.seed(1234)
    baseline_scores = {}  # Maps length of marker gene list to random overlaps
    for l in set([len(genes) for genes in marker_genes.values()]):
        pseudo_markers = np.random.choice(list(adata.var_names), size=l, replace=False)
        num_deseq_genes = len(adata.uns[uns_key]["names"])
        baseline_scores[l] = np.array(
            [
                method_str_to_func[method](
                    np.random.choice(
                        list(adata.var_names), size=num_deseq_genes, replace=False
                    ),
                    pseudo_markers,
                )
                for _i in range(1000)
            ]
        )
    baseline_thresholds = {k: np.percentile(v, 95) for k, v in baseline_scores.items()}
    celltype_thresholds = np.array(
        [baseline_thresholds[len(l)] for l in marker_genes.values()]
    )

    marker_matches = sc.tl.marker_gene_overlap(
        adata, marker_genes, method=method, key=uns_key
    )

    # Determine mapping of cluster index to celltype
    idx_to_celltype = {}
    for idx in marker_matches.columns:
        per_celltype_match_scores = marker_matches.loc[:, idx]
        # If does not exceed random cutoff, set to 0
        per_celltype_match_scores[
            np.where(per_celltype_match_scores < celltype_thresholds)[0]
        ] = 0.0
        idx_to_celltype[idx] = (
            per_celltype_match_scores.idxmax()
            if np.max(per_celltype_match_scores) > 0
            else "Unmatched"
        )
        logging.info(f"Mapping index {idx} -> {idx_to_celltype[idx]}")

    logging.info(f"Saving annotated celltypes to {groupby}_celltypes")
    key_to_add = f"{groupby}_celltypes"
    if store_key_suffix:
        key_to_add += "_" + store_key_suffix
    adata.obs[key_to_add] = [idx_to_celltype[idx] for idx in adata.obs[groupby]]

    if simplify_output:
        zero_cols = marker_matches.columns[marker_matches.sum(axis=0) == 0]
        logging.info(f"Dropping: {zero_cols}")
        marker_matches.drop(columns=zero_cols, inplace=True)

    return marker_matches


def reformat_marker_genes_to_dict(
    adata: AnnData,
    split_by: str = "leiden",
    padj_cutoff: float = 0.05,
    score_cutoff: float = 0.35,
    top_n: int = 0,
) -> Dict[str, List[str]]:
    """
    Take the (somewhat convoluted) data structure used to store marker genes
    and store as a plain dictionary, excluding pvals that exceed cutoff
    """
    # https://stackoverflow.com/questions/8530670/get-recarray-attributes-columns-python
    marker_dict = adata.uns[f"rank_genes_{split_by}"]
    d = collections.defaultdict(list)
    # Top level if array is for each of the N marker genes
    using_scores = False
    if "pvals_adj" in marker_dict:
        for _i, (level_list, level_pvals) in enumerate(
            zip(marker_dict["names"], marker_dict["pvals_adj"])
        ):
            cluster_names = level_pvals.dtype.names
            assert len(level_list) == len(level_pvals) == len(cluster_names)
            assert len(level_list) == len(d) == len(level_pvals)
            for group_id, gene, pval in zip(cluster_names, level_list, level_pvals):
                if pval < padj_cutoff:
                    d[group_id].append((pval, gene))
    else:
        logging.warn("p-values not found in marker gene struct")
        using_scores = True
        for _i, (level_list, level_scores) in enumerate(
            zip(marker_dict["names"], marker_dict["scores"])
        ):
            cluster_names = level_scores.dtype.names
            assert len(level_list) == len(level_scores) == len(cluster_names)
            for group_id, gene, score in zip(cluster_names, level_list, level_scores):
                if score >= score_cutoff:
                    d[group_id].append((score, gene))
    retval = {}
    if top_n:
        for k, v in d.items():
            # Sort by default is small -> big
            v_sorted = sorted(v)
            if using_scores:  # Reverse to big -> small
                v_sorted = v_sorted[::-1]
            retval[k] = [m[1] for m in v_sorted[:top_n]]
    else:
        retval = {k: [m[1] for m in v] for k, v in d.items()}
    return retval


def split_dataset_by_pred(
    func: Callable,
    dataset: Dataset,
    target: int,
    threshold: Union[float, Callable] = 0.0,
    device: str = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset, split it by the model predictions on the ith example
    Returns a tuple of baseline and positive examples
    Useful for doing something that isn't just the naive 0 baseline
    """
    pred_values, examples = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            single_batch = dataset[i][0].view(1, -1)
            assert single_batch.shape[0] == 1
            preds = func(single_batch.to(device)).cpu().numpy().flatten()
            assert len(preds.shape) == 1, "Must be one dimensional"

            pred_values.append(preds[target])
            examples.append(single_batch)
    pred_values = np.array(pred_values)
    examples = torch.cat(examples)

    # Determine cutoff
    if callable(threshold):
        cutoff = float(threshold(pred_values))
    else:
        cutoff = threshold
    assert isinstance(cutoff, float)

    pos_examples = examples[np.where(pred_values >= cutoff)]
    baseline_examples = examples[np.where(pred_values < cutoff)]
    return baseline_examples, pos_examples


def interpret_atac_to_rna(
    model: nn.Module,
    inputs: torch.Tensor,
    target: int,
    baseline: torch.Tensor = None,
    interpret_class=IntegratedGradients,
    progressbar: bool = False,
    device=DEVICE,
    batch_size: int = 8,
) -> sparse.csr_matrix:
    """
    For each output gene in RNA space, in the given single data point, perform
    interpretation
    Target is given as RNA index (ATAC is ignored in output space)
    """
    model.eval()
    model = model.to(device)

    if baseline is None:
        baseline = torch.zeros_like(inputs[0, :]).view(1, -1).to(device)
    model_forward = lambda x: model.forward(x, mode=(2, 1))[0]
    # 2 > 1 is ATAC > RNA mode
    explainer = interpret_class(forward_func=model_forward)

    # Establish index pairs
    baseline_indices = list(range(0, baseline.shape[0], batch_size))
    baseline_index_pairs = list(zip(baseline_indices[:-1], baseline_indices[1:]))
    if not baseline_index_pairs:
        baseline_index_pairs = [(0, baseline.shape[0])]
    inputs_indices = list(range(0, inputs.shape[0], batch_size))
    inputs_index_pairs = list(zip(inputs_indices[:-1], inputs_indices[1:]))
    if not inputs_index_pairs:
        inputs_index_pairs = [(0, inputs.shape[0])]

    # Loop through the examples and do attributions
    attrs = []
    pbar = tqdm.tqdm_notebook if isnotebook() else tqdm.tqdm
    for baseline_start, baseline_stop in pbar(
        baseline_index_pairs, disable=not progressbar
    ):
        for input_start, input_stop in inputs_index_pairs:
            attributions = (
                explainer.attribute(
                    inputs[input_start:input_stop].to(device),
                    baseline[baseline_start:baseline_stop].to(device),
                    target=target,
                )
                .cpu()
                .numpy()
            )
            attr_sparse = sparse.csr_matrix(attributions)
            # delta = delta.cpu().numpy()
            attrs.append(attr_sparse)
    # attrs = np.vstack(attrs)
    attrs = sparse.vstack(attrs)

    return attrs


def split_preds_by_chrom(
    x: Union[pd.Series, pd.DataFrame]
) -> Dict[str, Union[List[float], pd.DataFrame]]:
    """Split predictions by chrom. Assume that chr is labelled as such"""
    retval = collections.defaultdict(list)
    for i, val in x.iteritems():
        if not i.startswith("chr"):
            continue
        chrom, _pos = i.split(":")
        retval[chrom].append(val)
    if isinstance(x, pd.DataFrame):
        retval = {k: pd.DataFrame(v).T for k, v in retval.items()}
    return retval


def _interval_distance(x: Tuple[int, int], y: Tuple[int, int]) -> int:
    """
    Helper function for returning the absolute distance between intervals

    >>> _interval_distance((1, 3), (4, 5))
    1
    >>> _interval_distance((1, 3), (2, 5))
    0
    >>> _interval_distance((1, 10), (10, 15))
    0
    >>> _interval_distance((100, 1000), (0, 101))
    0
    >>> _interval_distance((1, 10), (2, 5))
    0
    >>> _interval_distance((100, 200), (0, 10))
    90
    >>> _interval_distance((170476, 170976), (126383, 542503))
    0
    """
    x_start, x_end = x
    assert x_start < x_end
    y_start, y_end = y
    assert y_start < y_end

    if (
        x_start <= y_start <= x_end
        or x_start <= y_end <= x_end
        or y_start <= x_start <= y_end
        or y_start <= x_end <= y_end
    ):
        return 0
    elif x_start > y_end:
        return np.abs(x_start - y_end)
    elif x_end < y_start:
        return np.abs(y_start - x_end)
    else:
        raise NotImplementedError(f"Cannot compare: {x} {y}")


def split_preds_proximal_distant(
    x: pd.Series, pos: Tuple[str, int, int], window: int = 50000
) -> Tuple[List[float], List[float]]:
    """
    Split predictions into those that are proximal and those that are distant
    These are defined by being within <window> of the <pos>
    """
    ref_chrom, ref_start, ref_end = pos
    assert ref_start < ref_end
    if not ref_chrom.startswith("chr"):
        ref_chrom = "chr" + ref_chrom

    proximal, distant = [], []
    for i, val in x.iteritems():
        if not i.startswith("chr"):  # Maybe extend to genes later
            continue
        chrom, span = i.split(":")
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        start, end = map(int, span.split("-"))
        assert start < end
        if chrom != ref_chrom:
            distant.append(val)
        elif _interval_distance((start, end), (ref_start, ref_end)) <= window:
            proximal.append(val)
        else:
            distant.append(val)

    return proximal, distant


def z_score_mat(
    mat: Union[sparse.csr_matrix, sparse.csc_matrix, np.ndarray],
    center: float = 0,
    axis: int = 0,
) -> np.ndarray:
    """
    Converts matrix of values to z-scores
    """
    if axis not in (0, 1):
        raise ValueError(f"Invalid value for axis: {axis}")
    if axis == 0:
        mat_comp = sparse.csc_matrix(mat) if isinstance(mat, sparse.csr_matrix) else mat
        sds = np.array(
            [
                np.std(mat_comp.getcol(i).toarray()).flatten()
                for i in range(mat_comp.shape[1])
            ]
        ).flatten()
        assert sds.size == mat.shape[1]
    else:
        mat_comp = sparse.csr_matrix(mat) if isinstance(mat, sparse.csc_matrix) else mat
        sds = np.array(
            [
                np.std(mat_comp.getrow(i).toarray()).flatten()
                for i in range(mat_comp.shape[0])
            ]
        ).flatten()
        assert sds.size == mat.shape[0]
    means = mat.mean(axis=axis).flatten()
    retval = (means / sds).flatten()
    retval[np.where(np.isnan(retval))] = 0.0
    if isinstance(retval, np.matrix) or not isinstance(retval, np.ndarray):
        retval = np.squeeze(np.asarray(retval))
    return retval


def reformat_rules(rules: Iterable[str], orig_features: Iterable[str]) -> List[str]:
    """Reformat so that something like 'feature_4' becomes readable"""
    retval = []
    for r in rules:
        tokens = r.split(" ")
        for i, t in enumerate(tokens):
            if t.startswith("feature_"):
                ft_id = int(t.strip("feature_"))
                tokens[i] = orig_features[ft_id]
        retval.append(" ".join(tokens))
    return retval


def involved_features_from_rules(rules: Iterable[str]) -> Set[str]:
    """Extract the features involved in the rules"""
    known_tokens = set(["<=", "&", ">"])
    retval = set()
    for r in rules:
        tokens = r.split(" ")
        for t in tokens:
            if t in known_tokens or is_numeric(t):
                continue
            retval.add(t)
    return retval


if __name__ == "__main__":
    import doctest

    doctest.testmod()
