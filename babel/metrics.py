"""
Code for evaluating predicted/truth tables
"""
import sys
import os
import multiprocessing
from typing import *
import functools

import numpy as np
import pandas as pd

from sklearn.metrics import (
    adjusted_mutual_info_score,
    cluster,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    log_loss,
)

from anndata import AnnData

import adata_utils
import sc_data_loaders

GENERAL_CELLTYPES_MAPPING = {
    "CD4 T cells": [
        "Naive CD4 T",
        "T helper 17",
        "CD4+ T follicular helper",
        "Regulatory CD4+ T cells",
        "T helper 1",
        "CD4_T_cells",
        "Tregs",
    ],
    "CD8 T cells": [
        "Naive CD8 T",
        "Memory CD8 T",
        "CD8 TEx",
        "Effector CD8 T",
        "CD8_act_T_cells",
        "CD8_ex_T_cells",
        "CD8_mem_T_cells",
    ],
    "Dendritic cells": ["DCs"],
    "B cells": ["B", "Plasma B", "B_cells_1", "B_cells_2"],
    "Natural killer": ["Natural killer 1", "Natural killer 2", "NK_cells"],
    "Tumor": ["Tumor 1", "Tumor 2", "Tumor 3", "Tumor 4", "Tumor_1", "Tumor_2"],
}

SEMISPECIFIC_CELLTYPES_MAPPING = {
    "B cells": ["B_cells_1", "B_cells_2"],
    "Tumor": ["Tumor 1", "Tumor 2", "Tumor 3", "Tumor 4", "Tumor_1", "Tumor_2"],
    "Natural killer": ["Natural killer 1", "Natural killer 2"],
}


def _is_all_unique(x) -> bool:
    """
    Helper function for checking if everything is unique
    >>> _is_all_unique([1, 2, 3])
    True
    >>> _is_all_unique([1, 1, 45])
    False
    """
    return len(x) == len(set(x))


def confusion_matrix(
    x: Iterable[Any],
    y: Iterable[Any],
    unique_x: Iterable[Any] = None,
    unique_y: Iterable[Any] = None,
) -> pd.DataFrame:
    """
    Construct a confusion matrix using x and y
    """
    assert len(x) == len(y)
    if unique_x is None:
        unique_x = sorted(list(set(x)))
    assert _is_all_unique(unique_x)
    assert all([_ in unique_x for _ in x])
    if unique_y is None:
        unique_y = sorted(list(set(y)))
    assert _is_all_unique(unique_y)
    assert all(
        [_ in unique_y for _ in y]
    ), f"{set(y) - set(unique_y)} not found in unique_y"

    retval = pd.DataFrame(0, index=unique_x, columns=unique_y)
    for i, j in zip(x, y):
        retval.loc[i, j] += 1

    return retval


def pool_confusion_matrix(
    cmat: pd.DataFrame, pooling_dict: Dict[str, List[str]] = GENERAL_CELLTYPES_MAPPING
):
    """
    Pool the confusion matrix using the given dictionary mapping general term to a list
    of more specific terms
    """

    @functools.lru_cache(maxsize=128)
    def find_general_match(query: str) -> str:
        """Find general match if it exists, else passthrough query"""
        for k, v in pooling_dict.items():
            if query in v:
                return k
        return query

    def dedup_retain_order(l: Iterable[str]) -> List[str]:
        """Dedup the l, preserving order"""
        retval = []
        for item in l:
            if item not in retval:
                retval.append(item)
        return retval

    retval = pd.DataFrame(
        0,
        index=dedup_retain_order([find_general_match(i) for i in cmat.index]),
        columns=dedup_retain_order([find_general_match(i) for i in cmat.columns]),
    )
    for i in cmat.index:
        for j in cmat.columns:
            retval.loc[find_general_match(i), find_general_match(j)] += cmat.loc[i, j]
    return retval


def overlap_coef(x, y) -> float:
    """
    Size of intersection divided by smaller of two sets
    >>> overlap_coef([1, 2, 3], [1, 2, 3])
    1.0
    >>> overlap_coef([1, 2, 3], [1])
    1.0
    >>> overlap_coef([1, 2, 3], [2, 4])
    0.5
    """
    # https://en.wikipedia.org/wiki/Overlap_coefficient
    assert _is_all_unique(x) and _is_all_unique(y)
    numerator = len(set(x).intersection(set(y)))
    denominator = min(len(x), len(y))
    return numerator / denominator


def overlap_count(x, y) -> float:
    """
    Size of intersection
    """
    assert _is_all_unique(x) and _is_all_unique(y)
    return float(len(set(x).intersection(set(y))))


def mean_squared_error(truths, preds) -> float:
    squared = np.square(truths - preds)
    return np.mean(squared.flatten())


def jaccard_index(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return jaccard index (intersection / union)
    >>> jaccard_index(["A", "B", "C"], ["A", "B", "C"])
    1.0
    >>> np.round(jaccard_index(np.array([1, 2]), np.array([1, 3])), 4)
    0.3333
    >>> np.round(jaccard_index(['hi', 'there'], ['there', 'friend']), 4)
    0.3333
    """
    # https://en.wikipedia.org/wiki/Jaccard_index
    numer = np.intersect1d(x, y)
    denom = np.union1d(x, y)
    retval = numer.size / denom.size
    return retval


def top_n_closest_cell(preds, truth, top_n: int = 5, threads: int = 12) -> float:
    """
    How often do the top n closest cells in preds coincide with the top n closest cell
    in the ground truth (originally measured) data (measured by Jaccard index)?
    This is not differentiable
    """
    # The first (zeroth) entry in argosrt will be the cell itself so we ignore that
    truth_cell_dist_matrix = sc_data_loaders.cell_distance_matrix(truth)
    preds_cell_dist_matrix = sc_data_loaders.cell_distance_matrix(preds)

    pool = multiprocessing.Pool(threads)
    # argsort returns index of *lowest* value first
    # indices are ordered in decreasing similarity
    truth_closest_cells = np.vstack(pool.map(np.argsort, truth_cell_dist_matrix))
    preds_closest_cells = np.vstack(pool.map(np.argsort, preds_cell_dist_matrix))
    pool.close()
    pool.join()

    if isinstance(top_n, int):
        # First element is self
        preds_closest_cells = preds_closest_cells[:, 1 : 1 + top_n]
        truth_closest_cells = truth_closest_cells[:, 1 : 1 + top_n]

        d = np.array(
            [
                jaccard_index(x, y)
                for x, y in zip(truth_closest_cells, preds_closest_cells)
            ]
        )
        retval = np.mean(d)
    else:
        retval = []
        for n in top_n:
            preds_closest_cells = preds_closest_cells[:, 1 : 1 + n]
            truth_closest_cells = truth_closest_cells[:, 1 : 1 + n]
            d = np.array(
                [
                    jaccard_index(x, y)
                    for x, y in zip(truth_closest_cells, preds_closest_cells)
                ]
            )
            retval.append(np.mean(d))
        retval = np.array(retval)

    return retval


def marker_gene_overlap(
    preds_anndata: AnnData, truth_anndata: AnnData, n_genes: int = 10
) -> Dict[Tuple[int, int], float]:
    """
    Compute overlap in marker genes
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html
    cont_mat = pd.DataFrame(  # Rows contain truth label, columns are pred
        cluster.contingency_matrix(truth_anndata.obs.leiden, preds_anndata.obs.leiden)
    )
    truth_tots = cont_mat.sum(axis=1)
    cont_mat_proportion = (cont_mat.T / truth_tots).T
    assert np.all(np.isclose(cont_mat_proportion.sum(axis=1), 1.0))

    # Indices of truth/reference label preds with majority overlap
    major_overlap = np.where(cont_mat_proportion.max(axis=1) >= 0.5)

    # Compute marker genes for both anndata
    adata_utils.find_marker_genes(preds_anndata, n_genes=n_genes)
    adata_utils.find_marker_genes(truth_anndata, n_genes=n_genes)

    retval = {}
    for i in major_overlap[0]:
        j = np.argmax(cont_mat.iloc[i, :])
        assert cont_mat_proportion.iloc[i, j] >= 0.5
        truth_genes = [
            truth_anndata.uns["rank_genes_groups"]["names"][_i][i]
            for _i in range(n_genes)
        ]
        preds_genes = [
            preds_anndata.uns["rank_genes_groups"]["names"][_i][j]
            for _i in range(n_genes)
        ]

        ji = jaccard_index(preds_genes, truth_genes)
        retval[(i, j)] = ji
    return retval


def top_n_accuracy(preds: List[List[Any]], truth: List[Any], n: int = 0) -> float:
    """
    Computes the top n accuracy
    >>> top_n_accuracy([['x', 'y'], ['a', 'b']], ['x', 'z'])
    0.5
    >>> top_n_accuracy([['x', 'y'], ['a', 'b']], ['y', 'z'], n=1)
    0.0
    """
    assert len(preds) == len(truth)
    if n == 0:
        correct_vec = np.array([t in p for t, p in zip(truth, preds)])
    else:
        correct_vec = np.array([t in p[:n] for t, p in zip(truth, preds)])
    return np.mean(correct_vec)


def main():
    """Misc things"""
    # Benchmark a random 10,000 x 10,000 jaccard index
    print("Top N Jaccard Index random baseline scores")
    idx = np.arange(10000)
    for size in [10, 25, 50, 100]:
        ji_vals = [
            jaccard_index(
                np.random.choice(idx, size=size, replace=False),
                np.random.choice(idx, size=size, replace=False),
            )
            for _ in range(50000)
        ]
        print(size, np.mean(ji_vals), np.std(ji_vals))
    """
    > python commonspace/metrics.py
    Top N Jaccard Index random baseline scores
    10 0.0004994152046783625 0.005153194474261492
    25 0.001282654146765089 0.005101156184618113
    50 0.0025152103987486907 0.005033709044165715
    100 0.00504921845713569 0.00502535557096633
    """


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # main()
