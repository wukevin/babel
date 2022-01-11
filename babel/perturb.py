import logging
import functools

import numpy as np
from scipy import sparse

import anndata as ad


def _csr_swap_in_row(
    row: sparse.csr_matrix, rng: np.random.Generator, p: float = 0.1
) -> sparse.csr_matrix:
    """
    Helper function for swapping nonzero values in a given row
    """
    assert row.shape[0] == 1, f"Did not get a row!"
    nonzero_idx = row.nonzero()[1]
    shuffle_idx = np.arange(len(nonzero_idx))
    # Randomly choose a proportion of the nonzero indices to shuffle
    n = int(round(len(shuffle_idx) * p))
    swap_idx = nonzero_idx[rng.choice(shuffle_idx, size=n, replace=False)]
    # Shuffle the indices we chose above
    dest_idx = rng.choice(swap_idx, size=len(swap_idx), replace=False)
    assert swap_idx.shape == dest_idx.shape

    arr = row.toarray().squeeze()
    assert np.all(arr[swap_idx] != 0)
    arr[dest_idx] = arr[swap_idx]
    retval = sparse.csr_matrix(arr)
    return retval


def _csr_swap_zero_nonzero_in_row(
    row: sparse.csr_matrix, rng: np.random.Generator, p: float = 0.1
) -> sparse.csr_matrix:
    """
    Swap 0 and nonzero values
    Typically, most values are 0, so we do this for p * num_zero entries
    This is exact, meaning we never swap fewer or more values
    """
    assert row.shape[0] == 1
    nonzero_idx = row.nonzero()[1]
    arr = row.toarray().squeeze()
    zero_idx = np.where(arr == 0)[0]
    # Because # nonzero << # zero, we use # nonzero to determine number of swaps
    n = int(round(len(nonzero_idx) * p))
    # Choose indices to swap
    zero_idx_swap = rng.choice(zero_idx, n, replace=False)
    nonzero_idx_swap = rng.choice(nonzero_idx, n, replace=False)
    # Transfer nonzero values to selected "aero" indices
    arr[zero_idx_swap] = arr[nonzero_idx_swap]
    # Zero out the original values at the nonzero indices
    arr[nonzero_idx_swap] = 0
    retval = sparse.csr_matrix(arr)
    assert retval.shape == row.shape
    assert len(retval.nonzero()[1]) == len(nonzero_idx)
    return retval


def swap_adata(
    adata: ad.AnnData,
    p: float = 0.1,
    mode: str = "zero_nonzero",
    copy: bool = True,
    seed: int = 6489,
) -> ad.AnnData:
    """
    Randomly swap the nonzero values from this proportion of cells
    Swapping is done within each example
    """
    rng = np.random.default_rng(seed)

    logging.info(f"Swapping {p} of values in each observation")
    if mode == "zero_zero":
        pfunc = functools.partial(_csr_swap_in_row, rng=rng, p=p)
    elif mode == "zero_nonzero":
        pfunc = functools.partial(_csr_swap_zero_nonzero_in_row, rng=rng, p=p)
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    swapped_rows = sparse.vstack([pfunc(row) for row in adata.X])
    assert swapped_rows.shape == adata.shape

    retval = adata.copy() if copy else adata
    retval.X = swapped_rows
    return retval
