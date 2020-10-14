"""
Utility functions

Some functions live here because otherwise managing their import
in other places would be overly difficult
"""
import os
import sys
import logging
from typing import *
import itertools
import collections
import gzip

import numpy as np
import pandas as pd
import scipy
import scanpy as sc
from anndata import AnnData

import torch


def ensure_arr(x) -> np.ndarray:
    """Return x as a np.array"""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return x.toarray()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


def get_file_extension_no_gz(fname: str) -> str:
    """
    Get the filename extension (not gz)
    >>> get_file_extension_no_gz("foo.txt.gz")
    'txt'
    >>> get_file_extension_no_gz("foo.bar")
    'bar'
    >>> get_file_extension_no_gz("foo")
    ''
    """
    assert fname, f"Got empty input"
    retval = ""
    while fname and (not retval or retval == ".gz"):
        fname, ext = os.path.splitext(fname)
        if not ext:
            break  # Returns empty string
        if ext != ".gz":
            retval = ext
    return retval.strip(".")


def sc_read_multi_files(
    fnames: List[str], reader: Callable = sc.read_10x_h5
) -> AnnData:
    """Given a list of files, read the adata objects and concatenate"""
    for fname in fnames:
        assert os.path.isfile(fname), f"File does not exist: {fname}"
        # assert fname.endswith(".h5"), f"Unrecognized file type: {fname}"
    parsed = [reader(fname) for fname in fnames]
    for f, p in zip(fnames, parsed):
        logging.info(f"Read in {f} for {p.shape}")
    genomes_present = set(
        itertools.chain.from_iterable(
            [p.var["genome"] for p in parsed if "genome" in p.var]
        )
    )
    assert len(genomes_present) <= 1, f"Got more than one genome: {genomes_present}"
    for fname, p in zip(fnames, parsed):  # Make variable names unique
        p.var_names_make_unique()
        p.obs["source_file"] = fname
    retval = parsed[0]
    if len(parsed) > 1:
        retval = retval.concatenate(*parsed[1:])
    return retval


def sc_read_10x_h5_ft_type(fname: str, ft_type: str) -> AnnData:
    """Read the h5 file, taking only features with specified ft_type"""
    assert fname.endswith(".h5")
    parsed = sc.read_10x_h5(fname, gex_only=False)
    parsed.var_names_make_unique()
    assert ft_type in set(
        parsed.var["feature_types"]
    ), f"Given feature type {ft_type} not in included types: {set(parsed.var['feature_types'])}"

    retval = parsed[
        :,
        [n for n in parsed.var_names if parsed.var.loc[n, "feature_types"] == ft_type],
    ]
    return retval


def extract_file(fname, overwrite: bool = False) -> str:
    """Extracts the file and return the path to extracted file"""
    out_fname = os.path.abspath(fname.replace(".gz", ""))
    if os.path.isfile(out_fname):  # If the file already
        # If the file already exists and we aren't overwriting, do nothing
        if not overwrite:
            return out_fname
        os.remove(out_fname)
    with open(out_fname, "wb") as sink, gzip.GzipFile(fname) as source:
        sink.write(source.read())
    return out_fname


def read_delimited_file(
    fname: str, delimiter: str = "\n", comment: str = "#"
) -> List[str]:
    """Read the delimited (typically newline) file into a list"""
    with open(fname) as source:
        contents = source.read().strip()
        retval = contents.split(delimiter)
    # Filter out comment entries
    retval = [item for item in retval if not item.startswith(comment)]
    return retval


def isnotebook() -> bool:
    """
    Returns True if the current execution environment is a jupyter notebook
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_device(i: int = None) -> str:
    """Returns the i-th GPU if GPU is available, else CPU"""
    if torch.cuda.is_available() and isinstance(i, int):
        devices = list(range(torch.cuda.device_count()))
        device_idx = devices[i]
        torch.cuda.set_device(device_idx)
        d = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(d)
    else:
        d = torch.device("cpu")
    return d


def is_numeric(x) -> bool:
    """Return True if x is numeric"""
    try:
        x = float(x)
        return True
    except ValueError:
        return False


def is_all_unique(x: Iterable[Any]) -> bool:
    """
    Return whether the given iterable is all unique
    >>> is_all_unique(['x', 'y'])
    True
    >>> is_all_unique(['x', 'x', 'y'])
    False
    """
    return len(set(x)) == len(x)


def shifted_sigmoid(x, center: float = 0.5, slope: float = 25):
    """Compute a shifted sigmoid with configurable center and slope (steepness)"""
    return 1.0 / (1.0 + np.exp(slope * (-x + center)))


def unit_rescale(vals):
    """Rescale the given values to be between 0 and 1"""
    vals = np.array(vals).astype(float)
    denom = float(np.max(vals) - np.min(vals))
    retval = (vals - np.min(vals)) / denom
    assert np.alltrue(retval <= 1.0) and np.alltrue(retval >= 0.0)
    return retval


def split_df_by_col(df: pd.DataFrame, col: str) -> List[pd.DataFrame]:
    """Splits the dataframe into multiple dataframes by value of col"""
    unique_vals = set(df[col])

    retval = {}
    for v in unique_vals:
        df_sub = df[df[col] == v]
        retval[v] = df_sub
    return retval


def main():
    """On the fly testing"""


if __name__ == "__main__":
    get_file_extension_no_gz("foo")
    import doctest

    doctest.testmod()
    main()
