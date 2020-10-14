"""
Code to combine a bunch of plaintext RNA files

Takes as input a series of filename *prefixes*

Example usage:
python combine_plaintext_atac_files.py GSM4119513 GSM4119514 GSM4119515 GSM4119516 GSM4119517 GSM4119518 GSM4119519 -o output.h5ad
"""

import os
import sys
import argparse
from typing import *
import multiprocessing
import functools
import logging
import glob
import gzip

logging.basicConfig(level=logging.INFO)

import pandas as pd
import scipy
import scanpy as sc
import anndata

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import sc_data_loaders


def separate_trio_files(trio: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """
    Organize the trio of files such that they are in the order:
    barcodes, peaks, matrix
    """
    barcodes_files = [f for f in trio if "barcodes" in f]
    assert len(barcodes_files) == 1
    peaks_files = [f for f in trio if "peaks" in f]
    assert len(peaks_files) == 1
    mat_files = [f for f in trio if "matrix" in f]
    assert len(mat_files) == 1
    return barcodes_files.pop(), peaks_files.pop(), mat_files.pop()


def read_barcodes(fname: str) -> List[str]:
    """Read the barcodes file"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        retval = [l.strip() for l in source]
        retval = [l.decode() if isinstance(l, bytes) else l for l in retval]
        return retval


def read_peaks(fname: str) -> List[str]:
    """Read the peaks file"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        tokens = [l.strip().decode() for l in source if l]
        tokens = [l.decode() if isinstance(l, bytes) else l for l in tokens]
        return tokens


def read_prefix(prefix: str) -> sc.AnnData:
    """
    Helper function for reading in a prefix
    """
    matches = glob.glob(prefix + "*")
    assert len(matches) == 3, f"Got unexpected matches with prefix {prefix}"

    barcodes_file, peaks_file, mat_file = separate_trio_files(matches)
    barcodes = read_barcodes(barcodes_file)
    logging.info(f"Read {len(barcodes)} barcodes from {barcodes_file}")

    peaks = read_peaks(peaks_file)
    logging.info(f"Read {len(peaks)} peaks from {peaks_file}")

    adata = sc.AnnData(
        scipy.sparse.csr_matrix(scipy.io.mmread(mat_file).T),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=peaks),
    )
    return adata


def build_parser():
    """Build commandline argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prefix",
        nargs="*",
        type=str,
        help="File prefixes denoting the files to combine",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output file to write to"
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=int(multiprocessing.cpu_count() / 2)
    )
    return parser


def main():
    """Run main body of the script"""
    parser = build_parser()
    args = parser.parse_args()
    assert args.output.endswith(".h5ad"), "Output file must be in .h5ad format"
    threads = min(args.threads, len(args.prefix))

    # Read in all the prefixes
    pool = multiprocessing.Pool(threads)
    adatas = list(pool.map(read_prefix, args.prefix))
    pool.close()
    pool.join()

    # After having read in all the files, aggregate them
    common_bins = adatas[0].var_names
    for adata in adatas[1:]:
        common_bins = sc_data_loaders.harmonize_atac_intervals(
            common_bins, adata.var_names
        )

    logging.info(f"Aggregated {len(args.prefix)} prefixes into {len(common_bins)} bins")

    pfunc = functools.partial(sc_data_loaders.repool_atac_bins, target_bins=common_bins)
    pool = multiprocessing.Pool(threads)
    adatas = list(pool.map(pfunc, adatas))
    pool.close()
    pool.join()

    retval = adatas[0]
    if len(adatas) > 1:
        retval = retval.concatenate(adatas[1:])
    logging.info(
        f"Concatenated {len(args.prefix)} prefixes into a single adata of {retval.shape}"
    )

    logging.info(f"Writing to {args.output}")
    retval.write(args.output)


if __name__ == "__main__":
    main()
