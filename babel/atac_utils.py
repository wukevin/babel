"""
Utilities for working with ATAC data, including code for calculating
gene activities, which serve as a baseline for our work
"""

import os
import sys
import logging
import gzip
import collections
import argparse
from typing import *

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy

import tqdm
import intervaltree as itree

import sc_data_loaders
import genomic_interval
import utils

logging.basicConfig(level=logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
assert os.path.isdir(DATA_DIR)
HG38_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.100.gtf.gz")
assert os.path.isfile(HG38_GTF)


def binarize_preds(
    preds: np.ndarray,
    raw: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
) -> np.ndarray:
    """
    Binarize the predicted matrix of floats based on the following:
    - 1 if imputed_ij > mean(raw[i, ]) and imputed_ij > mean(raw[:, j])
    - 0 otherwise

    Based on SCALE https://www.nature.com/articles/s41467-019-12630-7#Sec10
    """
    if not isinstance(raw, np.ndarray):
        raw = raw.toarray()  # Consistent runtime for row and col ops
    assert np.array_equal(raw, raw.astype(np.bool))

    row_means = np.mean(raw, axis=1)
    col_means = np.mean(raw, axis=0)

    row_passes = (preds.T > row_means).T
    col_passes = preds > col_means

    retval = np.logical_and(row_passes, col_passes)
    return retval


def fragments_from_frag_tsv(
    fname: str, return_duplicates: bool = False
) -> Iterable[Tuple[str, genomic_interval.GenomicInterval]]:
    """
    Given a fragments tsv file, return a generator of its fragments
    Fragments file format is 5 columns: chrom, start, stop, barcode, duplicate_count
    """
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            line = line.decode().strip()
            chrom, start, stop, barcode, dup_count = line.split("\t")
            gi = genomic_interval.GenomicInterval((chrom, start, stop))
            if return_duplicates:
                for i in range(int(dup_count)):
                    yield barcode, gi
            else:
                yield barcode, gi
    raise StopIteration


def closest_feature(
    query_regions: List[str], gtf_file: str = HG38_GTF, max_dist: int = 100000
) -> pd.DataFrame:
    """
    Find the closest feature to each of the given queries
    Very loosely based on: https://satijalab.org/signac/reference/ClosestFeature.html
    Returns a dataframe with columns:
    match_tx_id, match_gene_name, match_gene_id, match_gene_biotype, match_type, match_closest_region, distance
    """
    matched_rows = []
    reference_features = utils.read_gtf_pos_to_features()
    for region in query_regions:
        region_gi = genomic_interval.GenomicInterval(region)
        matches = list(
            reference_features[region_gi.chrom].overlap(
                region_gi.start - max_dist, region_gi.stop + max_dist
            )
        )
        if not matches:
            continue
        matches_gi = [
            genomic_interval.GenomicInterval((region_gi.chrom, m.begin, m.end))
            for m in matches
        ]
        distances = np.array([region_gi.difference(m) for m in matches_gi])
        min_dist = np.min(distances)
        min_indices = np.where(distances == min_dist)[0]
        for i in min_indices:
            # Cast as defaultdict to automatically return null when key missing
            match_coords = matches[i].begin, matches[i].end
            match = collections.defaultdict(str, matches[i].data)
            matched_rows.append(
                (
                    match["transcript_id"],
                    match["gene_name"],
                    match["gene_id"],
                    match["gene_biotype"],
                    match["entry_type"],
                    f"{region_gi.chrom}-{match_coords[0]}:{match_coords[1]}",
                    str(region_gi),
                    min_dist,
                )
            )
    retval = pd.DataFrame(
        matched_rows,
        columns=[
            "tx_id",
            "gene_name",
            "gene_id",
            "gene_biotype",
            "type",
            "closest_region",
            "query_region",
            "distance",
        ],
    )
    return retval


def gene_activity_matrix_from_frags(
    frag_file: str,
    annotation: str = sc_data_loaders.HG38_GTF,
    size_norm: bool = False,
) -> ad.AnnData:
    """
    Create a gene activity matrix
    Take the annotation, extend to 2kb upstream (strand dependent)
    Count number of overlapping fragments
    Source: https://satijalab.org/signac/articles/pbmc_vignette.html
    """
    gene_to_pos = utils.read_gtf_gene_to_pos(annotation, extend_upstream=2000)
    # Dict of chrom (without chr prefix) to intervaltree
    gene_intervaldict = sc_data_loaders.gene_pos_dict_to_range(gene_to_pos)

    # Intermediate data structure mapping a dok to a int
    cnt = collections.defaultdict(int)
    opener = gzip.open if frag_file.endswith(".gz") else open
    with opener(frag_file) as source:
        for i, line in tqdm.tqdm(enumerate(source)):
            line = line.decode().strip()
            tokens = line.split()
            chrom, start, stop, barcode, dup_count = tokens
            if chrom.startswith("chr"):
                chrom = chrom.strip("chr")
            start, stop, dup_count = int(start), int(stop), int(dup_count)
            assert start <= stop, f"Anomalous fragment: {line}"
            overlaps = gene_intervaldict[chrom].overlap(start, stop)
            for o in overlaps:
                v = 1
                if size_norm:
                    raise NotImplementedError
                cnt[(barcode, o.data)] += 1  # Ignores duplicates for now
            # if i > 100000:  # For debugging
            #     break
    barcodes, genes = zip(*cnt.keys())
    unique_barcodes = sorted(list(set(barcodes)))
    unique_genes = sorted(list(set(genes)))

    barcodes_to_idx = {b: i for i, b in enumerate(unique_barcodes)}
    genes_to_idx = {g: i for i, g in enumerate(unique_genes)}
    barcodes_idx = np.array([barcodes_to_idx[b] for b in barcodes])
    genes_idx = np.array([genes_to_idx[g] for g in genes])
    mat = scipy.sparse.csr_matrix(
        (list(cnt.values()), (barcodes_idx, genes_idx)),
        shape=(len(unique_barcodes), len(unique_genes)),
    )

    retval = ad.AnnData(mat)
    retval.obs_names = unique_barcodes
    retval.var_names = unique_genes
    return retval


def gene_activity_matrix_from_adata(
    adata: ad.AnnData,
    annotation: str = sc_data_loaders.HG38_GTF,
    size_norm: bool = False,
) -> ad.AnnData:
    """
    Create a gene activity matrix using h5ad input
    """
    gene_to_pos = utils.read_gtf_gene_to_pos(annotation, extend_upstream=2000)
    genes = list(gene_to_pos.keys())
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    # Dict of chrom (without chr prefix) to intervaltree
    gene_intervaldict = sc_data_loaders.gene_pos_dict_to_range(gene_to_pos)

    # Create a matrix that maps each feature to the overlapping genes
    weights = np.zeros((adata.shape[1], len(genes)), dtype=int)
    for i, atac_bin in enumerate(adata.var_names):
        chrom, span = atac_bin.split(":")
        if chrom.startswith("chr"):
            chrom = chrom.strip("chr")
        start, stop = map(int, span.split("-"))
        overlaps = gene_intervaldict[chrom][start:stop]
        for o in overlaps:
            weights[i, gene_to_idx[o.data]] = 1
    weights = scipy.sparse.csc_matrix(weights)

    # Map the ATAC bins to features
    mat = scipy.sparse.csr_matrix(adata.X @ weights, dtype=int)
    if size_norm:
        interval_span = lambda x: x[2] - x[1]
        spans = np.array([interval_span(gene_to_pos[g]) for g in genes])
        mat /= spans
    retval = ad.AnnData(mat)
    if hasattr(adata, "obs_names"):
        retval.obs_names = adata.obs_names
    retval.var_names = genes
    return retval


def archr_gene_activity_matrix_from_frags(
    fragments: Iterable[Tuple[str, genomic_interval.GenomicInterval]],
    annotation: str = sc_data_loaders.HG38_GTF,
    gene_model: Callable = lambda x: np.exp(-np.abs(x) / 5000) + np.exp(-1),
    extension: int = 100000,
    gene_upstream: int = 5000,
    use_gene_boundaries: bool = True,
    use_TSS: bool = False,
    gene_scale_factor: float = 5.0,  # Numeric scaling factor to weight genes based on inverse of length
    ceiling: float = 4.0,
    scale_to: float = 10000.0,
) -> ad.AnnData:
    """
    Use the more sophisiticated gene activity scoring method described here:
    https://www.archrproject.com/bookdown/calculating-gene-scores-in-archr.html
    https://github.com/GreenleafLab/ArchR/blob/ddcaae4a6093685875052219141e5ea41030fc55/R/MatrixGeneScores.R

    Note this isn't a FULL reimplementation since we do slightly different handling of distances
    and tiling and such, but this should be a very close approximation

    Some other notes:
    - By default, ArchR appears to be doing distance calculations based on the entire gene body, which we do
    """


def archr_gene_activity_matrix_from_adata(
    adata: ad.AnnData,
    annotation: str = sc_data_loaders.HG38_GTF,
    gene_model: Callable = lambda x: np.exp(-np.abs(x) / 5000) + np.exp(-1),
    extension: int = 100000,
    gene_upstream: int = 5000,
    use_gene_boundaries: bool = True,
    use_TSS: bool = False,
    gene_scale_factor: float = 5.0,  # Numeric scaling factor to weight genes based on inverse of length
    ceiling: float = 4.0,
    scale_to: float = 10000.0,
) -> ad.AnnData:
    """
    Use the more sophisiticated gene activity scoring method described here:
    https://www.archrproject.com/bookdown/calculating-gene-scores-in-archr.html
    https://github.com/GreenleafLab/ArchR/blob/ddcaae4a6093685875052219141e5ea41030fc55/R/MatrixGeneScores.R

    Note this isn't a FULL reimplementation since we do slightly different handling of distances
    and tiling and such, but this should be a very close approximation

    Some other notes:
    - By default, ArchR appears to be doing distance calculations based on the entire gene body, which we do
    """
    gene_to_pos = utils.read_gtf_gene_to_pos(annotation, extend_upstream=gene_upstream)
    genes = list(gene_to_pos.keys())
    # Map each gene and bin to a corresponding index in axis
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    bin_to_idx = {b: i for i, b in enumerate(adata.var_names)}

    # Map of chrom without chr prefix to intervaltree
    chrom_to_gene_intervals = utils.gene_pos_dict_to_range(gene_to_pos)
    # Create a mapping of where atac bins are, so we can easily grep for overlap later
    chrom_to_atac_intervals = collections.defaultdict(itree.IntervalTree)
    for atac_bin in adata.var_names:
        chrom, span = atac_bin.split(":")
        if chrom.startswith("chr"):
            chrom = chrom.strip("chr")
        start, stop = map(int, span.split("-"))
        chrom_to_atac_intervals[chrom][start:stop] = atac_bin

    # Create a matrix that maps each feature to overlapping genes with weight
    weights = scipy.sparse.lil_matrix((adata.shape[1], len(genes)), dtype=float)

    # Create per-gene weights based on size of the gene
    logging.info("Determining gene weights based on gene size")
    gene_widths = np.array([gene_to_pos[g][2] - gene_to_pos[g][1] for g in genes])
    assert np.all(gene_widths > 0)
    inv_gene_widths = 1 / gene_widths
    gene_weight = 1 + inv_gene_widths * (gene_scale_factor - 1) / (
        np.max(inv_gene_widths) - np.min(inv_gene_widths)
    )
    assert np.all(gene_weight >= 1)
    if not np.all(gene_weight <= gene_scale_factor):
        logging.warning(
            f"Found values exceeding gene scale factor {gene_scale_factor}: {gene_weight[np.where(gene_weight > gene_scale_factor)]}"
        )
    if not np.all(gene_weight >= 1.0):
        logging.warning(
            f"Found values below minimum expected value of 1: {gene_weight[np.where(gene_weight < 1.0)]}"
        )

    logging.info("Constructing bin to gene matrix")
    for gene in genes:  # Compute weight for each gene
        gene_gi = genomic_interval.GenomicInterval(
            gene_to_pos[gene], metadata_dict={"gene": gene}
        )
        chrom, start, stop = gene_to_pos[gene]
        assert start < stop
        # Get all ATAC bins
        gene_overlap_atac_bins = chrom_to_atac_intervals[chrom][
            start - extension : stop + extension
        ]
        # Drop the ATAC bins that overlap a gene that isn't this current gene
        filtered_gene_overlap_atac_bins = []
        for o in gene_overlap_atac_bins:
            atac_start, atac_end = o.begin, o.end
            atac_bin_gene_overlaps = chrom_to_gene_intervals[chrom][atac_start:atac_end]
            is_matched = [g.data != gene for g in atac_bin_gene_overlaps]
            if any(is_matched):
                continue
            filtered_gene_overlap_atac_bins.append(o)
        # Calculate the distance and the corresponding weight
        for o in filtered_gene_overlap_atac_bins:
            bin_gi = genomic_interval.GenomicInterval((chrom, o.begin, o.end))
            d = gene_gi.difference(bin_gi)
            assert d >= 0
            w = gene_model(d)
            assert weights[bin_to_idx[o.data], gene_to_idx[gene]] == 0
            # Note, ArchR works on 500bp bins, so may count a little differently
            # To adjust multiply by teh size of the bin divided by 500
            # This approximates how many times each bin might be counted
            # as fragments
            v = w * gene_weight[gene_to_idx[gene]] * max(bin_gi.size / 500, 1)
            weights[bin_to_idx[o.data], gene_to_idx[gene]] = v
    weights = scipy.sparse.csc_matrix(weights)

    if ceiling > 0:
        logging.info(
            f"Calculating maximum capped counts per bin at {ceiling} per 500bp"
        )
        bin_to_width = lambda x: genomic_interval.GenomicInterval(x).size
        per_bin_cap = np.array(
            [max(bin_to_width(b) / 500, 1) * ceiling for b in adata.var_names]
        )
        assert np.all(per_bin_cap >= ceiling)
        adata.X = np.minimum(utils.ensure_arr(adata.X), per_bin_cap)

    # Map the ATAC bins to features
    # Converting to an array is necessary for correct broadcasting
    logging.info("Calculating gene activity scores")
    mat = utils.ensure_arr(adata.X @ weights)

    # Normalize depths
    if scale_to > 0:
        logging.info(f"Depth normalizing gene activity scores to {scale_to}")
        per_cell_depths = np.array(mat.sum(axis=1)).flatten()
        per_cell_scaling = scale_to / per_cell_depths
        per_cell_scaling[np.where(per_cell_depths == 0)] = 0.0
        mat = scipy.sparse.csr_matrix(mat * per_cell_scaling[:, np.newaxis])
        assert np.all(
            np.logical_or(
                np.isclose(scale_to, mat.sum(axis=1)), np.isclose(0, mat.sum(axis=1))
            )  # Either is the correct size, or is all 0
        )

    retval = ad.AnnData(mat)
    if hasattr(adata, "obs_names"):
        retval.obs_names = adata.obs_names
    retval.var_names = genes
    return retval
