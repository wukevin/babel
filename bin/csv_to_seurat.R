# Seurat based analysis of GM12878 paired data scRNA component
# Preparation for integration with ArchR
# References
# https://satijalab.org/seurat/v3.0/pbmc3k_tutorial.html
# https://satijalab.org/seurat/v3.1/merge_vignette.html
# https://learn.gencore.bio.nyu.edu/single-cell-rnaseq/loading-your-own-data-in-seurat-reanalyze-a-different-dataset/
rm(list = ls(all.names = TRUE))  # Clear environment

library(dplyr)
library(Seurat)

option_list = list(
  make_option(c("-i", "--input"), type="character", default=NULL,
              help="Input csv file", metavar="character"),
  make_option(c("-p", "--project"), type="character", default="proj",
              help="Project name", metavar="character"),
  make_option(c("-o", "--output"), type="character", default=NULL, 
              help="output file name", metavar = "character")
);
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

raw_counts <- read.table(file=opt$input, sep=",")
sobj <- CreateSeuratObject(raw.data = raw_counts, min.cells = 3, min.genes = 200, project = opt$project)

sobj[["percent.mt"]] <- PercentageFeatureSet(sobj, pattern = "^MT-")
VlnPlot(sobj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

# Remove some unwanted cells
sobj <- subset(sobj, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 25)

# Normalize
sobj <- NormalizeData(sobj, normalization.method = "LogNormalize", scale.factor = 10000)

# Find variable features
sobj <- FindVariableFeatures(sobj, selection.method = "vst", nfeatures = 2000)

# Scale
sobj <- ScaleData(sobj, features = rownames(sobj))

# PCA
sobj <- RunPCA(sobj, features = VariableFeatures(object = sobj))
DimPlot(sobj, reduction="pca")
ElbowPlot(sobj)

# Cluster the cells
sobj <- FindNeighbors(sobj, dims=1:10)  # Appears to be the same for 20 or 10 dimensions
sobj <- FindClusters(sobj, resolution=1)

# Plot
sobj <- RunUMAP(sobj, dims=1:20)
DimPlot(sobj, reduction="umap")

# Save
saveRDS(sobj, opt$output)
