# Code to integrate atac and rna using archr
# References
# https://www.archrproject.com/bookdown/cross-platform-linkage-of-scatac-seq-cells-with-scrna-seq-cells.html
# We do this using the paired ATAC/RNA GM12878 data
rm(list = ls(all.names = TRUE))  # Clear environment

# Some other useful links
# https://stackoverflow.com/questions/9776064/how-do-i-test-if-r-is-running-as-rscript
# https://stackoverflow.com/questions/45325863/how-to-access-hidden-functions-that-are-very-well-hidden

library(tools)
library(optparse)
library(Matrix)
# https://www.r-bloggers.com/passing-arguments-to-an-r-script-from-command-lines/
# https://stackoverflow.com/questions/13790610/passing-multiple-arguments-via-command-line-in-r
# https://gallery.rcpp.org/articles/sparse-matrix-coercion/

if (!interactive()){
  option_list = list(
    make_option(c("-a", "--archr"), type="character", default=NULL,
                help="Archr file", metavar="character"),
    make_option(c("-s", "--seurat"), type="character", default=NULL,
                help="Seurat file", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="archr_gene_integration", 
                help="output file name prefix [default= %default]", metavar = "character"),
    make_option(c("-g", "--grouping"), type = "character", default = "RNA_snn_res.1",
                help="grouping to use for RNA object", metavar = "character")
  );
  
  opt_parser = OptionParser(option_list=option_list);
  opt = parse_args(opt_parser);
  
  if (is.null(opt$archr) | is.null(opt$seurat)){
    print_help(opt_parser)
    stop("At least one argument must be supplied (input file).n", call.=FALSE)
  }
  
  archr_dir <- opt$archr
  seurat_file <- opt$seurat
  output_prefix <- opt$out
  grouping_key <- opt$grouping
} else {
  setwd("/Users/kevin/Documents/Stanford/zou/single_cell_rep/commonspace_data/tmp")
  archr_dir <- "/Users/kevin/Documents/Stanford/zou/single_cell_rep/commonspace_data/archr_gene_activities/GM12878"
  seurat_file <- "/Users/kevin/Documents/Stanford/zou/single_cell_rep/commonspace_data/seurat_scrnaseq/GM12878_scraseq_seurat.rds"
  output_prefix <- "temptest"
  grouping_key <- "RNA_snn_res.1"
  imputed_genes <- as.matrix(
    read.csv(file = "/Users/kevin/Documents/Stanford/zou/single_cell_rep/commonspace_eval/evalGM_logsplit_DM_HSR_PBMC/atac_rna_table.csv", row.names = 0)
  )
}



library(ArchR)
library(Seurat)

gm.archr.proj = loadArchRProject(path = archr_dir)
# gm.archr.proj <- addIterativeLSI(ArchRProj = gm.archr.proj, useMatrix = "TileMatrix", name = "IterativeLSI")

if (interactive()) {
  ArchR:::.addMatToArrow(
    mat = imputed_genes,
    ArrowFile = getArrowFiles(gm.archr.proj)[2],
    Group = "GeneScoreMatrix/1",
    binarize = FALSE,
    addColSums = TRUE,
    addRowSums = TRUE,
    addRowVarsLog2 = TRUE,
  )
}

input_seurat_ext <- file_ext(seurat_file)
if (input_seurat_ext == "rds") {
  gm.seurat <- readRDS(seurat_file)
} else if (input_seurat_ext == "h5ad") {
  gm.seurat <- ReadH5AD(seurat_file)
} else {
  stop("Unrecognized file extension")
}


gm.archr.proj <- addGeneIntegrationMatrix(
  ArchRProj = gm.archr.proj, 
  useMatrix = "GeneScoreMatrix",
  matrixName = "GeneIntegrationMatrix",
  reducedDims = "IterativeLSI",
  seRNA = gm.seurat,
  addToArrow = TRUE,
  groupRNA = grouping_key,
  nameCell = "predictedCell_Un",
  nameGroup = "predictedGroup_Un",
  nameScore = "predictedScore_Un",
  force = TRUE,
)

mat = getMatrixFromProject(gm.archr.proj, useMatrix = "GeneIntegrationMatrix")
writeMM(assays(mat)[['GeneIntegrationMatrix']], paste(output_prefix, "_gene_integration.mtx", sep=""))  # Write the numerical matrix
write.csv(rowData(mat), paste(output_prefix, "_gene_integration_var.csv", sep=""))
write.csv(colData(mat), paste(output_prefix, "_gene_integration_obs.csv", sep=""))
