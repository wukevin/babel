# Code for running ArchR to get ene expression matrices out
# Useful links:
# https://www.archrproject.com/articles/Articles/tutorial.html
# https://www.archrproject.com/bookdown/calculating-gene-scores-in-archr.html
library(optparse)


# https://www.r-bloggers.com/passing-arguments-to-an-r-script-from-command-lines/
# https://stackoverflow.com/questions/13790610/passing-multiple-arguments-via-command-line-in-r
option_list = list(
  make_option(c("-f", "--files"), type="character", default=NULL, 
              help="Comma separated list of files", metavar="character"),
  make_option(c("-n", "--names"), type="character", default=NULL,
              help="Comma separated list of names", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="archr_gene_act", 
              help="output file name prefix [default= %default]", metavar = "character"),
  make_option(c("-g", "--genome"), type="character", default="hg38",
              help="Genome to use [default=%default]", metavar = "character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$files)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

library(ArchR)

addArchRGenome(opt$genome)

inputFiles = strsplit(opt$files, ",")[[1]]
inputSampleNames = strsplit(opt$names, ",")[[1]]

ArrowFiles <- createArrowFiles(inputFiles = inputFiles, sampleNames = inputSampleNames, filterTSS = 4, filterFrags = 1000, addTileMat = TRUE, addGeneScoreMat = TRUE)
proj <- ArchRProject(ArrowFiles = ArrowFiles, outputDirectory = opt$out, copyArrows = TRUE)
proj <- addIterativeLSI(ArchRProj = proj, useMatrix = "TileMatrix", name = "IterativeLSI")
proj <- addClusters(input = proj, reducedDims = "IterativeLSI")

# # https://www.bioconductor.org/help/course-materials/2019/BSS2019/04_Practical_CoreApproachesInBioconductor.html
mat = getMatrixFromProject(proj)
writeMM(assays(mat)[['GeneScoreMatrix']], paste(opt$out, "_gene_activity.mtx", sep=""))  # Write the numerical matrix
write.csv(rowData(mat), paste(opt$out, "_gene_activity_var.csv", sep=""))
write.csv(colData(mat), paste(opt$out, "_gene_activity_obs.csv", sep=""))

# Save the ArchR project
proj <- saveArchRProject(ArchRProj = proj)
