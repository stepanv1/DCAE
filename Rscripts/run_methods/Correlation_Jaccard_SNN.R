#########################################################################################
# R script to load prepare L1 based distances data using Samusik all
#to use in test runs
#
# Lukas Weber, August 2016
# Stepan Grinek February 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)
library(igraph)

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")
source("../helpers/helper_evaluate_NMI.R")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_graphs_distances.R')

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/PhenoGraph"
CALC_NAME="KdependencySamusik_NAN"
DATA_DIR <- "../../benchmark_data_sets"
DIST_DIR <- "  "
#load  data
data=read.table(paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/data_Samusik_all.txt"), header = F, stringsAsFactors = FALSE)


####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
  Samusik_all = file.path(DATA_DIR, "Samusik_all.fcs") 
)

# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

for (i in 1:length(clus_truth)) {
  # if (!is_subsampled[i]) {
  data_truth_i <- flowCore::exprs(flowCore::read.FCS(files_truth[[i]], transformation = FALSE, truncate_max_range = FALSE))
  #} else {
  #data_truth_i <- read.table(files_truth[[i]], header = TRUE, stringsAsFactors = FALSE)
  #}
  clus_truth[[i]] <- data_truth_i[, "label"]
}

sapply(clus_truth, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl_truth <- lapply(clus_truth, table)

tbl_truth
sapply(tbl_truth, length)

# store named objects (for other scripts)

files_truth_PhenoGraph <- files_truth
clus_truth_PhenoGraph <- clus_truth




###############################
### load PhenoGraph results ###
###############################

# load cluster labels
files_res <- list(
  Samusik_all15 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=15Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all30 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=30Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all45   = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=45Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all60  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=60Python_assigned_labels_phenoGraph_Samusik_all.txt"),
  Samusik_all75  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=75Python_assigned_labels_phenoGraph_Samusik_all.txt"),
  Samusik_all90  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=90Python_assigned_labels_phenoGraph_Samusik_all.txt"),
  Samusik_all105  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=105Python_assigned_labels_phenoGraph_Samusik_all.txt") 
)


clus <- vector("list", length(files_res))
names(clus) <- names(files_res)

for (i in 1:length(clus)){
  clus[[i]] <- read.table(files_res[[i]], header = F, stringsAsFactors = FALSE)[,1]
}

sapply(clus, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl <- lapply(clus, table)

tbl
sapply(tbl, length)

# contingency tables

for (i in 1:length(clus)) {
  print(table(clus[[i]], clus_truth[[1]]))
}

# store named objects (for other scripts)

files_PhesnoGraph <- files_out
clus_PhenoGraph <- clus



#find nearest neighbors
#system.time(neighborMatrix <- find_neighbors(data=data, query=data, k=60+1, metric='L1'))
dirname="/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/auto/PhenoGraphKdependencySamusik_NAN"
#system(paste0('mkdir ', dirname))
#save(neighborMatrix, file=paste0(dirname, '/L1_neighbors60.RData'))
load( file=paste0(dirname, '/L2_neighbors105.RData'))

#zzz=20000
#system.time(neighborMatrix <- find_neighbors(data=data, query=data, k=60+1, metric='L1'))
#dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME, "L1")
#system(paste0('mkdir ', dirname))
#save(neighborMatrix, file=paste0(dirname, '/L1_neighbors105.RData'))
#load(file=paste0(dirname, '/L1_neighbors105.RData'))



#jaccard=mclapply(seq(5,105, by=5), function(i){
#  system.time(links <- cytofkit:::jaccard_coeff(neighborMatrix[,1:i]))
#  system.time(links01 <- cytofkit:::jaccard_coeff(neighborMatrix01[,1:i]))
#  dt=data.table(links)
#  dt01=data.table(links01)
#  return(list('all'=dt, '01'=dt01))
#}, mc.cores=3)
#dirname=paste0(RES_DIR, CALC_NAME)
#system(paste0('mkdir ', dirname))
#save(jaccard, file=paste0(dirname, '/jaccard105.RData'))
load( file=paste0(dirname, '/L2_jaccard45.RData'))
View(neighborMatrix$nn.dists)
View(jaccard45)



