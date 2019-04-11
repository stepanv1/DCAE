#########################################################################################
# R script to load PhenoGraph results and calculate \delta Q for smaller clusters
#not separated by Phenograph
#
# Lukas Weber, August 2016
# Stepan Grinek February 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)
library(igraph)
library(data.table)
library(cytofkit)

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

#load  data
#data=read.table(paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/data_Samusik_all.txt"), header = F, stringsAsFactors = FALSE)


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

#for (i in 1:length(clus_truth)) {
  # if (!is_subsampled[i]) {
#  data_truth_i <- flowCore::exprs(flowCore::read.FCS(files_truth[[i]], transformation = FALSE, truncate_max_range = FALSE))
  #} else {
#  #data_truth_i <- read.table(files_truth[[i]], header = TRUE, stringsAsFactors = FALSE)
  #}
#  clus_truth[[i]] <- data_truth_i[, "label"]
#}

sapply(clus_truth, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl_truth <- lapply(clus_truth, table)

tbl_truth
sapply(tbl_truth, length)

# store named objects (for other scripts)

files_truth_PhenoGraph <- files_truth
clus_truth_PhenoGraph <- clus_truth

dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
system(paste0('mkdir ', dirname))
#save(clus_truth, file=paste0(dirname, '/clus_truth.RData'))
load(paste0(dirname, '/clus_truth.RData'))


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
#system.time(neighborMatrix <- find_neighbors(data=data[, 1:39], query=data[, 1:39], k=105+1, metric='L2'))
dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
#system(paste0('mkdir ', dirname))
#save(neighborMatrix, file=paste0(dirname, '/L2_neighbors105.RData'))
load( paste0(dirname, '/L2_neighbors105.RData'))



#jaccard=mclapply(seq(5,105, by=5), function(i){
#  system.time(links <- cytofkit:::jaccard_coeff(neighborMatrix$nn.idx[,1:(i+1)]))
#  dt=data.table(links)
#  return(dt)
#}, mc.cores=1)
#dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
#system(paste0('mkdir ', dirname))
#save(jaccard, file=paste0(dirname, '/L2_jaccard105.RData'))
#load( paste0(dirname, '/L2_jaccard105.RData'))

#TO DELETE: temp measure to get memoty extension
#jaccard45=(links45 <- data.table(cytofkit:::jaccard_coeff(neighborMatrix$nn.idx[,1:(46)])))
#remove self-loops
#jaccard45 <- jaccard45[ V1 != V2 ] 

#save(jaccard45, file=paste0(dirname, '/L2_jaccard45.RData'))  
#load(paste0(dirname, '/L2_jaccard45.RData'))  


#create weighted igraph of from links in 'jaccard' list
#rm(neighborMatrix)
#k=45
#edges = jaccard45
#gr<-graph_from_data_frame(jaccard45, directed = F, vertices = NULL)
#save(gr, file=paste0(dirname, '/L2_gr45.RData'))
#dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
#system(paste0('mkdir ', dirname))
#load(paste0(dirname, '/L2_gr45.RData'))
#E(gr)$weight=as.numeric(as.data.frame(jaccard45)[,3])
#save(gr, file=paste0(dirname, '/L2_gr45.RData'))
load(paste0(dirname, '/L2_gr45.RData'))

#an alternative way to do it
#cat("--- Creating graph... ")
#start <- proc.time()

#vertex.attrs <- list(name = unique(c(df$src, df$dst)))
#edges <- rbind(match(df$src, vertex.attrs$name),
#               match(df$dst,vertex.attrs$name))

#G <- graph.empty(n = 0, directed = T)
#G <- add.vertices(G, length(vertex.attrs$name), attr = vertex.attrs)
#G <- add.edges(G, edges)

#remove(edges)
#remove(vertex.attrs)

#cat(sprintf("--- elapsed user-time: %fs ", (proc.time() - start)[1]))

#weights=E(gr)$weight
#save(weights, file=paste0(dirname, '/weights_gr45.RData'))

#load(paste0(dirname, '/weights_gr45.RData'))
#IDX<-sample(1:841644, 10000)
#ind<-induced_subgraph(gr, vids=IDX, impl = "create_from_scratch")
#mmz<-modularity_matrix(ind, membership=clus[[3]][IDX], weights = E(ind)$weight)
#modularity(ind, membership=clus[[3]][IDX])

#Q_45 <- modularity_matrix(gr, membership=clus[[3]], weights = weights)
#dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
#save(Q_45, file=paste0(dirname, '/L2_gr45.RData'))
#load(paste0(dirname, '/L2_gr45.RData'))

#load population names

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/graphQcomputations.R')
#TODO: calculate modularities based on igraph 
Qfun(gr, clus[[3]])
#[1] 0.8935593



#calculate modularities dependending on indexi set, given claster assignments
#in ungated data, wrapper of rigrpah function 




