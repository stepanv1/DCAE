#outlier assignment using subspaces as reference set 
########################################################
# S. Grinek 26.07.17

#load data, labels etc.
library(cytofkit) 
library(fpc)
library(cluster) 
library(Rtsne)
library(rgl)
library(gclus)
library(data.table)
library(flowCore)
library(parallel)
library(Matrix)
library(scales)
library(beepr)
library(gplots)
seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_NAME<-'Louvain_L2cosine'
k=30
RES_DIR <- '../../results/outlier_compare'
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_create_snnk_graph.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')

#################
### LOAD DATA ###
#################
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
# filenames

DATA_DIR <- "../../benchmark_data_sets"

files <- list(
  Levine_32dim = file.path(DATA_DIR, "Levine_32dim.fcs"), 
  Levine_13dim = file.path(DATA_DIR, "Levine_13dim.fcs"), 
  Samusik_01   = file.path(DATA_DIR, "Samusik_01.fcs"), 
  Samusik_all  = file.path(DATA_DIR, "Samusik_all.fcs") 
  #Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  #Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs"), 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)


# FlowCAP data sets are treated separately since they require clustering algorithms to be
# run individually for each sample

is_FlowCAP <- c(FALSE, FALSE, FALSE, FALSE)

# load data files

data <- vector("list", length(files))
names(data) <- names(files)

#for (i in 1:1) {
for (i in 1:length(data)) {
  f <- files[[i]]
  
  if (!is_FlowCAP[i]) {
    data[[i]] <- flowCore::exprs(flowCore::read.FCS(f, transformation = FALSE, truncate_max_range = FALSE))
    
  } else {
    smp <- flowCore::exprs(flowCore::read.FCS(f, transformation = FALSE, truncate_max_range = FALSE))
    smp <- smp[, "sample"]
    d <- flowCore::read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
    d <- flowCore::split(d, smp)
    data[[i]] <- lapply(d, function(s) flowCore::exprs(s))
  }
}

head(data[[1]])

sapply(data, length)

sapply(data[!is_FlowCAP], dim)
sapply(data[is_FlowCAP], function(d) {
  sapply(d, function(d2) {
    dim(d2)
  })
})

#Remove cells without labels from data
#For now not done: subsampling for data sets with excessive runtime (> 12 hrs on server)

ix_subsample <- 1:4
n_sub <- 1000000000000

for (i in ix_subsample) {
  if (!is_FlowCAP[i]) {
    set.seed(123)
    data[[i]] <- data[[i]][, ]
    # save subsampled population IDs
    true_labels_i <- data[[i]][, "label", drop = FALSE]
    files_true_labels_i <- paste0("../../results/auto/DensVM/true_labels_DensVM_", 
                                  names(data)[i], ".txt")
    for (f in files_true_labels_i) {
      write.table(true_labels_i, file = f, row.names = FALSE, quote = FALSE, sep = "\t")
    }
    
  } else {
    # FlowCAP data sets
    for (j in 1:length(data[[i]])) {
      set.seed(123)
      data[[i]][[j]] <- data[[i]][[j]][, ]
      # save subsampled population IDs
      true_labels_ij <- data[[i]][[j]][, "label", drop = FALSE]
      files_true_labels_ij <- paste0("../../results/auto/DensVM/true_labels_DensVM_", 
                                     names(data)[i], "_", j, ".txt")
      for (f in files_true_labels_ij) {
        write.table(true_labels_ij, file = f, row.names = FALSE, quote = FALSE, sep = "\t")
      }
    }
  }
}


# indices of protein marker columns

marker_cols <- list(
  Levine_32dim = 5:36, 
  Levine_13dim = 1:13, 
  Samusik_01   = 9:47, 
  Samusik_all  = 9:47, 
  Nilsson_rare = c(5:7, 9:18), 
  Mosmann_rare = c(7:9, 11:21), 
  FlowCAP_ND   = 3:13, #keep the label 
  FlowCAP_WNV  = 3:8
)
sapply(marker_cols, length)

# subset data: protein marker columns only

for (i in 1:length(data)) {
  if (!is_FlowCAP[i]) {
    data[[i]] <- data[[i]][, marker_cols[[i]]]
  } else {
    for (j in 1:length(data[[i]])) {
      data[[i]][[j]] <- data[[i]][[j]][, marker_cols[[i]]]
    }
  }
}

sapply(data[!is_FlowCAP], dim)
sapply(data[is_FlowCAP], function(d) {
  sapply(d, function(d2) {
    dim(d2)
  })
})

marker_names<- lapply(data, colnames)
#save(marker_names,  file=paste0(DATA_DIR, '/marker_names.RData'))

####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)
MANUAL_DENSITYCUT <- "../../results/manual/densityCut"
files_truth <- list(
  Levine_32dim = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Levine_32dim.txt"), 
  Levine_13dim = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Levine_13dim.txt"), 
  Samusik_01   = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Samusik_01.txt"), 
  Samusik_all  = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Samusik_all.txt") 
  #Nilsson_rare = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Nilsson_rare.txt"), 
  #Mosmann_rare = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Mosmann_rare.txt") 
  #FlowCAP_ND   = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_FlowCAP_ND.txt"), 
  #FlowCAP_WNV  = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_FlowCAP_WNV.txt")
)
# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

#for (i in 1) {
for (i in 1:length(clus_truth)) {
  
  data_truth_i <- read.table(files_truth[[i]], header = TRUE, stringsAsFactors = FALSE)[, "label"]
  clus_truth[[i]] <- data_truth_i
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


###########################################################################################
#########################################################################################
#load precomuted L1 distances
JAC_DIR = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/kk30L2cosine'
files <- list(
  Levine_32dim = file.path(JAC_DIR, "j30Levine_32dim.txt"), 
  Levine_13dim = file.path(JAC_DIR, "j30Levine_13dim.txt"), 
  Samusik_01   = file.path(JAC_DIR, "j30Samusik_01.txt"), 
  Samusik_all  = file.path(JAC_DIR, "j30Samusik_all.txt")
  #Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  #Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs"), 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)


jaccard_l<- vector("list", length(files))
for (i in 1:length(data)) {
  f <- files[[i]]
  jaccard_l[[i]]<-read.table(file=f, header = FALSE)
}
lapply(jaccard_l, dim)
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
#########################################################################################################
clus_assign<-vector("list", length(files_truth))
clus_assign_2Step<-vector("list", length(files_truth))
clus_assign_walk<-vector("list", length(files_truth))
louvain_assign<-vector("list", length(files_truth))
outliersWalk <- vector("list", length(files_truth))
outliersWalkP <- vector("list", length(files_truth))
outliers2Step <- vector("list", length(files_truth))
######################################################
#outlier assignment using subspaces as reference set 
clus_truthL<-lapply(clus_truth, function(x) ifelse(is.na(x),1000, x ))

flist <- c("Levine_32dim", "Levine_13dim", "Samusik_01", "Samusik_all") 

lapply(1:length(data), function(x ){
  write.table(data[[x]], paste0('../../results/ELKI/', flist[[x]], '.csv'), row.names = FALSE,
            col.names = FALSE, sep=' ')
} )

write.table(data[[2]][1:1000,], paste0('../../results/ELKI/', 'testLevin13', '.csv'), row.names = FALSE,
          col.names = FALSE, sep=' ')

#extract ELKI results
####################################################################
#1. get SOD scores results and put them int0 RData files
ELKI_DIR<-'/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/ELKI'
files <- list(
  Levine_32dim = file.path(ELKI_DIR, "Levine32SOD", "sod-outlier_order.txt"), 
  Levine_13dim = file.path(ELKI_DIR, "Levine13SOD", "sod-outlier_order.txt"), 
  Samusik_01   = file.path(ELKI_DIR, "Samusik_01SOD", "sod-outlier_order.txt"), 
  Samusik_all  = file.path(ELKI_DIR, "Samusik_allSOD", "sod-outlier_order.txt") 
)
filesR <- list(
  Levine_32dim = file.path(ELKI_DIR, "Levine32SOD", "sod-outlier_order.RData"), 
  Levine_13dim = file.path(ELKI_DIR, "Levine13SOD", "sod-outlier_order.RData"), 
  Samusik_01   = file.path(ELKI_DIR, "Samusik_01SOD", "sod-outlier_order.RData"), 
  Samusik_all  = file.path(ELKI_DIR, "Samusik_allSOD", "sod-outlier_order.RData") 
)
library(parallel)
for (i in 1:length(files)) {
  f <- files[[i]]
  string <- scan(f, character(0), sep = "\n")
  string<-string[c(T,F,F,F,F)]
  ID_score<-mclapply(string, function(x) {
    s<-unlist(strsplit(x, split = " "))
    return(c(sub(".*=", "", s[1]), sub(".*=", "", s[length(s)])))}  , mc.core=10)
  ID_score = as.numeric(unlist(ID_score))
  out_ID<-as.data.frame(cbind(ID_score[c(T,F)], ID_score[c(F,T)]))
  colnames(out_ID) <-c('ID', 'score')
  out_ID <- out_ID[order(out_ID$ID),]
  save(out_ID, file=filesR[[i]])
}

#2. get OUTRANK-CLIQUE scores results and put them into RData files
ELKI_DIR<-'/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/ELKI'
files <- list(
  Levine_32dim = file.path(ELKI_DIR, "Levine_32dimOUTRANK_CLIQUE", "OUTRANK_S1_order.txt"), 
  Levine_13dim = file.path(ELKI_DIR, "Levine_13dimOUTRANK_CLIQUE", "OUTRANK_S1_order.txt"), 
  Samusik_01   = file.path(ELKI_DIR, "Samusik_01OUTRANK_CLIQUE", "OUTRANK_S1_order.txt"), 
  Samusik_all  = file.path(ELKI_DIR, "Samusik_allOUTRANK_CLIQUE", "OUTRANK_S1_order.txt") 
)
filesR <- list(
  Levine_32dim = file.path(ELKI_DIR, "Levine_32dimOUTRANK_CLIQUE", "OUTRANK_S1_order.RData"), 
  Levine_13dim = file.path(ELKI_DIR, "Levine_13dimOUTRANK_CLIQUE", "OUTRANK_S1_order.RData"), 
  Samusik_01   = file.path(ELKI_DIR, "Samusik_01OUTRANK_CLIQUE", "OUTRANK_S1_order.RData"), 
  Samusik_all  = file.path(ELKI_DIR, "Samusik_allOUTRANK_CLIQUE", "OUTRANK_S1_order.RData") 
)
library(parallel)
for (i in 1:length(files)) {
  f <- files[[i]]
  string <- scan(f, character(0), sep = "\n")
  ID_score<-mclapply(string, function(x) {
    s<-unlist(strsplit(x, split = " "))
    return(c(sub(".*=", "", s[1]), sub(".*=", "", s[length(s)])))} , mc.core=10)
  ID_score = as.numeric(unlist(ID_score))
  out_ID<-as.data.frame(cbind(ID_score[c(T,F)], ID_score[c(F,T)]))
  colnames(out_ID) <-c('ID', 'score')
  out_ID <- out_ID[order(out_ID$ID),]
  save(out_ID, file=filesR[[i]])
}








