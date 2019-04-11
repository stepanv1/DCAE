#Compare reulsts of Random Walk and LOF 
#for clustering improvement ability
#  Evaluation of outlier detection in flow cytometry data 
# using random walk. FLOCK Siluette index.
# We use full Samusik data here as Samusik_01 becomes to
# sparse after cleaning for the algorithm to run
#load data, labels etc.

#library(cytofkit) 
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
library(clusterCrit)

seed<-set.seed(12345)
CALC_NAME <- 'CheckinFASTnn'
ALG_NAME='Louvain_L2_k30'
RES_DIR <- '../../results/outlier_compare'
k=30
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')


#################
### LOAD DATA ###
#################


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
#load precomuted L2 distances
JAC_DIR = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/kk30L2'
files <- list(
  Levine_32dim = file.path(JAC_DIR, "j31Levine_32dim.txt"), 
  Levine_13dim = file.path(JAC_DIR, "j31Levine_13dim.txt"), 
  Samusik_01   = file.path(JAC_DIR, "j31Samusik_01.txt"), 
  Samusik_all  = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/KdependencySamusik_840kk30L2/j31Samusik_all.txt'
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

#load LOF results
load(file=paste0(RES_DIR, '/',  'lbod_results', '.RData'))

#calculate subdimensional outlierness measures
#########################################################################################################
ref_subs <- vector("list", length(files_truth))
glob_out <- vector("list", length(files_truth))
loc_out <- vector("list", length(files_truth))
SS_out <- vector("list", length(files_truth))
clus_assign<-vector("list", length(files_truth))
clus_assign_LOF<-vector("list", length(files_truth))
clus_assign_glob_out<-vector("list", length(files_truth))
clus_assign_loc_out<-vector("list", length(files_truth))
clus_assign_SS<-vector("list", length(files_truth))
######################################################

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_global_outliers.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
num.louvain.run=5
for (i in 1:length(clus_truth)){  
  ## create graph and cluster it
  ##############################
  colnames(jaccard_l[[i]])<- c("from","to","weight")
  jaccard_l[[i]]<-as.data.table(jaccard_l[[i]])
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  sym_w<-(asym_w+t(asym_w))/2
  
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  clus_assign[[i]] <- louvain_multiple_runs(g, num.run = num.louvain.run);
  print(i)
  print(helper_match_evaluate_multiple(clus_assign[[i]], clus_truth[[i]]))
  
  
  ref_subs[[i]]<-helper_find_subdimensions(data[[i]], clus_assign[[i]])
  #plot heatmap of subspaces
  heatmap.2(as.matrix(ifelse(ref_subs[[i]], 1,0)))
  # find global outliers
  glob_out[[i]]<-helper_global_outliers(data[[i]], ref_subs[[i]], clus_assign[[i]], mc.cores=5)
  #hist(glob_out[[i]],200)
  print(helper_match_evaluate_multiple(ifelse(glob_out[[i]]>2 , 1, 0), ifelse(is.na(clus_truth[[i]]), 1 ,0 )))
  print(table(ifelse(glob_out[[i]]>2 , 'out', 'in'), ifelse(is.na(clus_truth[[i]]), 1 ,0 )))
  # find local outliers
  system.time(loc_out[[i]]<-helper_local_outliersLOF(data[[i]], ref_subs[[i]],  clus_assign[[i]], k=25, mc.cores=5))
}
#save(ref_subs, glob_out, loc_out, clus_assign, file=paste0(RES_DIR, '/', CALC_NAME, 'subdim_out.RData'))
load(file=paste0(RES_DIR, '/', CALC_NAME, 'subdim_out.RData'))

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_topPercentile.R')
system.time(for (i in 1:length(clus_truth)){
  colnames(jaccard_l[[i]])<- c("from","to","weight")
  jaccard_l[[i]]<-as.data.table(jaccard_l[[i]])
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  
  #cluster using cleaning by LOF
  sym_w<-(asym_w+t(asym_w))/2
  idx <- helper_topPercentile(0.95, outl_resld[[i]]$lof[,2], direction='bottom')
  sym_w<-sym_w[idx, idx]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
  clus_assign_LOF[[i]] <- louvain_multiple_runs_par(g, num.run=5)
  gc()
  
  #cluster using cleaning by LOF in relevant dimensions
  sym_w<-(asym_w+t(asym_w))/2
  
  idx <- helper_topPercentile(0.95, loc_out[[i]]$lout_order, direction='bottom')
  sym_w<-sym_w[idx, idx]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
  clus_assign_loc_out[[i]] <- louvain_multiple_runs_par(g, num.run=5)
  gc()
  
  #cluster removing global outliers
  sym_w<-(asym_w+t(asym_w))/2
  idx <- helper_topPercentile(0.95, glob_out[[i]], direction='bottom')
  sym_w<-sym_w[idx, idx]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
  clus_assign_glob_out[[i]] <- louvain_multiple_runs_par(g, num.run=5)
  gc()
  
  #cluster removing outliers identified by combined measure
  #combined measure, SS
  SS_out[[i]]=sqrt(loc_out[[i]]$lout_order^2 +  glob_out[[i]]^2)
  
  sym_w<-(asym_w+t(asym_w))/2
  idx <- helper_topPercentile(0.95, SS_out[[i]], direction='bottom')
  sym_w<-sym_w[idx, idx]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
  clus_assign_SS[[i]] <- louvain_multiple_runs_par(g, num.run=5)
  gc()
  })
#save(SS_out, clus_assign_SS, clus_assign_LOF, clus_assign_glob_out, clus_assign_loc_out, file=paste0(RES_DIR, '/', CALC_NAME, 'subdim_out_cluster_reulsts.RData'))
load(file=paste0(RES_DIR, '/', CALC_NAME, 'subdim_out_cluster_reulsts.RData'))

PerfMeasures <- vector("list", length(files_truth))
num.louvain.run = 5

for (i in 1:length(clus_truth)){
  print(i)
  # Calculate 4 performance measures, supevised and 
  # unsupervised. Without outliers detection, and with Random Walk
  # and LOF detection.
  PerfMeasures[[i]] = matrix(NA, 5, 4)
  
  #supervised measures
  PerfMeasures[[i]][1,1] <- helper_match_evaluate_multiple(clus_assign[[i]], clus_truth[[i]])$mean_F1 
  idx <- helper_topPercentile(0.95, outl_resld[[i]]$lof[,2], direction='bottom')
  PerfMeasures[[i]][2,1] <- helper_match_evaluate_multiple(clus_assign_LOF[[i]], clus_truth[[i]][idx])$mean_F1
  idx <- helper_topPercentile(0.95, glob_out[[i]], direction='bottom')
  PerfMeasures[[i]][3,1] <- helper_match_evaluate_multiple(clus_assign_glob_out[[i]], clus_truth[[i]][idx])$mean_F1
  idx <- helper_topPercentile(0.95, loc_out[[i]]$lout_order, direction='bottom')
  PerfMeasures[[i]][4,1] <- helper_match_evaluate_multiple(clus_assign_loc_out[[i]], clus_truth[[i]][idx])$mean_F1
  idx <- helper_topPercentile(0.95, SS_out[[i]], direction='bottom')
  PerfMeasures[[i]][5,1] <- helper_match_evaluate_multiple(clus_assign_SS[[i]], clus_truth[[i]][idx])$mean_F1
  
  PerfMeasures[[i]][1,2] <- AMI(clus_assign[[i]], clus_truth[[i]])
  idx <- helper_topPercentile(0.95, outl_resld[[i]]$lof[,2], direction='bottom')
  PerfMeasures[[i]][2,2] <- AMI(clus_assign_LOF[[i]], clus_truth[[i]][idx])
  idx <- helper_topPercentile(0.95, glob_out[[i]], direction='bottom')
  PerfMeasures[[i]][3,2] <- AMI(clus_assign_glob_out[[i]], clus_truth[[i]][idx])
  idx <- helper_topPercentile(0.95, loc_out[[i]]$lout_order, direction='bottom')
  PerfMeasures[[i]][4,2] <- AMI(clus_assign_loc_out[[i]], clus_truth[[i]][idx])
  idx <- helper_topPercentile(0.95, SS_out[[i]], direction='bottom')
  PerfMeasures[[i]][5,2] <- AMI(clus_assign_SS[[i]], clus_truth[[i]][idx])
  
  
  idx <- sample(length(clus_assign[[i]]), 2000)
  int <- intCriteria(scale(data[[i]][idx, ]), as.integer(clus_assign[[i]][idx]), c("C_index","Gamma"))
  
  idx<-sample(clus_assign_LOF[[i]], 2000)
  intLOF<-intCriteria(scale(data[[i]][idx, ]), as.integer(clus_assign_LOF[[i]][idx]), c("C_index","Gamma"))
  
  idx<-sample(clus_assign_glob_out[[i]], 2000)
  intGLOB<-intCriteria(scale(data[[i]][idx,]),as.integer(clus_assign_glob_out[[i]][idx]),c("C_index","Gamma"))
  
  idx<-sample(clus_assign_loc_out[[i]], 2000)
  intLOC<-intCriteria(scale(data[[i]][idx,]),as.integer(clus_assign_loc_out[[i]][idx]),c("C_index","Gamma"))
  
  idx<-sample(clus_assign_SS[[i]], 2000)
  intSS<-intCriteria(scale(data[[i]][idx,]),as.integer(clus_assign_SS[[i]][idx]),c("C_index","Gamma"))
  
  PerfMeasures[[i]][1,3] <- 1-int$c_index 
  PerfMeasures[[i]][2,3] <- 1-intLOF$c_index 
  PerfMeasures[[i]][3,3] <- 1-intGLOB$c_index
  PerfMeasures[[i]][4,3] <- 1-intLOC$c_index 
  PerfMeasures[[i]][5,3] <- 1-intSS$c_index 
  
  PerfMeasures[[i]][1,4] <- int$gamma
  PerfMeasures[[i]][2,4] <- intLOF$gamma 
  PerfMeasures[[i]][3,4] <- intGLOB$gamma
  PerfMeasures[[i]][4,4] <- intLOC$gamma 
  PerfMeasures[[i]][5,4] <- intSS$gamma  
}

save(PerfMeasures, file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'clusterOutImprove', '.RData'))
load(file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'clusterOutImprove', '.RData'))

#generate performance measures for OUTRANK (lower scores for outliers)
#generate performance measures for SOD (higher scores for outliers)

clus_OUTRANK <- vector("list", length(files_truth))
clus_SOD <- vector("list", length(files_truth))
PerfMeasuresELKI <- vector("list", length(files_truth))
num.louvain.run = 5
ELKI_DIR<-'/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/ELKI'
filesROUTRANK <- list(
  Levine_32dim = file.path(ELKI_DIR, "Levine_32dimOUTRANK_CLIQUE", "OUTRANK_S1_order.RData"), 
  Levine_13dim = file.path(ELKI_DIR, "Levine_13dimOUTRANK_CLIQUE", "OUTRANK_S1_order.RData"), 
  Samusik_01   = file.path(ELKI_DIR, "Samusik_01OUTRANK_CLIQUE", "OUTRANK_S1_order.RData"), 
  Samusik_all  = file.path(ELKI_DIR, "Samusik_allOUTRANK_CLIQUE", "OUTRANK_S1_order.RData") 
)
filesRSOD <- list(
  Levine_32dim = file.path(ELKI_DIR, "Levine32SOD", "sod-outlier_order.RData"), 
  Levine_13dim = file.path(ELKI_DIR, "Levine13SOD", "sod-outlier_order.RData"), 
  Samusik_01   = file.path(ELKI_DIR, "Samusik_01SOD", "sod-outlier_order.RData"), 
  Samusik_all  = file.path(ELKI_DIR, "Samusik_allSOD", "sod-outlier_order.RData") 
)


for (i in 1:length(outliersWalkP)){
  load(file=filesROUTRANK[[i]]) 
  OUTRANKres <- out_ID$score
  load(file=filesRSOD[[i]]);
  SODres <- out_ID$score
  
  colnames(jaccard_l[[i]])<- c("from","to","weight")
  jaccard_l[[i]]<-as.data.table(jaccard_l[[i]])
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  sym_w<-(asym_w+t(asym_w))/2
  #run clustering on a subset below threshold
  ixOUTRANK<-frankv(OUTRANKres, ties.method = 'random', order=1) > sum(outliersWalkP[[i]]==0)
  sym_w<-sym_w[ixOUTRANK,ixOUTRANK]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F, add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  resflow <- louvain_multiple_runs(g, num.run = num.louvain.run);
  gc()
  clus_OUTRANK[[i]] <- rep(NA, length(outliersWalkP[[i]]))
  clus_OUTRANK[[i]][ixOUTRANK] <- resflow
  #print(helper_match_evaluate_multiple(clus_RW[[i]], clus_truth[[i]]))
  
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  sym_w<-(asym_w+t(asym_w))/2
  #run clustering on a subset below threshold
  ixSOD<-frankv(SODres, ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0)
  sym_w<-sym_w[ixSOD,ixSOD]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F, add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  resflow <- louvain_multiple_runs(g, num.run = num.louvain.run);
  gc()
  clus_SOD[[i]] <- rep(NA, length(outliersWalkP[[i]]))
  clus_SOD[[i]][ixSOD] <- resflow
  #print(helper_match_evaluate_multiple(clus_LOF[[i]], clus_truth[[i]]))  
  #print(helper_match_evaluate_multiple(clus_assign[[i]], clus_truth[[i]]))  
  
  # Calculate 4 performance measures, supevised and 
  # unsupervised. Without outliers detection, and with Random Walk
  # and LOF detection.
  PerfMeasuresELKI[[i]] = matrix(NA, 2,4)
  
  PerfMeasuresELKI[[i]][1,1] <- helper_match_evaluate_multiple(clus_SOD[[i]], clus_truth[[i]])$mean_F1 
  PerfMeasuresELKI[[i]][2,1] <- helper_match_evaluate_multiple(clus_OUTRANK[[i]], clus_truth[[i]])$mean_F1
  
  PerfMeasuresELKI[[i]][1,2] <- AMI(clus_SOD[[i]], clus_truth[[i]])
  PerfMeasuresELKI[[i]][2,2] <- AMI(clus_OUTRANK[[i]], clus_truth[[i]])
  
  outliersOUTRANK <- (1:length(outliersWalkP[[i]]))[ixOUTRANK]
  ii<-sample(outliersOUTRANK, 20000)
  intOUTRANK<-intCriteria(scale(data[[i]][ii,]),as.integer(clus_OUTRANK[[i]][ii]),c("C_index","Gamma"))
  
  outliersSOD <- (1:length(outliersWalkP[[i]]))[ixSOD]
  ii<-sample(outliersSOD, 20000)
  intSOD<-intCriteria(scale(data[[i]][ii,]),as.integer(clus_SOD[[i]][ii]),c("C_index","Gamma"))
  
  PerfMeasuresELKI[[i]][1,3] <- 1-intSOD$c_index 
  PerfMeasuresELKI[[i]][2,3] <- 1-intOUTRANK$c_index 
  
  PerfMeasuresELKI[[i]][1,4] <- intSOD$gamma
  PerfMeasuresELKI[[i]][2,4] <- intOUTRANK$gamma 
  
}
save(PerfMeasuresELKI, clus_OUTRANK, clus_SOD, file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'ELKIclusterOutImprove', '.RData'))
load(file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'ELKIclusterOutImprove', '.RData'))




##############################################################################
#tables for paper
##############################################################################
library(xtable)
options(xtable.floating = TRUE)
options(xtable.timestamp = "")
floating.environment = getOption( "table")
###################################################################################

load(file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'clusterOutImprove', '.RData'))
PerfMeasuresAllsets <- Reduce(rbind, PerfMeasures)
PerfMeasuresAllsets <- as.data.table(round(PerfMeasuresAllsets, digits=3))
names(PerfMeasures) <- names(clus_truth)

load(file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'ELKIclusterOutImprove', '.RData'))
PerfMeasuresAllsetsELKI <- Reduce(rbind, PerfMeasuresELKI)
PerfMeasuresAllsetsELKI <- as.data.table(round(PerfMeasuresAllsetsELKI, digits=3))
names(PerfMeasuresELKI) <- names(clus_truth)

PerfMeasuresAllsets<-rbindlist(list(PerfMeasuresAllsets[1:3, ], PerfMeasuresAllsetsELKI[1:2, ],
                                    PerfMeasuresAllsets[4:6, ], PerfMeasuresAllsetsELKI[3:4, ],
                                    PerfMeasuresAllsets[7:9, ], PerfMeasuresAllsetsELKI[5:6, ],
                                    PerfMeasuresAllsets[10:12, ], PerfMeasuresAllsetsELKI[7:8, ]))


# correct names of data sets
goodNames<-c('Levine32', 'Levine13', 'Samusik01'  ,'SamusikAll')
setNames <- unlist(lapply(goodNames, function(x) c(x, '', '', '', '')))
methodNames<-rep(c('No detection', 'Random Walk', 'LOF', 'SOD', 'OUTRANK'), 4)
PerfMeasuresAllsets <-cbind(setNames, cbind(methodNames, PerfMeasuresAllsets)) 
colnames(PerfMeasuresAllsets)<-c('Data set', 'Outlier detection algorithm', 'mean F1', 'AMI',  'C-measure', '\\parbox[t]{1.2cm}{Gamma}')

xres<-xtable(PerfMeasuresAllsets,  caption = "Tests results with Phenograph, 4 methods", digits=4, display=c('s', 's', 'fg','fg','fg','fg', 'fg'), include.rownames=FALSE, align = c("|c|c|c|c|c|c|c|"), label = "Phenograph_4methods")

print(xres,  include.rownames = FALSE, sanitize.text = identity,  table.placement = "h!",  caption.placement = "top", hline.after = c(-1,0, seq(5, nrow(PerfMeasuresAllsets), length.out=length(clus_truth))))



########################
#plot outliers for 2Step
library(pdfCluster)
i=2
IDXout<-which(!(1:dim(data[[i]])[1] %in% outliers2Step[[i]]))
#sample non-outlers
ii<-sample(outliers2Step[[i]], 10000)
hist(dist(data[[i]][ii,]),500, col='red')
io<-sample(IDXout, ifelse(length(IDXout)<2000,length(IDXout), 2000 ))
hist(dist(data[[i]][io,]),500, col='green', add=T)

library('flexclust')
hist(dist2(data[[i]][io,], data[[i]][ii,], method = "euclidean", p=2),500)

idxs <- sample(length(clus_assign[[i]]), 10000)
sind1<-silhouette(clus_assign[[i]][idxs], dist(scale(data[[i]][idxs,]), method = "manhattan"))
plot(sind1)
median(sind1[ ,3])
dbs1<-dbs(scale(data[[i]][idxs,]), clusters = clus_assign[[i]][idxs], prior=as.vector(table( clus_assign[[i]][idxs])/sum(table( clus_assign[[i]][idxs]))))
median(dbs1@dbs)

sind2<-silhouette(RFassign2Step[[i]][ii], dist(scale(data[[i]][ii,]), method = "manhattan"))
plot(sind2)
median(sind2[ ,3])
dbs2<-dbs(scale(data[[i]][ii,]), clusters = RFassign2Step[[i]][ii], prior=as.vector(table(RFassign2Step[[i]][ii])/sum(table(RFassign2Step[[i]][ii]))))
median(median(dbs2@dbs))

dbso<-dbs(scale(data[[i]][io,]), clusters = RFassign2Step[[i]][io], prior=as.vector(table(RFassign2Step[[i]][io])/sum(table(RFassign2Step[[i]][io]))))
median(median(dbso@dbs))



hist(sind1[ ,3], col=alpha('red', 0.8), 100, main=paste0('Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(sind2[ ,3], col=alpha('green', 0.7), add=T, 100)

hist(dbs1@dbs, col=alpha('red', 0.8), 100, main=paste0('Density Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(dbs2@dbs, col=alpha('green', 0.7), add=T, 100)







i=3 
table(clus_assign[[i]])
IDXout<-which(!(1:dim(data[[i]])[1] %in% outliersWalk[[i]]))
ic<-(1:nrow(data[[i]]))[clus_assign[[i]]==9]
fit <-  kdevine(scale(data[[i]][ic,]))
destALL <- dkdevine(scale(data[[i]][IDXout,]), fit)
destOUT <- dkdevine(scale(data[[i]][ic,]), fit)

i=3
#pairs(data[[i]][sample(1:nrow(data[[i]]), 2000) , ], pch='.')
IDXout<-sample(which(!(1:dim(data[[i]])[1] %in% outliersWalk[[i]])))
length(IDXout); dim(data[[i]])
pairs(data[[i]][c(sample(1:nrow(data[[i]]), 2000), IDXout[1:2000]), 20:23], pch='.', col = ifelse((1:(2000+length(IDXout[1:2000])))>=2001, 'red', 'black'))

i=1; table(clus_RW[[i]])
prRW=prcomp(data[[i]][!is.na(clus_RW[[k]]) & clus_RW[[i]]==19 ,], scale=T, retx=T)
plot(prRW$x[sample(nrow(prRW$x), 2000),c(1,2)], pch='.')
plot(prRW$x[sample(nrow(prRW$x), 2000),c(1,32)], pch='.')
for(k in 1:25){
  prRW=prcomp(data[[i]][!is.na(clus_RW[[i]]) & clus_RW[[i]]==k ,], scale=T, retx=T)
  print(print(prRW$sdev[1]/prRW$sdev[32]))
  plot(prRW$sdev)
}
table(clus_assign[[i]])
for(k in 1:32){
  prRW=prcomp(data[[i]][!is.na(clus_assign[[i]]) & clus_assign[[i]]==k ,], scale=T, retx=T)
  print(prRW$sdev[1]/prRW$sdev[32])
  plot(prRW$sdev)
}
table(clus_truth[[i]])
for(k in 1:14){
  prRW=prcomp(data[[i]][!is.na(clus_truth[[i]]) & clus_truth[[i]]==k ,], scale=T, retx=T)
  print(prRW$sdev[1]/prRW$sdev[32])
  plot(prRW$sdev, main=paste0('N= ', sum(!is.na(clus_truth[[i]]) & clus_truth[[i]]==k)))
}


#explorinf the dims of clusters
clus_truthL <- lapply(clus_truth, function(x) ifelse(is.na(x), 0, x))
i=2
table(clus_truth[[i]])
table(clus_truthL[[i]])
cln=2
pairs(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln),912) , 1:13], pch='.')
pairs(data[[i]][sample(nrow(data[[i]]) ,1000), 1:13], pch='.')

library(beanplot)
pairs(data[[i]][sample(nrow(data[[i]]) ,1000), 1:13], pch='.')
pairs(data[[i]][clus_truthL[[i]]!=0 ,][sample(sum(clus_truthL[[i]]!=0),1000) , 1:13], pch='.')
for(j in sort(unique(clus_truthL[[i]])) ){
  cln=j
  beanplot(lapply(as.data.frame(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(5000, sum(clus_truthL[[i]]==cln)) )) , ]), function(x) x), main=j)
}
#difference between 5th and 13thclusters
cln=5
boxplot(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(5000, sum(clus_truthL[[i]]==cln)) )) , ], main=j)
cln=24
boxplot(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(5000, sum(clus_truthL[[i]]==cln)) )) , ], main=j, add=T, col='red')




cln=5
princomp(data[[i]][clus_truthL[[i]]==cln,])
plot(princomp(data[[i]][clus_truthL[[i]]==cln,]), npc=13)
princomp(data[[i]][clus_truthL[[i]]==cln,][sample(sum(clus_truthL[[i]]==cln), 100), ])
plot(princomp(data[[i]][clus_truthL[[i]]==cln,][sample(sum(clus_truthL[[i]]==cln), 200), ]), npc=13)

prccl<-princomp(data[[i]][clus_truthL[[i]]==cln,])
View(prccl$loadings)
matplot((prccl$loadings[1:39,1:6]), type = 'l')

library(rgl)
plot3d(data[[i]][clus_truthL[[i]]==13 | clus_truthL[[i]]==5,][,c(13,11,12)])
plot3d(data[[i]][clus_truthL[[2]]!=0 ,][,c(1,3,11)], col =clus_truthL[[2]][clus_truthL[[2]]!=0]+1, cex=0.01, pch='.')
#without all other clusters bit with noise 
clnex<-c(5,13,24); 
plot3d(data[[i]][clus_truthL[[i]] %in% c(0, clnex), ][,c(13,11,12)][sample(sum(clus_truthL[[i]] %in% c(0, clnex)) ,50000),])
plot3d(data[[i]][clus_truthL[[i]] %in% c( clnex), ][,c(13,11,12)][sample(sum(clus_truthL[[i]] %in% c( clnex)) ,5000),])
#add the third cluster on top of 5 and 13 separated by cd3

samplecl<-sample(sum(clus_truthL[[i]] %in% c(0, clnex)) ,50000)
plot3d(data[[i]][clus_truthL[[i]] %in% c(0, clnex), ][,c(13,11,12)][samplecl,],
       col = clus_truthL[[2]][ clus_truthL[[i]] %in% c(0, clnex) ][samplecl]+2)

samplecl<-sample(sum(clus_truthL[[i]] %in% c( clnex)) ,5000)
plot3d(data[[i]][clus_truthL[[i]] %in% c( clnex), ][,c(13,11,12)],
       col = clus_truthL[[2]][ clus_truthL[[i]] %in% c( clnex) ]+1)

#assign clusters feature selection
table(clus_assign[[i]])
for(j in sort(unique(clus_assign[[i]])) ){
  cln=j
  boxplot(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000, sum(clus_assign[[i]]==cln)) )) , ], main=j)
}
library(beanplot)
for(j in sort(unique(clus_assign[[i]])) ){
  cln=j
  beanplot(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000, sum(clus_assign[[i]]==cln)) )),]), main=j, beanlines = 'quantiles')
}

library(moments)

cln=j
lapply(lapply(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000, sum(clus_assign[[i]]==cln)) )),]), function(x) x), skewness)
zzz=lapply(as.data.frame(data[[i]][clus_assign[[i]]==cln ,]), function(x)  list('skewness'=skewness(x), 'p.value'=agostino.test(x, alternative = 'less')))
zzz=lapply(as.data.frame(data[[i]][clus_assign[[i]]==cln ,]), function(x)  list('skewness'=skewness(x), 'p.value'=bonett.test(x, alternative = 'less')))

lapply(as.data.frame(data[[i]][clus_assign[[i]]==cln ,]), function(x)  skewness(x)>2  & agostino.test(x, alternative = 'less')$p.value<10^(-5))
sum(unlist(lapply(as.data.frame(data[[i]][clus_assign[[i]]==cln ,]), function(x)  skewness(x)>2  & agostino.test(x, alternative = 'less')$p.value<10^(-5))))

