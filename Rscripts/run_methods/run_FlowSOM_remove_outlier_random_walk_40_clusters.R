#  Evaluation of outlier detection in flow cytometry data 
# using random walk. FlowSOM at k=40 with the number of cluster given by 
# manual gating.

library(FlowSOM)

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


seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_NAME <- 'FlowSOM_40'
RES_DIR <- '../../results/outlier_compare'
k=30
cln=40
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_call_FlowSOM.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")

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


# indices of protein marker columns

marker_cols <- list(
  Levine_32dim = 5:36, 
  Levine_13dim = 1:13, 
  Samusik_01   = 9:47, 
  Samusik_all  = 9:47, 
  Nilsson_rare = c(5:7, 9:18), 
  Mosmann_rare = c(7:9, 11:21) 
  #FlowCAP_ND   = 3:12, 
  #FlowCAP_WNV  = 3:8
)
sapply(marker_cols, length)

#####################################################
# load data files: FlowSOM requires flowFrame objects
#####################################################
DATA_DIR_F <- "../../benchmark_data_sets"

filesF <- list(
  Levine_32dim = file.path(DATA_DIR_F, "Levine_32dim.fcs"), 
  Levine_13dim = file.path(DATA_DIR_F, "Levine_13dim.fcs"), 
  Samusik_01   = file.path(DATA_DIR_F, "Samusik_01.fcs"), 
  Samusik_all  = file.path(DATA_DIR_F, "Samusik_all.fcs") 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)

dataF <- vector("list", length(filesF))
names(dataF) <- names(filesF)

for (i in 1:length(dataF)) {
  f <- filesF[[i]]
  dataF[[i]] <- flowCore::read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
}



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
true_n_clus<-sapply(tbl_truth, length)
# store named objects (for other scripts)




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
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
#########################################################################################################
clus_assign<-vector("list", length(files_truth))
clus_assign_2Step<-vector("list", length(files_truth))
clus_assign_walk<-vector("list", length(files_truth))
louvain_assign<-vector("list", length(files_truth))
outliersWalk <- vector("list", length(files_truth))
outliers2Step <- vector("list", length(files_truth))
outliersWalkP <- vector("list", length(files_truth))
#########################################################################################################
#loop over all data sets cluster with and without outlier removal
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_walker.R')

for (i in 1:length(files)){
  lbls <- clus_truth[[i]]
  colnames(jaccard_l[[i]])<- c("from","to","weight")
  jaccard_l[[i]]<-as.data.table(jaccard_l[[i]])
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  sym_w<-(asym_w+t(asym_w))/2
  cat('number of sinks: ', '\n')
  sum(colSums(asym_w)==0)
  gL<-graph_from_adjacency_matrix((asym_w+t(asym_w))/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  
  gL<-simplify(gL, remove.loops=T, edge.attr.comb=list(weight="sum"));gc()
  #run on un-modified set  ########################################################
  #################################################################################
  system.time(resflow<-helper_call_FlowSOM40(dataF[[i]], marker_cols = marker_cols[[i]], numC=cln))
  table(resflow)
  clus_assign[[i]] <- resflow
  # use random walk to find isolated, weakly connected nodes  ######################
  ##################################################################################
  asym_wn<-norm_mat(asym_w)
  Pasym_w<-EnhanceDensity(P=(asym_wn), V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), smooth=FALSE, debug=TRUE, maxit=5000, alpha=1, eps=5e-06); gc()
  cat('number of outliers defined by zeroe value in the stationary state: ', '\n')
  sum(Pasym_w==0)
  sum(colSums(asym_wn)==0)
  outliersWalk[[i]] = (1:ncol(asym_wn))[!(Pasym_w==0)]
  outliersWalkP[[i]] = Pasym_w
  #run on modified set
  system.time(cl_res_walk<-helper_call_FlowSOM40(dataF[[i]], subset=outliersWalk[[i]], marker_cols = marker_cols[[i]], numC=cln)); gc()  
  table(cl_res_walk)
  table(lbls[Pasym_w==0])
  length(lbls[is.na(lbls) & Pasym_w==0])
  length(lbls[is.na(lbls) & colSums(asym_wn)==0])
  clus_assign_walk[[i]]=cl_res_walk
  
  #2 step procedure to identify outliers 
  ############################################################
  gr_res <- mcOutLocal(asym_w, addLoops = F, expansion = 2, inflation = 1,  max.iter = 1, ESM = TRUE); gc()
  gr_res[[1]]
  gr_w<-gr_res[[2]] 
  head(sort(colSums(gr_w),decreasing = T))
  hist(colSums(gr_w),  50000)
  hist(colSums(gr_w), xlim=c(0,0.005), 50000)
  sum(colSums(gr_w)==0)
  
  deg <- colSums(asym_w!=0)
  minweight <- deg/(2*k-1)
  IDXe <- which(colSums(asym_w)<=minweight & deg<=k)
  IDXw<-which(colSums(gr_w)==0)
  
  IDX <- which(!((1:gorder(gL)) %in% union(IDXe, IDXw)))
  table(lbls[IDX])
  table(lbls[union(IDXe, IDXw)])
  outliers2Step[[i]]<-IDX
  #outliers[[i]]<-!IDXw
  gc()
  #cl_res<-cluster_louvain(g)
  system.time(cl_resi<-helper_call_FlowSOM40(dataF[[i]], subset=IDX, marker_cols = marker_cols[[i]], numC=cln)); gc()
  
  clus_assign_2Step[[i]] <- cl_resi
  
  lbls=clus_truth[[i]]
  mbri<-clus_assign_2Step[[i]]
  
  table(mbri)
  table(lbls)
  
  comf<-rep(NA,length(lbls))
  comf[outliers2Step[[i]]]<-mbri
  comf[is.na(comf)]<-100
  table(comf)
  
  helper_match_evaluate_multiple(comf, lbls)
  helper_match_evaluate_multiple(clus_assign[[i]], lbls)
  
}

save(outliers2Step, outliersWalk, outliersWalkP, clus_assign, clus_assign_2Step, clus_assign_walk, file=paste0(RES_DIR, '/', CALC_NAME, '.RData'))
load(file=paste0(RES_DIR, '/', CALC_NAME, '.RData'))
###############################################################################################
#assignment of outliers to clusters and evaluation
#using silhouette and F1 measure. Louvain algorithm is run
#on the set without  outliers and on the set with
#outliers present
################################################################################################

labels_out<-vector("list", length(files_truth))
RFassignWalk<-vector("list", length(files_truth))

system.time(for (i in 1:length(files_truth)){
  lbls=clus_truth[[i]]
  mbri<-clus_assign_walk[[i]]#results by the clustering with the nodes removed by random walk
  mbr<-clus_assign[[i]]# algorithm was executed regrardless of outliers
  
  IDXout<-!(1:dim(data[[i]])[1] %in% outliersWalk[[i]])
  labels_out[[i]] <- helper_assign_outliers(bulk_data = data[[i]][outliersWalk[[i]], ], 
  out_data = data[[i]][IDXout, ], bulk_labels =mbri); gc()
  
  RFassignWalk[[i]]<-rep(NA,length(lbls))
  RFassignWalk[[i]][outliersWalk[[i]]]<-mbri
  RFassignWalk[[i]][IDXout]<-labels_out[[i]]
  print(table(RFassignWalk[[i]]))
  
  print(helper_match_evaluate_multiple(mbr, lbls))
  print(helper_match_evaluate_multiple(RFassignWalk[[i]], lbls))
  print(table(lbls))
})
beep()

labels_out2<-vector("list", length(files_truth))
RFassign2Step<-vector("list", length(files_truth))

system.time(for (i in 1:length(files_truth)){
  lbls=clus_truth[[i]]
  mbri<-clus_assign_2Step[[i]]
  mbr<-clus_assign[[i]]
  
  IDXout<-!(1:dim(data[[i]])[1] %in% outliers2Step[[i]])
  labels_out2[[i]] <- helper_assign_outliers(bulk_data = data[[i]][outliers2Step[[i]], ], 
  out_data = data[[i]][IDXout, ], bulk_labels =mbri); gc()
  
  RFassign2Step[[i]]<-rep(NA,length(lbls))
  RFassign2Step[[i]][outliers2Step[[i]]]<-mbri
  RFassign2Step[[i]][IDXout]<-labels_out2[[i]]
  table(RFassign2Step[[i]])
  
  print(helper_match_evaluate_multiple(mbr, lbls))
  print(helper_match_evaluate_multiple(RFassign2Step[[i]], lbls))
  print(table(lbls))
})
beep(8)

save(labels_out, labels_out2, RFassign2Step, RFassignWalk, file=paste0(RES_DIR, '/', 'RFassign_', CALC_NAME, '.RData'))

load(file=paste0(RES_DIR, '/', 'RFassign_', CALC_NAME, '.RData'))




############################
########visualisation#######
############################
#########################
library(pdfCluster)
#plot outliers for Walk
i=2
IDXout<-which(!(1:dim(data[[i]])[1] %in% outliersWalk[[i]]))
#sample non-outlers
ii<-sample(outliersWalk[[i]], 20000)
hist(dist(data[[i]][ii,]),500, col='red')
io<-sample(IDXout, ifelse(length(IDXout)<2000,length(IDXout), 2000 ))
hist(dist(data[[i]][io,]),500, col='green', add=T)

library('flexclust')
hist(dist2(data[[i]][io,], data[[i]][ii,], method = "euclidean", p=2),500)

idxs <- sample(length(clus_assign[[i]]), 20000)
sind1<-silhouette(clus_assign[[i]][idxs], dist(scale(data[[i]][idxs,]), method = "manhattan"))
plot(sind1)
median(sind1[ ,3])
dbs1<-dbs(scale(data[[i]][idxs,]), clusters = clus_assign[[i]][idxs], prior=as.vector(table( clus_assign[[i]][idxs])/sum(table( clus_assign[[i]][idxs]))))
median(dbs1@dbs)

sind2<-silhouette(RFassignWalk[[i]][ii], dist(scale(data[[i]][ii,]), method = "manhattan"))
plot(sind2)
median(sind2[ ,3])
dbs2<-dbs(scale(data[[i]][ii,]), clusters = RFassignWalk[[i]][ii], prior=as.vector(table( RFassignWalk[[i]][ii])/sum(table( RFassignWalk[[i]][ii]))))
median(median(dbs2@dbs))


hist(sind1[ ,3], col=alpha('red', 0.8), 150, main=paste0('Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(sind2[ ,3], col=alpha('green', 0.7), add=T, 150)

hist(dbs1@dbs, col=alpha('red', 0.8), 100, main=paste0('Density Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(dbs2@dbs, col=alpha('green', 0.7), add=T, 100)
########################
#plot outliers for 2Step
i=1
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

hist(sind1[ ,3], col=alpha('red', 0.8), 100, main=paste0('Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(sind2[ ,3], col=alpha('green', 0.7), add=T, 100)

hist(dbs1@dbs, col=alpha('red', 0.8), 100, main=paste0('Density Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(dbs2@dbs, col=alpha('green', 0.7), add=T, 100)




