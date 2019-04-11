#Compare reulsts of Random Walk and LOF 
#for clustering improvement ability
#  Evaluation of outlier detection in flow cytometry data 
# using random walk. FLOCK Siluette index.
# We use full Samusik data here as Samusik_01 becomes to
# sparse after cleaning for the algorithm to run
#load data, labels etc.
library(flowMeans)

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
CALC_NAME <- 'RW_vs_LOF'
ALG_NAME='CLIQUE'
RES_DIR <- '../../results/outlier_compare'
k=30
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_walker.R')

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')

#################
### LOAD DATA ###
#################


DATA_DIR <- "../../benchmark_data_sets"

files <- list(
  Levine_32dim = file.path(DATA_DIR, "Levine_32dim.fcs"), 
  Levine_13dim = file.path(DATA_DIR, "Levine_13dim.fcs"), 
  #Samusik_01   = file.path(DATA_DIR, "Samusik_01.fcs") 
  Samusik_all  = file.path(DATA_DIR, "Samusik_all.fcs") 
  #Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  #Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs"), 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)


# FlowCAP data sets are treated separately since they require clustering algorithms to be
# run individually for each sample

is_FlowCAP <- c(FALSE, FALSE, FALSE)

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

ix_subsample <- 1:3
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
  #Samusik_01   = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Samusik_01.txt") 
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
  #Samusik_01   = file.path(JAC_DIR, "j31Samusik_01.txt") 
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

###########################################################
#compute F1 measure for RW and LOF



###########################################################
#compute intersection of LOF and RANDOM WALK outliers
##################################
###LOF and random walk results ###
##################################

load(file=paste0(RES_DIR, '/', ALG_NAME, '.RData'))
load(file=paste0(RES_DIR, '/',  'lbod_results', '.RData'))
i=2
hist(-log10(outliersWalkP[[i]]+10^(-150)),1000, xlim=c(00,150), ylim=c(0,10))
#sum(-log10(outliersWalkP[[3]]+10^(-150))>50 & -log10(outliersWalkP[[3]]+10^(-150))<150)#cluster of low probability 
hist(log10(outl_resld[[i]]$lof[,i]),1000)
outlCut<-ifelse(frankv(outl_resld[[i]]$lof[,i], ties.method = 'random', order=-1)<=sum(outliersWalkP[[i]]==0),1,0)
helper_match_evaluate_multiple(outlCut, ifelse(outliersWalkP[[i]]==0, 1, 0) )
AMI(outlCut, ifelse(outliersWalkP[[i]]==0, 1, 0))
#pCardIntersection2(U=265627, G=6475, N=6475, c= 1798, log10=TRUE, mc.cores=5) #very slow
length(intersect((1:length(outliersWalkP[[i]]))[ifelse(outlCut,T,F)], (1:length(outliersWalkP[[i]]))[outliersWalkP[[i]]==0]))
sum(outliersWalkP[[i]]==0)
cor(outliersWalkP[[i]], frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1), method='spearman')
plot(-log10(outliersWalkP[[i]]+10^(-30)), (outl_resld[[i]]$lof[,2]), pch='.')
#cicle through 
library(subspace)
clus_LOF <- vector("list", length(files_truth))
clus_RW <- vector("list", length(files_truth))
for (i in 1:length(outliersWalkP)){
  system.time(resflow<-P3C(data[[i]][outliersWalkP[[i]]!=0,]))
  table(resflow)
  clus_RW[[i]] <- rep(NA, length(outliersWalkP[[i]]))
  clus_RW[[i]][outliersWalkP[[i]]!=0] <- resflow
  print(helper_match_evaluate_multiple(clus_RW[[i]], clus_truth[[i]]))
  
  system.time(resflow<-helper_call_Flock(data[[i]][frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0),], FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)
  table(resflow)
  clus_LOF[[i]] <- rep(NA, length(outliersWalkP[[i]]))
  clus_LOF[[i]][frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0)] <- resflow
  print(helper_match_evaluate_multiple(clus_LOF[[i]], clus_truth[[i]]))  
  print(helper_match_evaluate_multiple(clus_assign[[i]], clus_truth[[i]]))  
}
save(clus_RW, clus_LOF, file=paste0(RES_DIR, '/', CALC_NAME, '_', 'clusterOutImprove', '.RData'))

###siluete index (unsupevised measure of clustering performance)
################################################################




############################
########visualisation#######
############################
#########################
library(pdfCluster)
#plot outliers for Walk
i=2
#library('flexclust')
#hist(dist2(data[[i]][io,], data[[i]][ii,], method = "euclidean", p=2),500)

idxs <- sample(length(clus_assign[[i]]), 20000)
sind1<-silhouette(clus_assign[[i]][idxs], dist(scale(data[[i]][idxs,]), method = "manhattan"))
median(sind1[ ,3])
dbs1<-dbs(scale(data[[i]][idxs,]), clusters = clus_assign[[i]][idxs], prior=as.vector(table( clus_assign[[i]][idxs])/sum(table( clus_assign[[i]][idxs]))))
median(dbs1@dbs)

IDXout<-which(!(1:dim(data[[i]])[1] %in% outliersWalk[[i]]))
#sample non-outlers
ii<-sample(outliersWalk[[i]], 20000)
#hist(dist(data[[i]][ii,]),500, col='red')
io<-sample(IDXout, ifelse(length(IDXout)<2000,length(IDXout), 2000 ))
sind2<-silhouette(clus_RW[[i]][ii], dist(scale(data[[i]][ii,]), method = "manhattan"))
median(sind2[ ,3])
dbs2<-dbs(scale(data[[i]][ii,]), clusters = clus_RW[[i]][ii], prior=as.vector(table( clus_RW[[i]][ii])/sum(table( clus_RW[[i]][ii]))))
(median(dbs2@dbs))

outliersLOF <- (1:length(outliersWalkP[[i]]))[frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0)]
IDXout<-which(!(1:dim(data[[i]])[1] %in% outliersLOF))
#sample non-outlers
ii<-sample(outliersLOF, 20000)
#hist(dist(data[[i]][ii,]),500, col='red')
io<-sample(IDXout, ifelse(length(IDXout)<2000,length(IDXout), 2000 ))
sind3<-silhouette(clus_LOF[[i]][ii], dist(scale(data[[i]][ii,]), method = "manhattan"))
#plot(sind3)
median(sind3[ ,3])
dbs3<-dbs(scale(data[[i]][ii,]), clusters = clus_LOF[[i]][ii], prior=as.vector(table( clus_LOF[[i]][ii])/sum(table( clus_LOF[[i]][ii]))))
median(dbs3@dbs)


h1<-hist(sind1[ ,3], col=alpha('red', 0.9), 150, main=paste0('Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
h2<-hist(sind2[ ,3], col=alpha('green', 0.8), add=T, 150)
h3<-hist(sind3[ ,3], col=alpha('blue', 0.7), add=T, 150)

hist(dbs1@dbs, col=alpha('red', 0.9), 100, main=paste0('Density Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(dbs2@dbs, col=alpha('green', 0.8), add=T, 100)
hist(dbs3@dbs, col=alpha('blue', 0.7), add=T, 100)

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

detach("package:flowClust", unload=TRUE)
library(flowClust)
i=2
res_fC<-flowClust(data[[i]],  varNames = NULL, K=24, randomStart = 5, prior = NULL, usePrior = "no")
table(res_fC@flagOutliers[ !(1:(dim(data[[2]])[1]) %in% outliersWalk) ])
table(res_fC@flagOutliers[ which(!(1:(dim(data[[i]])[1]) %in% outliersWalk[[i]])) ])

print(helper_match_evaluate_multiple(res_fC@label, clus_truth[[i]]))
# test vine copula density
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


