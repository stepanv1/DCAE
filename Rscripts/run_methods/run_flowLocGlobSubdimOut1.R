#artificial set generation 
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
library(MASS)
seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_NAME<-'Louvain_L1'
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
JAC_DIR = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/kk30L1'
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


#accard_l<- vector("list", length(files))
#for (i in 1:length(data)) {
#  f <- files[[i]]
#  jaccard_l[[i]]<-read.table(file=f, header = FALSE)
#}
#lapply(jaccard_l, dim)
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

#algorithm starts here

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
ref_subs<- vector("list", length(files_truth))
glob_out<- vector("list", length(files_truth))
loc_out<- vector("list", length(files_truth))


#load clustering results without oulier assignments to learn reference set
load(file=paste0(RES_DIR, '/', CALC_NAME, '.RData'))
i=1
# find reference subdimensions
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
ref_subs[[i]]<-helper_find_subdimensions(data[[i]][!is.na(clus_truth[[i]]), ], clus_truth[[i]][!is.na(clus_truth[[i]])])
#plot heatmap of subspaces
heatmap.2(as.matrix(ifelse(ref_subs[[i]]$subdim, 1,0)))
#mean absolute correlations between reference subdimensions
corabs<-(abs(cor(data[[i]][clus_assign[[i]]==cln, (1:39)[ref_subs[[i]][cln,]] ][,], method='spearman')))
sum(abs(corabs-diag(diag(corabs)))>0.1)/((dim(corabs)[1] )*(dim(corabs)[1]-1))
# find global outliers
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_global_outliers.R')
glob_out[[i]]<-helper_global_outliers_Discrete(data[[i]], ref_subs[[i]]$subdim, clus_assign[[i]], mc.cores=5)
hist(glob_out[[i]],200)
glob_out[[i]][is.na(glob_out[[i]])] <- 0
helper_match_evaluate_multiple(ifelse(glob_out[[i]]>2 , 1, 0), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )
table(ifelse(glob_out[[i]]>2 , 'out', 'in'), ifelse(is.na(clus_truth[[i]]), 1 ,0 ))
#save(glob_out, file=paste0(RES_DIR, '/', CALC_NAME, 'glob_out.RData'))
# find local outliers
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
system.time(loc_out[[i]]<-helper_local_outliers(data[[i]], ref_subs[[i]], glob_out[[i]], clus_assign[[i]], k=25, mc.cores=5))

zzz=as.numeric(loc_out[[i]])
hist(zzz, 200)
#save(loc_out, file=paste0(RES_DIR, '/', CALC_NAME, 'loc_out.RData'))
helper_match_evaluate_multiple(ifelse(zzz>1 | is.na(zzz), 1, 0), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )
table(ifelse(zzz>1.2 | is.na(zzz), 'o', 'in'), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )
#evaluate results, runnung clustering algorithm with global outliers removed


#generate artificial data based on samulikAll,
i=4
rs<-ref_subs[[i]]
dm<-ncol(rs)
dataSam <- data[[i]][!is.na(clus_truth[[i]]), ]
#dataSam <- apply(dataSam, 2, function(x) x-min(x))

tcl <- clus_truth[[i]][!is.na(clus_truth[[i]])]
nclus <- length(unique(tcl))
table(tcl)

Fullcovariances <- lapply(1:nclus, function(x) cov(dataSam[tcl==x, ]))
subdimCovariances <- lapply(1:nclus, function(x) {
  Fullcovariances[[x]][!rs[x,], !rs[x,]]<-0
  return(Fullcovariances[[x]])})

Fullmeans <- lapply(1:nclus, function(x) colMeans(dataSam[tcl==x, ]))
subdimMeans <- lapply(1:nclus, function(x) {
  Fullmeans[[x]][!rs[x,]]<-0
  return(Fullmeans[[x]])})


library('mvtnorm')
populations<-table(tcl)
library(ks)
#generate noise
system.time(noisy_densities <- mclapply(1:nclus, function(z) {
  fhat <- apply( dataSam[tcl==z, !rs[z,]], 2, function(x) kde(x=x, h=hpi(x)))
  return(fhat)}, mc.cores=5))

#create means and covariances for noisy dimensions
Noisycovariances <- lapply(1:nclus, function(z) {
  diag(unlist(lapply(noisy_densities[[z]], function(x) var(rkde(x, n=5000)))))})
NoisyMeans <- lapply(1:nclus, function(z) {
  (unlist(lapply(noisy_densities[[z]], function(x) mean(rkde(x, n=5000)))))})

subdimCovariances <- lapply(1:nclus, function(x) {
  Fullcovariances[[x]][!rs[x,], !rs[x,]]<-0
  return(Fullcovariances[[x]])})

Fullmeans <- lapply(1:nclus, function(x) colMeans(dataSam[tcl==x, ]))
subdimMeans <- lapply(1:nclus, function(x) {
  Fullmeans[[x]][!rs[x,]]<-0
  return(Fullmeans[[x]])})
#end of preparation of parameters for noisy data

#generate non-noisy populations
artData <- lapply(1:nclus, function(x) {
  Y = matrix(0, ncol=dm, nrow=populations[x])
  Y[, rs[x,]]=rmvnorm(n=populations[x],  mean=subdimMeans[[x]][rs[x,]], sigma=subdimCovariances[[x]][rs[x,], rs[x,]]) 
  return(Y)})

#generate noisy dimensions
artData <- lapply(1:nclus, function(z) {
  artData[[z]][, !rs[z,]] <- matrix(unlist(lapply(noisy_densities[[z]], function(x) rkde(x, n=populations[z]))),
                            nrow=populations[z], ncol = sum(!rs[z,]), byrow=F)
  return(as.data.table(artData[[z]]))})

#temp hack, to be improved
artData<-lapply(artData, abs)

beanplot(as.data.frame(artData[[2]])[, rs[2,]], log = "")
beanplot(as.data.frame(artData[[2]])[, !rs[2,]], log = "")

aFrame<-rbindlist(artData)

aFrame <- as.matrix(aFrame)

lbls <- unlist(lapply(1:nclus, function(z) rep(z,populations[z])))

#caluclate mahalonobis distances for each point in relevan dimensions
Incov<-lapply(1:nclus, function(z) ginv(subdimCovariances[[z]][rs[z,],rs[z,]]))
mahListLocal<- mclapply(1:nclus, function(z){ mahalanobis(aFrame[lbls==z, rs[z,]], 
                            cov=Incov[[z]], center=subdimMeans[[z]][rs[z,]], inverted = T)/sum(rs[z,])
}, mc.cores=5)
hist(unlist(mahListLocal),200)
for(x in 1:nclus){hist((mahListLocal[[x]]),200, main=x)}
mahLocal<-unlist(mahListLocal)
sum(mahLocal>1.5)

#create global outlier score
invNoisycovariances <- lapply(Noisycovariances, ginv)
mahListGlobal<- mclapply(1:nclus, function(z) {mahalanobis(aFrame[lbls==z, !rs[z,]], 
                                                          cov=invNoisycovariances[[z]], center=NoisyMeans[[z]], inverted = T)/sum(!rs[z,])},
                         mc.cores=5)
hist(unlist(mahListGlobal),200)
for(x in 1:nclus){hist((mahListGlobal[[x]]),200, main=x)}
mahGlobal<-unlist(mahListGlobal)
sum(mahGlobal>1.5)

#calculate Full Mahalanobis distance  
Fullcovariances<-lapply(1:nclus, function(z) {Fullcovariances[[z]][!rs[z,], !rs[z,]] <- Noisycovariances[[z]]
return(Fullcovariances[[z]])
})
invFullcovariances <- lapply(Fullcovariances, ginv)
Fullmeans<-lapply(1:nclus, function(z) {Fullmeans[[z]][!rs[z,]] <- NoisyMeans[[z]]
return(Fullmeans[[z]])})

Invall <-lapply(1:nclus, function(z) {
  Fullcovariances[[z]]<-0*Fullcovariances[[z]]
  Fullcovariances[[z]][rs[z, ], rs[z, ]]<-Incov[[z]]
  print(eigen(Incov[[z]])$values); print(eigen(invNoisycovariances[[z]])$values)
  Fullcovariances[[z]][!rs[z, ], !rs[z, ]]<- invNoisycovariances[[z]]
  #print(det(Fullcovariances[[z]]))
  return(Fullcovariances[[z]])
})


mahListFull<- mclapply(1:nclus, function(z) {mahalanobis(abs(aFrame[lbls==z,]), 
                                                           cov=Invall[[z]], center=Fullmeans[[z]], inverted = T)/dim(aFrame)[2]}, mc.cores=5)
hist(unlist(mahListFull),200)
for(x in 1:nclus){hist((mahListFull[[x]]),200, main=x)}
mahFull<<-unlist(mahListFull)
sum(mahFull>1.5)


#calculate LOF and LOFSNN
k=30
#generate a list of cluster coordinates
#library(RANN)
#find_neighbors <- function(data, k){
#  nearest <- RANN::nn2(data, data, k, treetype = "bd", searchtype = "standard")
#  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
#}

#generate a list of cluster coordinates
#library(RANN.L1)
#find_neighbors <- function(data, k){
#  nearest <- RANN.L1::nn2(data, data, k, treetype = "bd", searchtype = "standard")
#  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
#}

#system.time(m2<-find_neighbors(aFrame, k=k+1))
#neighborMatrix <- (m2$nn.idx)[,-1]
#system.time(links <- cytofkit:::jaccard_coeff(m2$nn.idx))
#gc()
#save(m2, links, aFrame, lbls, g, clus_assign,  tsne_out, clus_assignOUT, file='/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/outlier_compare/ArtSetTestSamusik.RData')
#load(file="/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/outlier_compare/ArtSetTestSamusik.RData")
#links <- links[links[,1]>0, ]
relations <- as.data.frame(links)
colnames(relations)<- c("from","to","weight")
relations<-as.data.table(relations)
relations<-relations[to!=from,]
mat_dist<-m2$nn.dists[,-1]
#add distance to the ta
relations$dist <- c(t(mat_dist[,]))  
relations <- relations[relations$weight>0, ]
k=30
k_dist<-mat_dist[,k]
### compute lof and lrd for each k value specified by user
relations$k_dist <- k_dist[relations$to] 
#compute reachebility distance
relations$reach_dist <- relations[,pmax(k_dist,dist)]
lrd <- 1/relations[, sum(.SD$reach_dist) /.N, by=from]$V1
hist(lrd, 500)
relations$lrdto <-  lrd[relations$to] 
relations$lrdfrom <-  lrd[relations$from] 
relations<- relations[, lof:=sum(.SD$lrdto)/.N/.SD$lrdfrom[1], by=from]
lof1<-relations[, lof[1], by=from]$V1
hist(lof1,200)
############################################################
#compute lrd based on subdimensional distances

#relations<- relations[, subdim:=list(list(rep(0, dm))), by = from]
relations<- relations[, clus_to:=lbls[to], by = to]
relations<- relations[, clus_from:=lbls[from], by = from]

hist(relations$lrdfrom, 100)
relations<- relations[, lofSNN:=sum(.SD$lrdto * .SD$weight)/.SD$lrdfrom[1]/sum(.SD$weight), by=from]
lof2<-relations[, lofSNN[1], by=from]$V1
hist(lof2,200)

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_global_outliers.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
#try simple global outlier algorithm
GO<-helper_global_outliers(aFrame, rs, lbls, mc.cores=5)
for (i in 1:nclus){hist(GO[lbls==i], 200, main=paste0(i, ' mean ',  mean(GO[lbls==i]))) }

################################################################
helper_match_evaluate_multiple(ifelse(lof1>1.5, 1, 0), ifelse(mahLocal>1.5, 1, 0))
helper_match_evaluate_multiple(ifelse(lof2>1.5, 1, 0), ifelse(mahLocal>1.5, 1, 0))
table(ifelse(lof1>1.6, 1, 0), ifelse(mahLocal>2.0, 1, 0))
table(ifelse(lof2>1.568, 1, 0), ifelse(mahLocal>2.0, 1, 0))

table(ifelse(lof1>1.6, 1, 0), ifelse(mahGlobal>5.0, 1, 0))
table(ifelse(lof2>1.568, 1, 0), ifelse(mahGlobal>5.0, 1, 0))

#ROC curves
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahLocal>2, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- lof2
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 


library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahGlobal>4, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- lof2
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 

#ROC curves
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahLocal>2, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- GO
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 

#GO
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahGlobal>4, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- GO
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 

## create graph and cluster it
asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight, symmetric = F, index1=T);gc()
asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
#run one reweight cycle
sym_w<-(asym_w+t(asym_w))/2
g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
#clus_assign<-louvain_multiple_runs(g, num.run=5, time.lim=2000)
##############################
helper_match_evaluate_multiple(clus_assign, lbls)


rsClus<-helper_find_subdimensions(aFrame, clus_assign)
heatmap.2(as.matrix(ifelse(rsClus, 1,0)))
#try simple global outlier algorithm
GOclus<-helper_global_outliers(aFrame, rsClus, clus_assign, mc.cores=1)
hist(GOclus)
#GO toadd to thesis
hist(mahGlobal,200)
sum(mahGlobal>2.5)
X11()
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahGlobal>2.5, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- GOclus
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 

sym_w<-(asym_w+t(asym_w))/2
#sym_w<-sym_w[1:100, 1:100]
sym_w<-sym_w[GOclus<1.7, GOclus<1.7]
g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
clus_assignOUT<-louvain_multiple_runs_par(g, num.run=5)
gc()
helper_match_evaluate_multiple(clus_assignOUT, lbls[GOclus<1.7])
f11=helper_match_evaluate_multiple(clus_assign, lbls)$F1
f22=helper_match_evaluate_multiple(clus_assignOUT, lbls[GOclus<1.7])$F1
f22-f11
AMI(clus_assign, lbls)
AMI(clus_assignOUT, lbls[GOclus<1.7])



library(RColorBrewer)
X11()
plot(tsne_out$Y, col=lbls, pch='.')

plot(tsne_out$Y, col=ifelse(lbls==1, 'red', 'black'), pch='.')
library(beanplot)
beanplot(as.data.frame(aFrame[lbls==1, rs[1,] ]))
beanplot(as.data.frame(aFrame[lbls==1, !rs[1,] ]), log='')
library(rgl)
plot3d(aFrame[lbls==1, c(1,12,19) ], 
       type="p")


cols <- rev(brewer.pal(11, 'RdYlBu'))
X11()
dmah<-as.data.frame(cbind(tsne_out$Y,mahLocal))
ggplot(dmah, aes(V1, V2, colour = mahLocal)) + 
  geom_point(size = 0.1)+
  scale_colour_gradientn(colours = cols)

X11()
dmahGlob<-as.data.frame(cbind(tsne_out$Y,mahGlobal))
ggplot(dmahGlob, aes(V1, V2, colour = mahGlobal)) + 
  geom_point(size = 0.1)+
  scale_colour_gradientn(colours = cols)

X11()
dlof<-as.data.frame(cbind(tsne_out$Y,lof2))
ggplot(dlof, aes(V1, V2, colour = lof2)) + 
  geom_point(size = 0.1)+
  scale_colour_gradientn(colours = cols)
X11()
dlof1<-as.data.frame(cbind(tsne_out$Y,lof1))
ggplot(dlof1, aes(V1, V2, colour = lof1)) + 
  geom_point(size = 0.1)+
  scale_colour_gradientn(colours = cols)

#local outlers
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
system.time(LO<-helper_local_outliersLOF(aFrame, rsClus,  clus_assign=clus_assign, k=30, mc.cores=5))

hist(LO$lout_order,200)
#LO
hist(mahGlobal,200)
sum(mahGlobal>2.5)
X11()
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahGlobal>2.5, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- LO$lout_order
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 

hist(mahGlobal,200)
sum(mahLocal>1.6)#to include in thesis
X11()
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahLocal>1.6, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- LO$lout_order
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 


sym_w<-(asym_w+t(asym_w))/2
#sym_w<-sym_w[1:100, 1:100]
sym_w<-sym_w[LO$lout_order<1.25, LO$lout_order<1.25]
g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
#clus_assignOUTLoc<-louvain_multiple_runs_par(g, num.run=5)
gc()
helper_match_evaluate_multiple(clus_assignOUTLoc, lbls[LO$lout_order<1.25])
f11<-helper_match_evaluate_multiple(clus_assign, lbls)$F1
f22<-helper_match_evaluate_multiple(clus_assignOUTLoc, lbls[LO$lout_order<1.25])$F1
f22-f11
populations

AMI(clus_assign, lbls)
AMI(clus_assignOUTLoc, lbls[LO$lout_order<1.25])

#define combined score, 'super score'
SS=sqrt(LO$lout_order^2 + GOclus^2)
hist(SS,200)
sum(SS>2)
#total Mahalanobis distance
#check if normalisation is ok
Mah<-sqrt(mahGlobal^2+mahLocal^2)
hist(Mah,200)
hist(Mah[lbls==3],200)
sum(Mah>2)

X11()
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)
response <- ifelse(mahGlobal>2.5, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- LO$lout_order
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 

hist(mahGlobal,200)
sum(Mah>2.6)
X11()
library(pROC)
sampleIDX<-sample(1:length(lof1), 50000)# to include in abstract for conference
response <- ifelse(Mah>2.5, 1, 0)# this is 'weighted Mahalonobis'
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- SS
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 
cor(lof1, Mah)
cor(SS, Mah)
cor(GOclus,Mah)
cor(LO$lout_order, Mah)

#compare using true Mahlonobis distance
sampleIDX<-sample(1:length(lof1), 50000)# ti incluse into the abstract for the conference
response <- ifelse(mahFull>2.5, 1, 0)
predictor1 <- lof1
Wroc <- roc(response[sampleIDX], as.numeric(predictor1)[sampleIDX], direction='<')
plot(Wroc, col='red', main=i, print.auc=T)
predictor2 <- SS
TWroc <- roc(response[sampleIDX], predictor2[sampleIDX], direction='<', print.auc=T)
plot(TWroc, add=T, col='blue', print.auc=T) 
cor(lof1, mahFull)
cor(SS, mahFull)
cor(GOclus,mahFull)
cor(LO$lout_order, mahFull)





corSpearman<-function(x,y){cor(x,y, method='spearman')}
outer(list('LOF'=lof1, 'LocOut'=LO$lout_order, 'GlobOut'=GOclus, 'SummaryScore'=SS), list('mahLocal'=mahLocal, 'mahGlobal'=mahGlobal, 'MahTotal'=Mah, 'mahFull'=mahFull) , FUN=Vectorize(corSpearman))

topouts<-rank(mahFull)
idx <- (1:length(mahFull))[topouts<length(topouts) & topouts>length(topouts)-50000]

corSpearman<-function(x,y){cor(x,y, method='spearman')}
outer(list('LOF'=lof1[idx], 'LocOut'=LO$lout_order[idx], 'GlobOut'=GOclus[idx], 'SummaryScore'=SS[idx]), list('mahLocal'=mahLocal[idx], 'mahGlobal'=mahGlobal[idx], 'MahTotal'=Mah[idx], 'mahFull'=mahFull[idx]) , FUN=Vectorize(corSpearman))

#check the intersection of top ouliers with distances
scores <- lapply(list('LOF'=lof1, 'LocOut'=LO$lout_order, 'GlobOut'=GOclus, 'SummaryScore'=SS), function(x) 
  {topouts<-rank(x)
  idx2 <- (1:length(x))[topouts<length(topouts) & topouts>length(topouts)-50000]
  idx2}) 
distances <- lapply(list('mahLocal'=mahLocal, 'mahGlobal'=mahGlobal, 'MahTotal'=Mah, 'mahFull'=mahFull), function(x)  {topouts<-rank(x)
idx2 <- (1:length(x))[topouts<length(topouts) & topouts>length(topouts)-50000]
idx2})
intersectSize<-function(x,y){length(intersect(x,y))}
outer(scores , distances , FUN=Vectorize(intersectSize))

outer(list('LOF'=lof1, 'LocOut'=LO$lout_order, 'GlobOut'=GOclus, 'SummaryScore'=SS), list('mahLocal'=mahLocal, 'mahGlobal'=mahGlobal, 'MahTotal'=Mah) , FUN=Vectorize(corSpearman))

outer(list('LOF'=lof1, 'LocOut'=LO$lout_order, 'GlobOut'=GOclus, 'SummaryScore'=SS), list('mahLocal'=mahLocal, 'mahGlobal'=mahGlobal, 'MahTotal'=Mah) , FUN=Vectorize(corSpearman))


f22=helper_match_evaluate_multiple(clus_assignOUT, lbls[GOclus<1.7])$F1
f11=helper_match_evaluate_multiple(clus_assign, lbls)$F1

#try our combine measure
#clus_assignSS<-louvain_multiple_runs_par(g, num.run=5)
sym_w<-(asym_w+t(asym_w))/2
#sym_w<-sym_w[1:100, 1:100]
sum(SS>1.7)
sym_w<-sym_w[SS<1.7, SS<1.7]
g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
gc()
clus_assignSS<-louvain_multiple_runs_par(g, num.run=5)
gc()
helper_match_evaluate_multiple(clus_assignSS, lbls[SS<1.7])
f11<-helper_match_evaluate_multiple(clus_assign, lbls)$F1
f22<-helper_match_evaluate_multiple(clus_assignSS, lbls[SS<1.7])$F1
f22-f11
populations

AMI(clus_assign, lbls)
AMI(clus_assignSS, lbls[SS<1.7])

load("~/tsneout3DartSet1.RData")
library(rgl)

plot3d(tsne_out3D$Y,   type="p", col=mahFull)
plot3d(tsne_out3D$Y,   type="p", col=Mah)
plot3d(tsne_out3D$Y,   type="p", col=mahLocal)
plot3d(tsne_out3D$Y,   type="p", col=LO$lout_order)
plot3d(tsne_out3D$Y,   type="p", col=GO)
plot3d(tsne_out3D$Y,   type="p", col=SS)
plot3d(tsne_out3D$Y,   type="p", col=ifelse(SS<1.8, 'black', 'red'))

sym_w<-(asym_w+t(asym_w))/2
sum(lof1>1.3)
#sym_w<-sym_w[1:100, 1:100]
sym_w<-sym_w[lof1<1.3, lof1<1.3]
g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
clus_assignLOF<-louvain_multiple_runs_par(g, num.run=5)
gc()
helper_match_evaluate_multiple(clus_assignLOF, lbls[lof1<1.3])
f11<-helper_match_evaluate_multiple(clus_assign, lbls)$F1
f22<-helper_match_evaluate_multiple(clus_assignLOF, lbls[lof1<1.3])$F1
f22-f11
populations

#try FLOCK
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_call_FLOCK.R')
system.time(clus_FLOCK<-helper_call_Flock(aFrame, FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)



system.time(clus_FLOCK_LOF<-helper_call_Flock(aFrame[lof1<1.3, ], FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)
system.time(clus_FLOCK_SS<-helper_call_Flock(aFrame[SS<1.93, ], FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)
system.time(clus_FLOCK_GLOB<-helper_call_Flock(aFrame[GOclus<1.7, ], FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)

helper_match_evaluate_multiple(clus_FLOCK, lbls)
helper_match_evaluate_multiple(clus_FLOCK_LOF, lbls[lof1<1.3])
helper_match_evaluate_multiple(clus_FLOCK_SS, lbls[SS<1.93])
helper_match_evaluate_multiple(clus_FLOCK_GLOB, lbls[GOclus<1.7])

#create outlierness measures based on FLOCK results
rsClusFLOCK<-helper_find_subdimensions(aFrame, clus_FLOCK)
heatmap.2(as.matrix(ifelse(rsClusFLOCK, 1,0)))
#try simple global outlier algorithm
GOclusFLOCK<-helper_global_outliers(aFrame, rsClusFLOCK, clus_FLOCK, mc.cores=1)
hist(GOclusFLOCK)
#GO toadd to thesis
hist(mahGlobal,200)
sum(mahGlobal>2.5)

system.time(LOFLOCK<-helper_local_outliersLOF(aFrame, rsClusFLOCK,  clus_assign=clus_FLOCK, k=30, mc.cores=5))

SSFLOCK=sqrt(LOFLOCK$lout_order^2 + GOclusFLOCK^2)


system.time(clus_FLOCK_LOC_FLOCK<-helper_call_Flock(aFrame[LOFLOCK$lout_order<1.25, ], FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)

system.time(clus_FLOCK_SSFLOCK<-helper_call_Flock(aFrame[SSFLOCK<1.93, ], FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)
system.time(clus_FLOCK_GLOBFLOCK<-helper_call_Flock(aFrame[GOclusFLOCK<1.7, ], FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population)

helper_match_evaluate_multiple(clus_FLOCK, lbls)
helper_match_evaluate_multiple(clus_FLOCK_LOF, lbls[lof1<1.3])
helper_match_evaluate_multiple(clus_FLOCK_SSFLOCK, lbls[SSFLOCK<1.93])
helper_match_evaluate_multiple(clus_FLOCK_GLOBFLOCK, lbls[GOclusFLOCK<1.7])
helper_match_evaluate_multiple(clus_FLOCK_LOC_FLOCK, lbls[LOFLOCK$lout_order<1.25])

AMI(clus_FLOCK, lbls)
AMI(clus_FLOCK_LOF, lbls[lof1<1.3])
AMI(clus_FLOCK_SSFLOCK, lbls[SSFLOCK<1.93])
AMI(clus_FLOCK_GLOBFLOCK, lbls[GOclusFLOCK<1.7])
AMI(clus_FLOCK_LOC_FLOCK, lbls[LOFLOCK$lout_order<1.25])


save(m2, links, aFrame, lbls, g,   tsne_out, clus_assignOUT, LO, clus_assignOUTLoc, clus_assign, clus_assignSS, clus_assignLOF, file='/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/outlier_compare/ArtSetTestSamusik.RData')
load(file="/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/outlier_compare/ArtSetTestSamusik.RData")
f00=helper_match_evaluate_multiple(clus_assign[GOclus<1.7], lbls[GOclus<1.7])$F1
