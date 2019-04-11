# outlier detection using graph random walk on lof weights
#Growing neural gus art set clustering
library('clusterGeneration')
#library('gmum.r')
library('rgl')
library(FNN)
library(igraph)
library(parallel)
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers")
# helper functions to match clusters and evaluate
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_walker.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')

library(Matrix)
library(data.table)


chunks2<- function(n, nChunks) {split(1:n, ceiling(seq_along(1:n)/(n/nChunks)))}


clus_set<-genRandomClust(numClust=15,
                         sepVal=-0.1,
                         numNonNoisy=40,
                         numNoisy=0,
                         numOutlier=0.05,
                         numReplicate=1,
                         fileName="test",
                         clustszind=2,
                         clustSizeEq=100,
                         rangeN=c(200,4000),
                         clustSizes=NULL,
                         covMethod="eigen",
                         rangeVar=c(2, 30),
                         lambdaLow=1,
                         ratioLambda=10,
                         alphad=1,
                         eta=1,
                         rotateind=FALSE,
                         iniProjDirMethod="SL",
                         projDirMethod="newton",
                         alpha=0.05,
                         ITMAX=20,
                         eps=1.0e-10,
                         quiet=TRUE,
                         outputDatFlag=TRUE,
                         outputLogFlag=TRUE,
                         outputEmpirical=TRUE,
                         outputInfo=TRUE)

cl_coord=clus_set$datList$test_1
lbls=clus_set$memList$test_1
table(lbls)

k=90
#generate a list of cluster coordinates
library(RANN)
find_neighbors <- function(data, k){
  nearest <- nn2(data, data, k, treetype = "bd", searchtype = "standard")
  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
}

#generate a list of cluster coordinates
library(RANN.L1)
find_neighbors <- function(data, k){
  nearest <- RANN.L1::nn2(data, data, k, treetype = "bd", searchtype = "standard")
  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
}

system.time(m2<-find_neighbors(cl_coord, k=k+1))
neighborMatrix <- (m2$nn.idx)[,-1]
system.time(links <- cytofkit:::jaccard_coeff(m2$nn.idx))
gc()
links <- links[links[,1]>0, ]
relations <- as.data.frame(links)
#save(relations, file='../../results/relationsSamusik_01.RData')
colnames(relations)<- c("from","to","weight")
relations<-as.data.table(relations)

library(ldbod)
lof_res<-ldbod(cl_coord, k =  30, nsub = nrow(cl_coord), method = "lof")
relations$lof_from <-lof_res$lof[relations$from] 
relations$lof_to <- lof_res$lof[relations$to] 
relations$weight2 <- relations$lof_from / relations$lof_to
relations$lrd_from <-lof_res$lrd[relations$from] 
relations$lrd_to <- lof_res$lrd[relations$to] 


#generate normalised graph
asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight2, symmetric = F, index1=T);gc()
asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
#run one reweight cycle
sym_w<-(asym_w+t(asym_w))/2

asym_wn<-norm_mat(sym_w)
#asym_wn<-norm_mat(asym_w)
Pasym_w2<-EnhanceDensity(P=(asym_wn), V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), smooth=FALSE, debug=TRUE, maxit=5000, alpha=1, eps=1e-16)
sum(Pasym_w2==0)
table(lbls)

#same with snn neigbours
asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight, symmetric = F, index1=T);gc()
asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
#run one reweight cycle
sym_w<-(asym_w+t(asym_w))/2

asym_wn<-norm_mat(sym_w)
#asym_wn<-norm_mat(asym_w)
Pasym_w<-EnhanceDensity(P=(asym_wn), V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), smooth=FALSE, debug=TRUE, maxit=5000, alpha=1, eps=1e-16)
sum(Pasym_w==0)
table(lbls)

hist(lof_res$lof,200)
hist(log10(Pasym_w2+10^(-20)),200)
helper_match_evaluate_multiple(ifelse(lof_res$lof>1.6, 1, 0), ifelse(lbls==0, 1, 0))
helper_match_evaluate_multiple(ifelse(Pasym_w2==0, 1, 0), ifelse(lbls==0, 1, 0))
helper_match_evaluate_multiple(ifelse(Pasym_w==0, 1, 0), ifelse(lbls==0, 1, 0))


#check, what points were misassigned
setOut <- (1:length(lbls))[Pasym_w2==0 & lbls!=0]
dist_setOut<-lapply(setOut, function(x) (mahalanobis(cl_coord[x, ], colMeans(cl_coord[lbls==lbls[x], ]), cov(cl_coord[lbls==lbls[x], ])))
hist(unlist(dist_setOut), 200))
setIn <- (1:length(lbls))[Pasym_w2 > 0 & lbls!=0]
dist_setIn<-lapply(setIn, function(x) (mahalanobis(cl_coord[x, ], colMeans(cl_coord[lbls==lbls[x], ]), cov(cl_coord[lbls==lbls[x], ]))))
hist(unlist(dist_setIn), 200)

#snn-corrected lof, snnlof
norm_factor_snnlof <- lapply(1:length(lbls), function(i) sum(relations[i,]$weight))
norm_factor_snnlof <- relations[, sum(.SD$weight),by=from]
neighb_lrd_non_normalized <- relations[, sum(.SD$weight*.SD$lrd_to), by=from]  
lofsnn <- (neighb_lrd_non_normalized$V1 / norm_factor_snnlof$V1) / unlist(lof_res$lrd)[,1]
hist(lofsnn)

library(pROC)
rocLOFsnn<-roc(ifelse(lbls==0, 1, 0), lofsnn,
    direction = '<')
plot(rocLOFsnn)
rocLOF<-roc(ifelse(lbls==0, 1, 0), lof_res$lof[,1],
               direction = '<')
plot(rocLOF)

