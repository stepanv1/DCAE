#create snn corrected LOF estimator
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


clus_set<-genRandomClust(numClust=5,
                         sepVal=-0.7,
                         numNonNoisy=15,
                         numNoisy=25,
                         numOutlier=0.30,
                         numReplicate=1,
                         fileName="test",
                         clustszind=2,
                         clustSizeEq=100,
                         rangeN=c(200,4000),
                         clustSizes=NULL,
                         covMethod="eigen",
                         rangeVar=c(2, 30),
                         lambdaLow=1,
                         ratioLambda=20,
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
cl_coord=scale(cl_coord)
lbls=clus_set$memList$test_1
table(lbls)

k=90
#generate a list of cluster coordinates
#library(RANN)
#find_neighbors <- function(data, k){
#  nearest <- RANN::nn2(data, data, k, treetype = "bd", searchtype = "standard")
#  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
#}

#generate a list of cluster coordinates
library(RANN.L1)
find_neighbors <- function(data, k){
  nearest <- RANN.L1::nn2(data, data, k, treetype = "bd", searchtype = "standard")
  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
}

system.time(m2<-find_neighbors(cl_coord, query=cl_coord, k=k+1))
neighborMatrix <- (m2$nn.idx)[,-1]
system.time(links <- cytofkit:::jaccard_coeff(m2$nn.idx))
gc()
#links <- links[links[,1]>0, ]
relations <- as.data.frame(links)
colnames(relations)<- c("from","to","weight")
relations<-as.data.table(relations)
relations<-relations[to!=from,]
mat_dist<-m2$nn.dists[,-1]
#add distance to the table
relations$dist <- c(t(mat_dist[,]))  
relations <- relations[relations$weight>0, ]

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
#compute lofSNN
relations <-  relations[, lof:=sum(.SD$lrdto)/(.N)/(.SD$lrdfrom[1]), by=from] 
hist(relations$lrdfrom, 100)
relations<- relations[, lofSNN:=sum(.SD$lrdto * .SD$weight)/.SD$lrdfrom[1]/sum(.SD$weight), by=from]
lof2<-relations[, lofSNN[1], by=from]$V1
hist(lof2,200)

library(ldbod)
lof_res<-ldbod(cl_coord, k =  90, nsub = nrow(cl_coord), method = "lof")
#relations$lof_from <-lof_res$lof[relations$from] 
#relations$lof_to <- lof_res$lof[relations$to] 
#relations$weight2 <- relations$lof_from / relations$lof_to
#relations$lrd_from <-lof_res$lrd[relations$from] 
#relations$lrd_to <- lof_res$lrd[relations$to] 
hist(lof_res$lof,200)
hist(lof_res$lrd,500)
cor(lrd, lof_res$lrd)
cor(lof1, lof_res$lof, method='spearman')
cor(lof2, lof_res$lof, method='spearman')
#compare results
################################################################
helper_match_evaluate_multiple(ifelse(lof_res$lof>1.6, 1, 0), ifelse(lbls==0, 1, 0))
helper_match_evaluate_multiple(ifelse(lof2>1.5, 1, 0), ifelse(lbls==0, 1, 0))
#ROC curves
library(pROC)
response <- ifelse(lbls==0, 1, 0)
predictor1 <- lof_res$lof
Wroc <- roc(response, as.numeric(predictor1), direction='<')
plot(Wroc, col='red', main=i)
predictor2 <- lof2
TWroc <- roc(response, predictor2, direction='<')
plot(TWroc, add=T, col='blue') 


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


#add lrd to this
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
