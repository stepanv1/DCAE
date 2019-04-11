#INFOMAP on top of MCL coarsened
#graph
library('clusterGeneration')
library('gmum.r')
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
library(Matrix)
library(data.table)

chunks2<- function(n, nChunks) {split(1:n, ceiling(seq_along(1:n)/(n/nChunks)))}


clus_set<-genRandomClust(numClust=15,
                         sepVal=-0.1,
                         numNonNoisy=15,
                         numNoisy=1,
                         numOutlier=0.05,
                         numReplicate=1,
                         fileName="test",
                         clustszind=2,
                         clustSizeEq=100,
                         rangeN=c(200,4000),
                         clustSizes=NULL,
                         covMethod="eigen",
                         rangeVar=c(2, 30),
                         lambdaLow=3,
                         ratioLambda=7,
                         alphad=1,
                         eta=1,
                         rotateind=TRUE,
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

#generate a list of cluster coordinates
library(RANN)
find_neighbors <- function(data, k){
  nearest <- nn2(data, data, k, treetype = "bd", searchtype = "standard")
  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
}
gc()
k=30
m2<-find_neighbors(cl_coord, k=k+1);
neighborMatrix <- (m2$nn.idx)[,-1]
system.time(links <- cytofkit:::jaccard_coeff(m2$nn.idx))
gc()
links <- links[links[,1]>0, ]
relations <- as.data.frame(links)
#save(relations, file='../../results/relationsSamusik_01.RData')
colnames(relations)<- c("from","to","weight")
relations<-as.data.table(relations)
#relations<-relations[from!=to, ]# remove self-loops


asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight, symmetric = F, index1=T);gc()
asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
#run one reweight cycle
sym_w<-(asym_w+t(asym_w))/2
nnzero(sym_w)
table(round(colSums(sym_w)))
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods/helper_walker.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')

clus_assign<-vector()
clus_assign_ind<-vector()
louvain_assign<-vector()
outliers <- vector()
#reweight SYMMERIC cmatrix #####################################################
##################################################################################
asym_rw<-mcReweightLocal(asym_w, addLoops = FALSE, expansion = 2, inflation = 1.5,  max.iter = 1, ESM = TRUE , stop.cond=0.9)[[2]];gc() #version with assymetric input matrix
head(sort(colSums(asym_rw),decreasing = T)) 
table(round(colSums(asym_rw)))
hist(colSums(asym_rw),500)

hist((asym_rw@x),500)

g<-graph_from_adjacency_matrix((asym_rw+t(asym_rw))/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()

gL<-graph_from_adjacency_matrix((asym_w+t(asym_w))/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()

g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
gL<-simplify(gL, remove.loops=T, edge.attr.comb=list(weight="sum"))

cl_resL<-louvain_multiple_runs(gL, num.run = 1);gc()
cl_res<-louvain_multiple_runs(g, num.run = 1);gc()

louvain_assign<-membership(cl_resL)
clus_assign=membership(cl_res)
mbr<-clus_assign
lva<-louvain_assign
#mbr2<-membership(cl_res2)

table(mbr)
#table(mbr2)
table(lbls)
table(lva)


helper_match_evaluate_multiple(mbr, lbls)
helper_match_evaluate_multiple(lva, lbls)

#idx<-!is.na(lbls)
#helper_match_evaluate_multiple(mbr[idx], lbls[idx])
#helper_match_evaluate_multiple(mbr2[idx], lbls[idx])

##########################################################################
#outlier removal
gr_res <- mcOutLocal(asym_w, addLoops = F, expansion = 2, inflation = 1,  max.iter = 1, ESM = TRUE); gc()
gr_res[[1]]
gr_w<-gr_res[[2]] 
head(sort(colSums(gr_w),decreasing = T))
hist(colSums(gr_w),  50000)
hist(colSums(gr_w), xlim=c(0,0.005), 50000)
table(lbls[rowSums(gr_w)==0])
table(lbls[colSums(gr_w)==0])
table(lbls[colSums(gr_w)>0])
sum(colSums(gr_w)==0)
table(lbls)

#plot(colSums(gr_w), unlist(lapply(1:nrow(gr_w), function(x) nnzero(gr_w[x,]))))
table(lbls[colSums(gr_w)==0])

deg <- colSums(asym_w!=0)
minweight <- deg/(2*k-1)
IDXe <- which(colSums(asym_w)<=minweight & deg<=k)
#IDXe<-which(round(colSums(sym_w))==0)

IDXw<-which(colSums(gr_w)==0)

IDX <- !((1:length(lbls)) %in% union(IDXe, IDXw))
table(lbls[!IDX])
table(lbls[(1:length(lbls) %in% IDXw)])
table(lbls[(1:length(lbls) %in% IDXe)])
outliers<-IDX

indg<-induced_subgraph(g, (1:gorder(g))[IDX])

#cl_res<-cluster_louvain(g)
cl_resi<-louvain_multiple_runs(indg, num.run = 1);gc()


clus_assign_ind=membership(cl_resi)

mbri<-clus_assign_ind


table(mbr)
table(mbri)
table(lbls)

comf<-rep(NA,length(lbls))
comf[outliers]<-mbri
comf[!outliers]<-100
table(comf)

helper_match_evaluate_multiple(mbr, lbls)
helper_match_evaluate_multiple(comf, lbls)
helper_match_evaluate_multiple(lva, lbls)



######################################################################################################
#experiment with INFOMAP via aggregation and simplification of the clusters
######################################################################################################
#rum MCL till it converges and create coarsened vesion of the graph###################################
asym_rw<-mcReweightLocal(asym_w, addLoops = FALSE, expansion = 2, inflation = 10,  max.iter = 100, ESM = TRUE , stop.cond=10000)[[2]];gc() #version with assymetric input matrix
head(sort(colSums(asym_rw),decreasing = T)) 
table(round(colSums(asym_rw)))
hist(colSums(asym_rw),500)
g_asym_rw=graph_from_adjacency_matrix((asym_rw), mode =  "directed", weighted = TRUE, diag = F,  add.colnames = NULL, add.rownames = NA); gc()
comp<-components(g_asym_rw, mode = "weak")
comp


#g_small<- cut_at(cl_resL, no=10000)
gcon <- contract.vertices(gL, comp$membership, vertex.attr.comb = list(weight = "sum"))
gcon <- simplify(gcon, edge.attr.comb=list(weight="sum"))
gcon=delete.edges(gcon,which(E(gcon)$weight==0))
#gcon=delete.vertices(gcon,which(strength(gcon)<0.6))
V(gcon)$weight<-table(comp$membership)
#V(gcon)$weight<-0
#plot(gcon, edge.width=E(gcon)$weight)

cl_crs <- cluster_infomap(gcon, e.weights = E(gcon)$weight, v.weights = V(gcon)$weight, nb.trials = 10)
cl_crs <- cluster_optimal(gcon)
cl_crs <- cluster_fast_greedy(gcon)
table(membership(cl_crs))
