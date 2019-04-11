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

#generate a list of cluster coordinates
library(RANN)
find_neighbors <- function(data, k){
  nearest <- nn2(data, data, k, treetype = "bd", searchtype = "standard")
  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
}
k=90
#system.time(res_annoyKnn <- annoyKnn(cl_coord,k=k+1,ntree=1200 ))
#gc()

system.time(m2<-find_neighbors(cl_coord, k=k+1))
neighborMatrix <- (m2$nn.idx)[,-1]
system.time(links <- cytofkit:::jaccard_coeff(m2$nn.idx))
gc()
links <- links[links[,1]>0, ]
relations <- as.data.frame(links)
#save(relations, file='../../results/relationsSamusik_01.RData')
colnames(relations)<- c("from","to","weight")
relations<-as.data.table(relations)
#relations<-relations[from!=to, ]# remove self-loops
#sum((res_annoyKnn$ind[,1:31]-m2$nn.idx)!=0)

asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight, symmetric = F, index1=T);gc()
asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
#run one reweight cycle
sym_w<-(asym_w+t(asym_w))/2
nnzero(sym_w)
table(round(colSums(sym_w)))
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_walker.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')

clus_assign<-vector()
clus_assign_ind<-vector()
louvain_assign<-vector()
outliers <- vector()
#reweight SYMMERIC cmatrix #####################################################
##################################################################################
asym_rw <- mcReweightLocal(asym_w, addLoops = FALSE, expansion = 2, inflation = 1.5,  max.iter = 1, ESM = TRUE , stop.cond=dim(asym_w)[1])[[2]];gc() #version with assymetric input matrix
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

louvain_assign<-cl_resL
clus_assign=cl_res
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

deg <- rowSums(asym_w!=0) #degree of outcoming links
minweight <- deg/(2*k-1) #lowest possible value of the link is 1/(2*k-1), then minimal weight is this
IDXe <- which(rowSums(asym_w) <= (minweight & deg<=k))
#IDXe<-which(round(colSums(sym_w))==0)

IDXw<-which(colSums(gr_w)==0)#condition that there is no incoming links

table(lbls[(1:length(lbls) %in% which(colSums(asym_w)==0))]) #initial number of nodes with zero incoming links
lbls[intersect(which(colSums(asym_w)==0), IDXw)]#how many new weak connected links are discovered
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


###############################################################################################
#assignment of outliers to clusters and evaluation
#using silhouette and F1 measure. Louvain algorithm is run
#on the set without  outliers and on the set with
#outliers present
################################################################################################
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
labels_out <- helper_assign_outliers(bulk_data =cl_coord[outliers,], out_data = cl_coord[!outliers, ], bulk_labels = mbri); gc()
  
comRF<-rep(NA,length(lbls))
comRF[outliers]<-mbri
comRF[!outliers]<-labels_out
table(comRF)
  
helper_match_evaluate_multiple(mbr, lbls)
helper_match_evaluate_multiple(comRF, lbls)
helper_match_evaluate_multiple(lva, lbls)
  
table(lbls)


IDX <- outliers
#tsne visualisation
library(Rtsne)
tsne3D <- Rtsne(X=cl_coord, dims = 3, perplexity = 30, verbose = T, max_iter = 1000) 
tsne <- Rtsne(X=cl_coord, dims = 2, perplexity = 30, verbose = T, max_iter = 1000) 
ncolors=length(unique(lbls))
col_true=rainbow(ncolors)
colors<-unlist(lapply(lbls, function(x) col_true[x]))
colors[lbls==0]='black'
color2=ifelse(IDX, 'green', 'red' ); color2[lbls==0 & IDX]='black' #check hits and misses
color3=ifelse(IDX, 'green', 'red'); color3[lbls!=0 & !IDX]='black' #check hits and misses
#data<- cbind(f, clusters)
#plot(res_tsne$tsne_out$Y,col=colors, pch='.', cex=1)

plot(tsne$Y,  col=colors, pch='.')
plot(tsne$Y,  col=ifelse(IDX, 'green', 'red'), pch='.', cex=2.8)
plot(tsne$Y,  col=ifelse(lbls==0, 'red', 'green'), pch='.')
plot(tsne$Y,  col=color2, pch='.', cex=2.8, main='Blacks are false negatives')
plot(tsne$Y,  col=color3, pch='.', cex=2.8, main='Blacks are false positives')
resi<-helper_match_evaluate_multiple(membership(comi), lbls)



open3d()
plot3d(tsne3D$Y,  col=colors, pch='.') 

open3d()
plot3d(tsne3D$Y,  col=ifelse(IDX, 'green', 'red'), pch='.') 

open3d()
plot3d(tsne3D$Y,  col=ifelse(lbls==0, 'red', 'green'), pch='.') 

open3d()#check hits and misses
plot3d(tsne3D$Y,  col=color2, pch='o') 


#experiment with INFOMAP and brute force modularity via aggregation and simplification of the clusters

#g_small<- cut_at(cl_resL, no=10000)
gcon <- contract.vertices(gL, cl_res$membership, vertex.attr.comb = list(weight = "sum"))
gcon <- simplify(gcon, edge.attr.comb=list(weight="sum"))
gcon=delete.edges(gcon,which(E(gcon)$weight==0))
gcon=delete.vertices(gcon,which(strength(gcon)<0.6))
V(gcon)$weight<-table(cl_res$membership)
#V(gcon)$weight<-0
#plot(gcon, edge.width=E(gcon)$weight)

cl_crs <- cluster_infomap(gcon)
table(membership(cl_crs))
#############################################################################################################
## 
#############################################################################################################
asym_wn<-norm_mat(asym_w)
Pasym_w<-EnhanceDensity(P=(asym_wn), V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), smooth=FALSE, debug=TRUE, maxit=5000, alpha=1, eps=1e-16)
asym_wn2 <- as((asym_wn) * Pasym_w, "dgCMatrix") #get the probability for edge and inflate
asym_wn2 <- norm_mat(asym_wn2);gc()
#asym_rw <-  asym_wn

sum(asym_wn2==0)
min(asym_wn2)
hist(asym_wn2@x,50000, xlim=c(0, 1e-7))
hist(colSums(asym_wn2)[colSums(asym_wn2)>0],50000, xlim=c(0.0, 1e-7))
table(round(colSums(asym_wn)))

Pasym_w<-set_zero(Pasym_w, 1.7e-10)
hist(Pasym_w2@x,500000, xlim=c(0, 2e-9))
Pasym_w2<-norm_mat(Pasym_w2)


table(lbls[colSums(asym_w)==0])
table(lbls[colSums(asym_wn)==0])
table(lbls[Pasym_w==0])
table(lbls)
hist(Pasym_w,500000, xlim=c(0,1e-6))

#repeat to remove sources==outliers:
m <- asym_wn
IDX=1:ncol(m)
d=1
while (d!=0){
  l=length(IDX)
  IDX <- IDX[colSums(m[IDX, IDX])!=0]
  d=l-length(IDX)
  
}
table(lbls)-table(lbls[IDX])
table(lbls[IDX])
table(lbls)
asym_wn<-norm_mat(asym_w)
IDXo<-sample((1:length(lbls))[lbls==0], 400); IDXi <- sample((1:length(lbls))[lbls!=0], 400)
Ptrace<-EnhanceDensityTrace(P=(asym_wn), V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), IDX=c(IDXo, IDXi), smooth=FALSE, debug=TRUE, maxit=5000, alpha=0.999, eps=1e-20)
#Assign initial weights according incoming links
#Ptrace<-EnhanceDensityTrace(P=(asym_wn), V=(colSums(asym_wn))/sum(asym_wn), IDX=c(IDXo, IDXi), smooth=F, debug=TRUE, maxit=5000, alpha=1, eps=0)

matplot(log10(t(Ptrace$Piter[,])+min(Ptrace$Piter[,5001][Ptrace$Piter[,5001]>0])), pch='.', col = ifelse((1:800)>=401, 'red', 'black'), type = "l")
matplot(log(t(Ptrace$Piter[1:400,])+0), pch='.', type = "l", main='Outliers')
matplot(log(t(Ptrace$Piter[400:800,250:400])+0), pch='.', type = "l",  main='Core points')
matplot((t(Ptrace$Piter[,])+0), pch='.', col = ifelse((1:800)>=401, 'red', 'black'), type = "l")
hist(asym_wn[, lbls==0]@x,200)

x1=4;x2=3
plot(cl_coord[c(IDXi, IDXo) , c(x1,x2)], col = ifelse((1:800)>=401, 'red', 'black'))
pairs(cl_coord[c(IDXi, IDXo) , 1:5], col = ifelse((1:800)>=401, 'red', 'black'), pch='.')
pairs(cl_coord[c(IDXi, IDXo[150]) , ], col = ifelse((1:401)>=401, 'red', 'black'), pch='o')

#reciprocity check
#Outliers detected do not have many non-reciprocal 
#incedent edges
sum(asym_wn[lbls!=0, lbls==0]);nnzero(asym_wn[lbls!=0, lbls==0]);sum(asym_wn[lbls!=0, lbls==0])/nnzero(asym_wn[lbls!=0, lbls==0]);
sum(asym_wn[lbls==0, lbls==0]);nnzero(asym_wn[lbls==0, lbls==0]);sum(asym_wn[lbls==0, lbls==0])/nnzero(asym_wn[lbls==0, lbls==0]);
sum(asym_wn[lbls!=0, lbls!=0]);nnzero(asym_wn[lbls!=0, lbls!=0]);sum(asym_wn[lbls!=0, lbls!=0])/nnzero(asym_wn[lbls!=0, lbls!=0]);
sum(asym_wn[lbls==0, lbls!=0]);nnzero(asym_wn[lbls==0, lbls!=0]);sum(asym_wn[lbls==0, lbls!=0])/nnzero(asym_wn[lbls==0, lbls!=0]);
table(lbls)
hist(Ptrace$fin,500)

logP<-as.numeric(log(Ptrace$Piter[1:800,5000]))

library('Ckmeans.1d.dp')
minLog<-min(logP[logP!=-Inf])
#kmeansArg=ifelse(as.numeric(log(Ptrace$Piter[1:200,2]))==-Inf)
logP[logP==-Inf] <- minLog
kmeansres<-Ckmeans.1d.dp(logP, k=c(2), y=1,
              method= "loglinear",
              estimate.k="BIC")
plot(kmeansres)
#plot(Ckmeans.1d.dp(exp(logP), k=c(1:20), y=1,
#          method= "loglinear",
#          estimate.k="BIC")

table(kmeansres$cluster)
#kmeansres$cluster

#using 'damping coefficient'
logP<-log(as.numeric(Ptrace$k))

library('Ckmeans.1d.dp')
minLog<-min(logP[logP!=-Inf])
#kmeansArg=ifelse(as.numeric(log(Ptrace$Piter[1:200,2]))==-Inf)
logP[logP==-Inf] <- minLog
kmeansres<-Ckmeans.1d.dp(logP, k=c(2), y=1,
                         method= "loglinear",
                         estimate.k="BIC")
plot(kmeansres)
#plot(Ckmeans.1d.dp(exp(logP)), k=c(1:20), y=1, estimate.k="BIC")
     
table(kmeansres$cluster)
kmeansres$cluster
table(lbls[kmeansres$cluster==1])
table(lbls)
##################################################################
#clustering by tyhe last value of probability
#using 'damping coefficient'
logP<-log(as.numeric(Ptrace$fin))

znodes <- which(as.numeric(Ptrace$fin)==0)
nznodes <- which(as.numeric(Ptrace$fin)!=0)
library('Ckmeans.1d.dp')
#minLog<-min(logP[logP!=-Inf])
#kmeansArg=ifelse(as.numeric(log(Ptrace$Piter[1:200,2]))==-Inf)
#logP[logP==-Inf] <- minLog
hist(logP, 500)
kmeansres<-Ckmeans.1d.dp(logP[nznodes], k=c(2), y=1,
                         method= "loglinear",
                         estimate.k="BIC")
plot(kmeansres)

#plot(Ckmeans.1d.dp(exp(logP)), k=c(1:20), y=1, estimate.k="BIC")

table(kmeansres$cluster)

table(lbls)
table(lbls[kmeansres$cluster==1])
helper_match_evaluate_multiple(kmeansres$cluster, ifelse(lbls==0, 1, 0))
helper_match_evaluate_multiple(ifelse(as.numeric(Ptrace$fin)==0, 0,1), ifelse(lbls==0, 1, 0))

#by the value of k, damping coefficient
table(lbls[as.numeric(Ptrace$k)==0])
#by the value of probability
table(lbls[as.numeric(Ptrace$fin)==0])




# why some comverge to zero?
hist(colSums(asym_wn[,IDXo])-rowSums(asym_wn[IDXo,]),500)
hist(colSums(asym_wn[,IDXi])-rowSums(asym_wn[IDXi,]),500)
table(lbls[colSums(asym_wn[,])-rowSums(asym_wn[,])<0])

#testing package ldbod
library(ldbod)
library(Ckmeans.1d.dp)
system.time(outl_res<-ldbod(cl_coord, k = c(10, 20, 30, 60), nsub = nrow(cl_coord),  ldf.param = c(h = 1, c = 0.1), rkof.param = c(alpha = 1, C = 1, sig2 = 1), lpdf.param = c(cov.type = "full", sigma2 = 1e-05, tmax = 1, v = 1), treetype = "kd", searchtype = "standard", eps = 0, scale.data = TRUE, method = "lof"))

hist(log10(outl_res$lof[, 3]),50)
kmeansresLD<-Ckmeans.1d.dp(outl_res$lof[, 3], k=2, y=1,
                         method= "quadratic",
                         estimate.k="BIC")
plot(kmeansresLD)
#plot(Ckmeans.1d.dp(exp(logP)), k=c(1:20), y=1, estimate.k="BIC")
table(kmeansresLD$cluster)
table(lbls)
table(lbls[kmeansresLD$cluster==2])
helper_match_evaluate_multiple(kmeansresLD$cluster, ifelse(lbls==0, 1, 0))

plot(scale(-log10(Ptrace$fin + min(Ptrace$fin[Ptrace$fin>0]))) , outl_res$lof[, 3], pch='.', col=ifelse(lbls==0, 'red', 'black' ))
plot(scale(-log10(Ptrace$fin + min(Ptrace$fin[Ptrace$fin>0]))) , pch='.', col=ifelse(lbls==0, 'red', 'black' ))
table(ifelse(Ptrace$fin==0, 'o','i'), ifelse(lbls==0,1,0))
table(ifelse(kmeansresLD$cluster==1, 'i', 'o'), ifelse(lbls==0,1,0))
beep()
#testing package abodOutlier
library(abodOutlier)
#abod_res<-abod(cl_coord, method = "knn", k = 30)
hist(outl_res$lof[, 4],50)
kmeansresLD<-Ckmeans.1d.dp(outl_res$lof[, 3], k=2, y=1,
                           method= "loglinear",
                           estimate.k="BIC")
plot(kmeansresLD)
#plot(Ckmeans.1d.dp(exp(logP)), k=c(1:20), y=1, estimate.k="BIC")
table(kmeansresLD$cluster)
table(lbls)
table(lbls[kmeansresLD$cluster==2])
helper_match_evaluate_multiple(kmeansresLD$cluster, ifelse(lbls==0, 1, 0))

plot(scale(-log10(Ptrace$fin + min(Ptrace$fin[Ptrace$fin>0]))) , outl_res$lof[, 3], pch='.', col=ifelse(lbls==0, 'red', 'black' ))
plot(scale(-log10(Ptrace$fin + min(Ptrace$fin[Ptrace$fin>0]))) , pch='.', col=ifelse(lbls==0, 'red', 'black' ))
table(ifelse(Ptrace$fin==0, 'o','i'), ifelse(lbls==0,1,0))
table(ifelse(kmeansresLD$cluster==1, 'i', 'o'), ifelse(lbls==0,1,0))

# testing ‘HighDimOut’, namely SOD


#test flowMeans on artificial data
library(flowMeans)
system.time(cl_flowM<-flowMeans(cl_coord[ , ], NumC = 15)@Label) ;gc()
helper_match_evaluate_multiple(cl_flowM, lbls)
helper_match_evaluate_multiple(cl_flowM[lbls!=0], lbls[lbls!=0])

system.time(cl_flowMout<-flowMeans(cl_coord[lbls!=0 , ], NumC = 15)@Label) ;gc()
helper_match_evaluate_multiple(cl_flowMout, lbls[lbls!=0])

system.time(cl_flowM<-flowMeans(cl_coord[ , ], NumC = 15)@Label) ;gc()
helper_match_evaluate_multiple(cl_flowM, lbls)
helper_match_evaluate_multiple(cl_flowM[lbls!=0], lbls[lbls!=0])

system.time(cl_flowMout<-flowMeans(cl_coord[lbls!=0 , ], NumC = 15)@Label) ;gc()
helper_match_evaluate_multiple(cl_flowMout, lbls[lbls!=0])

