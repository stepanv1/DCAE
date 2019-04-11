#generate clusters with bridge points by overlaping them
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
library(Matrix)
library(data.table)

chunks2<- function(n, nChunks) {split(1:n, ceiling(seq_along(1:n)/(n/nChunks)))}
perc.rank <- function(x) trunc(rank(x))/length(x)
n_clust=3
system.time(clus_set<-genRandomClust(numClust=n_clust,
                         sepVal=0.0,
                         numNonNoisy=5,
                         numNoisy=0,
                         numOutlier=0,
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
                         rotateind=TRUE,
                         iniProjDirMethod="SL",
                         projDirMethod="newton",
                         alpha=0.05,
                         ITMAX=20,
                         eps=1.0e-10,
                         quiet=FALSE,
                         outputDatFlag=TRUE,
                         outputLogFlag=TRUE,
                         outputEmpirical=TRUE,
                         outputInfo=TRUE))


cl_coord=clus_set$datList$test_1
lbls=clus_set$memList$test_1
table(lbls)

cl_pca <- prcomp(cl_coord,
                 center = TRUE,
                 scale. = TRUE) 
library(ggfortify)
autoplot(cl_pca, alpha = 0.3)


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
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_walker.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')

clus_assign<-vector()
clus_assign_ind<-vector()
louvain_assign<-vector()
outliers <- vector()



#############################################################################################################
## 
#############################################################################################################
asym_wn<-norm_mat(asym_w)
Pasym_w<-EnhanceDensity(P=norm_mat(gr_res$Equilibrium.state.matrix - Diagonal(x = diag(gr_res$Equilibrium.state.matrix))), V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), smooth=FALSE, debug=TRUE, maxit=5000, alpha=1, eps=1e-16)

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
IDXo<-sample((1:length(lbls))[lbls==1], 400); IDXi <- sample((1:length(lbls))[lbls==2], 400)
Ptrace<-EnhanceDensityTrace(P=asym_wn, V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), IDX=c(IDXo, IDXi), smooth=FALSE, debug=TRUE, maxit=5000, alpha=1, eps=1e-20)
#Assign initial weights according incoming links
#Ptrace<-EnhanceDensityTrace(P=(asym_wn), V=(colSums(asym_wn))/sum(asym_wn), IDX=c(IDXo, IDXi), smooth=F, debug=TRUE, maxit=5000, alpha=1, eps=0)

matplot(log10(t(Ptrace$Piter[,])+min(Ptrace$Piter[,5001][Ptrace$Piter[,5001]>0])), pch='.', col = ifelse((1:800)>=401, 'red', 'black'), type = "l")
matplot(log(t(Ptrace$Piter[1:400,])+0), pch='.', type = "l", main='Outliers')
matplot(log(t(Ptrace$Piter[400:800,250:400])+0), pch='.', type = "l",  main='Core points')
matplot((t(Ptrace$Piter[,])+0), pch='.', col = ifelse((1:800)>=401, 'red', 'black'), type = "l")

logP<-as.numeric(log(Ptrace$fin))

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

cl_pca <- prcomp(cl_coord,
                 center = TRUE,
                 scale. = TRUE) 
library(ggfortify)
tkm=table(kmeansres$cluster)
autoplot(cl_pca, col= ifelse(kmeansres$cluster==as.numeric(names(which(tkm == min(tkm)))), 'red', 'green'), alpha=0.1)
autoplot(cl_pca, col= ifelse(Ptrace$fin == 0, 'red', 'green'), alpha=0.5)


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
allRes <- 
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
outl_res<-ldbod(cl_coord, k = c(10, 20, 30, 60), nsub = nrow(cl_coord),  ldf.param = c(h = 1, c = 0.1), rkof.param = c(alpha = 1, C = 1, sig2 = 1), lpdf.param = c(cov.type = "full", sigma2 = 1e-05, tmax = 1, v = 1), treetype = "kd", searchtype = "standard", eps = 0, scale.data = TRUE)

hist(outl_res$lof[, 3],50)
kmeansresLD<-Ckmeans.1d.dp(outl_res$lof[, 3], k=2, y=1,
                           method= "loglinear",
                           estimate.k="BIC")
plot(kmeansresLD)
#plot(Ckmeans.1d.dp(exp(logP)), k=c(1:20), y=1, estimate.k="BIC")
#stretch the LOF index

rbPal <- colorRampPalette(c('red', 'orange', 'yellow', 'green', 'blue'))
Col <- rbPal(101)[  round(perc.rank(outl_res$lof[, 3])*100)]
plot(cl_pca$x[,1], cl_pca$x[,2],  col=Col, pch='.', cex=2)

rbPal <- colorRampPalette(c('red', 'orange', 'yellow', 'green', 'blue'))
Col <- rbPal(101)[ 100-round(perc.rank(Ptrace$fin)*100)]
plot(cl_pca$x[,1], cl_pca$x[,2],  col=Col, pch='.', cex=2)

plot(cl_pca$x[,1], perc.rank(Ptrace$fin)*100, pch='.')
plot(cl_pca$x[,2], perc.rank(Ptrace$fin)*100, pch='.')

plot(cl_pca$x[,1], 100-perc.rank(outl_res$lof[, 3])*100, pch='.')
plot(cl_pca$x[,2], 100-perc.rank(outl_res$lof[, 3])*100, pch='.')

require(akima) ; require(rgl)

nx=length(cl_pca$x[,1])
x = cl_pca$x[,1]; y =  cl_pca$x[,2]; z = (outl_res$lof[, 3])*10;
tmp=data.frame(sx=frankv(x, ties.method='random'), sy=frankv(y, ties.method='random'), z=z)
zm=matrix(0, ncol=nx, nrow=nx)

for (i in 1:nrow(tmp)){
zm[tmp$sx[i], tmp$sy[i] ] <- z[i]
}

akima.li=bicubic.grid(x, y, zm,  nx=100, ny=100)
surface3d(akima.li$x,akima.li$y,akima.li$z)

s=interp(x, y, z,  nx=100, ny=100, linear=T)
surface3d(s$x,s$y,s$z)

iRbic <- bicubic.grid(x,y,z,nx=250,ny=250)
### Note that this interpolation tends to extreme values in large cells.
### Therefore zmin and zmax are taken from here to generate the same
### color scheme for the next plots.
zmin <- min(iRbic$z, na.rm=TRUE)
zmax <- max(iRbic$z, na.rm=TRUE)
breaks <- pretty(c(zmin,zmax),10)
colors <- heat.colors(length(breaks)-1)
image(iRbic,breaks=breaks,col = colors)
contour(iRbic,col="black",levels=breaks,add=TRUE)
points(xy$x,xy$y)
title(main="bicubic interpolation",
      xlab="bcubic.grid(...)",
      sub="Akimas regular grid version, ACM 760")

gd<-graph_from_adjacency_matrix(asym_w, mode =  "directed", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
dec.res<-decompose(gd, mode = "strong", max.comps = 100, min.vertices = 10)
c.res <- components(gd, mode = "strong")
table(c.res$membership)
mst_gd <- mst(gd)
hist(E(gd)$weight,500)
hist(E(mst_gd)$weight,500)
