#experiments with SIMLR
library(SIMLR)
resSIMLR<-SIMLR(X=t(cl_coord[1:1000,]), c=15, no.dim = NA, k = 10, if.impute = FALSE, normalize = FALSE,
              cores.ratio = 0);gc()

helper_match_evaluate_multiple(resSIMLR$y$cluster,lbls[1:1000])
resSIMLRLarge<-SIMLR_Large_Scale(X=t(cl_coord[,]), c=15, k = 30, kk = 16, if.impute = FALSE,
                   normalize = F)
table(resSIMLRLarge$y$cluster)
helper_match_evaluate_multiple(resSIMLRLarge$y$cluster,lbls[])
table(resSIMLRLarge$y$cluster[lbls==0])
View(resSIMLRLarge$S0)
View(resSIMLRLarge$val)

#extract SO
n=dim(cl_coord)[1]
Im=unlist(lapply(1:n, function(x) rep(x, 60-1)))
Jm=c(t(resSIMLRLarge$ind[,2:60]))
Xm=c(t(resSIMLRLarge$val[,2:60]))
SIMLm <- sparseMatrix(i = Im, j = Jm, x = Xm)

  resSIMLRLarge$ind

#calculate eigenvalue of similarity matrix using RSpectra
#Simple Laplacian
SL<-Diagonal(x=rowSums(asym_w))-asym_w
#generalised Laplacian
L= Diagonal(x=(rowSums(asym_w))^(-1)) %*% SL
#normalised Laplacian
L= Diagonal(x=(rowSums(asym_w))^(-1/2)) %*% SL %*% Diagonal(x=(rowSums(asym_w))^(-1/2))
library(RSpectra)
Lapl_eig<-eigs( SL , k=30, which = "SR", sigma = NULL);gc()
proj<-Re(Lapl_eig$vectors[,2:30]) / sqrt(rowSums((Lapl_eig$vectors[,2:(30)])^2))
plot(Re(Lapl_eig$values))
plot(diff(Re(Lapl_eig$values)))
plot(apply(Re(Lapl_eig$vectors), 2, function(x) sum((x)^4)))#localization
plot(log(apply(Re(Lapl_eig$vectors), 2, function(x) sum((x)^4))))#localization

gd <- graph_from_adjacency_matrix((t(asym_w)+asym_w)/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
#gd_info<-cluster_infomap(gd)

clus_assign_gd=membership(gd_info)
mbr<-clus_assign_gd

table(mbr)
#table(mbr2)
table(lbls)
table(lva)


helper_match_evaluate_multiple(mbr, lbls)
helper_match_evaluate_multiple(lva, lbls)
#Simple laplacian
SL_Sp_cluster<-function(mat, n){
  SL<-Diagonal(x=rowSums(mat))-mat
  print('Performing eigenvalue decomposition')
  Lapl_eig<-eigs( SL , k=n+1, which = "SR", sigma = NULL);gc()
  proj<-Re(Lapl_eig$vectors[,2:(n+1)]) / sqrt(colSums(Lapl_eig$vectors[,2:(n+1)]^2))
  print('Performing kmeans')
  return(kmeans(proj, n, nstart = 100, iter.max = 20)$cluster)
}
res1<-SL_Sp_cluster(asym_w,14)
helper_match_evaluate_multiple(res1, lbls)
#Random walk laplacian
Lrw_Sp_cluster<-function(mat, n){
  Lrw<- Diagonal(x=(rowSums(asym_w))^(-1)) %*% ( Diagonal(x=rowSums(mat))-mat )
  print('Performing eigenvalue decomposition')
  Lapl_eig<-eigs( Lrw , k=n+1, which = "SR", sigma = NULL);gc()
  proj<-Re(Lapl_eig$vectors[,2:(n+1)]) / sqrt(colSums(Lapl_eig$vectors[,2:(n+1)]^2))
  print('Performing kmeans')
  return(kmeans(proj, n, nstart = 100, iter.max = 100, algorithm="Lloyd")$cluster)
}
res2<-Lrw_Sp_cluster(asym_w,14)
helper_match_evaluate_multiple(res2, lbls)

LNG_Sp_cluster<-function(mat, n){
  LNG<- Diagonal(x=(rowSums(mat))^(-1/2)) %*% ( Diagonal(x=rowSums(mat))- mat ) %*% Diagonal(x=(rowSums(mat))^(-1/2))
  print('Performing eigenvalue decomposition')
  Lapl_eig<-eigs( LNG , k=n+1, which = "SR", sigma = NULL);gc()
  proj<-Re(Lapl_eig$vectors[,2:(n+1)]) / sqrt(colSums(Lapl_eig$vectors[,2:(n+1)]^2))
  print('Performing kmeans')
  return(kmeans(proj, n, nstart = 100, iter.max = 100, algorithm="Lloyd")$cluster)
}
res3<-LNG_Sp_cluster(sym_w[rowSums(sym_w)!=0,rowSums(sym_w)!=0],14)
helper_match_evaluate_multiple(res3, lbls)

res4<-LNG_Sp_cluster(SIMLm,15)
helper_match_evaluate_multiple(res4, lbls)

library('ClusterR')
resGMM <- GMM(cl_coord, gaussian_comps = 15, dist_mode = "eucl_dist",
    seed_mode = "random_subset", km_iter = 500, em_iter = 500,
    verbose = FALSE, var_floor = 1e-10, seed = 1)
helper_match_evaluate_multiple(apply(resGMM$Log_likelihood, 1, function(x) which.max(x)), lbls)#F1 0.78

resGMM <- GMM(proj, gaussian_comps = 15, dist_mode = "eucl_dist",
              seed_mode = "random_subset", km_iter = 500, em_iter = 100,
              verbose = FALSE, var_floor = 1e-10, seed = 1)
helper_match_evaluate_multiple(apply(resGMM$Log_likelihood, 1, function(x) which.max(x)), lbls)


resKMpp <- KMeans_rcpp(proj, 15, num_init = 100, max_iters = 100,
            initializer = "kmeans++", fuzzy = FALSE, threads = 10,
            verbose = FALSE, CENTROIDS = NULL, tol = 1e-04,
            tol_optimal_init = 0.3, seed = 1)$clusters
helper_match_evaluate_multiple(resKMpp, lbls)

resMed<-Cluster_Medoids(proj, 15, distance_metric = "euclidean",
                minkowski_p = 1, threads = 10, swap_phase = TRUE, fuzzy = FALSE,
                verbose = FALSE, seed = 1);gc()
helper_match_evaluate_multiple(resMed$clusters, lbls)#F1 0.66

#reorder according clustering and plot
IDX<-order(lbls)
sym_w_o<-sym_w[, IDX];sym_w_o<-sym_w_o[IDX, ]
sIDX<-sort(sample(1:length(IDX), 3500))
pal <- colorRampPalette(c("red", "yellow"), space = "rgb")
library(heatmap3)
heatmap(as.matrix(sym_w_o[sIDX,sIDX]),Rowv=NA, Colv=NA)
heatmap3(log(as.matrix(t(sym_w_o[sIDX,sIDX]))+0.000001),Rowv=NA, Colv=NA, useRaster=F)

table(lbls)
#compare connection in between and insede clusters 1 and 2
sum(sym_w[lbls==1,lbls==1])/nnzero(sym_w[lbls==1, lbls==1])#  0.173846
sum(sym_w[lbls==2,lbls==2])/nnzero(sym_w[lbls==2, lbls==2])#  0.4010785
sum(sym_w[lbls==1,lbls==2])/nnzero(sym_w[lbls==1, lbls==2])#  0.01673856

#matrix structure for real data
#load data in memory (first part of run_mcodc.R)
i=1 #choose set
IDX<-ifelse(is.na(clus_truth[[i]]), F, T)
cl_tr <- clus_truth[[i]][IDX]
data_tr<-data[[i]][IDX, ]
sym_w<-sym_w[, IDX];sym_w<-sym_w[IDX, ]

IDX<-order(cl_tr)
sym_w_o<-sym_w[, IDX];sym_w_o<-sym_w_o[IDX, ]
sIDX<-sort(sample(1:length(IDX), 3500))
pal <- colorRampPalette(c("red", "yellow"), space = "rgb")
library(heatmap3)
heatmap(as.matrix(sym_w_o[sIDX,sIDX]),Rowv=NA, Colv=NA)
heatmap3(log(as.matrix(t(sym_w_o[sIDX,sIDX]))+0.000001),Rowv=NA, Colv=NA, useRaster=F)

table(cl_tr)
i1=5;i2=4
sum(sym_w[cl_tr==i1,cl_tr==i1])/nnzero(sym_w[cl_tr==i1, cl_tr==i1])#  0.173846
sum(sym_w[cl_tr==i2,cl_tr==i2])/nnzero(sym_w[cl_tr==i2, cl_tr==i2])#  0.401078i1
sum(sym_w[cl_tr==i1,cl_tr==i2])/nnzero(sym_w[cl_tr==i1, cl_tr==i2])#  0.016738i16

sum(sym_w[cl_tr==i1,cl_tr==i1])
sum(sym_w[cl_tr==i2,cl_tr==i2])
sum(sym_w[cl_tr==i1,cl_tr==i2])

res3<-LNG_Sp_cluster(sym_w[rowSums(sym_w)!=0,rowSums(sym_w)!=0],24)
helper_match_evaluate_multiple(res3, cl_tr[rowSums(sym_w)!=0])

#affinity propagation clustering
library(apcluster, q=0)
resapc<-apcluster(s=sym_w, details=T, p=-3407.731, q=0, maxits=10000)
resapcK<-apclusterK(s=sym_w, details=T, K=15, verbose=TRUE, p=-3299.274)
preferenceRange(s=sym_w)


