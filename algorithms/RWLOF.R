# RWLOF algorithm
# To calculate outlier factor based on random walk
# comparing steady state probailities of the directed
# weighted graph
library(Matrix)
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_walker.R')

hist(rowSums(asym_w[lbls==0, ]), 50)
hist(rowSums(asym_w[lbls!=0, ]), 50)
hist(colSums(asym_w[, lbls==0 ]), 50)
hist(colSums(asym_w[, lbls!=0]), 50)
min(colSums(asym_w[, lbls!=0]))

asym_wn<-norm_mat(asym_w)
mat=norm_mat(asym_wn+norm_mat(sym_w))
mat=norm_mat(asym_w)
mat=norm_mat(sym_w)
RWLOF <- function(mat, alpha=0.99999, maxit=5000, eps=5e-20){
  mat=norm_mat(asym_w)
  PsAsym<-EnhanceDensity(P=mat, V=rep(1,  dim(mat)[1])/sum(rep(1,  dim(mat)[1])), debug=TRUE, maxit=maxit, alpha=alpha, eps= eps, smooth=FALSE);
  mat=norm_mat(sym_w)
  PsSym<-EnhanceDensity(P=mat, V=rep(1,  dim(mat)[1])/sum(rep(1,  dim(mat)[1])), debug=TRUE, maxit=maxit, alpha=alpha, eps= eps, smooth=FALSE);
  
  #mat <- (mat+t(mat))/2
  sumPsNeigh <- as.numeric((mat!=0) %*% Ps) / rowSums(mat!=0)
  sumPsNeigh <- as.numeric((norm_mat(asym_w)) %*% PsAsym) / rowSums(norm_mat(asym_w)!=0)
  RWLOF <- sumPsNeigh/PsSym
  
  
}

hist(RWLOF,50)
hist(log10(RWLOF),50)
kmeansres<-Ckmeans.1d.dp(log10(RWLOF), k=c(2), y=1,
                         method= "quadratic",
                         estimate.k="BIC")
plot(kmeansres)

#plot(Ckmeans.1d.dp(exp(logP)), k=c(1:20), y=1, estimate.k="BIC")

table(kmeansres$cluster)

table(lbls)
table(lbls[kmeansres$cluster==1])
helper_match_evaluate_multiple(kmeansres$cluster, ifelse(lbls==0, 1, 0))
helper_match_evaluate_multiple(kmeansres$cluster, ifelse(log10(RWLOF)>2, 1, 0))
table(log10(RWLOF)>2)

kmeansres<-Ckmeans.1d.dp(log10(P0+10^(-20)), k=c(2), y=1,
                         method= "loglinear",
                         estimate.k="BIC")
plot(kmeansres)

#plot(Ckmeans.1d.dp(exp(logP)), k=c(1:20), y=1, estimate.k="BIC")

table(kmeansres$cluster)

table(lbls)
table(lbls[kmeansres$cluster==1])
helper_match_evaluate_multiple(kmeansres$cluster, ifelse(lbls==0, 1, 0))
helper_match_evaluate_multiple(kmeansres$cluster, ifelse(P0==0, 1, 0))


