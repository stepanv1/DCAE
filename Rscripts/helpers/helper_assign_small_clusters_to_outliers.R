#assign clusters of size  1 to outliers #
#cl=c(1,2,3,3,4,4,4,5,5,5)
#outliers=rep(c(T,F), 10)
helper_assign_small_clusters_to_outliers<-function(cl, outliers, nmin=3){
  N<-length(outliers)
  tcl<-table(cl)
  newOutliers <- unlist(lapply(cl, function(x) ifelse(tcl[x]<nmin, T, F)))
  #add new outlers
  outliers[!outliers] <- newOutliers
  cl<-cl[!newOutliers]
  cl=as.integer(factor(cl,labels=1:length(unique(cl))))
  return(list('cl'=cl, 'outliers'=outliers))
}
library(mclust)
data=matrix(rnorm(1000), ncol=10)
 Mclust(as.matrix(data), G=1:5, modelNames = 'VVV')
 
 BIC <- mclustBIC(data)
 plot(BIC) 
summary(BIC) 