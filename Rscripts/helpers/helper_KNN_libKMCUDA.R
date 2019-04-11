#find nearest neighbors with CUDA and kmeans
library(kmcudaR)
find_neighborsCUDA <- function(data, k_, metric='euclidean', cores=12){
ca <- kmeans_cuda(data, 25, metric="L2", verbosity=0, seed=3, device=0)
neighbors <- knn_cuda(k_, data, centroids=ca$centroids, assignments=ca$assignments, metric="L2", verbosity=0, device=0)
return(list('dist'=NULL, 'idx' = neighbors))
}
#a=matrix(1:150000, nrow=10000, ncol=15)
#nbrs<-find_neighborsCUDA(a, k_, metric='euclidean', cores=12)