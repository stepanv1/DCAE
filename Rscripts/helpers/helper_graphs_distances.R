#helper functions to calculate n-nearest neighbours, 
#distances, weights of edges in data sets
library(RANN.L1)
library(RANN)
library(FNN)
# cytofkit:::find_neighbors
#modified not exported function from cytofkit 
find_neighbors <- function(data, query, k, metric ='L1'){
  #euclidian metric
  #if (k>=nrow(data)) {k<-k-2}
  if (metric=='L2'){
      #browser()  
      nearest <- get.knnx(data, query, k=k+1, algorithm="kd_tree")
  }
  if (metric=='L1'){
    nearest <- RANN.L1::nn2(data, query, k+1, treetype = "bd", searchtype = "standard")
  }
  
    return(nearest)
}

find_neighbors_vec <- function(data, query, k, metric ='L1'){
  #euclidian metric
  if (metric=='L2'){
    nearest <- RANN::nn2(data, query=matrix(query, byrow=T, nrow=1), k+1, treetype = "bd", searchtype = "standard")
  }
  if (metric=='L1'){
    nearest <- RANN.L1::nn2(data, query=matrix(query, byrow=T, nrow=1), k+1, treetype = "bd", searchtype = "standard")
  }
  
  return(nearest)
}

#a=matrix(1:150000, nrow=10000, ncol=15)
#nearest <- RANN::nn2(a, a, k+1, treetype = "bd", searchtype = "standard") 
#zzz=find_neighbors_vec(a[1,],a, k=5,metric='L2')