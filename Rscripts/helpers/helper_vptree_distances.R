#helper functions to calculate n-nearest neighbours, 
#distances, aprroximately
#One needes python package hnswlib installed
library(stsne)

#modified not exported function from cytofkit 
find_neighbors <- function(data,  k, metric ='L2', mc.cores=40){
  #euclidian metric
  #if (k>=nrow(data)) {k<-k-2}
  if (metric=='L2'){
    
    tree<-tree_create(data, mc.cores)
    neighbours<-tree_search(tree, data, k, num_threads=mc.cores) 
    indxs<-neighbours$idx
    dists<-neighbours$dist
    nearest <- list('nn.index'=indxs, 'nn.dist'=dists)
  }
  if (metric=='L1'){
    nearest <- RANN.L1::nn2(data, query, k, treetype = "bd", searchtype = "standard")
    }
  
    return(nearest)
}

find_neighbors_vec <- function(data, query, k, metric ='L2'){
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
find_neighborsTarget <- function(target, data, k, metric ='L2',  mc.cores=1){
  #euclidian metric
  #if (k>=nrow(data)) {k<-k-2}
  if (metric=='L2'){
    
    tree<-tree_create(data, mc.cores)
    neighbours<-tree_search(tree, target, k, num_threads=mc.cores) 

    indxs<-neighbours$idx
    dists<-neighbours$dist
    nearest <- list('nn.index'=indxs, 'nn.dist'=dists)
 
  }
  if (metric=='L1'){
    nearest <- RANN.L1::nn2(data, query, k, treetype = "bd", searchtype = "standard")
  }
  
  return(nearest)
}






