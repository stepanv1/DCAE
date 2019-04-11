#helper functions to calculate n-nearest neighbours, 
#distances, aprroximately
#One needes python package hnswlib installed
library(RANN.L1)
library(RANN)
library(FNN)

#modified not exported function from cytofkit 
find_neighbors <- function(data,  k, metric ='L2', mc.cores=40){
  #euclidian metric
  #if (k>=nrow(data)) {k<-k-2}
  if (metric=='L2'){
    
    k<-as.integer(k+1)
    library(reticulate)
    use_python('/home/sgrinek/anaconda3/envs/PycharmProjects/bin/python')
   
    nmslib<-import('nmslib')
    np<-import('numpy')
    
    index <- nmslib$init(method='hnsw', space='l2',  data_type=nmslib$DataType$DENSE_VECTOR)
    index$addDataPointBatch(data)
    index$createIndex(dict(c('post', 2, 'efConstruction',5000,'M',64)), print_progress=T)
    
    # query for the nearest neighbours of the first datapoint
    #ids, distances = index$knnQuery(data[0,], k=k)
    
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    #reticulate::np_array(data, dtype = np$int)
    neighbours <- index$knnQueryBatch(data, k = k, num_threads=as.integer(mc.cores))
    indxs<-do.call(rbind,lapply(neighbours,  function(x){l<-length(x[[1]]);
        return(c(x[[1]], rep(NA, k-l))) } ))
    dists<-do.call(rbind,lapply(neighbours,  function(x){l<-length(x[[2]]);
    return(c(x[[2]], rep(NA, k-l))) } ))
    #rm(labels_distances)
    
    errRowsI<-unlist(lapply(1:nrow(indxs), function(x) indxs[x,1]!=(x-1)))
    errRowsD<-unlist(lapply(1:nrow(dists), function(x) any(is.na(dists[x,]))))
    errRows<-errRowsI | errRowsD
    print(paste0('number of nmslib errors  ', sum(errRows)))
    
    #more error corrections
    msk<-sum(errRows)
    if (sum(msk)!=0){
      print('Correcting for nmslib bug..')
      query<-data[errRows,]
      dim(query)<-c(length(data[errRows,])/ncol(data),  ncol(data))
      nearestCorrected <- RANN::nn2(data, query, k, treetype = "bd", searchtype = "standard")
      indxs[errRows,] <- nearestCorrected$nn.idx
      dists[errRows,] <- nearestCorrected$nn.dists
    }
    
    nearest <- list('nn.index'=indxs[,2:(k)], 'nn.dist'=dists[,2:(k)])
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
find_neighborsTarget <- function(target, data, k, metric ='L2'){
  #euclidian metric
  #if (k>=nrow(data)) {k<-k-2}
  if (metric=='L2'){
    
    k<-as.integer(k+1)
    library(reticulate)
    use_python('/home/sgrinek/anaconda3/envs/PycharmProjects/bin/python')
    
    nmslib<-import('nmslib')
    np<-import('numpy')
    
    index <- nmslib$init(method='hnsw', space='l2',  data_type=nmslib$DataType$DENSE_VECTOR)
    index$addDataPointBatch(data)
    index$createIndex(dict(c('post', 2, 'efConstruction',5000,'M',64)), print_progress=T)
    
    # query for the nearest neighbours of the first datapoint
    #ids, distances = index$knnQuery(data[0,], k=k)
    
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    #reticulate::np_array(data, dtype = np$int)
    neighbours <- index$knnQueryBatch(target, k = k, num_threads=as.integer(mc.cores))
    indxs<-do.call(rbind,lapply(neighbours,  function(x){l<-length(x[[1]]);
    return(c(x[[1]], rep(NA, k-l))) } ))
    dists<-do.call(rbind,lapply(neighbours,  function(x){l<-length(x[[2]]);
    return(c(x[[2]], rep(NA, k-l))) } ))
    #rm(labels_distances)
    
    errRowsI<-unlist(lapply(1:nrow(indxs), function(x) indxs[x,1]!=(x-1)))
    errRowsD<-unlist(lapply(1:nrow(dists), function(x) any(is.na(dists[x,]))))
    errRows<-errRowsI | errRowsD
    print(paste0('number of nmslib errors  ', sum(errRows)))
    
    #more error corrections
    msk<-sum(errRows)
    if (sum(msk)!=0){
      print('Correcting for nmslib bug..')
      query<-target[errRows,]
      dim(query)<-c(length(query)/ncol(query),  ncol(query))
      nearestCorrected <- RANN::nn2(data, query, k, treetype = "bd", searchtype = "standard")
      indxs[errRows,] <- nearestCorrected$nn.idx
      dists[errRows,] <- nearestCorrected$nn.dists
    }
    
    nearest <- list('nn.index'=indxs[,2:(k)], 'nn.dist'=dists[,2:(k)])
  }
  if (metric=='L1'){
    nearest <- RANN.L1::nn2(data, query, k, treetype = "bd", searchtype = "standard")
  }
  
  return(nearest)
}






