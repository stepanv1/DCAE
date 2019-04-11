#identify global outliers based on
#the projections onto reference dimensions

#dat <- data[[i]]; clus_assig <- clus_assign[[i]]; ref_sub<-ref_subs[[i]];glob_ou<-glob_out[[i]]
#data <- cl_coord[,15:25]; query=cl_coord[,15:25]; k=30; ref_subset <- lbls==5


library(Matrix)
library(data.table)
#source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_graphs_distances.R')
chunks2<- function(n, nChunks) {split(1:n, ceiling(seq_along(1:n)/(n/nChunks)))}

#Returns lof and probability to belong to the reference set of each
#point in the data, reference set is specified by ref_subset
#Attention: data should contain only not-noisy dimensions.
#of the reference set
helper_local_outliersLOF2<-function(data,  ref_subset, k=30){
  if (k>=sum(ref_subset)) {k<-sum(ref_subset)-2}#for super small clusters
  data[data<0]=0
  
  nearest<-find_neighborsTarget(data, data[ref_subset,],  k=k, metric ='L2', mc.cores=1) 
  neighborMatrix <- nearest$nn.index
  #correct for squared output of nearest neighbors
  mat_dist<-sqrt(nearest$nn.dist)
  
  #remove self-neighbors
  #neighborMatrix[ref_subset,1:k]<-neighborMatrix[ref_subset,2:(k+1)]
  #mat_dist[ref_subset,1:k]<- mat_dist[ref_subset,2:(k+1)]
  #neighborMatrix <- neighborMatrix[,1:k]
  #mat_dist<- mat_dist[,1:k]
  
  #create table for pairwise connections between reference points 
  l<-k*nrow(data)
  relations <- data.table(from = rep(1:nrow(data), each=k), to  = c(t(neighborMatrix)), dist=c(t(mat_dist)))
  
  
  ## compute lof and lrd for each k value specified by user
  k_dist<-mat_dist[,k]
  relations$k_dist <- k_dist[relations$to] 
  #compute reachebility distance
  relations$reach_dist <- relations[,pmax(k_dist,dist)]
  lrd <- 1/relations[, sum(.SD$reach_dist) /.N, by=from]$V1
  relations$lrdto <-  lrd[relations$to] 
  relations$lrdfrom <-  lrd[relations$from] 
  relations<- relations[, lof:=sum(.SD$lrdto)/.N/.SD$lrdfrom[1], by=from]
  
  #create output lof = rnorm(150)
  lof<-relations[, lof[1], by=from]$V1
  ecdf_lof<-ecdf(lof[ref_subset])
  return(list('lof'= lof, 'ecdf_lof'=ecdf_lof, 'prob'= 1-ecdf_lof(lof)))
}
