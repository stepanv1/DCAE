# helper create snn_k graph out of coordinated
library(RANN)
library(RANN.L1)
library(igraph)
create_snnk_graph<-function(data,k, metric='L1'){
  
  find_neighbors <- function(data, k_, metric='L1'){
    if (metric=='L2'){  
    nearest <- RANN::nn2(data, data, k_, treetype = "bd", searchtype = "standard")
    return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
    }
    if (metric=='L1'){ 
    nearest <- RANN.L1::nn2(data, data, k_, treetype = "bd", searchtype = "standard")
      return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
    }
  }
  
  system.time(m2<-find_neighbors(data, k_=k+1, metric=metric))
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
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  return(list('graph'=g, 'relations'=relations))
}
#nn.idx<-nnind105[[3]]; subset<-list_out[[3]]
create_snnk_graph_from_subset<-function(nn.idx, k, subset, mc.cores=5){
  s<-(1:nrow(nn.idx))[subset]
  oi<-(1:nrow(nn.idx))[!subset]
  nn.idx<-nn.idx[subset,]
  nn.idx<-nn.idx[,-1]
  nn.idx<-as.matrix(nn.idx)
  sind<-1:length(s)
  names(sind)<-s
  id<- which((nn.idx) %in% oi, arr.ind=T);gc()
  #yi<-floor(id/nrow(nn.idx))+1
  #xi<-id-(yi-1)*nrow(nn.idx)
  
  #nn.idx[]<-0;gc()
  nn.idx2<-nn.idx
  nn.idx2[id]<-0
  nn.idx2<-t(apply(nn.idx2,1, function(x) x[x!=0][1:k] ))
  nn.idx2<-apply(nn.idx2, 2, function(x) sind[as.character(x)] )
  #which(is.na(nn.idx))
  print(system.time(links <- cytofkit:::jaccard_coeff(nn.idx)))
  gc()
  links <- links[links[,1]>0, ]
  relations <- as.data.frame(links)
 
  colnames(relations)<- c("from","to","weight")
  relations<-as.data.table(relations)
 
  asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  #run one reweight cycle
  sym_w<-(asym_w+t(asym_w))/2
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  return(list('graph'=g, 'relations'=relations, 'sind'=sind))
}

create_snnk_graph_vptree<-function(data,k, metric='L2', mc.cores=10){
  #data=matrix(rnorm(10000),1000,10)
  find_neighbors <- function(data, k_, metric='L2'){
    data<-as.matrix(data)
    tr<-tree_create(data,  mc.cores)
    system.time(nearest<-tree_search(tr, data, k_, mc.cores))
    
      return(list(nn.idx=nearest[[2]], nn.dists=nearest[[1]]))
    }
 
  
  View(neighborMatrix);
  View(neighborMatrix1);
  
  system.time(m2<-find_neighbors(data, k_=k, metric=metric))
  neighborMatrix <- m2$nn.idx
  print('Calculating Jaccard coefficient..')
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
  print('Creating graph...')
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  gc()
  return(list('graph'=g, 'relations'=relations, 'neighborMatrix'=neighborMatrix))
}

create_snnk_graph_approx<-function(data, k, metric='L2', mc.cores=10){
  #data=matrix(rnorm(10000),1000,10)
 
  system.time(m2<-find_neighbors(data, k=k, metric=metric))
  neighborMatrix <- m2$nn.index
  print('Calculating Jaccard coefficient..')
  system.time(links <- cytofkit:::jaccard_coeff(m2$nn.index))
  gc()
  links <- links[links[,1]>0, ]
  relations <- as.data.frame(links)
  rm(links)
  colnames(relations)<- c("from","to","weight")
  relations<-as.data.table(relations)
  #relations<-relations[from!=to, ]# remove self-loops
  #sum((res_annoyKnn$ind[,1:31]-m2$nn.idx)!=0)
  
  asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  #run one reweight cycle
  sym_w<-(asym_w+t(asym_w))/2
  print('Creating graph...')
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  gc()
  return(list('graph'=g, 'relations'=relations, 'neighborMatrix'=neighborMatrix, 'sym_w'=sym_w))
}


#data=matrix(rnorm(10000), ncol=10)

Louvain_multilevel<-function(data, k, mc.cores=10){
  setTimeLimit(cpu = Inf)
  relations=create_snnk_graph_approx(data, k=k, metric='L2', mc.cores=mc.cores)$relations
  library(purrr)
  library(purrrlyr)
  library(reticulate)
  use_condaenv('PhenographLevin13', conda='auto', required=T)
  reticulate::py_config()
  hnsw<-import('hnswlib')
  np<-import('numpy')
  ig<-import('igraph')
  scp<-import('scipy')
 
  edg <- as.list(as.data.frame(t(relations[,1:2])))
  edg<-lapply(edg,  function(x)  list((x[[1]]), (x[[2]])))
  edg<-unname( edg, force = TRUE)
  edg<-lapply(edg, function(x) as.integer(unlist(x)))
  G <- ig$Graph(edges=edg, directed=FALSE)
  G$es$set_attribute_values('weight', as.numeric(unlist(relations[,3]))) 
  G$vs$set_attribute_values('labels', 1:nrow(relations))
  print('running Louvain')
  
  Louv_res =  lapply(1:1, function(x) ig$Graph$community_multilevel(G, weights='weight', return_levels=TRUE))
  top_level_res=lapply(Louv_res, function(x) {
    levels=length(x)
    memberships=x[[levels]]$membership
    modularity = G$modularity(x[[levels]]$membership)
    return(list('levels'=levels, 'memberships'=memberships, 'modularity' = modularity))} )
  modularities=unlist(lapply(top_level_res, function(x) x$modularity)) 
  print(paste0('modularities  ', modularities )) 
  max_modularity_ind <- which(modularities == max(modularities))
  return(top_level_res[[max_modularity_ind]])
}

#clusters = G$community_multilevel(weights='weight', return_levels=TRUE)
#modularity_score = G$modularity(clusters[[5]]$membership)
#modularity_score













