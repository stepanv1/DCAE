#Luovain clustering is run multiple times, the best result (with highest modularity Q) is 
#returned
library(igraph)

louvain_multiple_runs<-function(graph, num.run=50, time.lim=2000){
  n=1; res_louvain=list()
  perm=list()
  start.time <- Sys.time()
 
  adj<-as_adjacency_matrix(graph, type = 'both', attr="weight", sparse = T)
  
  while(n <= num.run & Sys.time() - start.time < time.lim){
  print(n)  
    set.seed(n+1)  
  #permute nodes of the graph
  p <- sample(1:dim(adj)[1])
  adjp <- adj[p,];adjp<-adjp[,p]
  gp <- graph_from_adjacency_matrix(adjp, mode = "undirected", weighted=T)
  perm[[length(perm)+1]] <- p 
  res_louvain[[length(res_louvain)+1]] <- cluster_louvain(gp)
  print('modularity is:')
  print(res_louvain[[n]]$modularity)
  
  print(Sys.time() - start.time)
  n<-n+1
  }
  
  modularity <- unlist(lapply(res_louvain, function(x) max(x$modularity)))
  max_modularity <- which(modularity == max(modularity))
  
  print(res_louvain[[max_modularity[[1]]]])
  print('finished in')
  print(Sys.time() - start.time)
  print('after')
  print(paste0(n-1, ' iterations'))
  
  #inverse permutation
  ip <- invPerm(perm[[max_modularity]])
  return(res_louvain[[max_modularity[[1]]]]$membership[ip])
}

louvain_multiple_runs_par<-function(graph, num.run=5, mc.cores=5){
  perm=list()
  start.time <- Sys.time()
  
  adj<-as_adjacency_matrix(graph, type = 'both', attr="weight", sparse = T)
  
  res_louvain<-mclapply(1:num.run, function(n) {
    set.seed(n+1)  
    #permute nodes of the graph
    p <- sample(1:dim(adj)[1])
    adjp <- adj[p,];adjp<-adjp[,p]
    gp <- graph_from_adjacency_matrix(adjp, mode = "undirected", weighted=T)
    perm[[length(perm)+1]] <- p 
    return(list('cluster'=cluster_louvain(gp), 'perm'=perm))
  }, mc.cores=mc.cores)
  
  modularity <- unlist(lapply(res_louvain, function(x) max(x$cluster$modularity)))
  max_modularity <- which(modularity == max(modularity))
  
  print('finished in')
  print(Sys.time() - start.time)
  print('after')
  print(paste0(num.run, ' iterations'))
  
  #inverse permutation
  ip <- invPerm(unlist(res_louvain[[ max_modularity[[1]] ]]$perm))
  return(res_louvain[[ max_modularity[[1]] ]]$cluster$membership[ip])
}



