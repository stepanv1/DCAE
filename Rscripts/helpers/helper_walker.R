#Random walks on big graph, Based on densityCut implementation
library(Matrix)
library(igraph)
library(data.table)
#library(MCL)
#library(ff)
#library(bootSVD)

source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')




##====================================================================
# Enhance densities based on the transition matrix
EnhanceDensity = function(P, V, smooth=FALSE, debug=TRUE,
                          maxit=50, alpha=0.85, eps=1e-5) {
  iter = 0
  done = FALSE
  
  V0 = V
  while (!done) {    
    iter = iter + 1
    if (smooth == TRUE) {
      V1 =  alpha * P %*% V  + (1-alpha) * V0
    } else {
      V1 =  alpha * V %*% P   + (1-alpha) * V0
    }
    V1 = as.vector(V1)
    V1 = V1 / sum(V1)
    
    v.diff = sum(abs(V1 - V))
    if (debug == TRUE) {
      cat("Iter: ", iter, " V-diff: ", v.diff, "\n")
    }
    
    if (v.diff <= eps) {
      done = TRUE
    }  else if (iter > maxit) {
      cat(paste("WARNING! not converged"), "\n")
      done = TRUE
    }
    V = V1
  }
  
  return(as.vector(V))
}

##====================================================================
# Enhance densities based on the transition matrix
EnhanceDensityTrace = function(P, V, IDX, smooth=FALSE, debug=TRUE,
                          maxit=500, alpha=0.85, eps=1e-5) {
  iter <- 0
  done <- FALSE
  Piter <- matrix(rep(0,length(IDX)*(maxit+1)), nrow = length(IDX), ncol = (maxit+1))
  k=rep(1,length(V))
  
  V0 = V
  while (!done) {    
    iter = iter + 1
    if (smooth == TRUE) {
      V1 =  alpha * P %*% V  + (1-alpha) * V0
    } else {
      V1 =  alpha * V %*% P   + (1-alpha) * V0
    }
    V1 = as.vector(V1)
    V1 = V1 / sum(V1)
    k=k*ifelse(V==0, 0, V1/V)
    Piter[, iter] <- V1[IDX]
    v.diff = sum(abs(V1 - V))
    if (debug == TRUE) {
      cat("Iter: ", iter, " V-diff: ", v.diff, "\n")
    }
    
    if (v.diff <= eps) {
      done = TRUE
    }  else if (iter > maxit) {
      cat(paste("WARNING! not converged"), "\n")
      done = TRUE
    }
    V = V1
  }
  
  return(list('fin'=as.vector(V), 'Piter'=Piter[, 1:iter], 'k'=k))
}





norm_mat<-function(mat){
  mat <- as(mat, "dgTMatrix") 
  zero_mask <- mat@x!=0
  mat@x <- mat@x[zero_mask]
  mat@i <- mat@i[zero_mask]
  mat@j <- mat@j[zero_mask]
  mat <- as(mat, "dgCMatrix")
  mat <- mat*(1/rowSums(mat)); gc()
  mat <- as(mat, "dgCMatrix")
  return(mat)
}

set_zero <- function(mat, cut_off){
  mat <- as(mat, "dgTMatrix") 
  zero_mask <- mat@x>cut_off
  mat@x <- mat@x[zero_mask]
  mat@i <- mat@i[zero_mask]
  mat@j <- mat@j[zero_mask]
  mat <- as(mat, "dgCMatrix")
}


library(Matrix)
mclSparse <-
  function(x, addLoops = FALSE, expansion = 2, inflation = 2, allow1 = FALSE, max.iter = 100, ESM = FALSE ){
    
    if(is.null(addLoops)){
      stop("addLoops has to be TRUE or FALSE")
    }
    
    if (addLoops) diag(x) <- 1    
    
    adj.norm <- x*(1/rowSums(x))
    
    
    a <- 1
    
    
    repeat{
      
        expans <- adj.norm %*% adj.norm; gc()
        
        
        infl <- expans ^ inflation
        
        
        #leave just the same number of links as it was in original adjacency matrix
        infl <- as(infl, "dgTMatrix") 
        infl_order<-nnzero(infl)-data.table::frank(infl@x)+1
        infl_mask <- infl_order< nnz*a+1 
        infl@x <- infl@x[infl_mask]
        infl@i <- infl@i[infl_mask]
        infl@j <- infl@j[infl_mask]
        infl <- as(infl, "dgCMatrix")
        
        infl.norm <- infl*(1/rowSums(infl)); gc()
        
        if(a==max.iter) {
          a <- a+1
          break
        }
        
        
        adj.norm <- infl.norm
        a<- a+1
      
      
      
      
      if(identical(infl.norm,adj.norm)) {
        ident <- TRUE
        break
      }
      
      
      if(a==max.iter) {
        ident <- FALSE
        a <- a+1
        break
      }
      
      
      adj.norm <- infl.norm
      cat(adj.norm)
      a<- a+1
    }#end of interations of inflation expansion cycle
    
    
    
    if(!is.na(infl.norm[1,1]) & ident){
      
      count <- 0 
      for(i in 1:ncol(infl.norm)){
        if(sum(abs(infl.norm[i,])) != 0) {
          count <- count+1
        }
      }
      
      #neu <- matrix(nrow=count, ncol=ncol(infl.norm)) 
      neu <- Matrix(nrow=count, ncol=ncol(infl.norm))
      
      zeile <- 1
      for(i in 1:nrow(infl.norm)){
        if(sum(infl.norm[i,]) != 0) {
          neu[zeile,]<-infl.norm[i,]
          zeile <- zeile+1
        }
      }
      
      
      for(i in 1:nrow(neu)){
        for(j in 1:ncol(neu)) {
          if((neu[i,j] < 1) & (neu[i,j] > 0)){
            neu[,j] <- 0
            neu[i,j] <- 1
          }
        }
      }
      
      
      for(i in 1:nrow(neu)){
        for (j in 1:ncol(neu)){
          if(neu[i,j] != 0){
            neu[i,j] <- i
          }
        }
      }
      
      ClusterNummern <- sum(neu[,1])
      for(j in 2:ncol(neu)){
        ClusterNummern <- c(ClusterNummern,sum(neu[,j]))
      }
      
      
    } 
    
    ifelse(!(!is.na(infl.norm[1,1]) & ident), output <- paste("An Error occurred at iteration", a-1),
           {
             if(!allow1){
               dub <- duplicated(ClusterNummern) + duplicated(ClusterNummern,fromLast = T)
               for(i in 1:length(dub)){
                 if(dub[[i]]==0) ClusterNummern[[i]]<-0
               }
             }
             
             #### dimnames for infl.norm
             dimnames(infl.norm) <- list(1:nrow(infl.norm), 1:ncol(infl.norm))
             
             output <- list()
             output[[1]] <- length(table(ClusterNummern))
             output[[2]] <- a-1 
             output[[3]] <- ClusterNummern
             output[[4]] <- infl.norm
             
             names(output) <-c("K", "n.iterations","Cluster",
                               "Equilibrium.state.matrix")
           }
    )
    ifelse(ESM==TRUE,return(output),return(output[-4]))
  }


mcReweight <-
  function(x, addLoops = FALSE, expansion = 2, inflation = 2,  max.iter = 100, ESM = TRUE ,  stop.cond =60){
    
    if(is.null(addLoops)){
      stop("addLoops has to be TRUE or FALSE")
    }
    
    if (addLoops) diag(x) <- 1    
    
    #adj <- x
    #adj <- as(adj, "dgTMatrix") #some unpleasantries with elementwise operations
    nnz<-nnzero(x)
    #adj@x<-rep(1, nnz)
    #adj <- as(adj, "dgCMatrix")
    
    
    adj.norm <- x*(1/rowSums(x)); gc()
    degrees <-rowSums(x!=0)
    
    a <- 1
    
    
    repeat{
      print(paste0('iteration ', as.character(a)))
      
      print('calculating expansion')
      expans <- adj.norm %*% adj.norm; gc()
      
      print('calculating inflation')
      infl <- expans ^ inflation
      
      #leave just the same number of links as it was in original adjacency matrix
      
      print('pruning')
      infl <- as(infl, "dgTMatrix") 
      
      infl_order<-nnzero(infl)-data.table::frank(infl@x)+1
      infl_mask <- infl_order < round(nnz) 
      infl@x <- infl@x[infl_mask]
      infl@i <- infl@i[infl_mask]
      infl@j <- infl@j[infl_mask];gc()
      zero_mask <- infl@x!=0
      infl@x <- infl@x[zero_mask]
      infl@i <- infl@i[zero_mask]
      infl@j <- infl@j[zero_mask]
      #remove orphaned nodes, and store their indexes 
      
      
      infl <- as(infl, "dgCMatrix");gc()
      
      print('normalising inflated matrix')
      infl.norm <- infl*(1/rowSums(infl)); gc()
      #infl.norm <- infl; gc()
      
      #if(identical(infl.norm,adj.norm)) {
      #  ident <- TRUE
      #  break
      #}
      
      
      if(a==max.iter) {
        a <- a+1
        break
      }
      
      #if(max(colSums(infl.norm))>=stop.cond) {
      #  a <- a+1
      #  break
      #}
      
      
      
      adj.norm <- infl.norm
      a<- a+1
    }
    
    
    
   
             #### dimnames for infl.norm
    dimnames(infl.norm) <- list(1:nrow(infl.norm), 1:ncol(infl.norm))
             
    output <- list()
    output[[1]] <- a-1 
    output[[2]] <- infl.norm
    names(output) <-c("n.iterations", "Equilibrium.state.matrix")
           
    gc()
    ifelse(ESM==TRUE,return(output),return(output[-2]))
  }


mcOut <-
  function(x, addLoops = FALSE, expansion = 2, inflation = 5,  max.iter = 5, ESM = TRUE ){
    
    if(is.null(addLoops)){
      stop("addLoops has to be TRUE or FALSE")
    }
    
    if (addLoops) diag(x) <- 1    
    
    #adj <- x
    #adj <- as(adj, "dgTMatrix") #some unpleasantries with elementwise operations
    nnz<-nnzero(x)
    #adj@x<-rep(1, nnz)
    #adj <- as(adj, "dgCMatrix")
    
    
    adj.norm <- x*(1/rowSums(x))
    
    
    a <- 1
    
    
    repeat{
      print(paste0('iteration ', as.character(a)))
      
      print('calculating expansion')
      expans <- adj.norm %*% adj.norm; gc()
      
      print('calculating inflation')
      infl <- expans ^ inflation
      
      #leave just the same number of links as it was in original adjacency matrix
      
      print('pruning')
      infl <- as(infl, "dgTMatrix") 
      
      infl_order<-nnzero(infl)-data.table::frank(infl@x)+1
      infl_mask <- infl_order< nnz*4 
      infl@x <- infl@x[infl_mask]
      infl@i <- infl@i[infl_mask]
      infl@j <- infl@j[infl_mask]
      zero_mask <- infl@x!=0
      infl@x <- infl@x[zero_mask]
      infl@i <- infl@i[zero_mask]
      infl@j <- infl@j[zero_mask]
      
      infl <- as(infl, "dgCMatrix")
      
      print('normalising inflated matrix')
      infl.norm <- infl*(1/rowSums(infl)); gc()
      
      
      
      if(a==max.iter) {
        a <- a+1
        break
      }
      
      
      adj.norm <- infl.norm
      a<- a+1
    }
    
    
    
    
    #### dimnames for infl.norm
    dimnames(infl.norm) <- list(1:nrow(infl.norm), 1:ncol(infl.norm))
    
    output <- list()
    output[[1]] <- a-1 
    output[[2]] <- infl.norm
    names(output) <-c("n.iterations", "Equilibrium.state.matrix")
    
    gc()
    ifelse(ESM==TRUE,return(output),return(output[-2]))
  }


mcReweightHD <-
  function(x, addLoops = FALSE, expansion = 2, inflation = 2,  max.iter = 100, ESM = TRUE ){
    
    if(is.null(addLoops)){
      stop("addLoops has to be TRUE or FALSE")
    }
    
    if (addLoops) diag(x) <- 1    
    
    #adj <- x
    #adj <- as(adj, "dgTMatrix") #some unpleasantries with elementwise operations
    nnz<-nnzero(x)
    #adj@x<-rep(1, nnz)
    #adj <- as(adj, "dgCMatrix")
    
    
    adj.norm <- x*(1/rowSums(x))
    adj.norm<-as(adj.norm,"dgTMatrix")
    
    adj.norm_ff <- ff( initdata  = NULL , vmode="double", dim=dim(adj.norm))
    invisible(lapply(1:length(adj.norm@x), function(x) adj.norm_ff[adj.norm@i[x]+1, adj.norm@j[x]+1]<-adj.norm@x[x] ))
    
    
    a <- 1
    
    
    repeat{
      
      #expans <- adj.norm %*% adj.norm; gc()
      expans_ff <- ffmatrixmult(adj.norm_ff, xt=T, yt=F)
      
      infl_ff <- expans_ff ^ inflation
      
      #leave just the same number of links as it was in original adjacency matrix
      
      
      infl <- as(infl, "dgTMatrix") 
      
      infl_order<-nnzero(infl)-data.table::frank(infl@x)+1
      infl_mask <- infl_order< round(nnz/2) 
      infl@x <- infl@x[infl_mask]
      infl@i <- infl@i[infl_mask]
      infl@j <- infl@j[infl_mask]
      zero_mask <- infl@x!=0
      infl@x <- infl@x[zero_mask]
      infl@i <- infl@i[zero_mask]
      infl@j <- infl@j[zero_mask]
      
      infl <- as(infl, "dgCMatrix")
      
      infl.norm <- infl*(1/rowSums(infl)); gc()
      
      
      if(identical(infl.norm,adj.norm)) {
        ident <- TRUE
        break
      }
      
      
      if(a==max.iter) {
        a <- a+1
        break
      }
      
      
      adj.norm <- infl.norm
      a<- a+1
    }
    
    
    
    
    #### dimnames for infl.norm
    dimnames(infl.norm) <- list(1:nrow(infl.norm), 1:ncol(infl.norm))
    
    output <- list()
    output[[1]] <- a-1 
    output[[2]] <- infl.norm
    names(output) <-c("n.iterations", "Equilibrium.state.matrix")
    
    gc()
    ifelse(ESM==TRUE,return(output),return(output[-2]))
  }





mcReweightLocal <-
  function(x, addLoops = FALSE, expansion = 2, inflation = 2,  max.iter = 100, ESM = TRUE ,  stop.cond =dim(x)[1]){
    
    if(is.null(addLoops)){
      stop("addLoops has to be TRUE or FALSE")
    }
    
    if (addLoops) diag(x) <- 1    
    
    #adj <- x
    #adj <- as(adj, "dgTMatrix") #some unpleasantries with elementwise operations
    nnz<-nnzero(x)
    #adj@x<-rep(1, nnz)
    #adj <- as(adj, "dgCMatrix")
    
    
    adj.norm <- x*(1/rowSums(x)); gc()
    degrees <-rowSums(x!=0)
    
    a <- 1
    
    
    repeat{
      print(paste0('iteration ', as.character(a)))
      
      print('calculating expansion')
      expans <- adj.norm %*% adj.norm; gc()
      
      print('calculating inflation')
      infl <- expans ^ inflation
      
      #leave just the same number of links as it was in original adjacency matrix
      
      print('pruning')
      infl <- as(infl, "dgTMatrix") 
      
      #infl_order<-nnzero(infl)-data.table::frank(infl@x)+1
      infl_table<-data.table(from=infl@i, to=infl@j, weight=infl@x)
      setorderv(infl_table, c('from', 'weight', 'to' ), c(1, -1, 1))
      infl_table<-infl_table[, head(.SD, min(nrow(.SD), degrees[from])), keyby = from]
      nnzIDX <- infl_table$weight!=0
      infl@x <- infl_table$weight[nnzIDX]
      infl@i <- infl_table$from[nnzIDX]
      infl@j <- infl_table$to[nnzIDX]
       
      infl <- as(infl, "dgCMatrix");gc()
      
      print('normalising inflated matrix')
      infl.norm <- infl*(1/rowSums(infl)); gc()
      #infl.norm <- infl; gc()
      
      #if(identical(infl.norm,adj.norm)) {
      #  ident <- TRUE
      #  break
      #}
      
      nnz2<-nnzero(infl.norm)
      print(paste0('Edges left: ', as.character(nnz2)))
      
      if(a==max.iter) {
        a <- a+1
        break
      }
      
      
      temp_g=graph_from_adjacency_matrix((asym_rw), mode =  "directed", weighted = TRUE, diag = F,  add.colnames = NULL, add.rownames = NA); gc()
      comp<-components(temp_g, mode = "weak")
      
      print(paste0('Components #: ', as.character(comp$no)))
      if(comp$no >= stop.cond) {
        a <- a+1
       break
      }
      
      
      
      adj.norm <- infl.norm
      a<- a+1
    }
    
    
    
    
    #### dimnames for infl.norm
    dimnames(infl.norm) <- list(1:nrow(infl.norm), 1:ncol(infl.norm))
    
    output <- list()
    output[[1]] <- a-1 
    output[[2]] <- infl.norm
    names(output) <-c("n.iterations", "Equilibrium.state.matrix")
    
    gc()
    ifelse(ESM==TRUE,return(output),return(output[-2]))
  }


mcOutLocal <-
  function(x, addLoops = FALSE, expansion = 2, inflation = 5,  max.iter = 5, ESM = TRUE ){
    
    if(is.null(addLoops)){
      stop("addLoops has to be TRUE or FALSE")
    }
    
    if (addLoops) diag(x) <- 1    
    
    #adj <- x
    #adj <- as(adj, "dgTMatrix") #some unpleasantries with elementwise operations
    nnz<-nnzero(x)
    #adj@x<-rep(1, nnz)
    #adj <- as(adj, "dgCMatrix")
    degrees<-rowSums(x!=0)
    
    adj.norm <- norm_mat(x)
    a <- 1
    
    repeat{
      print(paste0('iteration ', as.character(a)))
      
      print('calculating expansion')
      expans <- adj.norm %*% adj.norm; gc()
      
      #print('calculating inflation')
      #infl <- expans ^ inflation
      infl <- expans
      rm(expans)
      #leave just the same number of links as it was in original adjacency matrix
      
      print('pruning')
      infl <- as(infl, "dgTMatrix") 
      
      #infl_order<-nnzero(infl)-data.table::frank(infl@x)+1
      infl_table<-data.table(from=infl@i, to=infl@j, weight=infl@x)
      setorderv(infl_table, c('from', 'weight', 'to' ), c(1, -1, 1))
      infl_table<-infl_table[, head(.SD, min(nrow(.SD), degrees[from])), keyby = from]
      nnzIDX <- infl_table$weight!=0
      infl@x <- infl_table$weight[nnzIDX]
      infl@i <- infl_table$from[nnzIDX]
      infl@j <- infl_table$to[nnzIDX]
      
      infl <- as(infl, "dgCMatrix");gc()
    
      infl.norm <- infl
      
      if(a==max.iter) {
        a <- a+1
        break
      }
      
      
      adj.norm <- infl.norm
      a<- a+1
    }
    
    
    
    
    #### dimnames for infl.norm
    dimnames(infl.norm) <- list(1:nrow(infl.norm), 1:ncol(infl.norm))
    
    output <- list()
    output[[1]] <- a-1 
    output[[2]] <- infl.norm
    names(output) <-c("n.iterations", "Equilibrium.state.matrix")
    
    gc()
    ifelse(ESM==TRUE,return(output),return(output[-2]))
  }



