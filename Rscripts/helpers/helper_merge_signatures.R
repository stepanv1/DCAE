#merge clusters with identical global noise signatures
library(partitions)
library(igraph)
library(car)
library(MASS)
library(Hotelling)
helper_merge_signatures<-function(subdim, clus_assign, data,  mc.cores=10){
  rownames(subdim)<-1:nrow(subdim)
  ncols<-ncol(subdim)
  ctl<-table(clus_assign)
  
  
  simb <- apply(subdim, 1, function(x)  Reduce(paste0, x) )
  subdimDup<-cbind(subdim,id=factor(simb,labels=1:length(unique(simb))))
  subdimDup[,'id']
  list_of_sim_dim<-lapply(unique(subdimDup[,'id']), function(x)  (1:nrow(subdim))[subdimDup[, 'id']==x]) 
  IDX<-lapply(list_of_sim_dim, length)
  if(!any(unlist(IDX)>1)){return(clus_assign)}
  list_of_sim_dim <- list_of_sim_dim[unlist(IDX)>1]

  
  
  #try Hoetelloiing test
  print('Performing Box-Cox transform and Hotelling tests..')
  library(Hotelling)
  dlt<-0.000001#to guaranty that data is positive for boxcox transform
  groups<-list()
  for (i in 1:length(list_of_sim_dim)){
    sim_d <- list_of_sim_dim[[i]]
    l_sim<-length(list_of_sim_dim[[i]])
    adjmat<-matrix(0, nrow=l_sim, ncol=l_sim)
    data_sub<-data[clus_assign %in% sim_d, subdim[sim_d[1],]]+dlt
    #try to descew data per dimension, to make powerTransform work
    pt1D<-lapply(as.data.frame(data_sub), function(x) {bc<-boxcox(x~1, lambda = seq(-2, 2, 1/100), eps=0.001, plotit = F); bc$x[bc$y==max(bc$y)] }) 
    data_subbc1D<-unlist(lapply(1:sum(subdim[sim_d[1],]),  function(x){lapply(data_sub[,x], 
                                                                       function(y) ifelse(pt1D[[x]]==0, log(y), (y^(pt1D[[x]])-1)/pt1D[[x]]))   }))
    data_subbc1D<-matrix(data_subbc1D, nrow = nrow(data_sub))
    #pt<-powerTransform(dlt-(data_subbc1D-max(data_subbc1D))~1)
    
    for(k in 1:(l_sim-1)){
      for(l in (k+1):l_sim){
        #create an induced graph in a subdimension
        cl2<-clus_assign[clus_assign %in% sim_d]
        cl2==sim_d[k]
        dat1 =  data_subbc1D[ cl2==sim_d[k], ] 
        dat2 =  data_subbc1D[ cl2==sim_d[l], ] 
        hot_res<-hotelling.test(dat1, dat2, shrinkage = T)
        adjmat[k,l]<-ifelse(hot_res$pval>0.01, 1, 0)
      }
    }
    adjsym <- (adjmat + t(adjmat))
    g  <- graph.adjacency(adjsym)  
    clu <- components(g)$membership
    groups[[length(groups)+1]]<-clu
  }
  
  num_groups<-length(groups)
  for (i in 1:num_groups) {
    sim_d <- list_of_sim_dim[[i]]
    grp <- groups[[i]]
    grp<-paste0(i, '_', grp)
    names(grp)<-sim_d
    clus_assign[clus_assign %in% sim_d]<-   grp[as.character(clus_assign[clus_assign %in% sim_d])]
    }
  #create new labels
  clus_assignMerge<-as.integer(as.factor(clus_assign))
  return(clus_assignMerge)
}
