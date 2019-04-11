########################################################
#calculate the modularity based on data.drame containing 
#connections and their links

#create anexample data-set
library(igraph)

#g1 <- graph.full(5)
#V(g1)$name <- 1:5    
#g2 <- graph.full(5)
#V(g2)$name <- 6:10
#g3 <- graph.ring(5)
#V(g3)$name <- 11:15
#g <- g1 %du% g2 %du% g3 + edge('1', '6') + edge('1', '11')
#set_graph_attr(g, "weight", 1)
#E(g)$weight <- 1 
#g <- g + edge('1', '6', weight=2.5) 
#plot(g, edge.label=E(g)$weight, margin=.5, layout=layout.circle, edge.width=E(g)$weight)

#fc <- cluster_louvain(g)
#modularity(simplify(g, remove.loops=FALSE, edge.attr.comb=list(weight="sum")), membership(fc))
#modularity(simplify(g, remove.loops=FALSE, edge.attr.comb=list(weight="sum")), membership(fc), weights=E(simplify(g, remove.loops=FALSE, edge.attr.comb=list(weight="sum")))$weight)
#modularity(g, membership(fc))

#modularity(g, membership(fc), weights=E(g)$weight)

#cg <- contract.vertices(g, membership(fc))
#modularity(simplify(cg, remove.loops=FALSE, edge.attr.comb=list(weight="sum")), 1:vcount(cg))
#E(cg)$weight <- 1
#cg2 <- simplify(cg, remove.loops=FALSE, edge.attr.comb=list(weight="sum"))
#plot(cg2, edge.label=E(cg2)$weight, margin=.5, layout=layout.circle)
#plot(cg, edge.label=E(cg)$weight, margin=.5, layout=layout.circle)

#get.edgelist(cg2, names=TRUE)
#E(cg2)$weight

#modularity_matrix(cg2, 1:vcount(cg2))
#modularity(cg2, 1:vcount(cg2))
#modularity(cg2, 1:vcount(cg2), weights=E(cg2)$weight)

convert_lbl2natural<-function(lbl){
  as.integer(as.factor(lbl))
}

Qfun<-function(g, lbl, plt=F){
  membership<-convert_lbl2natural(lbl)
  cg <- contract.vertices(g, membership)
  cg <- simplify(cg, remove.loops=FALSE, edge.attr.comb=list(weight="sum"))
  if(plt==T){plot(cg, edge.width=E(cg)$weight)}
  Q=modularity(cg, 1:vcount(cg), weights=E(cg)$weight)
  return(Q)  
}


