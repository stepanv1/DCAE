#small communities for experiments with graph algorithms
library(igraph)
library(MCL)
ge1 <- graph.full(5)
V(ge1)$name <- 1:5  
E(ge1)$weight <- 2
ge2 <- graph.full(5)
V(ge2)$name <- 6:10
E(ge2)$weight <- 2
ge3 <- graph.full(5)
V(ge3)$name <- 11:15
E(ge3)$weight <- 2
#add outlier
geO<-graph.full(1)
V(geO)$name <- 16
gf <- ge1 %du% ge2 %du% ge3 %du% geO +edge('1', '6', weight=1.95 ) + edge('1', '11',  weight=1.95 )+ edge('9', '13',  weight=1.95 )+edge('16','14', weight=0.2) + edge('16','3', weight=0.2)
#E(gf)$weight<-abs(rnorm(length(E(gf))))
plot(gf,  edge.width=E(gf)$weight)

g_wf=as_adjacency_matrix(gf, attr="weight", type = c("both"), 
                         edges = FALSE, names = TRUE, sparse = igraph_opt("sparsematrices"))


g_wf_r<-mcReweight(g_wf, addLoops = FALSE, expansion = 2, inflation = 2,  max.iter = 2, ESM = TRUE )
g_wf_r
g_wf_r<-1/2*(g_wf_r[[2]]+t(g_wf_r[[2]]))
gf_r<-graph_from_adjacency_matrix(g_wf_r, mode =  "undirected", weighted = TRUE, diag = F,  add.colnames = NULL, add.rownames = NA); gc()
plot(gf_r,  edge.width=E(gf)$weight)

g_wf_rl<-mcReweightLocal(g_wf, addLoops = FALSE, expansion = 2, inflation = 3,  max.iter = 10, ESM = TRUE )
g_wf_rl
g_wf_rl<-1/2*(g_wf_rl[[2]]+t(g_wf_rl[[2]]))
gf_rl<-graph_from_adjacency_matrix(g_wf_rl, mode =  "undirected", weighted = TRUE, diag = F,  add.colnames = NULL, add.rownames = NA); gc()
plot(gf_rl,  edge.width=E(gf)$weight)



#res_mclSparse<-mclSparse(g_wf, addLoops = FALSE, expansion = 2, inflation = 2,  max.iter = 130, ESM = TRUE )[[2]]

plot(graph_from_adjacency_matrix(res_mclSparse$Equilibrium.state.matrix))

res_mcl<-mcl(g_wf, addLoops = FALSE, expansion = 2, inflation = 2,  max.iter = 2, ESM = TRUE )

res<-cluster_infomap(gf)
res2<-cluster_louvain(gf)
res3<-mcl(gf, addLoops = FALSE, expansion = 2, inflation = 3,  max.iter = 20, ESM = TRUE )

plot(graph_from_adjacency_matrix((res3$Equilibrium.state.matrix+t(res3$Equilibrium.state.matrix))/2, mode =  "undirected", weighted = TRUE, diag = F,  add.colnames = NULL, add.rownames = NA))

