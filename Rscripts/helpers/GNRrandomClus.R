#Growing neural gus art set clustering
library('clusterGeneration')
library('gmum.r')
library('rgl')
library(FNN)
library(igraph)
library(parallel)
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers")
# helper functions to match clusters and evaluate
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")
library(Matrix)
library(data.table)

chunks2<- function(n, nChunks) {split(1:n, ceiling(seq_along(1:n)/(n/nChunks)))}


clus_set<-genRandomClust(numClust=15,
               sepVal=-0.1,
               numNonNoisy=15,
               numNoisy=1,
               numOutlier=0.05,
               numReplicate=1,
               fileName="test",
               clustszind=2,
               clustSizeEq=100,
               rangeN=c(200,4000),
               clustSizes=NULL,
               covMethod="eigen",
               rangeVar=c(2, 30),
               lambdaLow=3,
               ratioLambda=7,
               alphad=1,
               eta=1,
               rotateind=TRUE,
               iniProjDirMethod="SL",
               projDirMethod="newton",
               alpha=0.05,
               ITMAX=20,
               eps=1.0e-10,
               quiet=TRUE,
               outputDatFlag=TRUE,
               outputLogFlag=TRUE,
               outputEmpirical=TRUE,
               outputInfo=TRUE)

cl_coord=clus_set$datList$test_1
lbls=clus_set$memList$test_1
table(lbls)

#generate a list of cluster coordinates
library(RANN)
find_neighbors <- function(data, k){
  nearest <- nn2(data, data, k, treetype = "bd", searchtype = "standard")
  return(list(nn.idx=nearest[[1]], nn.dists=nearest[[2]]))
}
gc()
m2<-find_neighbors(cl_coord, k=30+1);
            neighborMatrix <- (m2$nn.idx)[,-1]
system.time(links <- cytofkit:::jaccard_coeff(m2$nn.idx))
gc()
links <- links[links[,1]>0, ]
relations <- as.data.frame(links)
#save(relations, file='../../results/relationsSamusik_01.RData')
colnames(relations)<- c("from","to","weight")
relations<-as.data.table(relations)
relations<-relations[from!=to, ]# remove self-loops


asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight);gc()
#run one reweight cycle
asym_rw<-mcReweight(asym_w, addLoops = FALSE, expansion = 2, inflation = 3,  max.iter = 1, ESM = TRUE )[[2]];gc()
 
hist((asym_rw@x),500, xlim=c(0,0.1))

g2<-graph_from_adjacency_matrix((asym_rw+t(asym_rw))/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()

g<-graph_from_adjacency_matrix((asym_w+t(asym_w))/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()

g2<-simplify(g2, remove.loops=T, edge.attr.comb=list(weight="sum"))
g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))

cl_res<-cluster_louvain(g)
cl_res2<-cluster_louvain(g2)

mbr<-membership(cl_res)
mbr2<-membership(cl_res2)

table(mbr)
table(mbr2)
table(lbls)


helper_match_evaluate_multiple(mbr, lbls)
helper_match_evaluate_multiple(mbr2, lbls)

idx<-lbls!=0
helper_match_evaluate_multiple(mbr[idx], lbls[idx])
helper_match_evaluate_multiple(mbr2[idx], lbls[idx])

##########################################################################
#outlier removal
gr_res <- mcOut(asym_w, addLoops = F, expansion = 2, inflation = 10,  max.iter = 2, ESM = TRUE); gc()
gr_res[[1]]
gr_w<-gr_res[[2]] 
head(sort(colSums(gr_w),decreasing = T))
hist(colSums(gr_w),  500000)
hist(colSums(gr_w), xlim=c(0,1), 5000)
table(lbls[rowSums(gr_w)==0])
table(lbls[colSums(gr_w)==0])
table(lbls[colSums(gr_w)>0])
sum(colSums(gr_w)==0)
table(lbls)

plot(colSums(gr_w), unlist(lapply(1:nrow(gr_w), function(x) nnzero(gr_w[x,]))))
table(lbls[colSums(gr_w)==0])

IDXe<-which(colSums(asym_w)==0)
IDXw<-which(colSums(gr_w)==0)

IDX <- !((1:length(lbls)) %in% union(IDXe, IDXw))
table(lbls[!IDX])
table(lbls[!(!((1:length(lbls)) %in% IDXw)])

indg<-induced_subgraph(g2, (1:gorder(g2))[IDX])

cl_res<-cluster_louvain(g)
cl_resi<-cluster_louvain(indg)

mbr<-membership(cl_res)
mbri<-membership(cl_resi)

table(mbr)
table(mbri)
table(lbls)

comf<-rep(NA,length(lbls))
comf[IDX]<-membership(cl_resi)
comf[!IDX]<-100
table(comf)

helper_match_evaluate_multiple(mbr, lbls)
helper_match_evaluate_multiple(comf, lbls)







IDX=sample(nrow(cl_coord), 1000)
matplot(t(m2$nn.dists[IDX[lbls[IDX]==0],]), type='l')
matlines(t(m2$nn.dists[IDX[lbls[IDX]==8],]), type='l')

plot3d((princomp(cl_coord, scores = TRUE)$scores)[, c(1,2,3)], pch='.')

cl_coord<-scale(cl_coord, center = TRUE, scale = TRUE)
hist(dist(cl_coord[sample(nrow(cl_coord), 1000), ]))


maxnodes=5000
system.time(g <- GNG(cl_coord, max.nodes=maxnodes,  max.iter = 6000,  verbosity=0, min.improvement=0.005))
gngSave(g, file=paste0("../../results", "/GNGtests/", "g10M500.RData"))
gngLoad( file=paste0("../../results", "/GNGtests/", "g10M500.RData"))

table(clustering(g))
meanError(g)
nNodes<-numberNodes(g)
errorStatistics(g)
plot(errorStatistics(g))
nodeError<-unlist(lapply(1:numberNodes(g), function(x) node(g,x)$error))
plot(nodeError, pch='.')

#plot(g, mode=gng.plot.2d.errors)#, layout=gng.plot.layout.v2d, vertex.color=gng.plot.color.cluster)
#plot(g)
#centr <- calculateCentroids(g)
#predictComponent(g, cl_coord[70000,])

#create a bacbone of the cluster from the nodes
#TODO:try walktrap.community
ig<-convertToIGraph(g)
#plot(ig)
vrtx<-vertex_attr(ig)
ed<-edge_attr(ig)
ed_w<-1/(ed$dists+3)
hist(ed$dists, 150)
#ed_w<-ifelse(ed$dists>1, 0,1)
system.time(cl_InfoMap<-cluster_infomap(ig, e.weights = ed_w, v.weights = nodeError, nb.trials = 100,  modularity = TRUE))
#optimal mudularity
system.time(cl_InfoMap<-cluster_edge_betweenness(ig, directed = F, weights = ed_w))

sizes(cl_InfoMap)
membershipInfoMap<-membership(cl_InfoMap)

#get the closest node for each data point
#nodeAssign<-unlist(mclapply(chunks2(nrow(cl_coord),60), function(i) findClosests(g, 1:nNodes, cl_coord[i,]), mc.cores=6, mc.preschedule=F), recursive=T)
#fast assignment of the points to INFO_map clusters 
surv_nodes<-length(unique(clustering(g)))
nodecoordsL<-lapply(1:surv_nodes, function(x) node(g, x)$pos)
nodecoords<-do.call(rbind, nodecoordsL)
system.time(nodeAssign<-unlist(mclapply(chunks2(nrow(cl_coord),30), function(i) get.knnx(nodecoords, cl_coord[i,], k=1, algorithm="kd_tree")$nn.index, mc.cores=6, mc.preschedule=F), recursive=T))

names(membershipInfoMap) = 1:(length(membershipInfoMap)) 
clus=unlist(mclapply(nodeAssign, function(i) membershipInfoMap[as.character(i)], mc.cores=6))

res=helper_match_evaluate_multiple(clus, lbls) 
res


rcentr <- calculateCentroids(g)
findClosests(g, centr, cl_coord[7000,])#quick hack for the cluster assignment
predict(g, cl_coord[70000,])
predictComponent(g, cl_coord[20000,])#quick hack for the cluster assignment




surv_nodes<-length(unique(clustering(g)))
nodecoordsL<-lapply(1:surv_nodes, function(x) node(g, x)$pos)
nodecoords<-do.call(rbind, nodecoordsL)


#plot3d((princomp(nodecoords, scores = TRUE)$scores)[, c(1,2,3)]) 


d <- dist(nodecoords)
hist(d)
fit <- cmdscale(d,eig=TRUE, k=3)
plot3d(fit$points)

plot3d((princomp(cl_coord, scores = TRUE)$scores)[, c(1,2,3)], pch='.', col = ifelse(lbls==0, 'green', 'black'))
plot3d((nodecoords %*% (princomp(cl_coord, scores = TRUE)$loadings)[, c(1,2,3)]) ,  col='red', add=T, type='s', size=0.5)

#color by cluster
ncolors=length(unique(lbls))
col=rainbow(ncolors)
colors<-(unlist(lapply(lbls, function(x) col[x])))
plot3d((princomp(cl_coord, scores = TRUE)$scores)[, c(1,2,3)], pch='.', col = colors)

ncolors=length(unique(clus))
col=rainbow(ncolors)
colors<-(unlist(lapply(clus, function(x) col[x])))
plot3d((princomp(cl_coord, scores = TRUE)$scores)[, c(1,2,3)], pch='.', col = colors)





helper_match_evaluate_multiple(clustering(g), lbls)

#evaluate clustering based on centroid coordiantes
centr_coords(node,centr)



mydata <- nodecoords
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:25) wss[i] <- sum(kmeans(mydata,
                                     centers=i)$withinss)
plot(1:25, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")





