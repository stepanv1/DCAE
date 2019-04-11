#Experiment:
#1. Extract outliers from samusik data and extract core points. Core points are the ones
#within the most dense areas in clusters 
#2. Investigate difference in d(k) (distance vs number of the neighbour) for the core points and for the outliers, "coreness" measure shows how likely the point to be the core point of the cluster
#3. Build the rating on "coreness" of the curve based on the difference above.
#4. Build the algorithm to continuosly grow clusters from core ponts towards outliers.
#UPDATE
#   This algorithm should proceed in the following way: create an initial backbone from top most   core points. Consequently include more outlying pointsa in the order from higher to lower coreness.
#Try a quick hack after "coreness" is implemented: run phenograph on top 95% most core points
#5.   
library(parallel)
library(data.table)
library(igraph)
library(gatepoints)
library(rgl)
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_multiple.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')

#run_phenograph before (preambula of the script)
##and evaluate
cos2<- function(a,b) {
  crossprod(a,b)/sqrt(crossprod(a)*crossprod(b))
}
#################################################################
#calculate sequential cosine differences between nearest neighbors 
cosDistNearest<-function(dat, vec){
  dat<-sweep(dat,2,vec)
  dat<-t(apply(dat,1, function(x) x/sum(abs(x))))
  unlist(lapply(1:(nrow(dat)-1), function(x) cos2(dat[x,], dat[x+1, ])))
}
###################################################
#given output of cytofkit:::jaccard_coeff function
#adds to it a column witha reciprocity score

chunk <- function(x,n) split(x, cut(seq_along(x), n, labels = FALSE)) 
dirtable<-function(tbl, mc.cores=4){
  colnames(tbl)<-c('V1','V2','V3')
  tbl <-as.data.table(tbl)
  #create key column base on node names
  chunks <- chunk(1:nrow(tbl), mc.cores)
  EdgeKey <- unlist(mclapply(chunks, function(x) paste0(pmin(tbl[x, V1 ], tbl[x,V2 ]), ' ',  pmax(tbl[x, V1 ], tbl[x,V2 ])), mc.cores=mc.cores))
  tbl[, EdgeKey := EdgeKey]
  setkey(tbl, NULL)
  setkey(tbl, EdgeKey)
  #count if there is reciprocal link
  tbl[, recCount := .N, by = EdgeKey]
  tbl[, linkCount := sum(recCount==2), by = V1]
  #dup<-duplicated(tbl[, EdgeKey])
  return(tbl)
}
######################################################################
#takes a data.table with recCount column (output of dirtable) and adds
#rank for 'loneliness' of points
perc.rank <- function(x) trunc(rank(x))/length(x)
Loners<-function(dirtable, step=0.01){
  dirtable[, perc := perc.rank(linkCount)]
  return(dirtable)
}
#######################################################################
# calculates the pvalue of two sets intersection
# which is the probability, that under null hypothesis H_0
# that there is no correlation between G an N sets, 
# to have the size of intersection large than c
# U -cardinality of universal set, G - cardinality of smaller set, 
# N - of bigger set, c - of intersection of N and G 
#uses gmp functions for big arithmetics
library(gmp)
pCardIntersection<-function(U,G,N,c, mc.cores=10){
  nom<-sum.bigz(unlist(mclapply(chunk(c:G, mc.cores), function(x) chooseZ(U,x)*chooseZ(U-(x), G-(x))*chooseZ(U-G, N-(x)), mc.cores=mc.cores)))
  denom<-sum.bigz(unlist(mclapply(chunk(0:G, mc.cores), function(x) chooseZ(U,x)*chooseZ(U-(x), G-(x))*chooseZ(U-G, N-(x)), mc.cores=mc.cores)))
  return(div.bigz(nom, denom))
}





setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
#load("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/auto/phenoGraph/Neighbours/SamusikNeighborsK105.RData")
#load("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/results/auto/phenoGraph/k=40_2016-12-0511:10:51/SamusikNeighborsK20.RData")

RES_DIR  <- "../../results/"
CALC_NAME="KdependencySamusik_840k"

#ind<-read.table(file=paste0("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/",  CALC_NAME ,"/indSamusik_all.txt"),  header = FALSE)
#dist<-read.table(file=paste0("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/",  CALC_NAME ,"/distSamusik_all.txt"),  header = FALSE)
#ind<-as.matrix(ind[,-1])
#dist<-dist[,-1]

#jaccard45<-read.table(file=paste0("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/",  #CALC_NAME ,"/j45Samusik_all.txt"),  header = FALSE)
#jaccard45[,1:2]<-jaccard45P[,1:2]+1
#jaccard45<-as.data.table(jaccard45P)
dirname=paste0(RES_DIR, CALC_NAME)
#system(paste0('mkdir ', dirname))
#save(jaccard45, file=paste0(dirname, '/jaccard45.RData'))
load( file=paste0(dirname, '/jaccard45.RData'))


#########################################################
#to plot the d(k) curves 
load("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/benchmark_data_sets/all_sets.RData")
#idx<-mclapply(1:nrow(ind), function(x) ind[x, ], mc.cores=8)
#md<-mclapply(1:nrow(ind), function(x) as.matrix(dist(data[[4]][c(x,idx[[x]]),]))[1, ], mc.cores=4)
#nndF<-matrix(unlist(md), byrow=T, ncol=76)
#nndF<-as.data.frame(nndF)
#plot(unlist(lapply(nndF[nndF$V2<150, ], mean)))
#matplot(t(nndF[sample(1:nrow(nndF), 15),2:ncol(nndF)]), pch='.', type = 'l', xlim=c(2, 41))
#matplot(diff(t(nndF[sample(1:nrow(nndF), 1),2:ncol(nndF)])), pch='.', type = 'l', xlim=c(2, 41))
i=4
dd<-as.matrix(dist(data[[i]][sample(1:nrow(data[[i]]), 10000),], method='manhattan'))
hist(dd,500, main='manhattan')
dd<-as.matrix(dist(data[[i]][sample(1:nrow(data[[i]]), 10000),]))
hist(dd,500, main='euclidian')

# plot neighbour distances 
##############################################################

#load labels, and use thise to separate pre-gated cells 
files_truth <- list(
  Samusik_all = file.path("../../benchmark_data_sets", "Samusik_all.fcs") 
)

# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

for (i in 1:length(clus_truth)) {
  data_truth_i <- flowCore::exprs(flowCore::read.FCS(files_truth[[i]], transformation = FALSE, truncate_max_range = FALSE))
  clus_truth[[i]] <- data_truth_i[, "label"]
}
clus_truth<-clus_truth[[i]]
data<-as.matrix(data)

#####################################################################
#add the loneliness index

#jaccard45 <- dirtable(jaccard45)
#jaccard45<-Loners(jaccard45)
#dirname=paste0(RES_DIR, CALC_NAME)
#system(paste0('mkdir ', dirname))
#save(jaccard45, file=paste0(dirname, '/jaccard45.RData'))
#load( file=paste0(dirname, '/jaccard45.RData'))
olidx<-unique(jaccard45[linkCount<=2,V1])# Remove potential bridges
table(clus_truth[olidx])
table(clus_truth)
#olidx<-unique(jaccard45[perc<=0.1,V1])

isna<-which(is.na(clus_truth))
belongNA<-olidx %in% as.integer(isna)
sum( belongNA)
#35789
#About a half of loners belongs to gated out 327000 cells. Statistically significant at ..
graphTbl<-jaccard45[!((V1 %in% olidx) | (V2 %in% olidx)), 1:3]
length(unique(graphTbl[,V1]))
graphTbl<-graphTbl[V3!=0,]
graphTbl<-graphTbl[order(V1)]
setnames(graphTbl,c('V1', 'V2', 'V3'),  c('from', 'to', 'weight'))
gr<-graph_from_data_frame(as.data.frame(graphTbl[,1:3]), directed = F, vertices = NULL)
#E(gr)$weight=as.numeric(as.data.frame(graphTbl)[,3])
gr <- simplify(gr, remove.loops=FALSE, edge.attr.comb=list(weight="sum"))
print(gr, max.lines = 10)
hist(E(gr)$weight,500)

#save(gr, file=paste0(dirname, '/L1_gr45.RData'))
A.  Lancichinetti,  S.  Fortunato,  and  J.  Kertesz,  New  J.
Physics
11
, 033015 (2009).
#zzz=induced.subgraph(graph=gr,vids=1:10000)
#com<-cluster_louvain(zzz)
#system.time(com<-cluster_louvain(gr, weights = E(gr)$weight))
load(file=paste0(dirname, '/L1_com45.RData'))
#save(com, file=paste0(dirname, '/L1_com45.RData'))

clus_core<-membership(com)
clus<-rep(NaN, length(clus_truth))
clus[!((1:length(clus_truth)) %in% olidx)] <- clus_core

IDX<-intersect(which(!is.na(clus_truth)), which(!is.na(clus)))

helper_match_evaluate_multiple(clus[IDX], clus_truth[IDX])
helper_match_evaluate_multiple_SweightedN(clus[IDX], clus_truth[IDX])

##########################################################################
# experiment with Rborist to speed up the random forest
library(Rborist)
dataset.train=as.data.frame(cbind(data[[4]][!((1:nrow(data[[4]])) %in% olidx),], clus_core))
system.time(rb <- Rborist(dataset.train[,-40], as.factor(dataset.train[,40]), classWeight = rep(1, length(unique(clus_core))), nTree=500))
save(rb, file=paste0(RES_DIR,'RandomForestSamusikRboristBalance.RData'))
system.time(pred <- predict(rb, data[[4]][((1:nrow(data[[4]])) %in% olidx),]))
yPred <- pred$yPred
clusForest<-clus
clusForest[((1:length(clus_truth)) %in% olidx)]<-yPred

#write.table(clusForest, file = 'clusForestBalance.txt', row.names = FALSE, quote = FALSE, sep = "\t")
clus<-unlist(read.table(paste0(RES_DIR, '/KdependencySamusik_840k/', 'clusForestBalance.txt'), header = T, stringsAsFactors = FALSE))

res1<-helper_match_evaluate_multiple(clusForest, clus_truth)
helper_match_evaluate_multiple_SweightedN(clusForest, clus_truth)

#check the separation of clusters 7 and 10
table(clus[clus_truth==7])
table(clus[clus_truth==10])
res1$n_cells_matched
res1$labels_matched
table(clusForest[clus_truth==7])
table(clusForest[clus_truth==10])
res1$n_cells_matched
res1$labels_matched
#correct assignment, separation is observed
#Try L2 metric
#Algorithm runs in 20 30 mins after neighbors calculated
#neighbours calculations are extremely slow, need to think about possible re-writing
#of nearest neighbor package for R, since the existing one is very slow
# so, dilemma: R or Python if to go with this application
# Imbalance learnining. Synthetic data. Gradiant boosing methods, by adding weak learners
# Directed Louvain,  ModMax package: genetic algorithm, mome
library(modMax)
mome()
######################################################################
# Label propagation ###########################
#com<-cluster_louvain(zzz)
#system.time(com<-cluster_label_prop(gr, weights = E(gr)$weight))
load(file=paste0(dirname, '/L1_com45.RData'))
#save(com, file=paste0(dirname, '/L1_com45.RData'))

clus_core<-membership(com)
clus<-rep(NaN, length(clus_truth))
clus[!((1:length(clus_truth)) %in% olidx)] <- clus_core

IDX<-intersect(which(!is.na(clus_truth)), which(!is.na(clus)))

helper_match_evaluate_multiple(clus[IDX], clus_truth[IDX])
helper_match_evaluate_multiple_SweightedN(clus[IDX], clus_truth[IDX])
AMI(clus[IDX], clus_truth[IDX])

##########################################################################
# Rborist to speed up the random forest
library(Rborist)
dataset.train=as.data.frame(cbind(data[[4]][!((1:nrow(data[[4]])) %in% olidx),], clus_core))
system.time(rb <- Rborist(dataset.train[,-40], as.factor(dataset.train[,40]), classWeight = rep(1, length(unique(clus_core))), nTree=500))
save(rb, file=paste0(RES_DIR,'RandomForestSamusikRboristBalance.RData'))
system.time(pred <- predict(rb, data[[4]][((1:nrow(data[[4]])) %in% olidx),]))
yPred <- pred$yPred
clusForest<-clus
clusForest[((1:length(clus_truth)) %in% olidx)]<-yPred

write.table(clusForest, file = 'clusForestBalance.txt', row.names = FALSE, quote = FALSE, sep = "\t")

res1<-helper_match_evaluate_multiple(clusForest, clus_truth)
helper_match_evaluate_multiple_SweightedN(clusForest, clus_truth)

#check the separation of clusters 7 and 10
table(clus[clus_truth==7])
table(clus[clus_truth==10])
res1$n_cells_matched
res1$labels_matched
table(clusForest[clus_truth==7])
table(clusForest[clus_truth==10])
res1$n_cells_matched
res1$labels_matched
#correct assignment, separation is observed
#Try L2 metric
#Algorithm runs in 20 30 mins after neighbors calculated
#neighbours calculations are extremely slow, need to think about possible re-writing
#of nearest neighbor package for R, since the existing one is very slow
# so, dilemma: R or Python if to go with this application
# Imbalance learnining. Synthetic data. Gradiant boosing methods, by adding weak learners
# Directed Louvain,  ModMax package: genetic algorithm, mome
library(modMax)
mome()



#transform into cosin distances
#cos between gated out
unassigned <- is.na(clus_truth)
idx<-sample(1:nrow(data) , 1000/0.4)
idx<-unlist(lapply(1:nrow(data), function(x) unassigned[x] & (x %in% idx)))
idx<-which(idx==T)
lst<-lapply(idx, function(x) (cosDistNearest(data[ind[x, ], ], data[x,])))
cosDist1<-Reduce(rbind, lst)
matplot(t(cosDist1),  pch='.', type = 'l', main='Gated out', ylim=c(-0.5,0.8))

#cos between in-cluster
unassigned <- is.na(clus_truth)
idx<-sample(1:nrow(data) , 1000/0.6)
idx<-unlist(lapply(1:nrow(data), function(x) !unassigned[x] & (x %in% idx)))
idx<-which(idx==T)
lst<-lapply(idx, function(x) (cosDistNearest(data[ind[x, ], ], data[x,])))
cosDist2<-Reduce(rbind, lst)
matplot(t(cosDist2),  pch='.', type = 'l', main='In cluster', ylim=c(-0.5,0.8))

boxplot(cosDist1)
boxplot(cosDist2)


#plot neighbor distances
matplot(t(distClus[sample(1:nrow(distClus), 50), 1:ncol(distClus)]), pch='.', type = 'l', main='In clusters')
matplot(t(distGate[sample(1:nrow(distGate), 50), 1:ncol(distGate)]), pch='.', type = 'l', main='Gated out')

matplot(diff(t(distClus[sample(1:nrow(distClus), 5000), 1:ncol(distClus)])), pch='.', type = 'l', main='In clusters')
matplot(diff(t(distGate[sample(1:nrow(distGate), 5000), 1:ncol(distGate)])), pch='.', type = 'l', main='Gated out')

#angular distances


#########################################################################################
#pick ip the data from the screen and find put their d(k) curve properties depending on position
load("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/results/auto/phenoGraph/KdependencySamusik_all/tsne.RData")
#run evaluate_phenograph preambule with correct CALC_ID
pop_names=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names$label=rownames(pop_names)
pop_names$population=pop_names$population.name


library(grDevices)
i=4
ncolors=length(unique(clus_truth[[i]]))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(clus_truth[[i]])
colors<-unlist(lapply(clus_truth[[i]], function(x) col_true[as.character(x)]))

#data<- cbind(f, clusters)

pc=princomp(data[[4]])
plot3d(pc$scores[,c(1,2,3)],  col=colors, pch='.') 

plot(res_tsne[[1]]$tsne_out$Y,col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names$population,col = col_true[as.character(pop_names$label)], pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(res_tsne[[1]]$tsne_out3D$Y,  col=colors, pch='.') 
legend3d("topright", legend = pop_names$population, pch = 16, col = col_true[as.character(pop_names$label)], cex=1, inset=c(0.02))

#color subset of  clusters  
cells=c(10,7)
colors1<-unlist(lapply(clus_truth[[i]], function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(res_tsne[[1]]$tsne_out$Y,col=colors1, pch='.', cex=1, main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(res_tsne[[1]]$tsne_out3D$Y,  col=colors1, pch='.', main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

# pick a point from 3d plot
f <- select3d()
selectedPoints3D=f(pc$scores[,c(1,2,3)]); sum(selectedPoints3D)
selectedPoints3D=f(res_tsne[[1]]$tsne_out3D$Y); sum(selectedPoints3D)
matplot(t(nndF[selectedPoints3D ,2:ncol(nndF)]), pch='.', type = 'l', xlim=c(2, 41))
hist(unlist(lapply(as.data.frame(t(nndF[selectedPoints3D ,2:ncol(nndF)])), function(x) sd(diff(x))/abs(mean(diff(x))))), 50)
hist(nl[selectedPoints3D,1], main="Degree 3d samples", breaks=50)
hist(nl[selectedPoints3D,2], main="Strength 3d samples", breaks=50)
plot(unlist(nl[selectedPoints3D,1]), unlist(nl[selectedPoints3D,2]), pch='.')
hist(unlist(nl[selectedPoints3D,2])/unlist(nl[selectedPoints3D,1]))
dd<-as.matrix(dist(data[[4]][selectedPoints3D,]))
hist(dd)

plot(res_tsne[[1]]$tsne_out$Y,col=colors, pch='.', cex=1)
selectedPoints <- as.numeric(fhs(res_tsne[[1]]$tsne_out$Y))
matplot(t(nndF[selectedPoints ,2:ncol(nndF)]), pch='.', type = 'l', xlim=c(2, 41))
hist(nl[selectedPoints,1], main="Degree 3d samples", breaks=50)
hist(nl[selectedPoints,2], main="Strength 3d samples", breaks=50)
plot(unlist(nl[selectedPoints,1]), unlist(nl[selectedPoints,2]), pch='.')
hist(unlist(nl[selectedPoints,2])/unlist(nl[selectedPoints,1]))


#TODO: plot 2 small clusters in PC components and see how outliers behave

IDX=clus_truth[[i]] %in% c(10,7)
pc=princomp(data[[4]][IDX,])
plot3d(pc$scores[,c(1,2,3)],  col=colors[IDX], pch='.') 
# pick a point from 3d plot
f <- select3d()
selectedPoints3D=f(pc$scores[,c(1,2,3)]); sum(selectedPoints3D)
matplot(t(nndF[IDX,][selectedPoints3D ,2:ncol(nndF)]), pch='.', type = 'l', xlim=c(2, 41))
hist(unlist(lapply(as.data.frame(t(nndF[IDX,][selectedPoints3D ,2:ncol(nndF)])), function(x) sd(diff(x))/abs(mean(diff(x))))), 50)
hist(nl[IDX,][selectedPoints3D,1], main="Degree 3d samples", breaks=50)
hist(nl[IDX,][selectedPoints3D,2], main="Strength 3d samples", breaks=50)
plot(unlist(nl[IDX,][selectedPoints3D,1]), unlist(nl[IDX,][selectedPoints3D,2]), pch='.')
hist(unlist(nl[IDX,][selectedPoints3D,2])/unlist(nl[IDX,][selectedPoints3D,1]))



match_table <-res[[4]]$labels_matched
lbl_mapped = unlist(lapply(clus[[4]], function(x) {
  ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[x])))
plot(res_tsne[[1]]$tsne_out$Y,col=colors_matched, pch='.', cex=1, main=CALC_NAME)
#pick a point from 2d plot
selectedPoints <- as.numeric(fhs(res_tsne[[i]]$tsne_out$Y))
matplot(t(nndF[selectedPoints ,2:ncol(nndF)]), pch='.', type = 'l', xlim=c(2, 41))
hist(unlist(lapply(as.data.frame(t(nndF[selectedPoints ,2:ncol(nndF)])), function(x) sd(diff(x))/abs(mean(diff(x))))), 50)
matplot(diff(t(nndF[selectedPoints ,2:ncol(nndF)])), pch='.', type = 'l', xlim=c(2, 41))


library(geometry)
cluster1<-data[[i]][clus_truth[[i]]==4, ]
#hull<-convhulln(cluster1[,], options = "Fx W1e-1 C1e-2 TF2000")
vertices=unique(as.vector(hull))
insiders<-unlist(lapply(1:513, function(x) ifelse(x %in% vertices, F, T)))
mean(dist((cluster1[insiders,])))
col2=unlist(lapply(1:513, function(x) ifelse(x %in% vertices, 'red', 'black')))
plot(cluster1[,c(5,9)], col=col2)
matplot(t(nndF[clus_truth[[i]]==14, 3:ncol(nndF)][vertices,]), pch='.', type = 'l', xlim=c(2, 41))
matplot(t(nndF[clus_truth[[i]]==14, 3:ncol(nndF)][insiders,]), pch='.', type = 'l', xlim=c(2, 41))
#todo: full search for vertices, based on .. SVM?

axis_set=replicate(10000,sample(1:39, 5 ,replace = F))
#system.time(resHull<-mclapply(1:10000, function(x) convhulln(cluster1[,axis_set[,x]], options = "Fx W1e-1 C1e-2 TF2000"), mc.cores=10))
dirname=paste0(RES_DIR, CALC_NAME)
system(paste0('mkdir ', dirname))
#save(resHull, file=paste0(dirname, '/convexHull.RData'))
load( file=paste0(dirname, '/convexHull.RData'))

n=1
ddcluster<-as.matrix(dist(data[[i]][clus_truth[[i]]==n, ][sample(sum(clus_truth[[i]]==n), 4000),]))
hist(ddcluster,50)

verticesS=unique(as.vector(unlist(resHull)))
insidersS<-unlist(lapply(1:12479 , function(x) ifelse(x %in% verticesS, F, T)))
mean(dist((cluster1[insidersS,])))
mean(dist((cluster1[verticesS,])))
col2=unlist(lapply(1:12479 , function(x) ifelse(x %in% verticesS, 'red', 'black')))
plot(cluster1[,c(35,14)], col=col2, pch='.')

vk=nndF[clus_truth[[i]]==i , 1:ncol(nndF)][verticesS,]
ik=nndF[clus_truth[[i]]==i , 1:ncol(nndF)][insidersS,]
matplot(t(vk[sample(nrow(vk),25),]), pch='.', type = 'l', xlim=c(1, 41), main='vertices',log = "x")
matplot(t(ik[sample(nrow(ik),25),]), pch='.', type = 'l', xlim=c(1, 41), main='insiders', log = "x")
matplot(diff(t(vk[sample(nrow(vk),25),])), pch='.', type = 'l', xlim=c(0, 41), main='vertices')
matplot(diff(t(ik[sample(nrow(ik),25),])), pch='.', type = 'l', xlim=c(0, 41), main='insiders')

hist(unlist(lapply(as.data.frame(t(vk[,])), function(x) sd(diff(x))/abs(mean(diff(x))))), 50, main='vertices')
hist(unlist(lapply(as.data.frame(t(ik[,])), function(x) sd(diff(x))/abs(mean(diff(x))))), 50, main='insiders')
#further attemt to identify oter points in clusters as aoutliers
library(Rlof)
lofres<-lof(cluster1, 15, cores = 8)
hist(lofres)
ok=nndF[clus_truth[[i]]==i , 1:ncol(nndF)][lofres>1.4,]
matplot(t(ok[,]), pch='.', type = 'l', xlim=c(0, 41), main='outliers')
library("HighDimOut")
#FBODres<-Func.FBOD(cluster1, 30, k.nn=15)
ok=nndF[clus_truth[[i]]==i , 3:ncol(nndF)][lofres>1.6,]
matplot(t(ok[sample(nrow(ok),25),]), pch='.', type = 'l', xlim=c(2, 41), main='outliers')

###########################################################
#check the mean angles between knearest neighbors in border vectors and core ones
#to identify teh difference


cosI<-(unlist(mclapply(as.integer(rownames(ik)), function(x){
  nbIDX<-neighborMatrix[x,]
  nb<-t(as.data.frame(t(data[[i]][nb,])))
  nd<-data[[i]][x,]
  diff<-as.data.frame(nb-nd)
  lapply(diff, function(z) lapply(diff, function(y) cos2(z, y) ))
}, mc.cores=5)))

cosO<-(unlist(mclapply(as.integer(rownames(ok)), function(x){
  nbIDX<-neighborMatrix[x,]
  nb<-t(as.data.frame(t(data[[i]][nb,])))
  nd<-data[[i]][x,]
  diff<-as.data.frame(nb-nd)
  lapply(diff, function(z) lapply(diff, function(y) cos2(z, y) ))
}, mc.cores=5)))

cosV<-(unlist(mclapply(as.integer(rownames(vk)), function(x){
  nbIDX<-neighborMatrix[x,]
  nb<-t(as.data.frame(t(data[[i]][nb,])))
  nd<-data[[i]][x,]
  diff<-as.data.frame(nb-nd)
  lapply(diff, function(z) lapply(diff, function(y) cos2(z, y) ))
}, mc.cores=5)))



#network properties
#strength for the full Samusik data.set, k=20, 40, 10 million lines
i=8
dt<-as.data.table(jaccard[[i]]['all'])
colnames(dt) <-  c('V1','V2','V3')
dt01<-as.data.table(jaccard[[i]]['01'])
colnames(dt01) <-  c('V1','V2','V3')
system.time(nlinks<-mclapply(1:max(dt[,V1]), function(x) 
{degree=nrow(dt[V1==x])
strength=sum(dt[V1==x, V3])
return(list('degree'=degree, 'strength'=strength))},
mc.cores=8))
#124 seconds on 8 cores, never finished with data.frame 
nl<-rbindlist(nlinks)

system.time(nlinks01<-mclapply(1:max(dt01[,V1]), function(x) 
{degree=nrow(dt01[V1==x])
strength=sum(dt01[V1==x, V3])
return(list('degree'=degree, 'strength'=strength))},
mc.cores=8))
nl01<-rbindlist(nlinks01)

hist(nl[,1], main="Degree all samples")
hist(nl01[,1], main="Degree 1 sample")

hist(nl[,2], main='Strength all samples')
hist(nl01[,2],  main='Strength 1 sample')

hist(log(nl[,2]+0.0000001), main='Strength all samples')
hist(log(nl01[,2]+0.0000001),  main='Strength 1 sample')
########strength per degree
hist(log(nl40[,2]/nl40[,1]+0.0000001), main='Strength per degree')
hist(log(nl0140[,2]/nl0140[,1]), main='Strength per degree')

hist(nl40[,2], main='Strength all samples')
hist(nl0140[,2],  main='Strength 1 sample')

#There is a difference in strength at k=40 maximum is achieved at s=2 for 01, and at roughly at 1 for all samples. There is no unconnected nodes at k=40 for 01
##########################################################################
#Build neighb object for spdep
#
system.time(nb01<-mclapply(1:max(dt01[,V1]), function(x) 
{nbrs=(dt01[V1==x, V2])
return(nbrs)},
mc.cores=8))
system.time(w01<-mclapply(1:max(dt01[,V1]), function(x) 
{wghts=(dt01[V1==x, V3])
return(wghts)},
mc.cores=6))

##############################################################################
#calculate mean distance between connected dots. As soon as ther is a jump in characteristic
#distance we might observe the k_max 




##############################################################################
#topological properties

plot(unlist(nl[,1]), log(unlist(nl[,2])), pch='.')
plot(unlist(nl40[,1]), log(unlist(nl40[,2])), pch='.')

plot(unlist(nl01[,1]), log(unlist(nl01[,2])), pch='.')
plot(unlist(nl0140[,1]), log(unlist(nl0140[,2])), pch='.')

summary(lm(log(unlist(nl01[nl01$degree>10,2]+0.0000000001))~unlist(nl01[nl01$degree>10,1])))

##################################################################################
#2d density
library(aplpack)
bagplot(as.data.frame(data[[3]])[,c('Ly6C','MHCII')])
library(hexbin)
library(RColorBrewer)
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))
r <- rf(32)

# Create hexbin object and plot
h <- hexbin(as.data.frame(data[[3]])[,c('Ly6C','MHCII')])
plot(h)
plot(h, colramp=rf)
h <- hexbin(as.data.frame(data[[3]])[,c(12,25)])
plot(h)
plot(h, colramp=rf)



#Look up operation speed up in data.table
system.time(nlinks<-unlist(mclapply((1:max(links[,1]))[1:10000], function(x) 
{nrow(links[links[,1]==x, ])
},
mc.cores=8)))
#approx 3 mins
#system.time(nlinks<-unlist(mclapply((1:max(links[,1]))[1:10000], function(x) 
system.time(nlinks<-unlist(mclapply((1:max(links[,1]))[1:100000], function(x) 
{nrow(dt[V1==x] )
},
mc.cores=8)))
#1 second on 10000
#
##########TEMP
#########
idx<-mclapply(1:nrow(neighborMatrix01), function(x) neighborMatrix01[x, ], mc.cores=8)
md<-mclapply(1:nrow(neighborMatrix01), function(x) as.matrix(dist(data[[3]][c(x,idx[[x]]),]))[1, ], mc.cores=8)
nnd<-matrix(unlist(md), byrow=T, ncol=41)
nnd<-as.data.frame(nnd)
plot(unlist(lapply(nnd[nnd$V2<150, ], mean)))
matplot(t(nnd[sample(1:nrow(nnd), 15),]), pch='.', type = 'l')
dd<-as.matrix(dist(data[[3]][sample(1:nrow(data[[3]]), 1000),]))
hist(dd)

idx<-mclapply(1:nrow(neighborMatrix), function(x) neighborMatrix[x, ], mc.cores=8)
md<-mclapply(1:nrow(neighborMatrix), function(x) as.matrix(dist(data[[4]][c(x,idx[[x]]),]))[1, ], mc.cores=8)
nndF<-matrix(unlist(md), byrow=T, ncol=41)
nndF<-as.data.frame(nndF)
plot(unlist(lapply(nndF[nndF$V2<150, ], mean)))
matplot(t(nndF[sample(1:nrow(nndF), 15),]), pch='.', type = 'l')
dd<-as.matrix(dist(data[[4]][sample(1:nrow(data[[4]]), 1000),]))
hist(dd)



plot(md[1,])
library(modMax)
#experiments with simulating annealing
randomgraph <- erdos.renyi.game(10000, 0.3, type="gnp",directed = FALSE, loops = FALSE)
vertices <- which(clusters(randomgraph)$membership==1)
graph <- induced.subgraph(randomgraph,vertices)
adj <- get.adjacency(graph)
plot(graph)
result <- simulatedAnnealing(adj, fixed=10)
system.time(result <- simulatedAnnealing(adj, initial='greedy', fixed=10))
result <- mome(adj)

