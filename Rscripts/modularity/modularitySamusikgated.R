#########################################################################################
# R script to load PhenoGraph results and calculate \delta Q for smaller clusters
#not separated by Phenograph
#
# Lukas Weber, August 2016
# Stepan Grinek February 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)
library(igraph)
library(data.table)
library(cytofkit)

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_graphs_distances.R')

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/PhenoGraph"
CALC_NAME="KdependencySamusik_all"
DATA_DIR <- "../../benchmark_data_sets"


####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
  Samusik_all = file.path(DATA_DIR, "Samusik_all.fcs") 
)

# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

#for (i in 1:length(clus_truth)) {
#  data_truth_i <- flowCore::exprs(flowCore::read.FCS(files_truth[[i]], transformation = FALSE, truncate_max_range = FALSE))
#  clus_truth[[i]] <- data_truth_i[, "label"]
#}

sapply(clus_truth, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl_truth <- lapply(clus_truth, table)

tbl_truth
sapply(tbl_truth, length)

# store named objects (for other scripts)

unassigned <- is.na(clus_truth[[1]])
clus_truth[[1]] <- clus_truth[[1]][!unassigned]
#data<-data[!unassigned, ]

files_truth_PhenoGraph <- files_truth
clus_truth_PhenoGraph <- clus_truth

dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
system(paste0('mkdir ', dirname))
#save(clus_truth, file=paste0(dirname, '/clus_truth.RData'))
load(paste0(dirname, '/clus_truth.RData'))


###############################
### load PhenoGraph results ###
###############################

# load cluster labels
files_res <- list(
  Samusik_all15 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=15Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all30 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=30Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all45   = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=45Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all60  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=60Python_assigned_labels_phenoGraph_Samusik_all.txt"),
  Samusik_all75  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=75Python_assigned_labels_phenoGraph_Samusik_all.txt"),
  Samusik_all90  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=90Python_assigned_labels_phenoGraph_Samusik_all.txt"),
  Samusik_all105  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=105Python_assigned_labels_phenoGraph_Samusik_all.txt") 
)


clus <- vector("list", length(files_res))
names(clus) <- names(files_res)

for (i in 1:length(clus)){
  clus[[i]] <- read.table(files_res[[i]], header = F, stringsAsFactors = FALSE)[,1]
}

sapply(clus, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl <- lapply(clus, table)

tbl
sapply(tbl, length)

# contingency tables

#for (i in 1:length(clus)) {
#  print(table(clus[[i]], clus_truth[[1]]))
#}

# store named objects (for other scripts)

files_PhesnoGraph <- files_out
clus_PhenoGraph <- clus


#source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_graphs_distances.R')
#find nearest neighbors
#system.time(neighborMatrix <- find_neighbors(data=data[, 1:39], query=data[,1:39], k=105+1, metric='L2'))
dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
#system(paste0('mkdir ', dirname))
#save(neighborMatrix, file=paste0(dirname, '/L2_neighbors105.RData'))
#load( paste0(dirname, '/L2_neighbors105.RData'))



#jaccard=mclapply(seq(5,105, by=5), function(i){
#  system.time(links <- cytofkit:::jaccard_coeff(neighborMatrix$nn.idx[,1:(i+1)]))
#  dt=data.table(links)
#  return(dt)
#}, mc.cores=1)
#dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
#system(paste0('mkdir ', dirname))
#save(jaccard, file=paste0(dirname, '/L2_jaccard105.RData'))
#dirname

#TO DELETE: temp measure to save memory extension
#system.time(jaccard45<-(links45 <- data.table(cytofkit:::jaccard_coeff(neighborMatrix$nn.idx[,2:(46)]))))
#remove self-loops
#jaccard45 <- jaccard45[ V1 != V2 ] 
clus<-read.table("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/auto/phenoGraph/Samusik_gated45/k=15Python_assigned_labels_phenoGraph_Samusik_all.txt", header = F, stringsAsFactors = FALSE)
jaccard45<-read.table(paste0("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/KdependencySamusik_all",'/j45Samusik_all.txt'), header = F, stringsAsFactors = FALSE) 

jaccard45<-as.data.table(jaccard45)
jaccard45[, 1:2] = jaccard45[, 1:2]+1
save(jaccard45, file=paste0(dirname, '/L2_jaccard45.RData'))  

#Vtemp<-jaccard45[, paste0(min(V1, V2), ' ', max(V1, V2)), by = 1:nrow(jaccard45)][,2]
#jaccard45[, Vtemp:=Vtemp]
#setkey(jaccard45, NULL)
#setkey(jaccard45, Vtemp)
#tables()
#dup<-duplicated(jaccard45[, Vtemp])
jaccard45[, nW:=sum(V3), by=Vtemp]
jaccard45[, nW:=nW/2]
jaccard45dr<-jaccard45[!dup,]
jaccard45dr<-jaccard45dr[order(V1,V2)]
#jaccard45dr<-jaccard45dr[V3!=0, ]
jaccard45dr0<-jaccard45dr[nW!=0, ]
UnAs<-data.table(V1=1:length(unassigned), Un = unassigned)
setkey(UnAs, V1)
setkey(jaccard45, NULL)
setkey(jaccard45dr, V1)

jaccard45dr<-merge(jaccard45dr, UnAs, all.x=TRUE)
jaccard45dr<-jaccard45dr[Un!=T,]
#save(jaccard45dr, file=paste0(dirname, '/L2_jaccard45_dubRemoved.RData'))  
#load(paste0(dirname, '/L2_jaccard45_dubRemoved.RData')) 
#create weighted igraph of from links in 'jaccard' list
#rm(neighborMatrix)
#k=45
#gr<-graph_from_data_frame(jaccard45dr[, ], directed = F, vertices = NULL)
#gr0<-graph_from_data_frame(jaccard45dr0[, ], directed = F, vertices = NULL)
#save(gr, file=paste0(dirname, '/L2_gr45.RData'))
#load(paste0(dirname, '/L2_gr45.RData'))
#E(gr)$weight=as.numeric(as.data.frame(jaccard45dr)[,'nW'])
#E(gr0)$weight=as.numeric(as.data.frame(jaccard45dr0)[,'nW'])
#hist(E(gr)$weight, 500)
#save(gr, file=paste0(dirname, '/L2_gr45.RData'))
load(paste0(dirname, '/L2_gr45.RData'))

#an alternative way to do it
#cat("--- Creating graph... ")
#start <- proc.time()

#vertex.attrs <- list(name = unique(c(df$src, df$dst)))
#edges <- rbind(match(df$src, vertex.attrs$name),
#               match(df$dst,vertex.attrs$name))

#G <- graph.empty(n = 0, directed = T)
#G <- add.vertices(G, length(vertex.attrs$name), attr = vertex.attrs)
#G <- add.edges(G, edges)

#remove(edges)
#remove(vertex.attrs)

#cat(sprintf("--- elapsed user-time: %fs ", (proc.time() - start)[1]))

#weights=E(gr)$weight
#save(weights, file=paste0(dirname, '/weights_gr45.RData'))

#load(paste0(dirname, '/weights_gr45.RData'))
#IDX<-sample(1:841644, 10000)
#ind<-induced_subgraph(gr, vids=IDX, impl = "create_from_scratch")
#mmz<-modularity_matrix(ind, membership=clus[[3]][IDX], weights = E(ind)$weight)
#modularity(ind, membership=clus[[3]][IDX])

#Q_45 <- modularity_matrix(gr, membership=clus[[3]], weights = weights)
#dirname=paste0(RES_DIR_PHENOGRAPH, CALC_NAME)
#save(Q_45, file=paste0(dirname, '/L2_gr45.RData'))
#load(paste0(dirname, '/L2_gr45.RData'))

#load population names

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/graphQcomputations.R')
#TODO: calculate modularities based on igraph 
Qfun(gr0, clus[[3]][subs])
# 0.8959616
Qfun(gr, clus_truth[[1]])
# 0.8770701
#Now calculate change in modularity if split true populations merged
#by Phenograph
#######################################################################

#1. Modularity of true clustering
Qfun(gr,clus_truth[[1]])
# 0.8770701
#Modularity of true clustering is lower! Probably an artifact of gating strategy.
#To check in Python

#Now compare splitting merged clusters:
load("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/benchmark_data_sets/tsne.RData")

pop_names=list()
pop_names[[1]]=read.table("../../benchmark_data_sets/attachments/population_names_Levine_32dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[2]]=read.table("../../benchmark_data_sets/attachments/population_names_Levine_13dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[3]]=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[3]]$label=rownames(pop_names[[3]])
pop_names[[3]]$population=pop_names[[3]]$population.name
pop_names[[4]]=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[4]]$label=rownames(pop_names[[4]])
pop_names[[4]]$population=pop_names[[4]]$population.name
 
cl_tr<-clus_truth<-clus_truth[[1]]
clus<-clus[[3]]
table(clus[clus_truth==7])
#   9   14 
# 2034    7 
table(clus[clus_truth==10])
# 3    9   11 
# 11 6055    2 
table(clus_truth[clus==9])
#  1    2    5    6    7    8   10   12   13   14   15   16   18   19   20   21   22   23 
# 31   18  108    2 2034  265 6055    1    4    8   11   50    7   24   33    4    2    1 
#                      !          !
# So the populations 7(CMP) and 10(GMP) are essentially united in one cluster 11.
#http://www.sciencedirect.com/science/article/pii/S0301472X06000683
#Common myeloid progenitor 7(CMP), the latter further committing to granulocyte/monocyte progenitors (GMPs) 
#CMP markers:  CD34, IL-3Rα, CD45RA  CD19−/CD34+/IL-3Rαlo/−/CD45RA−/TpoR−
#Based on Xshift paper: CD19- CD3- CD49- Sca1- CD150+ CD16_32-
#10 GMP markers: CD19−/CD34+/IL-3Rαlo/CD45RA+/TpoR− 
#Based on Xshift paper: CD19- CD3- CD49- Sca1- CD150+ CD16_32+ Ly6C+ CD34+ CD27+

#Now, let us reassign 7 and 10 to their separated clusters:
clus_sep<-clus
clus_sep[clus_truth==7]<-107
clus_sep[clus_truth==7]<-110
Qfun(gr, clus_sep)
#[1] 0.8955079
#This is less than 0.8959616 calculated for merged clusters.
#So indeed, Q does not catch the best clustering of CMP and and GMP

tsne3D<- res_tsne[[4]]$tsne_out3D$Y
tsne2D<-res_tsne[[4]]$tsne_out$Y
cl_tr<-lapply(clus_truth, function(x) x[which(!is.na(x))])
#clus<-lapply(1:length(cl_tr), function(x) clus[[i]][which(!is.na(clus_truth[[x]]))])

#i<-4
ncolors=length(unique(cl_tr))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr)
colors<-unlist(lapply(cl_tr, function(x) col_true[as.character(x)]))
par(oma = c(1, 1, 1, 1))
plot(tsne2D, col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names[[4]]$population, col = col_true[as.character(pop_names[[4]]$label)],  pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(tsne3D,  col=colors, pch='.') 
legend3d("topright", legend = pop_names[[4]]$population, pch = 16, col = col_true[as.character(pop_names[[4]]$label)], cex=1, inset=c(0.02))

#color only a subset of cells  
cells=c(7,10)
colors1<-unlist(lapply(cl_tr, function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(tsne2D,col=colors1, pch='.', cex=1, main = unlist(pop_names[[4]][pop_names[[4]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(tsne3D,  col=colors1, pch='.', main = unlist(pop_names[[4]][pop_names[[4]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

#match the clusters and plot F_1 consensus results 
for (i in (1:7)){
  ncolors=length(unique(cl_tr[[1]]))
  col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
  names(col_true) = unique(cl_tr[[1]])
  
  match_table <-res[[i]]$labels_matched
  lbl_mapped = unlist(lapply(clus[[i]][which(!is.na(clus_truth[[1]]))], function(x) {
    ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
  colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[as.character(x)])))
  
  #par(mar=c(5, 4, 4, 2) + 0.1)
  #par(xpd=T, mar=par()$mar+c(0,0,0,12))
  plot(tsne2D[[1]],col=colors_matched, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2', main=i)
  #legend(x = "right", bty = 'n', inset=c(-0.42,0), c(pop_names[[4]]$population, 'Unassigned'), col = c(col_true[as.character(pop_names[[4]]$label)], 'black'), pch = 16, xpd=T, cex=1.1, pt.cex=1.5)
  #par(mar=c(5, 4, 4, 2) + 0.1)
  
  plot3d(tsne3D[[1]],  col=colors_matched, pch='.', , xlab='tsne 1', ylab='tsne 2', zlab='tsne 3')
  legend3d("topright", legend = c(pop_names[[4]]$population, 'Unassigned'), pch = 16, col = c(col_true[as.character(pop_names[[4]]$label)], 'black'), cex=1, inset=c(0.02))
}
#colors by automatic cluster assignment 
ncolors_auto=length(unique(clus[[i]]))
col_auto=sample(rainbow(ncolors_auto), ncolors_auto , replace=F)
names(col_auto) = unique(clus[[i]])
colors_auto<-unlist(lapply(clus[[i]], function(x) col_auto[as.character(x)]))
plot(tsne2D[[i]],col=colors_auto, pch='.', cex=1)

plot3d(tsne3D[[i]],  col=colors_auto, pch='.') 

#3d plot if just tese two clusters:

########################################################
#find markers differentiated in population(s) ##########
########################################################
#load marker names
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
#load data  set
data=read.table(paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/data_Samusik_all.txt"), header = F, stringsAsFactors = FALSE)
colnames(data)<-c(marker_names[[4]], 'label')

ncolors=length(unique(cl_tr))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr)
colors<-unlist(lapply(cl_tr, function(x) col_true[as.character(x)]))
par(oma = c(1, 1, 1, 1))
plot(tsne2D, col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names[[4]]$population,
       col = col_true[as.character(pop_names[[4]]$label)],  pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

#Common myeloid progenitor (CMP), the latter further committing to granulocyte/monocyte progenitors (GMPs) 
#CMP (7) markers:  CD34, IL-3Rα, CD45RA  CD19−/CD34+/IL-3Rαlo/−/CD45RA−/TpoR−
#Based on Xshift paper, gating : CD19- CD3- CD49- Sca1- CD150+ CD16_32-
#granulocyte-macrophage progenitor (GMP)
#GMP (10) markers: CD19−/CD34+/IL-3Rαlo/CD45RA+/TpoR− 
#Based on Xshift paper: CD19- CD3- CD49- Sca1- CD150+ CD16_32+ Ly6C+ CD34+ CD27+


cells=c(7,10)

IDXsset<-clus_truth %in% cells
sset<-data[IDXsset, 1:39]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
pc_set<-princomp(sset[,1:39])
pcLoad<-as.data.frame(pc_set$loadings[,1])
colnames(sset)[order(abs(pcLoad),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(pcLoad),decreasing=T)][c(1,2,4)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])
plot3d(sset[,c("Ly6C" ,   "CD27"  ,  "CD16_32")], pch='.', col=colors[IDXsset])
#hist(sset[, 'CD20'], 50)
#hist(data[, 'CD20'], 500)
#plot(sset[, c('CD20', "CD123")], pch='.')
plot3d(sset[,c('CD34', 'CD27', 'CD16_32')], pch='.', col=colors[IDXsset])
plot(sset[,c( 'CD27', 'CD16_32')], pch=19, cex=0.5, col=colors[IDXsset])

#Emphasize the difference between clusters
#1. Asymmetric discriminant coordinates
library(fpc)

cells=c(7, 10)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:39]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
adc<-adcoord(sset, clus_vec, clnum=10)
plot3d(adc$proj[,1:3], col=colors[IDXsset])
units<-adc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])

#2. Discriminant coordinates
library(fpc)
cells=c(7,10)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:39]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
dc<-discrcoord(sset,clus_vec, pool = "equal")
plot3d(dc$proj[,1:3], col=colors[IDXsset])
units<-dc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])

#3. Asymmetric weighted discriminant coordinates
cells=c(7,10)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:39]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
awdc<-awcoord(sset,clus_vec,  clnum=c(10), method="classical")
plot3d(awdc$proj[,1:3], col=colors[IDXsset])
units<-awdc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])



#4. Neighborhood based discriminant coordinates
library(fpc)
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(12,13,14)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:32]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
ndc<-ncoord(sset, clus_vec, weighted=F)
plot3d(ndc$proj[,1:3], col=colors[IDXsset])
units<-ndc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])

#5. Asymmetric neighborhood based discriminant coordinates
library(fpc)

cells=c(7,10)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:39]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
andc<-ancoord(sset, clus_vec,  clnum=c(10))
plot3d(andc$proj[,1:3], col=colors[IDXsset])
units<-andc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])

#Now what is the difference of our B-cells from the rest.
#Selected cell population versus the rest
#4. Neighborhood based discriminant coordinates
library(fpc)
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(7,10)
#IDXsset<-clus_truth %in% cells
clus_sub<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, T, ifelse(runif(1)<0.02, T, F) )))#need subset the 'bulk'
sset<-data[clus_sub, 1:39]
clus_vec<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, x , -1) ))[clus_sub]
ndc<-ncoord(sset, clus_vec, weighted=F)
plot3d(ndc$proj[,1:3], col=colors[clus_sub])
units<-ndc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[clus_sub])


#Selected cell population versus the rest
#4. Neighborhood based discriminant coordinates
library(fpc)

cells=c(7,10)
#IDXsset<-clus_truth %in% cells
clus_sub<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, T, ifelse(runif(1)<0.02, T, F) )))#need subset the 'bulk'
sset<-data[clus_sub, 1:39]
clus_vec<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, x , -1) ))[clus_sub]
awdc<-awcoord(sset, clus_vec, clnum=-1)
plot3d(awdc$proj[,1:3], col=colors[clus_sub])
units<-awdc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[clus_sub])
plot3d(sset[, c("CD34", "CD27", "CD16_32")], pch='.', col=colors[clus_sub])


#Selected cell population versus the rest
#4. Neighborhood based discriminant coordinates
library(fpc)

cells=c(7,10)
#IDXsset<-clus_truth %in% cells
clus_sub<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, T, ifelse(runif(1)<0.05, T, F) )))#need subset the 'bulk'
sset<-data[clus_sub, 1:39]
sclus_vec<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, 1 , -1) ))[clus_sub]
andc<-ancoord(sset, sclus_vec, clnum=1)
plot3d(andc$proj[,1:3], col=colors[clus_sub])
units<-andc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[clus_sub])
plot3d(sset[,c("CD11b", "CD34", "CD16_32")], pch='.', col=colors[clus_sub])

#Check the resolution limit
#Common myeloid progenitor 7(CMP), the latter further committing to granulocyte/monocyte progenitors (GMPs) 
#CMP markers:  CD34, IL-3Rα, CD45RA  CD19−/CD34+/IL-3Rαlo/−/CD45RA−/TpoR−
#Based on Xshift paper: CD19- CD3- CD49- Sca1- CD150+ CD16_32-
#10 GMP markers: CD19−/CD34+/IL-3Rαlo/CD45RA+/TpoR− 
#Based on Xshift paper: CD19- CD3- CD49- Sca1- CD150+ CD16_32+ Ly6C+ CD34+ CD27+
load(paste0(dirname, '/L2_jaccard45.RData'))
clt1tbl<- data.table(V1=1:length(clus_truth), label1=clus_truth)
clt2tbl<- data.table(V2=1:length(clus_truth), label2=clus_truth)
setkey(clt1tbl,V1)
setkey(clt2tbl,V2)
setkey(jaccard45,V1)
jaccard45 <- merge(jaccard45, clt1tbl, all.x=TRUE)               
setkey(jaccard45,V2)
jaccard45 <- merge(jaccard45, clt2tbl, all.x=TRUE)

#save(jaccard45 , file=paste0(dirname, '/jaccard45_lbl.RData'))
load(file=paste0(dirname, '/jaccard45_lbl.RData'))
#23mln links
#number 0f intercluster links:
nrow(jaccard45[label1!=label2,])
#[1] 486214, just 2.5% links

epsilon <- sum(jaccard45[(label1 == 7 & label2 == 10) | (label1 == 10 & label2 == 7), V3])#weight of links between communities 
rhoCMP <- sum(jaccard45[(label1==7 & label2!=10) | ((label1!=10 & label2==7)), V3])#weigts connencting to the rest of network
rhoGMP <- sum(jaccard45[(label1==10 & label2!=7) | (label1!=7 & label2==10), V3])#weigts connencting to the rest of network  
omegaCMP <- sum(jaccard45[(label1 == 7) & (label2 == 7), V3]) #internal weights CMP
omegaGMP <-  sum(jaccard45[(label1 == 10) & (label2 == 10), V3]) #internal weights GMP
W <- sum(jaccard45[, V3])#sum of edges weights of the full network
#884746.6
sqrt(W*epsilon)  
omegaCMP
omegaGMP
rhoCMP
rhoGMP
#Delta_W:

Delta_W <- 2*epsilon- 1/W*(epsilon*(omegaCMP+omegaGMP+rhoCMP+rhoGMP)+epsilon^2)-1/W*(omegaCMP*omegaGMP+rhoCMP*rhoGMP+omegaCMP*rhoGMP+rhoCMP*omegaGMP)
Delta_W
#[1]  580.4205
#Modularity is higher for CMP and GMP merged if and only if Delta_W > 0
Delta_W/(2*W)  

W_crit = (2*epsilon*(epsilon*(omegaCMP+omegaGMP+rhoCMP+rhoGMP)+epsilon^2+(omegaCMP*omegaGMP+rhoCMP*rhoGMP+omegaCMP*rhoGMP+rhoCMP*omegaGMP))^(-1))^(-1)
#395864.2
0.0003280151
# Subsample so that W is lees that W_crit
subJaccard45 <- jaccard45[(label1 != 7) & (label2 != 10) & (label1 != 10) & (label2 != 7), ]
IDX<-unique(c(subJaccard45[, V1], subJaccard45[, V2]))
IDXsub<-sample(IDX, as.integer(length(IDX)/3))
setkey(subJaccard45, V1, V2)
subJaccard45 <- subJaccard45[(V1 %in% IDXsub) & (V2 %in% IDXsub),]
Wsubcrit <- sum(subJaccard45[, V3])

sample(1:nrow(jaccard45))

