#########################################################################################
# R script to load and evaluate results for PhenoGraph
#
# Lukas Weber, August 2016
# Stepan Grinek December 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)
library(mclust)

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")
source("../helpers/helper_evaluate_NMI.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')



# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/phenoGraph"
CALC_NAME="KdependencySamusik_SubSamplegatedL1"
DATA_DIR <- "../../benchmark_data_sets"

#load  data
data=read.table(paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/data_Samusik_all.txt"), header = F, stringsAsFactors = FALSE)


####################################################
### load truth (manual gating population labels) ###
####################################################


clus_truth <- data[, 40]

sapply(clus_truth, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl_truth <- table(clus_truth)


# store named objects (for other scripts)



###############################
### load PhenoGraph results ###
###############################

# load cluster labels
files_res <- list(
  Samusik_all15 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=15Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all30 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=30Python_assigned_labels_phenoGraph_Samusik_all.txt"), 
  Samusik_all45   = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=45Python_assigned_labels_phenoGraph_Samusik_all.txt") 
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
#  print(table(clus[[i]], clus_truth))
#}

# store named objects (for other scripts)





###################################
### match clusters and evaluate ###
###################################

# see helper function scripts for details on matching strategy and evaluation

res <- vector("list", length(clus))
names(res) <- names(clus)

for (i in 1:length(clus)) {
  
  res[[i]] <- helper_match_evaluate_multiple(clus[[i]], clus_truth)
  
}
lapply(res, function(x) c(x$mean_F1, x$n_clus))
#number of un-discovered sets based on F1=0
lapply(res, function(x) sum(x$F1==0))
lapply(res, function(x) x$F1)

res_S_H <- vector("list", length(clus))
names(res_S_H) <- names(clusS_H)



resS_H <- mclapply(1:3, function(i) helper_match_evaluate_multiple_SweightedH(clus[[i]], clus_truth), mc.cores = 4)
lapply(resS_H, function(x) c(x$mean_SweightedH, x$n_clus, x$total_SweightedH))


resS_Hu <- vector("list", length(clus))
names(resS_Hu) <- names(clus)
resS_Hu <- mclapply(1:3, function(i) helper_match_evaluate_multiple_SunweightedH(clus[[i]], clus_truth), mc.cores = 4)
lapply(resS_Hu, function(x) c(x$mean_SunweightedH, x$n_clus, x$total_SunweightedH))

resS_N <- vector("list", length(clus))
names(resS_N) <- names(clus)
resS_N <- mclapply(1:3, function(i) helper_match_evaluate_multiple_SweightedN(clus[[i]], clus_truth), mc.cores = 4)
lapply(resS_N, function(x) c(x$mean_SweightedN, x$n_clus, x$total_SweightedN))

resS_Nu <- vector("list", length(clus))
names(resS_Nu) <- names(clus)
resS_Nu <- mclapply(1:3, function(i) helper_match_evaluate_multiple_SunweightedN(clus[[i]], clus_truth), mc.cores = 7)
lapply(resS_Nu, function(x) c(x$mean_SunweightedN, x$n_clus, x$total_SunweightedN))

resMoCu <- vector("list", length(clus))
names(resMoCu) <- names(clus)
resMoCu <- mclapply(1:3, function(i) helper_match_evaluate_multiple_MoCu(clus[[i]], clus_truth), mc.cores = 7)
lapply(resMoCu, function(x) c(x$mean_MoCu, x$n_clus, x$total_MoCu))

resAdjR <- mclapply(1:3, function(i) adjustedRandIndex(clus[[i]], clus_truth), mc.cores = 7)
resAdjR 

resNMI <- mclapply(1:3, function(i) helper_evaluate_NMI(clus[[i]], clus_truth), mc.cores = 7)
resNMI
lapply(resNMI, function(x) c( x$n_clus, x$NMI))

resAMI <- lapply(1:3, function(i) AMI(clus[[i]], clus_truth, mc.cores=3))
resAMI


# store named object (for plotting scripts)

res_PhenoGraph <- res

############################
########visualisation#######
############################

library(Rtsne)

load("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/benchmark_data_sets/tsne.RData")

pop_names=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names$label=rownames(pop_names)
pop_names$population=pop_names$population.name
clus_truth<-clus_truth

tsne2D<-res_tsne[[4]]$tsne_out$Y[!unassign,]
tsne3D<-res_tsne[[4]]$tsne_out3D$Y[!unassign,]

i=4
ncolors=length(unique(clus_truth))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(clus_truth)
colors<-unlist(lapply(clus_truth, function(x) col_true[as.character(x)]))
#data<- cbind(f, clusters)
par(oma = c(1, 1, 1, 9))
#par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(tsne2D,col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names$population,col = col_true[as.character(pop_names$label)], , pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(tsne3D,  col=colors, pch='.') 
legend3d("topright", legend = pop_names$population, pch = 16, col = col_true[as.character(pop_names$label)], cex=1, inset=c(0.02))

#color only one cluster  
cells=c(9,21)
colors1<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(tsne2D,col=colors1, pch='.', cex=1, main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(tsne3D,  col=colors1, pch='.', main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

#match the clusters and plot F_1 consensus results 
match_table <-res[[i]]$labels_matched
lbl_mapped = unlist(lapply(clus[[i]], function(x) {
  ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[as.character(x)])))
plot(tsne2D,col=colors_matched, pch='.', cex=1)
legend(x = 'right', bty='n', inset=c(-0.28,0), c(pop_names$population, 'Unassigned'), col = c(col_true[as.character(pop_names$label)], 'black'), pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(tsne3D,  col=colors_matched, pch='.')
legend3d("topright", legend = c(pop_names$population, 'Unassigned'), pch = 16, col = c(col_true[as.character(pop_names$label)], 'black'), cex=1, inset=c(0.02))

#colors by automatic cluster assignment 
ncolors_auto=length(unique(clus[[i]]))
col_auto=sample(rainbow(ncolors_auto), ncolors_auto , replace=F)
names(col_auto) = unique(clus[[i]])
colors_auto<-unlist(lapply(clus[[i]], function(x) col_auto[as.character(x)]))
plot(tsne2D,col=colors_auto, pch='.', cex=1)
plot3d(tsne3D,  col=colors_auto, pch='.') 

#find markers differentiated in population(s)
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[4]], 'label')
cells=c(12)
sset<-data[clus_truth %in% cells, 1:39]
plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
pc_set<-princomp(sset[1:39])
pcLoad<-as.data.frame(pc_set$loadings[,1])
colnames(sset)[order(abs(pcLoad),decreasing=T)][1:3]
top_comp<-colnames(sset)[order(abs(pcLoad),decreasing=T)][1:3]
plot3d(sset2[,top_comp])

cells2<-c(12,13,14)
sset2<-data[clus_truth %in% cells2, 1:39]
plot3d(sset2[,c('IgD', 'IgM', 'MHCII')])
pc_set2<-princomp(sset2[1:39])
pcLoad2<-as.data.frame(pc_set2$loadings[,1])
colnames(sset2)[order(abs(pcLoad2),decreasing=T)][1:3]
top_comp2<-colnames(sset2)[order(abs(pcLoad2),decreasing=T)][1:3]
plot3d(sset2[,top_comp2])

pc_setALL<-princomp(data[,1:39])
pcLoadALL<-as.data.frame(pc_setALL$loadings[,1])
colnames(data[,1:39])[order(abs(pcLoadALL),decreasing=T)][1:3]
top_compAL<-colnames(data[,1:39])[order(abs(pcLoadALL),decreasing=T)][1:3]
plot3d(data[,1:39][,top_comp2])

hist(as.matrix(dist(sset[sample(1:nrow(sset), 1000),])))
hist(as.matrix(dist(sset2[sample(1:nrow(sset2), 1000),])))
dd_all<-as.matrix(dist(data[sample(1:nrow(data), 20000),1:39]))
hist(dd_all,300)

#calculate avarage intra-cluster distance
clus_size<-list()
labels=data[,'label']
cellsALL=unique(labels)
for(j in cellsALL){
  clus_j<-data[data[, 'label']==j, 1:39]
  if (nrow(clus_j)>=1000) {
    clus_size[[length(clus_size)+1]]<-list('n'=nrow(clus_j),
                                           'meanD'=mean(as.matrix(dist(clus_j[sample(1:nrow(clus_j), 1000),]))))}
  else {clus_size[[length(clus_size)+1]]<-list('n'=nrow(clus_j),
                                               'meanD'=mean(as.matrix(dist(clus_j))))}
  
}

lapply(clus_size, function(x)  c(x$meanD, x$n))
plot(x=unlist(lapply(clus_size, function(x)  x$n)), y=unlist(lapply(clus_size, function(x) x$meanD)))

dirname=paste0("../../plots/PhenoGraph/", CALC_NAME)
dir.create(file.path(dirname), showWarnings = T)
for(i in 1:4){
  ncolors=length(unique(clus_truth[[i]]))
  col_true=rainbow(ncolors)
  colors<-unlist(lapply(clus_truth[[i]], function(x) col_true[x]))
  #data<- cbind(f, clusters)
  
  png(filename=paste0(dirname, '/', names(data)[i], 'TsneTRUTH.png' ))
  plot(res_tsne2[[i]]$tsne_out$Y,col=colors[res_tsne2[[i]]$idx], pch='.', cex=1, 
       main=paste0(names(data)[i], ', true labels'), xlab='tsne 1', ylab='tsne 2')
  dev.off()
  #plot3d(res_tsne2[[i]]$tsne_out3D$Y,  col=colors[res_tsne2[[i]]$idx], pch='.') 
  
  #match the clusters and plot F_1 consensus results 
  match_table <-res[[i]]$labels_matched
  lbl_mapped = unlist(lapply(clus[[i]], function(x) {
    ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
  colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[x])))
  
  png(filename=paste0(dirname, '/', names(data)[i], 'TsneMATCH.png' ))
  plot(res_tsne2[[i]]$tsne_out$Y,col=colors_matched[res_tsne2[[i]]$idx], pch='.', cex=1, 
       main=paste0(names(data)[i], ', matched labels'), xlab='tsne 1', ylab='tsne 2')
  dev.off()
  plot3d(res_tsne2[[i]]$tsne_out3D$Y,  col=colors_matched[res_tsne2[[i]]$idx], pch='.',  main=paste0(names(data)[i], ', matched labels'), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3')
  
  #colors by automatic cluster assignment 
  ncolors_auto=length(unique(clus[[i]]))
  col_auto=sample(rainbow(ncolors_auto), ncolors_auto, replace =F )
  colors_auto<-unlist(lapply(clus[[i]], function(x) col_auto[x]))
  plot3d(res_tsne2[[i]]$tsne_out3D$Y,col=colors_auto[res_tsne2[[i]]$idx], pch='.', cex=1)
  
  png(filename=paste0(dirname, '/', names(data)[i], 'TsneAUTO.png' ))
  plot(res_tsne2[[i]]$tsne_out$Y,  col=colors_auto[res_tsne2[[i]]$idx], pch='.', 
       main=paste0(names(data)[i], ', auto labels'), xlab='tsne 1', ylab='tsne 2')
  dev.off()
}

#summary tables for Latex report
library(xtable)
options(xtable.floating = TRUE)
options(xtable.timestamp = "")
floating.environment = getOption( "table")

rnames<-c('15', '30', '45', '60', '75', '90', '105')
df<-matrix(unlist(lapply(1:length(clus), function(x) {z=res[[x]]; c(as.integer(rnames[x]), as.numeric(z$mean_F1), as.numeric(NMIscore[x]), as.numeric(z$mean_re), as.numeric(z$mean_pr),  as.integer(z$n_clus), as.integer(length(unique(clus_truth))-1))})), ncol=7, byrow=T )
colnames(df)<-c('k', 'mean F1', '\\parbox[t]{1cm}{NMI \\\\ score}', '\\parbox[t]{1cm}{mean \\\\ recall}', '\\parbox[t]{1.2cm}{mean\\\\ precision}',  '\\parbox[t]{2cm}{detected \\\\ N of clusters}', '\\parbox[t]{2cm}{true \\\\ N of clusters}')
df[,6:7]<-as.integer(df[,6:7])

xres<-xtable(df,  caption = "PhenoGraph results, ungated Samusik data set", digits=c(2,3,4,4,4,4,2,2), display=c('d','d','fg','fg','fg','fg','d','d'), include.rownames=FALSE, align = c("c ", "c ", "c ", "c ", "c ", "c ", "c ", "c "), label = "tab:Samusik7kUngated")

print(xres,  include.rownames = FALSE, sanitize.text = identity,  table.placement = "h!",  caption.placement = "top")




#marker cloured plots

rbPal <- colorRampPalette(c('red', 'green','blue'))
for(k in 1:(ncol(data[[i]]))){
  rbPal <- colorRampPalette(c('red', 'green','blue'))
  columnCol <- rbPal(10)[as.numeric(cut(data[[i]][IDX, k]+1 ,breaks = 10))]
  plot3d(tsne_out3D$Y[, ],  col=columnCol , pch='.') 
  cat ("Press [enter] to continue")
  line <- readline()
}

rbPal <- colorRampPalette(c('red', 'green','blue'))
for(k in 1:(ncol(data[[i]]))){
  rbPal <- colorRampPalette(c('red', 'green','blue'))
  columnCol <- rbPal(10)[as.numeric(cut(data[[i]][IDX, k]+1 ,breaks = 10))]
  plot(tsne_out$Y[, ],  col=columnCol , pch='.') 
  cat ("Press [enter] to continue")
  line <- readline()
}

#check the connectivity of the graph
library(RANN)
find_neighbors <- function(data, k=105){
  nearest <- nn2(data, data, k, treetype = "bd", searchtype = "standard")
  return(nearest)
}
system.time(neighborMatrix <- find_neighbors(data, k=105+1))
#save(neighborMatrix, file=paste0(dirname, '/neighborMatrix105.RData'))
################################################################# 
#Interactively pick-up subregion

res_tsne=res_tsne2
i=1
ncolors=length(unique(clus_truth[[i]]))
col_true=rainbow(ncolors)
colors<-unlist(lapply(clus_truth[[i]], function(x) col_true[x]))
#data<- cbind(f, clusters)
plot(res_tsne[[i]]$tsne_out$Y,col=colors[res_tsne[[i]]$idx], pch='.', cex=1)

plot3d(res_tsne[[i]]$tsne_out3D$Y,  col=colors[res_tsne[[i]]$idx], pch='.') 

#match the clusters and plot F_1 consensus results 
match_table <-res[[i]]$labels_matched
lbl_mapped = unlist(lapply(clus[[i]], function(x) {
  ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[x])))


plot(res_tsne[[i]]$tsne_out$Y,col=colors_matched[res_tsne[[i]]$idx], pch='.', cex=1, main='CALC_NAME')

library("gatepoints")
df<-as.data.frame(res_tsne[[i]]$tsne_out$Y)
selectedPoints <- fhs(res_tsne[[i]]$tsne_out$Y)

plot(df[selectedPoints, ], col=colors_matched[res_tsne[[i]]$idx][as.integer(selectedPoints)], pch='o', cex=1, main='CALC_NAME')

df3D<-as.data.frame(res_tsne[[i]]$tsne_out3D$Y)
plot3d(df3D[selectedPoints,],  col=colors_matched[res_tsne[[i]]$idx][as.integer(selectedPoints)], pch='.')

##Violaplot of the data used in the Phenograph run
dataset=data
colnames(dataset)[40]='label'
library(vioplot)
for (i in (c(1:24))){
  vioplot(dataset[dataset$label==i , 1], dataset[dataset$label==i , 2], dataset[dataset$label==i , 3], 
          dataset[dataset$label==i , 4],dataset[dataset$label==i , 5], dataset[dataset$label==i , 6],
          dataset[dataset$label==i , 7], dataset[dataset$label==i , 8],dataset[dataset$label==i , 9],
          dataset[dataset$label==i , 10],
          dataset[dataset$label==i , 11], dataset[dataset$label==i , 12], dataset[dataset$label==i , 13], 
          dataset[dataset$label==i , 14], dataset[dataset$label==i , 15], dataset[dataset$label==i , 16],
          dataset[dataset$label==i , 17], dataset[dataset$label==i , 18], dataset[dataset$label==i , 19],
          dataset[dataset$label==i , 20],
          dataset[dataset$label==i , 21], dataset[dataset$label==i , 22], dataset[dataset$label==i , 23], 
          dataset[dataset$label==i , 24], dataset[dataset$label==i , 25], dataset[dataset$label==i , 26],
          dataset[dataset$label==i , 27], dataset[dataset$label==i , 28], dataset[dataset$label==i , 29],
          dataset[dataset$label==i , 30],
          dataset[dataset$label==i , 31], dataset[dataset$label==i , 32], dataset[dataset$label==i , 33], 
          dataset[dataset$label==i , 34], dataset[dataset$label==i , 35], dataset[dataset$label==i , 36],
          dataset[dataset$label==i , 37], dataset[dataset$label==i , 38], dataset[dataset$label==i , 39])
  title(i)
}

dataset[, !(colnames(dataset) %in% 'label')]<-tanh(scale(dataset[, !(colnames(dataset) %in% 'label')], center=T))







