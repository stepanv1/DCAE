#########################################################################################
# R script to load and evaluate results for PhenoGraph
#
# Lukas Weber, August 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)
library(gplots)
# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')
 

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/PhenoGraph"
CALC_NAME="KdependencySamusik_NAN"




#load subsampled data
load(paste0("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/Rscripts/run_methods/",CALC_NAME,"RphenoGraphSubset.RData"))

# which data sets required subsampling for this method (see parameters spreadsheet)
is_subsampled <- c(TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE)

is_rare <- c(FALSE, FALSE, FALSE, FALSE, TRUE,  TRUE,  FALSE, FALSE)

# note: no FlowCAP data sets for this method




####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
  Levine_32dim = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Levine_32dim.txt"), 
  Levine_13dim = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Levine_13dim.txt"), 
  Samusik_01   = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Samusik_01.txt"), 
  Samusik_all  = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Samusik_all.txt"), 
  Nilsson_rare = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Nilsson_rare.txt"), 
  Mosmann_rare = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Mosmann_rare.txt") 
  #FlowCAP_ND   = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_FlowCAP_ND.txt"), 
  #FlowCAP_WNV  = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_FlowCAP_WNV.txt")
)



# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

#for (i in 1) {
for (i in 1:length(clus_truth)) {
  
  data_truth_i <- read.table(files_truth[[i]], header = TRUE, stringsAsFactors = FALSE)[, "label"]
  clus_truth[[i]] <- data_truth_i
}

sapply(clus_truth, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl_truth <- lapply(clus_truth, table)

tbl_truth
sapply(tbl_truth, length)

# store named objects (for other scripts)

files_truth_PhenoGraph <- files_truth
clus_truth_PhenoGraph <- clus_truth

###############################
### load PhenoGraph results ###
###############################

# load cluster labels
files_res <- list(
  Levine_32dim = file.path(paste0(RES_DIR_PHENOGRAPH, "/", CALC_NAME), "Levine_32dim.txt"), 
  Levine_13dim = file.path(paste0(RES_DIR_PHENOGRAPH, "/", CALC_NAME), "Levine_13dim.txt"), 
  Samusik_01   = file.path(paste0(RES_DIR_PHENOGRAPH, "/", CALC_NAME), "Samusik_01.txt"), 
  Samusik_all  = file.path(paste0(RES_DIR_PHENOGRAPH, "/", CALC_NAME), "Samusik_all.txt"), 
  Nilsson_rare = file.path(paste0(RES_DIR_PHENOGRAPH, "/", CALC_NAME), "Nilsson_rare.txt"), 
  Mosmann_rare = file.path(paste0(RES_DIR_PHENOGRAPH, "/", CALC_NAME), "Mosmann_rare.txt") 
  #FlowCAP_ND   = file.path(RES_DIR_PHENOGRAPH, "phenoGraph_labels_FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(RES_DIR_PHENOGRAPH, "phenoGraph_labels_FlowCAP_WNV.fcs")
)


clus <- vector("list", length(files_res))
names(clus) <- names(files_res)

for (i in 1:length(clus)){
  clus[[i]] <- read.table(files_res[[i]], header = TRUE, stringsAsFactors = FALSE)[,1]
}

sapply(clus, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl <- lapply(clus, table)

tbl
sapply(tbl, length)

# contingency tables

for (i in 1:length(clus)) {
  print(table(clus[[i]], clus_truth[[i]]))
}

# store named objects (for other scripts)

files_PhesnoGraph <- files_out
clus_PhenoGraph <- clus






###################################
### match clusters and evaluate ###
###################################

# see helper function scripts for details on matching strategy and evaluation

res <- vector("list", length(clus))
names(res) <- names(clus)

for (i in 1:7) {
     res[[i]] <- helper_match_evaluate_multiple(clus[[i]], clus_truth[[1]])
}
lapply(res, function(x) c(x$mean_F1, x$n_clus))

#calculate weighted S_H
resS_H <- vector("list", length(clus))
names(resS_H) <- names(clus)

resS_H <- mclapply(1:7, function(i) helper_match_evaluate_multiple_SweightedH(clus[[i]], clus_truth[[1]]), mc.cores = 4)
lapply(resS_H, function(x) c(x$mean_SweightedH, x$n_clus, x$total_SweightedH))


resS_Hu <- vector("list", length(clus))
names(resS_Hu) <- names(clus)

resS_Hu <- mclapply(1:7, function(i) helper_match_evaluate_multiple_SunweightedH(clus[[i]], clus_truth[[1]]), mc.cores = 4)
lapply(resS_Hu, function(x) c(x$mean_SunweightedH, x$n_clus, x$total_SunweightedH))

resS_N <- vector("list", length(clus))
names(resS_N) <- names(clus)

resS_N <- mclapply(1:7, function(i) helper_match_evaluate_multiple_SweightedN(clus[[i]], clus_truth[[1]]), mc.cores = 4)
lapply(resS_N, function(x) c(x$mean_SweightedN, x$n_clus, x$total_SweightedN))

resS_Nu <- vector("list", length(clus))
names(resS_Nu) <- names(clus)

resS_Nu <- mclapply(1:7, function(i) helper_match_evaluate_multiple_SunweightedN(clus[[i]], clus_truth[[1]]), mc.cores = 7)
lapply(resS_Nu, function(x) c(x$mean_SunweightedN, x$n_clus, x$total_SunweightedN))

resMoCu <- vector("list", length(clus))
names(resMoCu) <- names(clus)

resMoCu <- mclapply(1:7, function(i) helper_match_evaluate_multiple_MoCu(clus[[i]], clus_truth[[1]]), mc.cores = 7)
lapply(resMoCu, function(x) c(x$mean_MoCu, x$n_clus, x$total_MoCu))

resAdjR <- mclapply(1:7, function(i) adjustedRandIndex(clus[[i]], clus_truth[[1]]), mc.cores = 7)

resNMI <- mclapply(1:7, function(i) helper_evaluate_NMI(clus[[i]], clus_truth[[1]]), mc.cores = 7)





#for now only Levin32
res[[1]] <- helper_match_evaluate_multiple(clus[[1]], clus_truth[[1]])
print(res[[1]])

resInd <- vector("list", length(clus))
names(resInd) <- names(clus)

for (i in 1:length(clus)) {
  if (!is_rare[i]) {
    resInd[[i]] <- helper_match_evaluate_multiple_RandInd(clus[[i]], clus_truth[[i]])
  }
}

helper_match_evaluate_multiple_Fweight








# store named object (for plotting scripts)

res_PhenoGraph <- res

############################
########visualisation#######
############################
res <- resS_Nu
load("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/benchmark_data_sets/tsne.RData")

pop_names=list()
pop_names[[1]]=read.table("../../benchmark_data_sets/attachments/population_names_Levine_32dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')

pop_names[[2]]=read.table("../../benchmark_data_sets/attachments/population_names_Levine_13dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')

pop_names[[3]]=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[3]]$label=rownames(pop_names[[3]])
pop_names[[3]]$population=pop_names[[3]]$population.name

pop_names[[4]]=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[4]]$label=rownames(pop_names[[4]])
pop_names[[4]]$population=pop_names[[4]]$population.name

cl_tr<-lapply(clus_truth, function(x) x[which(!is.na(x))])

tsne3D<-lapply(4, function(x) res_tsne[[x]]$tsne_out3D$Y[which(!is.na(clus_truth[[1]])),])
tsne2D<-lapply(4, function(x) res_tsne[[x]]$tsne_out$Y[which(!is.na(clus_truth[[1]])),])

#clus<-lapply(1:length(cl_tr), function(x) clus[[i]][which(!is.na(clus_truth[[x]]))])

#i<-4
ncolors=length(unique(cl_tr[[1]]))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr[[1]])
colors<-unlist(lapply(cl_tr[[1]], function(x) col_true[as.character(x)]))
par(oma = c(1, 1, 1, 1))
plot(tsne2D[[1]],col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names[[4]]$population, col = col_true[as.character(pop_names[[4]]$label)],  pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(tsne3D[[1]],  col=colors, pch='.') 
legend3d("topright", legend = pop_names[[4]]$population, pch = 16, col = col_true[as.character(pop_names[[4]]$label)], cex=1, inset=c(0.02))

#color only a subset of cells  
cells=c(1,2,3)
colors1<-unlist(lapply(cl_tr[[1]], function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(tsne2D[[1]],col=colors1, pch='.', cex=1, main = unlist(pop_names[[4]][pop_names[[4]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(tsne3D[[1]],  col=colors1, pch='.', main = unlist(pop_names[[4]][pop_names[[4]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

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

#A plot of Levine 13 for the report
i=2
png(filename = '../../plots/GNG/prot01_Levine_13.png', width = 1440, height = 1000)
ncolors=length(unique(cl_tr[[i]]))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr[[i]])

match_table <-res[[i]]$labels_matched
lbl_mapped = unlist(lapply(clus[[i]][which(!is.na(clus_truth[[i]]))], function(x) {
  ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[as.character(x)])))
par(mar=c(5, 4, 4, 2) + 0.1)
par(xpd=T, mar=par()$mar+c(0,0,0,26))
plot(tsne2D[[i]],col=colors_matched, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = "right", bty = 'n', inset=c(-0.4,0), c(pop_names[[i]]$population, 'Unassigned'), col = c(col_true[as.character(pop_names[[i]]$label)], 'black'), pch = 16, xpd=T, cex=2.2, pt.cex=2.2)
par(mar=c(5, 4, 4, 2) + 0.1)
dev.off()

#ground truth plot
png(filename = '../../plots/ground_truth_Levine_13.png', width = 1440, height = 1000)
i<-2
ncolors=length(unique(cl_tr[[i]]))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr[[i]])
colors<-unlist(lapply(cl_tr[[i]], function(x) col_true[as.character(x)]))
par(mar=c(5, 4, 4, 2) + 0.1)
par(xpd=T, mar=par()$mar+c(0,0,0,26))
plot(tsne2D[[i]],col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.4,0), pop_names[[i]]$population, col = col_true[as.character(pop_names[[i]]$label)],  pch = 16, xpd=TRUE, cex=2.2, pt.cex=2.2)
par(mar=c(5, 4, 4, 2) + 0.1)
dev.off()





#summary tables for Latex report
library(xtable)
options(xtable.floating = TRUE)
options(xtable.timestamp = "")
floating.environment = getOption( "table")

rnames<-c('Levine 32', 'Levine 13', 'Samusik 01', 'Samusik all')
df<-data.frame(matrix(unlist(lapply(1:4, function(x) {z=res[[x]]; c(rnames[x], as.numeric(z$mean_F1),  as.numeric(z$mean_re), as.numeric(z$mean_pr),  as.integer(z$n_clus), as.integer(length(unique(clus_truth[[x]]))))})), ncol=6, byrow=T ), stringsAsFactors=FALSE)
colnames(df)<-c('Data set', 'mean F1',  '\\parbox[t]{1cm}{mean \\\\ recall}', '\\parbox[t]{1.2cm}{mean\\\\ precision}',  '\\parbox[t]{2cm}{detected \\\\ N of clusters}', '\\parbox[t]{2cm}{true \\\\ N of clusters}')
df[,5:6]<-lapply((df[,5:6]), function(x) as.integer(x))
df[,2:4]<-lapply((df[,2:4]), function(x) as.numeric(x))

xres<-xtable(df,  caption = "Prototype 0.1 results", digits=c(0,0,4,4,4,2,2), display=c('s', 's','fg','fg','fg','d','d'), include.rownames=FALSE, align = c("c", "c ",  "c ", "c ", "c ", "c ", "c "), label = "tab:Prot1")

print(xres,  include.rownames = FALSE, sanitize.text = identity,  table.placement = "h!",  caption.placement = "top")









library("gatepoints")
df<-as.data.frame(res_tsne[[i]]$tsne_out$Y)
selectedPoints <- fhs(res_tsne[[i]]$tsne_out$Y)

plot(df[selectedPoints, ], col=colors_matched[res_tsne[[i]]$idx][as.integer(selectedPoints)], pch='o', cex=1, main='CALC_NAME')

df3D<-as.data.frame(res_tsne[[i]]$tsne_out3D$Y)
plot3d(df3D[selectedPoints,],  col=colors_matched[res_tsne[[i]]$idx][as.integer(selectedPoints)], pch='.')

#build the table to represent expression of markers per cluster
i=5
sc_data<-scale(data[[i]], center = F, scale=T)
tbl_mark_cl<-matrix(unlist(lapply(1:ncol(sc_data), function(x) lapply(unique(clus_truth[[i]]), function(y) (mean(sc_data[clus_truth[[i]]==y, x]))))), nrow=length(unique(clus_truth[[i]])))

sc_heatmap <- heatmap.2((tbl_mark_cl), Rowv=NA, Colv=NA, col = "heat.colors", scale="none", margins=c(5,10))

sc_heatmap2 <- heatmap.2(tbl_mark_cl, Rowv=NA, Colv=NA, col = "heat.colors", scale="none", margins=c(5,10), trace='row')

