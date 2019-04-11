#########################################################################################
# R script to load and evaluate results for flowMeans
#
# Lukas Weber, August 2016
#########################################################################################


library(flowCore)
library(clue)
library(rgl)

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
RES_DIR_FLOWMEANS <- "../../results/manual/flowMeans"

DATA_DIR <- "../../benchmark_data_sets"

# which data sets required subsampling for this method (see parameters spreadsheet)
is_subsampled <- c(FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE)

# alternate FlowCAP results at the end
is_rare    <- c(FALSE, FALSE, FALSE, FALSE, TRUE,  TRUE,  FALSE, FALSE)
is_FlowCAP <- c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE,  TRUE)
n_FlowCAP <- 2




####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
  Levine_32dim = file.path(DATA_DIR, "Levine_32dim.fcs"), 
  Levine_13dim = file.path(DATA_DIR, "Levine_13dim.fcs"), 
  Samusik_01   = file.path(DATA_DIR, "Samusik_01.fcs"), 
  Samusik_all  = file.path(RES_DIR_FLOWMEANS, "true_labels_flowMeans_Samusik_all.txt"), 
  Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs")#, 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)

# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

for (i in 1:length(clus_truth)) {
  if (!is_subsampled[i]) {
    data_truth_i <- flowCore::exprs(flowCore::read.FCS(files_truth[[i]], transformation = FALSE, truncate_max_range = FALSE))
  } else {
    data_truth_i <- read.table(files_truth[[i]], header = TRUE, stringsAsFactors = FALSE)
  }
  clus_truth[[i]] <- data_truth_i[, "label"]
}

sapply(clus_truth, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl_truth <- lapply(clus_truth, table)

tbl_truth
sapply(tbl_truth, length)

# store named objects (for other scripts)

files_truth_flowMeans <- files_truth
clus_truth_flowMeans <- clus_truth




##############################
### load flowMeans results ###
##############################

# load cluster labels

files_out <- list(
  Levine_32dim = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_Levine_32dim.txt"), 
  Levine_13dim = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_Levine_13dim.txt"), 
  Samusik_01   = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_Samusik_01.txt"), 
  Samusik_all  = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_Samusik_all.txt"), 
  Nilsson_rare = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_Nilsson_rare.txt"), 
  Mosmann_rare = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_Mosmann_rare.txt")#, 
  #FlowCAP_ND   = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_FlowCAP_ND.txt"), 
  #FlowCAP_WNV  = file.path(RES_DIR_FLOWMEANS, "flowMeans_labels_FlowCAP_WNV.txt")
)

clus <- lapply(files_out, function(f) {
  read.table(f, header = TRUE, stringsAsFactors = FALSE)[, "label"]
})

sapply(clus, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl <- lapply(clus, table)

tbl
sapply(tbl, length)

# contingency tables
# (excluding FlowCAP data sets since population IDs are not consistent across samples)

for (i in 1:length(clus)) {
  if (!is_FlowCAP[i]) {
    print(table(clus[[i]], clus_truth[[i]]))
  }
}

# store named objects (for other scripts)

files_flowMeans <- files_out
clus_flowMeans <- clus




###################################
### match clusters and evaluate ###
###################################

# see helper function scripts for details on matching strategy and evaluation

#res <- vector("list", length(clus) + n_FlowCAP)
res <- vector("list", length(clus) )
names(res)[1:length(clus)] <- names(clus)
#names(res)[-(1:length(clus))] <- paste0(names(clus)[is_FlowCAP], "_alternate")

for (i in 1:length(clus)) {
  if (!is_rare[i] & !is_FlowCAP[i]) {
    res[[i]] <- helper_match_evaluate_multiple(clus[[i]], clus_truth[[i]])
    
  } else if (is_rare[i]) {
    res[[i]] <- helper_match_evaluate_single(clus[[i]], clus_truth[[i]])
    
  } else if (is_FlowCAP[i]) {
    res[[i]]             <- helper_match_evaluate_FlowCAP(clus[[i]], clus_truth[[i]])
    res[[i + n_FlowCAP]] <- helper_match_evaluate_FlowCAP_alternate(clus[[i]], clus_truth[[i]])
  }
}
res_flowMeans <- res
# store named object (for plotting scripts)
############################
########visualisation#######
############################
dirname=paste0("../../results/auto/phenoGraph/", CALC_NAME)
system(paste('mkdir ',dirname))
load(paste0(DATA_DIR,"tsne.RData"))

pop_names=list()
pop_names[[1]]=read.table("../../benchmark_data_sets/attachments/population_names_Levine_32dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')

pop_names[[2]]=read.table("../../benchmark_data_sets/attachments/population_names_Levine_13dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')

pop_names[[3]]=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[3]]$label=rownames(pop_names[[3]])
pop_names[[3]]$population=pop_names[[3]]$population.name

pop_names[[4]]=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names[[4]]$label=rownames(pop_names[[4]])
pop_names[[4]]$population=pop_names[[4]]$population.name

cl_tr<-lapply(clus_truth, function(x) x[which(!is.nan(x))])

tsne3D<-lapply(1:length(cl_tr), function(x) res_tsne[[x]]$tsne_out3D$Y[which(!is.nan(clus_truth[[x]])),])
tsne2D<-lapply(1:length(cl_tr), function(x) res_tsne[[x]]$tsne_out$Y[which(!is.nan(clus_truth[[x]])),])

tsne3D[[4]]
tsne2D[[4]]

i<-4
ncolors=length(unique(cl_tr[[i]]))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr[[i]])
colors<-unlist(lapply(cl_tr[[i]], function(x) col_true[as.character(x)]))
par(oma = c(1, 1, 1, 1))
plot(tsne2D[[i]],col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names[[i]]$population, col = col_true[as.character(pop_names[[i]]$label)],  pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(tsne3D[[i]],  col=colors, pch='.') 
legend3d("topright", legend = pop_names[[i]]$population, pch = 16, col = col_true[as.character(pop_names[[i]]$label)], cex=1, inset=c(0.02))

#color only a subset of cells  
cells=c(1,2,3)
colors1<-unlist(lapply(cl_tr[[i]], function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(tsne2D[[i]],col=colors1, pch='.', cex=1, main = unlist(pop_names[[i]][pop_names[[i]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(tsne3D[[i]],  col=colors1, pch='.', main = unlist(pop_names[[i]][pop_names[[i]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

#match the clusters and plot F_1 consensus results 
for (i in (1:4)){
  ncolors=length(unique(cl_tr[[i]]))
  col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
  names(col_true) = unique(cl_tr[[i]])
  
  match_table <-res[[i]]$labels_matched
  lbl_mapped = unlist(lapply(clus[[i]][which(!is.nan(clus_truth[[i]]))], function(x) {
    ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
  colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[as.character(x)])))
  
  par(mar=c(5, 4, 4, 2) + 0.1)
  par(xpd=T, mar=par()$mar+c(0,0,0,12))
  plot(tsne2D[[i]],col=colors_matched, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
  legend(x = "right", bty = 'n', inset=c(-0.42,0), c(pop_names[[i]]$population, 'Unassigned'), col = c(col_true[as.character(pop_names[[i]]$label)], 'black'), pch = 16, xpd=T, cex=1.1, pt.cex=1.5)
  par(mar=c(5, 4, 4, 2) + 0.1)
  
    plot3d(tsne3D[[i]],  col=colors_matched, pch='.', , xlab='tsne 1', ylab='tsne 2', zlab='tsne 3')
  legend3d("topright", legend = c(pop_names[[i]]$population, 'Unassigned'), pch = 16, col = c(col_true[as.character(pop_names[[i]]$label)], 'black'), cex=1, inset=c(0.02))
}
#colors by automatic cluster assignment 
ncolors_auto=length(unique(clus[[i]]))
col_auto=sample(rainbow(ncolors_auto), ncolors_auto , replace=F)
names(col_auto) = unique(clus[[i]])
colors_auto<-unlist(lapply(clus[[i]], function(x) col_auto[as.character(x)]))
plot(tsne2D[[i]],col=colors_auto, pch='.', cex=1)

plot3d(tsne3D[[i]],  col=colors_auto, pch='.') 

#A plot of Levine 13 for the report
png(filename = '../../plots/flowMeans/flowMeans_Levine_13.png', width = 1400, height = 1000)
par(mar=c(5, 4, 4, 2) + 0.1)
par(xpd=T, mar=par()$mar+c(0,0,0,25))
plot(tsne2D[[i]],col=colors_matched, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = "right", bty = 'n', inset=c(-0.40,0), c(pop_names[[i]]$population, 'Unassigned'), col = c(col_true[as.character(pop_names[[i]]$label)], 'black'), pch = 16, xpd=T, cex=2.2, pt.cex=2.2)
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

xres<-xtable(df,  caption = "flowMeans results", digits=c(0,0,4,4,4,2,2), display=c('s', 's','fg','fg','fg','d','d'), include.rownames=FALSE, align = c("c", "c ",  "c ", "c ", "c ", "c ", "c "), label = "tab:flowMeans")

print(xres,  include.rownames = FALSE, sanitize.text = identity,  table.placement = "h!",  caption.placement = "top")






