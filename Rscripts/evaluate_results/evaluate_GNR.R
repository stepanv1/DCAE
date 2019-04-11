#########################################################################################
# R script to load and evaluate results for densityCut
#
# Lukas Weber, August 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
AUTO_GNR <- "../../results/auto/GNR"
RES_DIR_GNR <- "../../results/auto/GNR"
#CALC_NAME="densityCut_2017-01-0511:30:45"
DATA_DIR <- "../../benchmark_data_sets"
CALC_ID="1000units2MInter"


#load  data


# which data sets required subsampling for this method (see parameters spreadsheet)
is_subsampled <- c(F, F, F, F, F, F, F)

is_rare <- c(FALSE, FALSE, FALSE, FALSE, TRUE,  TRUE,  FALSE, FALSE)

# note: no FlowCAP data sets for this method




####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
    Levine_32dim = file.path(AUTO_GNR, "true_labels_GNR_Levine_32dim.txt"), 
    Levine_13dim = file.path(AUTO_GNR, "true_labels_GNR_Levine_13dim.txt"), 
    Samusik_01   = file.path(AUTO_GNR, "true_labels_GNR_Samusik_01.txt"), 
    Samusik_all  = file.path(AUTO_GNR, "true_labels_GNR_Samusik_all.txt"), 
    Nilsson_rare = file.path(AUTO_GNR, "true_labels_GNR_Nilsson_rare.txt"), 
    Mosmann_rare = file.path(AUTO_GNR, "true_labels_GNR_Mosmann_rare.txt") 
    #FlowCAP_ND   = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_FlowCAP_ND.txt"), 
    #FlowCAP_WNV  = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_FlowCAP_WNV.txt")
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
    Levine_32dim = file.path(AUTO_GNR, paste0(CALC_ID, "Levine_32dim.txt")), 
    Levine_13dim = file.path(AUTO_GNR, paste0(CALC_ID, "Levine_13dim.txt")), 
    Samusik_01   = file.path(AUTO_GNR, paste0(CALC_ID, "Samusik_01.txt")), 
    Samusik_all  = file.path(AUTO_GNR, paste0(CALC_ID, "Samusik_all.txt"))#, 
    #Nilsson_rare = file.path(AUTO_GNR, paste0(CALC_ID, "Nilsson_rare.txt"), 
    #Mosmann_rare = file.path(AUTO_GNR, paste0(CALC_ID, "Mosmann_rare.txt") 
    #FlowCAP_ND   = file.path(RES_DIR_GNR, "densityCut_labels_FlowCAP_ND.fcs"), 
    #FlowCAP_WNV  = file.path(RES_DIR_GNR, "densityCut_labels_FlowCAP_WNV.fcs")
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

files_densityGNR <- files_out
clus_densityGNR <- clus






###################################
### match clusters and evaluate ###
###################################

# see helper function scripts for details on matching strategy and evaluation

res <- vector("list", length(clus))
names(res) <- names(clus)

for (i in 1:length(clus)) {
    if (!is_rare[i]) {
        res[[i]] <- helper_match_evaluate_multiple(clus[[i]], clus_truth[[i]])
    } else if (is_rare[i]) {
        res[[i]] <- helper_match_evaluate_single(clus[[i]], clus_truth[[i]])
    }
}
lapply(res, function(x) c(x$mean_F1, x$n_clus))
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

resFw <- vector("list", length(clus))
names(resInd) <- names(clus)

for (i in 1:length(clus)) {
    if (!is_rare[i]) {
        resFw[[i]] <- helper_match_evaluate_multiple_Fweight(clus[[i]], clus_truth[[i]])
    }
}






# store named object (for plotting scripts)

res_GNG <- res

############################
########visualisation#######
############################

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

tsne3D<-lapply(1:length(cl_tr), function(x) res_tsne[[x]]$tsne_out3D$Y[which(!is.na(clus_truth[[x]])),])
tsne2D<-lapply(1:length(cl_tr), function(x) res_tsne[[x]]$tsne_out$Y[which(!is.na(clus_truth[[x]])),])

#clus<-lapply(1:length(cl_tr), function(x) clus[[i]][which(!is.na(clus_truth[[x]]))])

i<-2
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
  lbl_mapped = unlist(lapply(clus[[i]][which(!is.na(clus_truth[[i]]))], function(x) {
    ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
  colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[as.character(x)])))
  
  par(mar=c(5, 4, 4, 2) + 0.1)
  par(xpd=T, mar=par()$mar+c(0,0,0,12))
  plot(tsne2D[[i]],col=colors_matched, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2', main=i)
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








