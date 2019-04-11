#########################################################################################
# R script to load and evaluate results for PhenoGraph
#
# Lukas Weber, August 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/PhenoGraph"
CALC_NAME="k=20_2016-12-0220:29:47"




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

for (i in 1:length(clus)) {
    if (!is_rare[i]) {
        res[[i]] <- helper_match_evaluate_multiple(clus[[i]], clus_truth[[i]])
    } else if (is_rare[i]) {
        res[[i]] <- helper_match_evaluate_single(clus[[i]], clus_truth[[i]])
    }
}

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

res_PhenoGraph <- res

############################
########visualisation#######
############################
library(Rtsne)
system.time(res_tsne<-mclapply(1:4, function(x){
    tsne_out <- Rtsne(data[[i]][, ], pca=F)
    tsne_out3D <- Rtsne(data[[i]][, ], pca=F, dims=3)
    return(list('tsne_out'=tsne_out, 'tsne_out3D'=tsne_out3D))
}, mc.cores=4))
dirname=paste0("../../results/auto/phenoGraph/", CALC_NAME)
system(paste('mkdir ','dirname'))
save(res_tsne, file=paste0(dirname, '/tsnePCAF.RData'))

############################################################
library(tsne)
system.time(res_tsne2<-mclapply(1:4, function(x){
    idx=sample(1:length((clus_truth[[x]])), 1000)
    tsne_out <- tsne(data[[x]][idx, ])
    ncolors=length(unique(clus_truth[[x]]))
    col=rainbow(ncolors)
    colors<-(unlist(lapply(clus_truth[[x]], function(x) col[x])))[idx]
    #tsne_out3D <- tsne(data[[i]][1:1000, ], dims=3)
    return(list('tsne_out'=tsne_out, 'colors'=colors))
}, mc.cores=4))

i=1
ncolors=length(unique(clus_truth[[i]]))
col=rainbow(ncolors)
colors<-unlist(lapply(clus_truth[[i]], function(x) col[x]))
#data<- cbind(f, clusters)
plot(res_tsne2[[i]]$tsne_out,col=res_tsne2[[i]]$colors, pch=19, cex=0.5)

##########################################################


library(tsne)
system.time(res_tsne<-mclapply(1:4, function(x){
    idx=sample(1:length((clus_truth[[x]])), 50000)
    ncolors=length(unique(clus_truth[[x]]))
    col=rainbow(ncolors)
    colors<-(unlist(lapply(clus_truth[[x]], function(x) col[x])))[idx]
    tsne_out <- Rtsne(data[[x]][idx, ])
    tsne_out3D <- Rtsne(data[[x]][idx, ], dims=3)
    return(list('tsne_out'=tsne_out, 'tsne_out3D'=tsne_out3D, 'colors'=colors, 'idx' = idx))
}, mc.cores=4))
dirname=paste0("../../results/auto/phenoGraph/", CALC_NAME)
system(paste('mkdir ','dirname'))
save(res_tsne, file=paste0(dirname, '/tsne.RData'))

#TO DELETE
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
plot(res_tsne2[[i]]$tsne_out$Y,col=colors_matched[res_tsne2[[i]]$idx], pch='.', cex=1, main='Accuracy')

plot3d(res_tsne2[[i]]$tsne_out3D$Y,  col=colors_matched[res_tsne2[[i]]$idx], pch='.')

#colors by automatic cluster assignment 
ncolors_auto=length(unique(clus[[i]]))
col_auto=sample(rainbow(ncolors_auto), ncolors_auto , replace=F)

colors_auto<-unlist(lapply(clus[[i]], function(x) col_auto[x]))
plot(res_tsne2[[i]]$tsne_out$Y,col=colors_auto[res_tsne2[[i]]$idx], pch='.', cex=1)

plot3d(res_tsne2[[i]]$tsne_out3D$Y,  col=colors_auto[res_tsne2[[i]]$idx], pch='.') 

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
options(xtable.floating = FALSE)
options(xtable.timestamp = "")

df<-matrix(unlist(lapply(1:4, function(x) {z=res[[x]]; c(z$mean_re, z$mean_pr, z$mean_F1, as.integer(z$n_clus), as.integer(length(unique(clus_truth[[x]]))))})), ncol=5, byrow=T )
colnames(df)<-c('mean recall', 'mean precision', 'mean F1', 'Detected N of clusters', 'true N of clusters')
rownames(df)<-names(res[1:4])
df[,4:5]<-as.integer(df[,4:5])

xtable(df, display=c('s','f','f','f','d','d'))



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
find_neighbors <- function(data, k=20){
    nearest <- nn2(data, data, k, treetype = "bd", searchtype = "standard")
    return(nearest[[1]])
}

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



