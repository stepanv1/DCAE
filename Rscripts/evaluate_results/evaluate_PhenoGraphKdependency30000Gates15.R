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

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparison/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/PhenoGraph"
CALC_NAME="Kdependency15Gates30000"




#load  data
data=read.table(paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/data_Levine_13dim.txt"), header = F, stringsAsFactors = FALSE)


####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
    #Levine_32dim = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Levine_32dim.txt"), 
    Levine_13dim = file.path(MANUAL_PHENOGRAPH, CALC_NAME,"Python_true_labels_phenoGraph_Levine_13dim.txt")
    #Samusik_01   = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Samusik_01.txt"), 
    #Samusik_all  = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Samusik_all.txt"), 
    #Nilsson_rare = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Nilsson_rare.txt"), 
    #Mosmann_rare = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_Mosmann_rare.txt") 
    #FlowCAP_ND   = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_FlowCAP_ND.txt"), 
    #FlowCAP_WNV  = file.path(MANUAL_PHENOGRAPH, "true_labels_phenoGraph_FlowCAP_WNV.txt")
)



# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

#for (i in 1) {
for (i in 1:length(clus_truth)) {
    
    data_truth_i <- read.table(files_truth[[i]], header = F, stringsAsFactors = FALSE)[, 1]
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
    Levine_13dim15 = file.path(RES_DIR_PHENOGRAPH, 'Kdependency15Gates30000', "k=15Python_assigned_labels_phenoGraph_Levine_13dim.txt"), 
    Levine_13dim30 = file.path(RES_DIR_PHENOGRAPH, 'Kdependency15Gates30000', "k=30Python_assigned_labels_phenoGraph_Levine_13dim.txt"), 
    Levine_13dim45   = file.path(RES_DIR_PHENOGRAPH, 'Kdependency15Gates30000', "k=45Python_assigned_labels_phenoGraph_Levine_13dim.txt"), 
    Levine_13dim60  = file.path(RES_DIR_PHENOGRAPH, 'Kdependency15Gates30000', "k=60Python_assigned_labels_phenoGraph_Levine_13dim.txt"), 
    Levine_13dim75  = file.path(RES_DIR_PHENOGRAPH, 'Kdependency15Gates30000', "k=75Python_assigned_labels_phenoGraph_Levine_13dim.txt"),
    Levine_13dim90  = file.path(RES_DIR_PHENOGRAPH, 'Kdependency15Gates30000', "k=90Python_assigned_labels_phenoGraph_Levine_13dim.txt"),
    Levine_13dim105  = file.path(RES_DIR_PHENOGRAPH, 'Kdependency15Gates30000', "k=105Python_assigned_labels_phenoGraph_Levine_13dim.txt") 
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

for (i in 1:length(clus)) {
    print(table(clus[[i]], clus_truth[[1]]))
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
    
    res[[i]] <- helper_match_evaluate_multiple(clus[[i]], clus_truth[[1]])
    
}
lapply(res, function(x) c(x$mean_F1, x$n_clus))
library(NMI)
NMIscore<-unlist(lapply(1:length(clus), function(i) NMI(cbind(1:length(clus[[1]]), clus[[i]]), cbind(1:length(clus[[1]]),clus_truth))))
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
#system.time(res_tsne<-mclapply(1, function(x){
#    #idx=sample(1:length((clus_truth[[x]])), 50000)
#    ncolors=length(unique(clus_truth[[x]]))
#    col=rainbow(ncolors)
#    colors<-(unlist(lapply(clus_truth[[x]], function(x) col[x])))#[idx]
#    tsne_out <- Rtsne(data[,  1:13])
#    tsne_out3D <- Rtsne(data[,  1:13], dims=3)
#    return(list('tsne_out'=tsne_out, 'tsne_out3D'=tsne_out3D, 'colors'=colors, 'idx' = NULL))
#}, mc.cores=4))
dirname=paste0("../../results/auto/phenoGraph/", CALC_NAME)
system(paste('mkdir ',dirname))
#save(res_tsne, file=paste0(dirname, '/tsne.RData'))
load(file=paste0(dirname, '/tsne.RData'))

pop_names=read.table("../../benchmark_data_sets/attachments/population_names_CURATEDLevine_13dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')

clus_truth<-clus_truth[[1]]

i=6
ncolors=length(unique(clus_truth))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(clus_truth)
colors<-unlist(lapply(clus_truth, function(x) col_true[as.character(x)]))
#data<- cbind(f, clusters)
par(oma = c(1, 1, 1, 9))
#par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(res_tsne[[1]]$tsne_out$Y,col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names$population, col = col_true[as.character(pop_names$label)], , pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(res_tsne[[1]]$tsne_out3D$Y,  col=colors, pch='.') 
legend3d("topright", legend = pop_names$population, pch = 16, col = col_true[as.character(pop_names$label)], cex=1, inset=c(0.02))

#color only a subset of cells  
cells=c(23,24)
colors1<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(res_tsne[[1]]$tsne_out$Y,col=colors1, pch='.', cex=1, main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(res_tsne[[1]]$tsne_out3D$Y,  col=colors1, pch='.', main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

#match the clusters and plot F_1 consensus results 
match_table <-res[[i]]$labels_matched
lbl_mapped = unlist(lapply(clus[[i]], function(x) {
    ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[as.character(x)])))
plot(res_tsne[[1]]$tsne_out$Y,col=colors_matched, pch='.', cex=1)
legend(x = 'right', bty='n', inset=c(-0.28,0), c(pop_names$population, 'Unassigned'), col = c(col_true[as.character(pop_names$label)], 'black'), pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(res_tsne[[1]]$tsne_out3D$Y,  col=colors_matched, pch='.')
legend3d("topright", legend = c(pop_names$population, 'Unassigned'), pch = 16, col = c(col_true[as.character(pop_names$label)], 'black'), cex=1, inset=c(0.02))

#colors by automatic cluster assignment 
ncolors_auto=length(unique(clus[[i]]))
col_auto=sample(rainbow(ncolors_auto), ncolors_auto , replace=F)
names(col_auto) = unique(clus[[i]])
colors_auto<-unlist(lapply(clus[[i]], function(x) col_auto[as.character(x)]))
plot(res_tsne[[1]]$tsne_out$Y,col=colors_auto, pch='.', cex=1)

plot3d(res_tsne[[1]]$tsne_out3D$Y,  col=colors_auto, pch='.') 

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
df<-matrix(unlist(lapply(1:length(clus), function(x) {z=res[[x]]; c(as.integer(rnames[x]), as.numeric(z$mean_F1), as.numeric(NMIscore[x]), as.numeric(z$mean_re), as.numeric(z$mean_pr),  as.integer(z$n_clus), as.integer(length(unique(clus_truth))))})), ncol=7, byrow=T )
colnames(df)<-c('k', 'mean F1', '\\parbox[t]{1cm}{NMI \\\\ score}', '\\parbox[t]{1cm}{mean \\\\ recall}', '\\parbox[t]{1.2cm}{mean\\\\ precision}',  '\\parbox[t]{2cm}{detected \\\\ N of clusters}', '\\parbox[t]{2cm}{true \\\\ N of clusters}')
df[,6:7]<-as.integer(df[,6:7])

xres<-xtable(df,  caption = "Estimates of linear model for father Muro CB", digits=c(2,3,4,4,4,4,2,2), display=c('d','d','fg','fg','fg','fg','d','d'), include.rownames=FALSE, align = c("c ", "c ", "c ", "c ", "c ", "c ", "c ", "c "), label = "tab:mytable")
             
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



