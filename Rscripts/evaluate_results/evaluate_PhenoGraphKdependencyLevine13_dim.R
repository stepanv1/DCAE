#########################################################################################
# R script to load and evaluate results for PhenoGraph
#
# Lukas Weber, August 2016
#########################################################################################


library(flowCore)
library(clue)
library(parallel)
library(rgl)
library(vcd)

# helper functions to match clusters and evaluate
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

# which set of results to use: automatic or manual number of clusters (see parameters spreadsheet)
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/PhenoGraph"
CALC_NAME="KdependencyNoGAtes"
DATA_DIR = "../../benchmark_data_sets/"



#load  data
data=read.table(paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/data_Levine_13dim.txt"), header = F, stringsAsFactors = FALSE)


####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
    Levine_13dim = file.path(DATA_DIR, "Levine_13dim.fcs") 
)

# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

for (i in 1:length(clus_truth)) {
    # if (!is_subsampled[i]) {
    data_truth_i <- flowCore::exprs(flowCore::read.FCS(files_truth[[i]], transformation = FALSE, truncate_max_range = FALSE))
    #} else {
    #data_truth_i <- read.table(files_truth[[i]], header = TRUE, stringsAsFactors = FALSE)
    #}
    clus_truth[[i]] <- data_truth_i[, "label"]
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
    Levine_13dim15 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=15Python_assigned_labels_phenoGraph_Levine_13dim.txt"), 
    Levine_13dim30 = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=30Python_assigned_labels_phenoGraph_Levine_13dim.txt"), 
    Levine_13dim45   = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=45Python_assigned_labels_phenoGraph_Levine_13dim.txt"), 
    Levine_13dim60  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=60Python_assigned_labels_phenoGraph_Levine_13dim.txt"),
    Levine_13dim75  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=75Python_assigned_labels_phenoGraph_Levine_13dim.txt"),
    Levine_13dim90  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=90Python_assigned_labels_phenoGraph_Levine_13dim.txt"),
    Levine_13dim105  = file.path(RES_DIR_PHENOGRAPH, CALC_NAME, "k=105Python_assigned_labels_phenoGraph_Levine_13dim.txt")
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

#visualise contingency table
#1 Correspondence analysis
library(ca)
for (k in 1:7){
#names(dimnames(tab)) = c("activity", "period")
#rownames(tab)        = c("feed", "social", "travel")
#colnames(tab)        = c("morning", "noon", "afternoon", "evening")
#         period
# activity morning noon afternoon evening
#   feed        28    4         0      56
#   social      38    5         9      10
#   travel       6    6        14      13
tab=table( clus_truth[[1]], clus[[k]])
tab=tab[,colSums(tab != 0) != 0] 
plot(ca(tab), main=k)
plot3d(ca(tab))}

dirname=paste0("../../plots/PhenoGraph/", CALC_NAME)
dir.create(file.path(dirname), showWarnings = T)
par(mar=c(5,4,4,2)+0.1)
for (k in 1:7){
    png(filename=paste0(dirname, '/', k, CALC_NAME, 'Mosaic.png' ))
    tab=(table(clus_truth[[1]], clus[[k]],dnn=c("True","Assign")))
    rownames(tab)        = sort(unique(clus_truth[[1]]))
    colnames(tab)        = sort(unique(clus[[k]]))
    #tab=tab[,colSums(tab != 0) != 0] 
    mosaic(tab, shade=F, legend=F,  cex=0.5, cex.lab=0.2,  pop = FALSE, zero_size=0.5, direction='h', labeling_args=list(rot_labels=c(bottom=90,top=90),gp_labels=(gpar(fontsize=6))))
    dev.off()    
}

library("corrplot")
par(mar=c(5,4,4,2)+0.1)
for (k in 1:7){
    tab=(table(clus_truth[[1]], clus[[k]],dnn=c("True","Assign")))
    rownames(tab)        = sort(unique(clus_truth[[1]]))
    colnames(tab)        = sort(unique(clus[[k]]))
    tab=as.matrix(tab)
    #tab=tab[,colSums(tab != 0) != 0] 
    corrplot(log2(tab+0.000001), is.corr=FALSE, main=k,method =  "shade")
    #mtext(side = 1, "Category1", line = 0.5, col="green")    
}








# store named objects (for other scripts)

#files_PhesnoGraph <- files_out
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
lapply(res, function(x) c(x$mean_F1, x$F1))
library(NMI)
#NMIscore<-unlist(lapply(1:length(clus), function(i) NMI(cbind(1:length(clus[[1]]), clus[[i]]), cbind(1:length(clus[[1]]),clus_truth))))
#calcuated in Python, since NMI calulation in R is extremly slow
NMIscore<-c(0.508024161932,
            0.512152062883,
            0.521264612452,
            0.516990482619,
            0.521156337758,
            0.525550236312,
            0.525699640456)


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

#check clustering properties of the data
nb<-NbClust(data = data[sample(1:nrow(data), 10000),1:13], distance = "euclidean", min.nc = 2, max.nc = 30, method = "complete", index ="all") 
save(nb, file=paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/NbClustResults.RData"))

data=read.table(paste0(MANUAL_PHENOGRAPH , '/',CALC_NAME,"/data_Levine_13dim.txt"), header = F, stringsAsFactors = FALSE)

# store named object (for plotting scripts)

res_PhenoGraph <- res

############################
########visualisation#######
############################
dirname=paste0("../../results/auto/phenoGraph/", CALC_NAME)
system(paste('mkdir ',dirname))
load(paste0(DATA_DIR,"tsne.RData"))

     
pop_names=read.table("../../benchmark_data_sets/attachments/population_names_Levine_13dim.txt", header = T, stringsAsFactors = FALSE, sep='\t')
pop_names$population <- paste0(pop_names$label, ' ', pop_names$population)
cl_tr<-clus_truth[[1]][which(!is.nan(clus_truth[[1]]))]

tsne3D<-res_tsne[[1]]$tsne_out3D$Y[which(!is.nan(clus_truth[[1]])),]
tsne2D<-res_tsne[[1]]$tsne_out$Y[which(!is.nan(clus_truth[[1]])),]

i<-7
ncolors=length(unique(cl_tr))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr)
colors<-unlist(lapply(cl_tr, function(x) col_true[as.character(x)]))
par(oma = c(1, 1, 1, 1))
plot(tsne2D,col=colors, pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names$population, col = col_true[as.character(pop_names$label)],  pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(tsne3D,  col=colors, pch='.') 
legend3d("topright", legend = pop_names$population, pch = 16, col = col_true[as.character(pop_names$label)], cex=1, inset=c(0.02))

#color only a subset of cells  
cells=c(1,2,3)
colors1<-unlist(lapply(cl_tr, function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(tsne2D,col=colors1, pch='.', cex=1, main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(tsne3D,  col=colors1, pch='.', main = unlist(pop_names[pop_names$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

#match the clusters and plot F_1 consensus results 
for (i in (1:7)){
match_table <-res[[i]]$labels_matched
lbl_mapped = unlist(lapply(clus[[i]][which(!is.nan(clus_truth[[1]]))], function(x) {
    ifelse(length(which(x==match_table))==0, 0, as.numeric(names(match_table[which(x==match_table)])))}))
colors_matched<-unlist(lapply(lbl_mapped, function(x) ifelse(x==0, 'black', col_true[as.character(x)])))
plot(tsne2D,col=colors_matched, pch='.', cex=1, main = 15*i)
legend(x = 'right', bty='n', inset=c(-0.28,0), c(pop_names$population, 'Unassigned'), col = c(col_true[as.character(pop_names$label)], 'black'), pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

plot3d(tsne3D,  col=colors_matched, pch='.')
legend3d("topright", legend = c(pop_names$population, 'Unassigned'), pch = 16, col = c(col_true[as.character(pop_names$label)], 'black'), cex=1, inset=c(0.02))
}
#colors by automatic cluster assignment 
ncolors_auto=length(unique(clus[[i]]))
col_auto=sample(rainbow(ncolors_auto), ncolors_auto , replace=F)
names(col_auto) = unique(clus[[i]])
colors_auto<-unlist(lapply(clus[[i]], function(x) col_auto[as.character(x)]))
plot(tsne2D,col=colors_auto, pch='.', cex=1)

plot3d(tsne3D,  col=colors_auto, pch='.') 


########################################################
#find markers differentiated in population(s) ##########
########################################################
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[2]], 'label')
cells=c(6,7)
IDXsset<-clus_truth %in% cells
sset<-data[IDXsset, 1:13]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
pc_set<-princomp(sset[,1:13])
pcLoad<-as.data.frame(pc_set$loadings[,1])
colnames(sset)[order(abs(pcLoad),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(pcLoad),decreasing=T)][c(1,2,4)]
plot3d(sset[,top_comp], pch='.', col=colors_matched[IDXsset])
plot3d(sset[,c("HLA-DR", "CXCR4",  "CD44")], pch='.', col=colors_matched[IDXsset])
hist(sset[, 'CD20'], 50)
hist(data[, 'CD20'], 500)
plot(sset[, c('CD20', "CD123")], pch='.')

#Emphasize the difference between clusters
#1. Asymmetric discriminant coordinates
library(fpc)
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(12,13,14)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:32]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
adc<-adcoord(sset, clus_vec, clnum=13)
plot3d(adc$proj[,1:3], col=colors[IDXsset])
units<-adc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])

#2. Discriminant coordinates
library(fpc)
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(12,13,14)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:32]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
dc<-discrcoord(sset,clus_vec, pool = "equal")
plot3d(dc$proj[,1:3], col=colors[IDXsset])
units<-dc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[IDXsset])

#3. Asymmetric weighted discriminant coordinates
library(fpc)
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(12,13,14)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:32]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
awdc<-awcoord(sset,clus_vec,  clnum=c(13), method="classical")
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
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(12,13,14)
IDXsset<-clus_truth %in% cells
clus_vec<-clus_truth[IDXsset]
sset<-data[IDXsset, 1:32]
#plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
andc<-ancoord(sset, clus_vec,  clnum=c(13))
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
cells=c(13,14)
#IDXsset<-clus_truth %in% cells
clus_sub<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, T, ifelse(runif(1)<0.02, T, F) )))#need subset the 'bulk'
sset<-data[clus_sub, 1:32]
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
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(1)
#IDXsset<-clus_truth %in% cells
clus_sub<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, T, ifelse(runif(1)<0.02, T, F) )))#need subset the 'bulk'
sset<-data[clus_sub, 1:32]
clus_vec<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, x , -1) ))[clus_sub]
awdc<-awcoord(sset, clus_vec, clnum=-1)
plot3d(awdc$proj[,1:3], col=colors[clus_sub])
units<-awdc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[clus_sub])

#Selected cell population versus the rest
#4. Neighborhood based discriminant coordinates
library(fpc)
load(  file=paste0(DATA_DIR, '/marker_names.RData'))
colnames(data)<-c(marker_names[[1]], 'label')
cells=c(10)
#IDXsset<-clus_truth %in% cells
clus_sub<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, T, ifelse(runif(1)<0.05, T, F) )))#need subset the 'bulk'
sset<-data[clus_sub, 1:32]
clus_vec<-unlist(lapply(clus_truth, function(x) ifelse(x %in% cells, x , -1) ))[clus_sub]
andc<-ancoord(sset, clus_vec, clnum=-1)
plot3d(andc$proj[,1:3], col=colors[clus_sub])
units<-andc$units[,1] 
colnames(sset)[order(abs(units),decreasing=T)][1:9]
top_comp<-colnames(sset)[order(abs(units),decreasing=T)][c(1,2,3)]
plot3d(sset[,top_comp], pch='.', col=colors[clus_sub])





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

xres<-xtable(df,  caption = "PhenoGraph results, Samusik data set", digits=c(2,3,4,4,4,4,2,2), display=c('d','d','fg','fg','fg','fg','d','d'), include.rownames=FALSE, align = c("c ", "c ", "c ", "c ", "c ", "c ", "c ", "c "), label = "tab:Samusik7k")

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



