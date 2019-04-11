# test mcodc: markov chain outlier detection and ######################################
# clustering
########################################################################################

#load data, labels etc.
library(cytofkit) 
library(fpc)
library(cluster) 
library(Rtsne)
library(rgl)
library(gclus)
library(data.table)
library(flowCore)
library(parallel)
library(Matrix)
library(scales)

library(silhouette )

seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_NAME<-'mcodc_L2_k30'
k=30
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')

#################
### LOAD DATA ###
#################
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
# filenames

DATA_DIR <- "../../benchmark_data_sets"

files <- list(
  Levine_32dim = file.path(DATA_DIR, "Levine_32dim.fcs"), 
  Levine_13dim = file.path(DATA_DIR, "Levine_13dim.fcs"), 
  Samusik_01   = file.path(DATA_DIR, "Samusik_01.fcs") 
  #Samusik_all  = file.path(DATA_DIR, "Samusik_all.fcs"), 
  #Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  #Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs"), 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)


# FlowCAP data sets are treated separately since they require clustering algorithms to be
# run individually for each sample

is_FlowCAP <- c(FALSE, FALSE, FALSE)

# load data files

data <- vector("list", length(files))
names(data) <- names(files)

#for (i in 1:1) {
for (i in 1:length(data)) {
  f <- files[[i]]
  
  if (!is_FlowCAP[i]) {
    data[[i]] <- flowCore::exprs(flowCore::read.FCS(f, transformation = FALSE, truncate_max_range = FALSE))
    
  } else {
    smp <- flowCore::exprs(flowCore::read.FCS(f, transformation = FALSE, truncate_max_range = FALSE))
    smp <- smp[, "sample"]
    d <- flowCore::read.FCS(f, transformation = FALSE, truncate_max_range = FALSE)
    d <- flowCore::split(d, smp)
    data[[i]] <- lapply(d, function(s) flowCore::exprs(s))
  }
}

head(data[[1]])

sapply(data, length)

sapply(data[!is_FlowCAP], dim)
sapply(data[is_FlowCAP], function(d) {
  sapply(d, function(d2) {
    dim(d2)
  })
})

#Remove cells without labels from data
#For now not done: subsampling for data sets with excessive runtime (> 12 hrs on server)

ix_subsample <- 1:8
n_sub <- 1000000000000

for (i in ix_subsample) {
  if (!is_FlowCAP[i]) {
    set.seed(123)
    data[[i]] <- data[[i]][, ]
    # save subsampled population IDs
    true_labels_i <- data[[i]][, "label", drop = FALSE]
    files_true_labels_i <- paste0("../../results/auto/DensVM/true_labels_DensVM_", 
                                  names(data)[i], ".txt")
    for (f in files_true_labels_i) {
      write.table(true_labels_i, file = f, row.names = FALSE, quote = FALSE, sep = "\t")
    }
    
  } else {
    # FlowCAP data sets
    for (j in 1:length(data[[i]])) {
      set.seed(123)
      data[[i]][[j]] <- data[[i]][[j]][, ]
      # save subsampled population IDs
      true_labels_ij <- data[[i]][[j]][, "label", drop = FALSE]
      files_true_labels_ij <- paste0("../../results/auto/DensVM/true_labels_DensVM_", 
                                     names(data)[i], "_", j, ".txt")
      for (f in files_true_labels_ij) {
        write.table(true_labels_ij, file = f, row.names = FALSE, quote = FALSE, sep = "\t")
      }
    }
  }
}


# indices of protein marker columns

marker_cols <- list(
  Levine_32dim = 5:36, 
  Levine_13dim = 1:13, 
  Samusik_01   = 9:47, 
  Samusik_all  = 9:47, 
  Nilsson_rare = c(5:7, 9:18), 
  Mosmann_rare = c(7:9, 11:21), 
  FlowCAP_ND   = 3:13, #keep the label 
  FlowCAP_WNV  = 3:8
)
sapply(marker_cols, length)

# subset data: protein marker columns only

for (i in 1:length(data)) {
  if (!is_FlowCAP[i]) {
    data[[i]] <- data[[i]][, marker_cols[[i]]]
  } else {
    for (j in 1:length(data[[i]])) {
      data[[i]][[j]] <- data[[i]][[j]][, marker_cols[[i]]]
    }
  }
}

sapply(data[!is_FlowCAP], dim)
sapply(data[is_FlowCAP], function(d) {
  sapply(d, function(d2) {
    dim(d2)
  })
})

marker_names<- lapply(data, colnames)
#save(marker_names,  file=paste0(DATA_DIR, '/marker_names.RData'))

####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)
MANUAL_DENSITYCUT <- "../../results/manual/densityCut"
files_truth <- list(
  Levine_32dim = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Levine_32dim.txt"), 
  Levine_13dim = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Levine_13dim.txt"), 
  Samusik_01   = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Samusik_01.txt") 
  #Samusik_all  = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Samusik_all.txt"), 
  #Nilsson_rare = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Nilsson_rare.txt"), 
  #Mosmann_rare = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Mosmann_rare.txt") 
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


###########################################################################################
#########################################################################################
#load precomuted L2 distances
JAC_DIR = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/kk30L2'
files <- list(
  Levine_32dim = file.path(JAC_DIR, "j31Levine_32dim.txt"), 
  Levine_13dim = file.path(JAC_DIR, "j31Levine_13dim.txt"), 
  Samusik_01   = file.path(JAC_DIR, "j31Samusik_01.txt") 
  #Samusik_all  = file.path(DATA_DIR, "Samusik_all.fcs"), 
  #Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  #Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs"), 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)


jaccard_l<- vector("list", length(files))
for (i in 1:length(data)) {
  f <- files[[i]]
  jaccard_l[[i]]<-read.table(file=f, header = FALSE)
}
lapply(jaccard_l, dim)
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
#########################################################################################################
clus_assign<-vector("list", length(files_truth))
clus_assign_ind<-vector("list", length(files_truth))
louvain_assign<-vector("list", length(files_truth))
outliers <- vector("list", length(files_truth))
#########################################################################################################
#loop over all data sets cluster with and without outlier removal
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods/helper_walker.R')
for (i in 1:length(files)){
  
  colnames(jaccard_l[[i]])<- c("from","to","weight")
  jaccard_l[[i]]<-as.data.table(jaccard_l[[i]])
  #relations<-relations[from!=to, ]# remove self-loops
  
  
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  #run one reweight cycle
  sym_w<-(asym_w+t(asym_w))/2
  nnzero(sym_w)
  table(round(colSums(sym_w)))
  
  #reweight SYMMETRIC cmatrix #####################################################
  ##################################################################################
  asym_rw<-mcReweightLocal(asym_w, addLoops = FALSE, expansion = 2, inflation = 3,  max.iter = 1, ESM = TRUE , stop.cond=0.9)[[2]];gc() #version with assymetric input matrix
  sym_rw <-  (asym_rw+t(asym_rw))/2
  head(sort(colSums(asym_rw),decreasing = T)) 
  table(round(colSums(asym_rw)))
  hist(colSums(asym_rw),500)
  
  hist((asym_rw@x),500)
  
  g<-graph_from_adjacency_matrix((asym_rw+t(asym_rw))/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  
  gL<-graph_from_adjacency_matrix((asym_w+t(asym_w))/2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  gL<-simplify(gL, remove.loops=T, edge.attr.comb=list(weight="sum"))
  
  cl_resL<-louvain_multiple_runs(gL, num.run = 1);gc()
  cl_res<-louvain_multiple_runs(g, num.run = 1);gc()
  
  louvain_assign[[i]]<-membership(cl_resL)
  clus_assign[[i]]=membership(cl_res)
  lbls=clus_truth[[i]]
  mbr<-clus_assign[[i]]
  lva<-louvain_assign[[i]]
  #mbr2<-membership(cl_res2)
  
  table(mbr)
  #table(mbr2)
  table(lbls)
  table(lva)
  
  
  helper_match_evaluate_multiple(mbr, lbls)
  helper_match_evaluate_multiple(lva, lbls)
  
  #idx<-!is.na(lbls)
  #helper_match_evaluate_multiple(mbr[idx], lbls[idx])
  #helper_match_evaluate_multiple(mbr2[idx], lbls[idx])
  
  ##########################################################################
  #outlier removal
  gr_res <- mcOutLocal(asym_w, addLoops = F, expansion = 2, inflation = 1,  max.iter = 1, ESM = TRUE); gc()
  gr_res[[1]]
  gr_w<-gr_res[[2]] 
  head(sort(colSums(gr_w),decreasing = T))
  hist(colSums(gr_w),  50000)
  hist(colSums(gr_w), xlim=c(0,0.005), 50000)
  table(lbls[rowSums(gr_w)==0])
  table(lbls[colSums(gr_w)==0])
  table(lbls[colSums(gr_w)>0])
  sum(colSums(gr_w)==0)
  table(lbls)
  
  #plot(colSums(gr_w), unlist(lapply(1:nrow(gr_w), function(x) nnzero(gr_w[x,]))))
  table(lbls[colSums(gr_w)==0])
  
  deg <- colSums(asym_w!=0)
  minweight <- deg/(2*k-1)
  IDXe <- which(colSums(asym_w)<=minweight & deg<=k)
  #IDXe<-which(round(colSums(sym_w))==0)
  
  
  IDXw<-which(colSums(gr_w)==0)
  
  IDX <- !((1:length(lbls)) %in% IDXw)
  table(lbls[!IDX])
  table(lbls[(1:length(lbls) %in% IDXw)])
  table(lbls[(1:length(lbls) %in% IDXe)])
  outliers[[i]]<-IDX
  #outliers[[i]]<-!IDXw
  
  indg<-induced_subgraph(g, (1:gorder(g))[IDX])
  
  #cl_res<-cluster_louvain(g)
  cl_resi<-louvain_multiple_runs(indg, num.run = 1);gc()
  
   
  clus_assign_ind[[i]]=membership(cl_resi)
  
  lbls=clus_truth[[i]]
  mbri<-clus_assign_ind[[i]]
  
  
  table(mbr)
  table(mbri)
  table(lbls)
  
  comf<-rep(NA,length(lbls))
  comf[outliers[[i]]]<-mbri
  comf[!outliers[[i]]]<-100
  table(comf)
  
  helper_match_evaluate_multiple(mbr, lbls)
  helper_match_evaluate_multiple(comf, lbls)

  
}

lapply(outliers, function(x) sum(!x))

#check the modularity
igraph::modularity(g, mbr, weights = E(g)$weight)
igraph::modularity(indg, membership(cl_resi), weights = E(indg)$weight)
igraph::modularity(gL, membership(cl_resL), weights = E(gL)$weight)

#experiment with INFOMAP and brute force modularity via aggregation and simplification of the clusters

#g_small<- cut_at(cl_resL, no=10000)
gcon <- contract.vertices(gL, cl_res$membership, vertex.attr.comb = list(weight = "sum"))
gcon <- simplify(gcon, edge.attr.comb=list(weight="sum"))
gcon=delete.edges(gcon,which(E(gcon)$weight==0))
gcon=delete.vertices(gcon,which(strength(gcon)<0.6))
V(gcon)$weight<-table(cl_res$membership)
#V(gcon)$weight<-0
#plot(gcon, edge.width=E(gcon)$weight)

cl_crs <- cluster_infomap(gcon)
table(membership(cl_crs))

###############################################################################################
#assignment of outliers to clusters and evaluation
#using silhouette and F1 measure. Louvain algorithm is run
#on the set without  outliers and on the set with
#outliers present
################################################################################################

labels_out<-vector("list", length(files_truth))
comRF<-vector("list", length(files_truth))

for (i in 1:length(files_truth)){
  lbls=clus_truth[[i]]
  mbri<-clus_assign_ind[[i]]
  mbr<-clus_assign[[i]]
  lva<-louvain_assign[[i]]
  
  labels_out[[i]] <- helper_assign_outliers(bulk_data =data[[i]][outliers[[i]],], out_data =data[[i]][!outliers[[i]],], bulk_labels =mbri); gc()
  
  comRF[[i]]<-rep(NA,length(lbls))
  comRF[[i]][outliers[[i]]]<-mbri
  comRF[[i]][!outliers[[i]]]<-labels_out[[i]]
  table(comRF[[i]])
  
  helper_match_evaluate_multiple(mbr, lbls)
  helper_match_evaluate_multiple(comRF[[i]], lbls)
  helper_match_evaluate_multiple(lva, lbls)
  
  table(lbls)
}
beep()


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

#pop_names[[4]]=read.table("../../benchmark_data_sets/attachments/population_names_Samusik.txt", header = T, stringsAsFactors = FALSE, sep='\t')
#pop_names[[4]]$label=rownames(pop_names[[4]])
#pop_names[[4]]$population=pop_names[[4]]$population.name

cl_tr<-clus_truth
cl_tr<-lapply(cl_tr, function(x) ifelse(is.na(x), 0, x))


tsne3D <- lapply(res_tsne, function(x) x$tsne_out3D$Y)[1:3]
tsne2D <- lapply(res_tsne, function(x) x$tsne_out$Y)[1:3]

i<-1
ncolors=length(unique(cl_tr[[i]]))
col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
names(col_true) = unique(cl_tr[[i]])
colors<-unlist(lapply(cl_tr[[i]], function(x) ifelse(x!=0, col_true[as.character(x)], 'grey')))
alp <- unlist(lapply(cl_tr[[i]], function(x) ifelse(x!=0, 1, 0.2)))
par(oma = c(1, 1, 1, 1))
plot(tsne2D[[i]],col=alpha(colors, alp), pch='.', cex=1, xlab='tsne 1', ylab='tsne 2')
legend(x = 'right', bty='n', inset=c(-0.28,0), pop_names[[i]]$population, col = col_true[as.character(pop_names[[i]]$label)],  pch = 16, xpd=TRUE, cex=0.8, pt.cex=1.5)

png(filename='zzz.png')
plot3d(tsne3D[[i]],  col=colors, pch='.') 
dev.off()
legend3d("topright", legend = pop_names[[i]]$population, pch = 16, col = col_true[as.character(pop_names[[i]]$label)], cex=1, inset=c(0.02))

#color only a subset of cells  
cells=c(1,2,3)
colors1<-unlist(lapply(cl_tr[[i]], function(x) ifelse(x %in% cells, col_true[as.character(x)], 'black')))
#data<- cbind(f, clusters)
plot(tsne2D[[i]],col=colors1, pch='.', cex=1, main = unlist(pop_names[[i]][pop_names[[i]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2')
plot3d(tsne3D[[i]],  col=colors1, pch='.', main = unlist(pop_names[[i]][pop_names[[i]]$label %in% cells,'population']), xlab='tsne 1', ylab='tsne 2', zlab='tsne 3') 

#match the clusters and plot F_1 consensus results 
res=lapply(1:length(clus_assign), function(x) helper_match_evaluate_multiple(clus_assign[[x]], cl_tr[[x]]))
for (i in (1:3)){
  ncolors=length(unique(cl_tr[[i]]))
  col_true=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
  names(col_true) = unique(cl_tr[[i]])
  
  match_table <-res[[i]]$labels_matched
  lbl_mapped = unlist(lapply(clus_assign[[i]][which(!is.nan(clus_truth[[i]]))], function(x) {
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
ncolors_auto=length(unique(clus_assign[[i]]))
col_auto=sample(rainbow(ncolors_auto), ncolors_auto , replace=F)
names(col_auto) = unique(clus_assign[[i]])
colors_auto<-unlist(lapply(clus_assign[[i]], function(x) col_auto[as.character(x)]))
plot(tsne2D[[i]],col=colors_auto, pch='.', cex=1)

plot3d(tsne3D[[i]],  col=colors_auto, pch='.') 

#plot outliers
i=1
plot(tsne2D[[i]],col=ifelse(outliers[[i]], 'black', 'red'), pch='.', cex=1)
plot3d(tsne3D[[i]],  col=ifelse(outliers[[i]], 'black', 'red'), pch='.') 

#sample non-outlers
ii<-sample(1:nrow(data[[i]][outliers[[i]], ]), 10000)
hist(dist(data[[i]][outliers[[i]], ][ii,]),500, col='red')
io<-sample(1:nrow(data[[i]][!outliers[[i]], ]), 2000)
hist(dist(data[[i]][!outliers[[i]], ][io,]),500, col='green', add=T)

library('flexclust')
hist(dist2(data[[i]][!outliers[[i]], ][io,], data[[i]][outliers[[i]], ][ii,], method = "euclidean", p=2),500)

#siluhette index for outliers

ii<-sample((1:nrow(data[[i]]))[outliers[[i]]], 10000)
#io<-sample(1:nrow(data[[i]][!outliers[[i]], ]), 2000)
labels_sil <- louvain_assign[[i]][ii]
labels_out <-  louvain_assign[[i]][ (1:nrow(data[[i]]))[!outliers[[i]] ] ]
data_sil <- data[[i]][ii,]
data_out <- data[[i]][ !outliers[[i]] , ]

sind<-silhouette(labels_sil,dist(data_sil))
plot(sind)

sind_in_out<-silhouette(c(labels_out, labels_sil) ,dist(rbind(data_out, data_sil)))
plot(sind_in_out)

sind_out<-silhouette(labels_out, dist( data_out))
plot(sind_out)

hist(sind_in_out[1:length(labels_out) ,3],500)
median(sind_in_out[1:length(labels_out) ,3])
hist(sind_in_out[!(1:nrow(sind_in_out) %in% 1:length(labels_out)) ,3],500)
median(sind_in_out[!(1:nrow(sind_in_out) %in% 1:length(labels_out)) ,3])

hist(sind[ ,3],500)
median(sind[ ,3])

hist(sind_out[ ,3],500)
median(sind_out[ ,3])

#now ierate per each outlier
#sind_1out <- rep(0, length(labels_out))
#system.time(for (l in (1:length(labels_out))){
#  sind_1out[l] <- silhouette(c(labels_out[l], labels_sil), dist(rbind(data_out[l,], data_sil)))[1,3]
#  print(c(l, ' '))
#})

hist(sind_1out, 500)
median(sind_1out)

#siluhette index for different clusterings
i=3
ii<-sample((1:nrow(data[[i]])), 20000)
#io<-sample(1:nrow(data[[i]][!outliers[[i]], ]), 2000)
labels_sil <- louvain_assign[[i]][ii]
data_sil <- scale(data[[i]][ii,])
boxplot(data[[i]][ii,])
boxplot(data_sil)

sind1<-silhouette(labels_sil,dist(data_sil))
plot(sind1)
median(sind1[ ,3])


labels_sil <- clus_assign[[i]][ii]
data_sil <- data[[i]][ii,]

sind2<-silhouette(labels_sil,dist(data_sil))
plot(sind2)
median(sind2[ ,3])

hist(sind1[ ,3], col=alpha('red', 0.8), 50, main=paste0('Silhouette histogram','\n', 'red - louvain, green - us'))
hist(sind2[ ,3], col=alpha('green', 0.7), add=T, 50)




