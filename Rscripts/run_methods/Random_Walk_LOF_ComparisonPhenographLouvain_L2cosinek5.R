#Compare reulsts of Random Walk and LOF 
#for clustering improvement ability
#  Evaluation of outlier detection in flow cytometry data 
# using random walk. FLOCK Siluette index.
# We use full Samusik data here as Samusik_01 becomes to
# sparse after cleaning for the algorithm to run
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
library(beepr)
library(clusterCrit)

seed<-set.seed(12345)
CALC_NAME <- 'RW_vs_LOF'
ALG_NAME='Louvain_L2cosine'
RES_DIR <- '../../results/outlier_compare'
k=30
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_walker.R')

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')

#################
### LOAD DATA ###
#################


DATA_DIR <- "../../benchmark_data_sets"

files <- list(
  Levine_32dim = file.path(DATA_DIR, "Levine_32dim.fcs"), 
  Levine_13dim = file.path(DATA_DIR, "Levine_13dim.fcs"), 
  Samusik_01   = file.path(DATA_DIR, "Samusik_01.fcs"), 
  Samusik_all  = file.path(DATA_DIR, "Samusik_all.fcs") 
  #Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  #Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs"), 
  #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)


# FlowCAP data sets are treated separately since they require clustering algorithms to be
# run individually for each sample

is_FlowCAP <- c(FALSE, FALSE, FALSE, FALSE)

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

ix_subsample <- 1:4
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
  Samusik_01   = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Samusik_01.txt"), 
  Samusik_all  = file.path(MANUAL_DENSITYCUT, "true_labels_densityCut_Samusik_all.txt") 
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
JAC_DIR = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/kk30L2cosine'
files <- list(
  Levine_32dim = file.path(JAC_DIR, "j30Levine_32dim.txt"), 
  Levine_13dim = file.path(JAC_DIR, "j30Levine_13dim.txt"), 
  Samusik_01   = file.path(JAC_DIR, "j30Samusik_01.txt"), 
  Samusik_all  = file.path(JAC_DIR, "j30Samusik_all.txt")
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


###########################################################
#compute intersection of LOF and RANDOM WALK outliers
##################################
###LOF and random walk results ###
##################################

load(file=paste0(RES_DIR, '/', ALG_NAME, '.RData'))
load(file=paste0(RES_DIR, '/',  'lbod_results', '.RData'))
i=2
hist(-log10(outliersWalkP[[i]]+10^(-150)),1000, xlim=c(00,150), ylim=c(0,10))
#sum(-log10(outliersWalkP[[3]]+10^(-150))>50 & -log10(outliersWalkP[[3]]+10^(-150))<150)#cluster of low probability 
hist(log10(outl_resld[[i]]$lof[,i]),1000)
outlCut<-ifelse(frankv(outl_resld[[i]]$lof[,i], ties.method = 'random', order=-1)<=sum(outliersWalkP[[i]]==0),1,0)
helper_match_evaluate_multiple(outlCut, ifelse(outliersWalkP[[i]]==0, 1, 0) )
AMI(outlCut, ifelse(outliersWalkP[[i]]==0, 1, 0))
#pCardIntersection2(U=265627, G=6475, N=6475, c= 1798, log10=TRUE, mc.cores=5) #very slow
length(intersect((1:length(outliersWalkP[[i]]))[ifelse(outlCut,T,F)], (1:length(outliersWalkP[[i]]))[outliersWalkP[[i]]==0]))
sum(outliersWalkP[[i]]==0)
cor(outliersWalkP[[i]], frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1), method='spearman')
plot(-log10(outliersWalkP[[i]]+10^(-30)), (outl_resld[[i]]$lof[,2]), pch='.')
#cicle through 
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_call_FLOCK.R')
clus_LOF <- vector("list", length(files_truth))
clus_RW <- vector("list", length(files_truth))
PerfMeasures <- vector("list", length(files_truth))
num.louvain.run = 5

for (i in 1:length(outliersWalkP)){
  colnames(jaccard_l[[i]])<- c("from","to","weight")
  jaccard_l[[i]]<-as.data.table(jaccard_l[[i]])
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  sym_w<-(asym_w+t(asym_w))/2
  sym_w<-sym_w[outliersWalkP[[i]]!=0, outliersWalkP[[i]]!=0]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  resflow <- louvain_multiple_runs(g, num.run = num.louvain.run);
  gc()
  clus_RW[[i]] <- rep(NA, length(outliersWalkP[[i]]))
  clus_RW[[i]][outliersWalkP[[i]]!=0] <- resflow
  #print(helper_match_evaluate_multiple(clus_RW[[i]], clus_truth[[i]]))
  
  asym_w <- sparseMatrix(i=jaccard_l[[i]]$from+1,j=jaccard_l[[i]]$to+1,x=jaccard_l[[i]]$weight, symmetric = F, index1=T);gc()
  asym_w<-asym_w-Diagonal(x=diag(asym_w));gc()
  sym_w<-(asym_w+t(asym_w))/2
  ixO<-frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0)
  sym_w<-sym_w[ixO, ixO]
  g<-graph_from_adjacency_matrix(sym_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
  g<-simplify(g, remove.loops=T, edge.attr.comb=list(weight="sum"))
  resflow <- louvain_multiple_runs(g, num.run = num.louvain.run);
  gc()
  clus_LOF[[i]] <- rep(NA, length(outliersWalkP[[i]]))
  clus_LOF[[i]][frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0)] <- resflow
  #print(helper_match_evaluate_multiple(clus_LOF[[i]], clus_truth[[i]]))  
  #print(helper_match_evaluate_multiple(clus_assign[[i]], clus_truth[[i]]))  
  
  # Calculate 4 performance measures, supevised and 
  # unsupervised. Without outliers detection, and with Random Walk
  # and LOF detection.
  PerfMeasures[[i]] = matrix(NA, 3,4)
  
  PerfMeasures[[i]][1,1] <- helper_match_evaluate_multiple(clus_assign[[i]], clus_truth[[i]])$mean_F1 
  PerfMeasures[[i]][2,1] <- helper_match_evaluate_multiple(clus_RW[[i]], clus_truth[[i]])$mean_F1
  PerfMeasures[[i]][3,1] <- helper_match_evaluate_multiple(clus_LOF[[i]], clus_truth[[i]])$mean_F1
  
  PerfMeasures[[i]][1,2] <- AMI(clus_assign[[i]], clus_truth[[i]])
  PerfMeasures[[i]][2,2] <- AMI(clus_RW[[i]], clus_truth[[i]])
  PerfMeasures[[i]][3,2] <- AMI(clus_LOF[[i]], clus_truth[[i]])
  
  idxs <- sample(length(clus_assign[[i]]), 20000)
  int<-intCriteria(scale(data[[i]][idxs,]),as.integer(clus_assign[[i]][idxs]),c("C_index","Gamma"))
  
  ii<-sample(outliersWalk[[i]], 20000)
  intRW<-intCriteria(scale(data[[i]][ii,]),as.integer(clus_RW[[i]][ii]),c("C_index","Gamma"))
  
  outliersLOF <- (1:length(outliersWalkP[[i]]))[frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0)]
  ii<-sample(outliersLOF, 20000)
  intLOF<-intCriteria(scale(data[[i]][ii,]),as.integer(clus_LOF[[i]][ii]),c("C_index","Gamma"))
  
  PerfMeasures[[i]][1,3] <- 1-int$c_index 
  PerfMeasures[[i]][2,3] <- 1-intRW$c_index 
  PerfMeasures[[i]][3,3] <- 1-intLOF$c_index 
  
  PerfMeasures[[i]][1,4] <- int$gamma
  PerfMeasures[[i]][2,4] <- intRW$gamma 
  PerfMeasures[[i]][3,4] <- intLOF$gamma  
}
save(PerfMeasures, clus_RW, clus_LOF, file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'clusterOutImprove', '.RData'))
load(file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'clusterOutImprove', '.RData'))

###siluete index (unsupevised measure of clustering performance)
################################################################




############################
########visualisation#######
############################
#########################
library(pdfCluster)
library(clusterCrit)
#plot outliers for Walk
i=3
#library('flexclust')
#hist(dist2(data[[i]][io,], data[[i]][ii,], method = "euclidean", p=2),500)

idxs <- sample(length(clus_assign[[i]]), 20000)
sind1<-silhouette(clus_assign[[i]][idxs], dist(scale(data[[i]][idxs,]), method = "manhattan"))
median(sind1[ ,3])
dbs1<-dbs(scale(data[[i]][idxs,]), clusters = clus_assign[[i]][idxs], prior=as.vector(table( clus_assign[[i]][idxs])/sum(table( clus_assign[[i]][idxs]))))
median(dbs1@dbs)
int1<-intCriteria(scale(data[[i]][idxs,]),clus_assign[[i]][idxs],c("C_index","Gamma"))
1-int1$c_index; int1$gamma;

IDXout<-which(!(1:dim(data[[i]])[1] %in% outliersWalk[[i]]))
#sample non-outlers
ii<-sample(outliersWalk[[i]], 20000)
#hist(dist(data[[i]][ii,]),500, col='red')
io<-sample(IDXout, ifelse(length(IDXout)<2000,length(IDXout), 2000 ))
sind2<-silhouette(clus_RW[[i]][ii], dist(scale(data[[i]][ii,]), method = "manhattan"))
median(sind2[ ,3])
dbs2<-dbs(scale(data[[i]][ii,]), clusters = clus_RW[[i]][ii], prior=as.vector(table( clus_RW[[i]][ii])/sum(table( clus_RW[[i]][ii]))))
(median(dbs2@dbs))
int2<-intCriteria(scale(data[[i]][ii,]),clus_RW[[i]][ii],c("C_index","Gamma"))
1-int2$c_index; int2$gamma;

outliersLOF <- (1:length(outliersWalkP[[i]]))[frankv(outl_resld[[i]]$lof[,2], ties.method = 'random', order=-1) > sum(outliersWalkP[[i]]==0)]
IDXout<-which(!(1:dim(data[[i]])[1] %in% outliersLOF))
#sample non-outlers
ii<-sample(outliersLOF, 20000)
#hist(dist(data[[i]][ii,]),500, col='red')
io<-sample(IDXout, ifelse(length(IDXout)<2000,length(IDXout), 2000 ))
sind3<-silhouette(clus_LOF[[i]][ii], dist(scale(data[[i]][ii,]), method = "manhattan"))
#plot(sind3)
median(sind3[ ,3])
dbs3<-dbs(scale(data[[i]][ii,]), clusters = clus_LOF[[i]][ii], prior=as.vector(table( clus_LOF[[i]][ii])/sum(table( clus_LOF[[i]][ii]))))
median(dbs3@dbs)
int3<-intCriteria(scale(data[[i]][ii,]),clus_LOF[[i]][ii],c("C_index","Gamma"))
1-int3$c_index; int3$gamma;

h1<-hist(sind1[ ,3], col=alpha('red', 0.9), 150, main=paste0('Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
h2<-hist(sind2[ ,3], col=alpha('green', 0.8), add=T, 150)
h3<-hist(sind3[ ,3], col=alpha('blue', 0.7), add=T, 150)

hist(dbs1@dbs, col=alpha('red', 0.9), 100, main=paste0('Density Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(dbs2@dbs, col=alpha('green', 0.8), add=T, 100)
hist(dbs3@dbs, col=alpha('blue', 0.7), add=T, 100)

##############################################################################
#tables for paper
##############################################################################
library(xtable)
options(xtable.floating = TRUE)
options(xtable.timestamp = "")
floating.environment = getOption( "table")
###################################################################################
i=1
# clustering improvement after outlier removal
df<-PerfMeasures[[i]]
rnames<-c('No detection', 'Random Walk', 'LOF')
#df<-data.frame(matrix(unlist(lapply(1:4, function(x) {z=res[[x]]; c(rnames[x], as.numeric(z$mean_F1),  as.numeric(z$mean_re), as.numeric(z$mean_pr),  as.integer(z$n_clus), as.integer(length(unique(clus_truth[[x]]))))})), ncol=6, byrow=T ), stringsAsFactors=FALSE)
rownames(df) <- rnames
df <- round(df, digits=3)
df <- cbind(rnames , df )
colnames(df)<-c('Outlier detection algorithm', 'mean F1', 'AMI',  'C-measure', '\\parbox[t]{1.2cm}{Gamma}')
xres<-xtable(df,  caption = "Tests results with FLOCK", digits=c(4,4,4,4,4,4), display=c('s', 'fg','fg','fg','fg', 'fg'), include.rownames=FALSE, align = c("|c|c|c|c|c|c|"), label = "FLOCK_Levine32")
print(xres,  include.rownames = FALSE, sanitize.text = identity,  table.placement = "h!",  caption.placement = "top")

############################################################
# Same as above, but using a ftable, all datasets together##
############################################################
library(xtable)
load(file=paste0(RES_DIR, '/', CALC_NAME, '_', ALG_NAME,  'clusterOutImprove', '.RData'))
PerfMeasuresAllsets <- Reduce(rbind, PerfMeasures)
PerfMeasuresAllsets <- round(PerfMeasuresAllsets, digits=3)
names(PerfMeasures) <- names(clus_truth)
# correct names of data sets
goodNames<-c('Levine32', 'Levine13', 'Samusik01', 'SamusikAll')
setNames <- unlist(lapply(goodNames, function(x) c(x, '', '')))
methodNames<-rep(c('No detection', 'Random Walk', 'LOF'), 3)
PerfMeasuresAllsets <-cbind(setNames, cbind(methodNames, PerfMeasuresAllsets)) 
colnames(PerfMeasuresAllsets)<-c('Data set', 'Outlier detection algorithm', 'mean F1', 'AMI',  'C-measure', '\\parbox[t]{1.2cm}{Gamma}')

xres<-xtable(PerfMeasuresAllsets,  caption = "Tests results with Phenograph cosine metric", digits=4, display=c('s', 's', 'fg','fg','fg','fg', 'fg'), include.rownames=FALSE, align = c("|c|c|c|c|c|c|c|"), label = "Phenograph_cosine")

print(xres,  include.rownames = FALSE, sanitize.text = identity,  table.placement = "h!",  caption.placement = "top", hline.after = c(-1,0, seq(3, nrow(PerfMeasuresAllsets), length.out=length(clus_truth))))



########################
#plot outliers for 2Step
library(pdfCluster)
i=2
IDXout<-which(!(1:dim(data[[i]])[1] %in% outliers2Step[[i]]))
#sample non-outlers
ii<-sample(outliers2Step[[i]], 10000)
hist(dist(data[[i]][ii,]),500, col='red')
io<-sample(IDXout, ifelse(length(IDXout)<2000,length(IDXout), 2000 ))
hist(dist(data[[i]][io,]),500, col='green', add=T)

library('flexclust')
hist(dist2(data[[i]][io,], data[[i]][ii,], method = "euclidean", p=2),500)

idxs <- sample(length(clus_assign[[i]]), 10000)
sind1<-silhouette(clus_assign[[i]][idxs], dist(scale(data[[i]][idxs,]), method = "manhattan"))
plot(sind1)
median(sind1[ ,3])
dbs1<-dbs(scale(data[[i]][idxs,]), clusters = clus_assign[[i]][idxs], prior=as.vector(table( clus_assign[[i]][idxs])/sum(table( clus_assign[[i]][idxs]))))
median(dbs1@dbs)

sind2<-silhouette(RFassign2Step[[i]][ii], dist(scale(data[[i]][ii,]), method = "manhattan"))
plot(sind2)
median(sind2[ ,3])
dbs2<-dbs(scale(data[[i]][ii,]), clusters = RFassign2Step[[i]][ii], prior=as.vector(table(RFassign2Step[[i]][ii])/sum(table(RFassign2Step[[i]][ii]))))
median(median(dbs2@dbs))

dbso<-dbs(scale(data[[i]][io,]), clusters = RFassign2Step[[i]][io], prior=as.vector(table(RFassign2Step[[i]][io])/sum(table(RFassign2Step[[i]][io]))))
median(median(dbso@dbs))



hist(sind1[ ,3], col=alpha('red', 0.8), 100, main=paste0('Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(sind2[ ,3], col=alpha('green', 0.7), add=T, 100)

hist(dbs1@dbs, col=alpha('red', 0.8), 100, main=paste0('Density Silhouette histogram','\n', 'red - algorithm, green - algorithm after data cleaning'))
hist(dbs2@dbs, col=alpha('green', 0.7), add=T, 100)







i=3 
table(clus_assign[[i]])
IDXout<-which(!(1:dim(data[[i]])[1] %in% outliersWalk[[i]]))
ic<-(1:nrow(data[[i]]))[clus_assign[[i]]==9]
fit <-  kdevine(scale(data[[i]][ic,]))
destALL <- dkdevine(scale(data[[i]][IDXout,]), fit)
destOUT <- dkdevine(scale(data[[i]][ic,]), fit)

i=3
#pairs(data[[i]][sample(1:nrow(data[[i]]), 2000) , ], pch='.')
IDXout<-sample(which(!(1:dim(data[[i]])[1] %in% outliersWalk[[i]])))
length(IDXout); dim(data[[i]])
pairs(data[[i]][c(sample(1:nrow(data[[i]]), 2000), IDXout[1:2000]), 20:23], pch='.', col = ifelse((1:(2000+length(IDXout[1:2000])))>=2001, 'red', 'black'))

i=1; table(clus_RW[[i]])
prRW=prcomp(data[[i]][!is.na(clus_RW[[k]]) & clus_RW[[i]]==19 ,], scale=T, retx=T)
plot(prRW$x[sample(nrow(prRW$x), 2000),c(1,2)], pch='.')
plot(prRW$x[sample(nrow(prRW$x), 2000),c(1,32)], pch='.')
for(k in 1:25){
  prRW=prcomp(data[[i]][!is.na(clus_RW[[i]]) & clus_RW[[i]]==k ,], scale=T, retx=T)
  print(print(prRW$sdev[1]/prRW$sdev[32]))
  plot(prRW$sdev)
}
table(clus_assign[[i]])
for(k in 1:32){
  prRW=prcomp(data[[i]][!is.na(clus_assign[[i]]) & clus_assign[[i]]==k ,], scale=T, retx=T)
  print(prRW$sdev[1]/prRW$sdev[32])
  plot(prRW$sdev)
}
table(clus_truth[[i]])
for(k in 1:14){
  prRW=prcomp(data[[i]][!is.na(clus_truth[[i]]) & clus_truth[[i]]==k ,], scale=T, retx=T)
  print(prRW$sdev[1]/prRW$sdev[32])
  plot(prRW$sdev, main=paste0('N= ', sum(!is.na(clus_truth[[i]]) & clus_truth[[i]]==k)))
}


#explorinf the dims of clusters
clus_truthL <- lapply(clus_truth, function(x) ifelse(is.na(x), 0, x))
i=2
table(clus_truth[[i]])
table(clus_truthL[[i]])
cln=2
pairs(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln),912) , 1:13], pch='.')
pairs(data[[i]][sample(nrow(data[[i]]) ,1000), 1:13], pch='.')

pairs(data[[i]][sample(nrow(data[[i]]) ,1000), 1:13], pch='.')
pairs(data[[i]][clus_truthL[[i]]!=0 ,][sample(sum(clus_truthL[[i]]!=0),1000) , 1:13], pch='.')
for(j in sort(unique(clus_truthL[[i]])) ){
  cln=j
  beanplot(lapply(as.data.frame(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(5000, sum(clus_truthL[[i]]==cln)) )) , ]), function(x) x), main=j)
}
#difference between 5th and 13thclusters
cln=5
boxplot(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(5000, sum(clus_truthL[[i]]==cln)) )) , ], main=j)
cln=24
boxplot(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(5000, sum(clus_truthL[[i]]==cln)) )) , ], main=j, add=T, col='red')




cln=5
princomp(data[[i]][clus_truthL[[i]]==cln,])
plot(princomp(data[[i]][clus_truthL[[i]]==cln,]), npc=13)
princomp(data[[i]][clus_truthL[[i]]==cln,][sample(sum(clus_truthL[[i]]==cln), 100), ])
plot(princomp(data[[i]][clus_truthL[[i]]==cln,][sample(sum(clus_truthL[[i]]==cln), 200), ]), npc=13)

prccl<-princomp(data[[i]][clus_truthL[[i]]==cln,])
View(prccl$loadings)
matplot((prccl$loadings[1:39,1:6]), type = 'l')

library(rgl)
plot3d(data[[i]][clus_truthL[[i]]==13 | clus_truthL[[i]]==5,][,c(13,11,12)])
plot3d(data[[i]][clus_truthL[[2]]!=0 ,][,c(1,3,11)], col =clus_truthL[[2]][clus_truthL[[2]]!=0]+1, cex=0.01, pch='.')
#without all other clusters bit with noise 
clnex<-c(5,13,24); 
plot3d(data[[i]][clus_truthL[[i]] %in% c(0, clnex), ][,c(13,11,12)][sample(sum(clus_truthL[[i]] %in% c(0, clnex)) ,50000),])
plot3d(data[[i]][clus_truthL[[i]] %in% c( clnex), ][,c(13,11,12)][sample(sum(clus_truthL[[i]] %in% c( clnex)) ,5000),])
#add the third cluster on top of 5 and 13 separated by cd3

samplecl<-sample(sum(clus_truthL[[i]] %in% c(0, clnex)) ,50000)
plot3d(data[[i]][clus_truthL[[i]] %in% c(0, clnex), ][,c(13,11,12)][samplecl,],
       col = clus_truthL[[2]][ clus_truthL[[i]] %in% c(0, clnex) ][samplecl]+2)

samplecl<-sample(sum(clus_truthL[[i]] %in% c( clnex)) ,5000)
plot3d(data[[i]][clus_truthL[[i]] %in% c( clnex), ][,c(13,11,12)],
       col = clus_truthL[[2]][ clus_truthL[[i]] %in% c( clnex) ]+1)

#assign clusters feature selection
table(clus_assign[[i]])
for(j in sort(unique(clus_assign[[i]])) ){
  cln=j
  boxplot(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000, sum(clus_assign[[i]]==cln)) )) , ], main=j)
}
library(beanplot)
for(j in sort(unique(clus_assign[[i]])) ){
  cln=j
  beanplot(lapply(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000, sum(clus_assign[[i]]==cln)) )),]), function(x) x), main=j)
}


