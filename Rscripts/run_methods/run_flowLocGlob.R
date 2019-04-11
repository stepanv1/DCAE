#outlier assignment using subspaces as reference set 
########################################################
# S. Grinek 26.07.17

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
library(gplots)
seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_NAME<-'Louvain_L1'
k=30
RES_DIR <- '../../results/outlier_compare'
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_create_snnk_graph.R')
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

individual<-data[[1]][,'individual']

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
#load precomuted L1 distances
JAC_DIR = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/results/kk30L1'
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


#jaccard_l<- vector("list", length(files))
#for (i in 1:length(data)) {
#  f <- files[[i]]
#  jaccard_l[[i]]<-read.table(file=f, header = FALSE)
#}
#lapply(jaccard_l, dim)
#source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
#source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
#########################################################################################################
clus_assign<-vector("list", length(files_truth))
clus_assign_2Step<-vector("list", length(files_truth))
clus_assign_walk<-vector("list", length(files_truth))
louvain_assign<-vector("list", length(files_truth))
outliersWalk <- vector("list", length(files_truth))
outliersWalkP <- vector("list", length(files_truth))
outliers2Step <- vector("list", length(files_truth))
######################################################
#outlier assignment using subspaces as reference set 
clus_truthL<-lapply(clus_truth, function(x) x[!is.na(x)])
dataL<-lapply(1:length(data), function(x) data[[x]][!is.na(clus_truth[[x]]), ] )
#algorithm starts here

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
ref_subs<- vector("list", length(files_truth))
glob_out<- vector("list", length(files_truth))
loc_out<- vector("list", length(files_truth))
ref_subsT<- vector("list", length(files_truth))
library(beanplot)
#load clustering results without oulier assignments to learn reference set
load(file=paste0(RES_DIR, '/', CALC_NAME, '.RData'))
i=1
# find reference subdimensions
sub_res<-helper_find_subdimensions(data[[i]], clus_assign[[i]])
ref_subs[[i]]<-sub_res$subdim
#sub_resL<-helper_find_subdimensions(data[[i]][!is.na(clus_truth[[i]]),], clus_truth[[i]][!is.na(clus_truth[[i]])])

ref_subs[[i]]<-sub_res$subdim
table(clus_truth[[i]])
table(clus_assign[[i]]) 
table(clus_truth[[i]], clus_assign[[i]])
#plot heatmap of subspaces
X11();heatmap.2(as.matrix(-log10(sub_res$pv+10^-10)))
X11();heatmap.2(as.matrix(ifelse(sub_res$subdim, 1,0)))
X11();heatmap.2(as.matrix(sub_res$sk))
cl=13
cor(data[[i]][clus_assign[[i]]==cl, !sub_res$subdim[cl,]], method='spearman')
X11();heatmap.2(cor(data[[i]][clus_assign[[i]]==cl, sub_res$subdim[cl,]], method='spearman'))
ref_subs<-helper_find_subdimensions(data[[i]], clus_assign[[i]])
cln=13
X11();beanplot(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000000, sum(clus_assign[[i]]==cln)) )), ]), main=cln, beanlines = 'median',  names= names(ref_subs$subdim[cln, ]) , what = c(1,1,1,0), lab.cex=0.1)

library(scales)
X11();plot(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000000, sum(clus_assign[[i]]==cln)) )), c(1,5,12,13,15,18,20,23,24,29)], col=alpha('red', alpha = 0.05)), pch='.', cex=0.5)#cln=22
mean((cor(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000000, sum(clus_assign[[i]]==cln)) )), c(1,5,12,13,15,18,20,23,24,29)]), method='spearman')))

X11();plot(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000000, sum(clus_assign[[i]]==cln)) )), !((1:32) %in% c(1,5,12,13,15,18,20,23,24,29))], col=alpha('red', alpha = 0.05)), pch='.', cex=0.5)#cln=22
mean((cor(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000000, sum(clus_assign[[i]]==cln)) )), !((1:32) %in% c(1,5,12,13,15,18,20,23,24,29))]), method='spearman')))

X11();hist(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000000, sum(clus_assign[[i]]==cln)) )), !((1:32) %in% c(1,5,12,13,15,18,20,23,24,29))], col=alpha('red', alpha = 0.05)))#cln=22

X11();hist(as.data.frame(data[[i]][clus_assign[[i]]==cln ,][sample(sum(clus_assign[[i]]==cln), min(c(5000000, sum(clus_assign[[i]]==cln)) )),  c(1,5,12,13,15,18,20,23,24,29)], col=alpha('red', alpha = 0.05)))#cln=22

1           2           3           4           5           6           7           8           9          10          11 
0.907504363 0.764257028 0.001037883 0.773519164 0.012146978 0.001996008 0.712152284 0.649225196 0.983505969 0.764760957 0.840916866 
12          13          14 
0.786545925 0.952017900 0.000000000 
cln=14
ref_subsT<-helper_find_subdimensions(data[[i]][!is.na(clus_truth[[i]]),], clus_truth[[i]][!is.na(clus_truth[[i]])])
X11();beanplot(as.data.frame(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(45000, sum(clus_truthL[[i]]==cln)) )), ]), main=cln, beanlines = 'median',  names= names(ref_subsT$subdim[cln, ]) , what = c(1,1,1,0), lab.cex=0.1)
X11();beanplot(as.data.frame(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), min(c(45000, sum(clus_truthL[[i]]==cln)) )), ref_subsT$subdim[cln,] ]), main=cln, beanlines = 'median',  names= names(ref_subsT$subdim[cln, ref_subsT$subdim[cln,]]) , what = c(1,1,1,0), lab.cex=0.1);abline(a=0, b=0)
X11();heatmap.2(as.matrix(ifelse(ref_subsT$subdim, 1,0)))



#small part of a cluster
X11();beanplot(as.data.frame(data[[i]][clus_truthL[[i]]==cln ,][sample(sum(clus_truthL[[i]]==cln), 50) , ]), main=cln, beanlines = 'median',  names= names(ref_subsT$subdim[[i]][cln, ]) , what = c(1,1,1,0), lab.cex=0.1)

cor((as.data.frame(data[[i]][clus_truthL[[i]]==cln ,!ref_subsT$subdim[cln,]] ) ))

clNoise<- (as.data.frame(data[[i]][clus_truthL[[i]]==cln ,!ref_subsT$subdim[cln,]] ) )
lapply(1:ncol(clNoise),function(x) sum(clNoise[,x]<0)/nrow(clNoise))

X11();hist(as.data.frame(data[[i]][clus_assign[[i]]==13 ,21:32]),50)
X11();hist(as.data.frame(data[[i]][clus_truthL[[i]]==13 ,]),50)

#summary of intencity in irrelevant dimensions per cluster
summary_irrel_subs <- lapply(colnames(data[[i]]),  function(x)  data[[i]][,x][clus_assign[[i]] %in% (1:ncol(data[[i]]))[!ref_subs$subdim[,x]] ] )
names(summary_irrel_subs)<-colnames(data[[i]])
X11();beanplot(summary_irrel_subs,  what = c(1,1,1,0), beanlines = 'median')

summary_irrel_subsT <- lapply(colnames(data[[i]]),  function(x)  data[[i]][,x][clus_truth[[i]] %in% (1:ncol(data[[i]]))[!ref_subsT$subdim[,x]] ] )
names(summary_irrel_subsT)<-colnames(data[[i]])
X11();beanplot(summary_irrel_subsT,  what = c(1,1,1,0), beanlines = 'median')


summary_all_subs <- lapply(colnames(data[[i]]),  function(x)  data[[i]][,x])
names(summary_irrel_subs)<-colnames(data[[i]])
X11();beanplot(summary_irrel_subs,  what = c(1,1,1,0), beanlines = 'median')


#mean absolute correlations between reference subdimensions
corabs<-(abs(cor(data[[i]][clus_assign[[i]]==cln, (1:39)[ref_subs[[i]][cln,]] ][,], method='spearman')))
sum(abs(corabs-diag(diag(corabs)))>0.1)/((dim(corabs)[1] )*(dim(corabs)[1]-1))
# find global outliers
glob_out[[i]]<-helper_global_outliers(data[[i]], ref_subs[[i]], clus_assign[[i]], mc.cores=5)
hist(glob_out[[i]],200)
glob_out[[i]][is.na(glob_out[[i]])] <- 0
helper_match_evaluate_multiple(ifelse(glob_out[[i]]>2 , 1, 0), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )
table(ifelse(glob_out[[i]]>2 , 'out', 'in'), ifelse(is.na(clus_truth[[i]]), 1 ,0 ))
#save(glob_out, file=paste0(RES_DIR, '/', CALC_NAME, 'glob_out.RData'))
# find local outliers
system.time(loc_out[[i]]<-helper_local_outliers(data[[i]], ref_subs[[i]], glob_out[[i]], clus_assign[[i]], k=25, mc.cores=5))

zzz=as.numeric(loc_out[[i]])
hist(zzz, 200)
#save(loc_out, file=paste0(RES_DIR, '/', CALC_NAME, 'loc_out.RData'))
helper_match_evaluate_multiple(ifelse(zzz>1 | is.na(zzz), 1, 0), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )
table(ifelse(zzz>1.2 | is.na(zzz), 'o', 'in'), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )
#evaluate results, runnung clustering algorithm with global outliers removed

#evaluate results, runnung clustering algorithm with local outliers removed
table(clus_assign[[i]])
helper_match_evaluate_multiple(ifelse(glob_out[[i]]>2  & zzz>1.1, 1, 0), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )
table(ifelse(glob_out[[i]]>1.3  & zzz>1, 'o', 'in'), ifelse(is.na(clus_truth[[i]]), 1 ,0 ) )

#compare with lof
out_cut=c(3, 1.25); cln=15
sum(glob_out[[i]]>out_cut[1]  |  zzz>out_cut[2])
outgc<- glob_out[[i]]>out_cut[1]  |  zzz>out_cut[2]
color=ifelse(outgc, 'red', 'black')[clus_assign[[i]]==cln]
size = ifelse(outgc, 5, 0.1)[clus_assign[[i]]==cln]
plot(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][,c(3,2) ], pch='.', cex=size, col=color)
plot(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][,c(3,2) ], pch='.', cex=size, col=ifelse(is.na(clus_truth[[i]][clus_assign[[i]]==cln]), 'black', clus_truth[[i]][clus_assign[[i]]==cln]+1))
plot(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][,c(3,2) ], pch='.', cex=size, col=ifelse(is.na(clus_truth[[i]][clus_assign[[i]]==cln]), 'black', clus_truth[[i]][clus_assign[[i]]==cln]+1))
ix<-sample(sum(clus_assign[[i]]==cln),2000 )
pairs(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][ix, ], pch='.', cex=size[ix], col=color[ix])
library(rgl)
plot3d(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][, ], pch='.', cex=size, col=color, xlim=c(0,1.5), ylim=c(0,1.5), zlim=c(0,1.5))
plot3d(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][, ], pch='.', cex=size, col=color)
plot3d(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][, ], pch='.', cex=size, col=ifelse(is.na(clus_truth[[i]]), 'black', clus_truth[[i]]+1))
#pairs(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]], pch='.', col=ifelse(is.na(clus_truth[[i]][clus_assign[[i]]==cln]), 'black', clus_truth[[i]][clus_assign[[i]]==cln]+1))
plot3d(data[[i]][, ref_subs[[i]][cln,]][, ], pch='.',  col=ifelse(clus_assign[[i]]==cln, 'red', 'black'))


load(file=paste0(RES_DIR, '/',  'lbod_results', '.RData'))
#compare results for i=4 and cln=2, 7, which is 3d without subspace, just using lof
# for i = 1 cln=23, 25, 13
# for i = 2 cln 13 - almost no outlers!!
cln<-2
lof_score<-outl_resld[[i]]$lof[,2]
lof_cut=c(1.4);
sum(lof_score>lof_cut)
color=ifelse(lof_score>lof_cut, 'red', 'black')[clus_assign[[i]]==cln]
size = ifelse(lof_score>lof_cut, 5, 0.1)[clus_assign[[i]]==cln]
plot(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][,c(3,2) ], pch='.', cex=size, col=color)
plot3d(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][, ], pch='.', cex=size, col=color)
ix<-sample(sum(clus_assign[[i]]==cln),2000 )
pairs(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][ix, ], pch='.', cex=size[ix], col=color[ix])
pairs(data[[i]][clus_assign[[i]]==cln, ref_subs[[i]][cln,]][ix, ], pch='.',  col=ifelse(is.na(clus_truth[[i]][clus_assign[[i]]==cln][ix]),'red','black' ))

#checking merger on subdivisions
# i =4 merge 9, 15, 24
clnset<-c(9,15,14)
helper_match_evaluate_multiple(clus_assign[[i]][clus_truthL[[i]]!=1000], clus_truth[[i]][clus_truthL[[i]]!=1000])
helper_match_evaluate_multiple(ifelse(clus_assign[[i]] %in% clnset, 99, clus_assign[[i]])[clus_truthL[[i]]!=1000] , clus_truth[[i]][clus_truthL[[i]]!=1000])
plot(data[[i]][clus_assign[[i]] %in% clnset, ref_subs[[i]][cln,]][,c(3,2) ], pch='.', cex=1.5, col=ifelse((clus_truthL[[i]][clus_assign[[i]] %in% c(  14)])==1000, 'black', clus_truthL[[i]][clus_assign[[i]] %in% clnset]+3))
table(clus_truthL[[i]][clus_assign[[i]] %in% clnset])

helper_match_evaluate_multiple(clus_assign[[i]][!outgc], clus_truth[[i]][!outgc])
#try to reculster clusters in clnset
system.time(subdim_graph<-create_snnk_graph(data[[i]][clus_assign[[i]] %in% clnset, ref_subs[[i]][9,]] ,k=30, metric='L1'))
system.time(res_sdg<-louvain_multiple_runs(subdim_graph$graph, num.run=5, time.lim=200))
res_sdg <- kmeans(data[[i]][clus_assign[[i]] %in% clnset, ref_subs[[i]][9,]], centers=3)$cluster
table(res_sdg)
subassign<-clus_assign[[i]]; subassign[clus_assign[[i]] %in% clnset]<-res_sdg
helper_match_evaluate_multiple(subassign, clus_truth[[i]])
helper_match_evaluate_multiple(res_sdg, clus_truth[[i]][clus_assign[[i]] %in% clnset])
ix<-sample(sum(clus_assign[[i]] %in% clnset),2000 )
pairs(data[[i]][clus_assign[[i]] %in% clnset, ref_subs[[i]][9,]][, ], pch='.', col= res_sdg )

