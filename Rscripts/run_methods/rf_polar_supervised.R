#experiment with random forest in transformed coordinates (cartesian - polar - cartesian)
library(cytofkit) 
library(fpc)
library(cluster) 
library(Rtsne)
library(rgl)
library(gclus)
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_cluster_plot.R')
library(flowCore)
library(parallel)

library(SphericalCubature)
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
MANUAL_PHENOGRAPH <- "../../results/manual/phenoGraph"
RES_DIR_PHENOGRAPH <- "../../results/auto/PhenoGraph"
CALC_NAME="KdependencySamusik_all"
DATA_DIR <- "../../benchmark_data_sets"
RES_DIR <- "../../results/auto/"

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_S^w_N.R")
source("../helpers/helper_match_evaluate_greedy.R")

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
  Samusik_all  = file.path(DATA_DIR, "Samusik_all.fcs"), 
  Nilsson_rare = file.path(DATA_DIR, "Nilsson_rare.fcs"), 
  Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs"), 
  FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
  FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
)


# FlowCAP data sets are treated separately since they require clustering algorithms to be
# run individually for each sample

is_FlowCAP <- c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE)

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
head(data[[8]][[1]])

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

# Samusik labels and data
####################################################
### load truth (manual gating population labels) ###
####################################################

# files with true population labels (subsampled labels if subsampling was required for
# this method; see parameters spreadsheet)

files_truth <- list(
  Samusik_all = file.path(DATA_DIR, "Samusik_all.fcs") 
)

# extract true population labels

clus_truth <- vector("list", length(files_truth))
names(clus_truth) <- names(files_truth)

for (i in 1:length(clus_truth)) {
  data_truth_i <- flowCore::exprs(flowCore::read.FCS(files_truth[[i]], transformation = FALSE, truncate_max_range = FALSE))
  clus_truth[[i]] <- data_truth_i[, "label"]
}

sapply(clus_truth, length)

# cluster sizes and number of clusters
# (for data sets with single rare population: 1 = rare population of interest, 0 = all others)

tbl_truth <- lapply(clus_truth, table)

tbl_truth
sapply(tbl_truth, length)

# store named objects (for other scripts)

unassigned <- is.na(clus_truth[[1]])
clus_truth[[1]] <- clus_truth[[1]][!unassigned]
data<-data_truth_i[!unassigned, ]
dataset<-as.data.frame(cbind(data[, c(9:47, 54)]))
cn<-colnames(dataset[1:39, ])









#remove NaNs
#dataset<-as.data.frame(data$FlowCAP_ND[[1]])

dataset[, !(colnames(dataset) %in% 'label')]<-tanh(scale(dataset[, !(colnames(dataset) %in% 'label')], center=T))
dataset[,'label']<-as.factor(dataset[,'label'])

helper_cluster_plot(cells=c(1,3), cl=dataset[,40],   data=dataset[,1:39],
                    popnames=as.character(1:24), Nmark=9, markers=NULL)


library(vioplot)
for (i in as.character(c(1:24))){
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




#transform to hypersphere
dataset_polar<-rect2polar(t(dataset[, 1:39]))
dataset_rect<-polar2rect(r=rep(1, nrow(dataset)), phi=as.matrix(dataset_polar$phi))
dataset<-as.data.frame(cbind(t(dataset_rect), dataset[ , 'label']))
colnames(dataset)[40]<-'label'
dataset[,'label']<-as.factor(dataset[,'label'])
colnames(dataset)[1:39]<-cn[1:39]

helper_cluster_plot(cells=c(1,2,3), cl=dataset[,40],   data=dataset[,1:39],
                    popnames=as.character(1:24), Nmark=9, markers=NULL)



set.seed(12345)
IDX<-sample(1:nrow(dataset) , 300000)
dataset.train<-dataset[IDX,]
dataset.valid<-dataset[!(1:nrow(dataset) %in% IDX),]


######################################################################################
#check with random forests
library(ranger)
rg.nd1 <- ranger(label ~ ., data = dataset.train, write.forest = T, num.threads = 10)
save(rg.nd1, file=paste0(RES_DIR,'RandomForestSamusik_sphere.RData'))
#Samusik set


unassigned <- is.na(dataset.valid[,'label'])
dataset.valid <- dataset.valid[!unassigned,]
#dataset.test[,!(colnames(dataset.test) %in% 'label')]<-tanh(scale(dataset.test[,!(colnames(dataset.test) %in% 'label')], center=T))

dataset_polar.valid<-rect2polar(t(dataset.valid[, 1:39]))
dataset_rect.valid<-polar2rect(r=rep(1, nrow(dataset.valid)), phi=as.matrix(dataset_polar.valid$phi))
dataset.valid<-as.data.frame(cbind(t(dataset_rect.valid), dataset.valid[ , 'label']))
colnames(dataset.valid)[40]<-'label'
dataset.valid[,'label']<-as.factor(dataset.valid[,'label'])
colnames(dataset.valid)[1:39]<-cn[1:39]


rg.pred <- predict(rg.nd1, dat = dataset.valid,num.threads = 5)
table(as.integer(rg.pred$predictions))
table(as.integer(dataset.valid[,'label']))


AMI(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]), mc.cores=1)
helper_match_evaluate_multiple(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]))$mean_F1
helper_match_evaluate_multiple_SweightedN(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]))$total_SweightedN
helper_match_evaluate_greedy(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]))$mean_F1


#AMI(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]), mc.cores=1)
#mi=  2.273518 nmi=  0.9962643 emi=  0.001196327 h1=  2.280488 h2=  2.282043 
#ami=  0.9962624 
#[1] 0.9962624
#> helper_match_evaluate_multiple(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]))$mean_F1
#[1] 0.9576236
#> helper_match_evaluate_multiple_SweightedN(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]))$total_SweightedN
#1778 11644 12349 15276 141454 1200 1845 122311 1178 12534 112 120555 147667 18918 134364 11272 1433 12194 1514 #11412 13538 13230 111040 1668 1[1] 0.995061
#> helper_match_evaluate_greedy(as.integer(rg.pred$predictions), as.integer(dataset.valid[,40]))$mean_F1
#[1] 0.9989482





#set ND
f1<-rep(0,29); fg<-rep(0,29)
tbl_lst <-list()
for (i in 2:30){
  dataset.test<-as.data.frame(data$FlowCAP_ND[[i]])
  unassigned <- is.na(dataset.test[,'label'])
  dataset.test <- dataset.test[!unassigned,]
  dataset.test[,!(colnames(dataset.test) %in% 'label')]<-tanh(scale(dataset.test[,!(colnames(dataset.test) %in% 'label')], center=T))
  
  dataset_polar.test<-rect2polar(t(dataset.test[, 1:10]))
  dataset_rect.test<-polar2rect(r=rep(1, nrow(dataset.test)), phi=as.matrix(dataset_polar.test$phi))
  dataset.test<-as.data.frame(cbind(t(dataset_rect.test), dataset.test[ , 'label']))
  colnames(dataset.test)[11]<-'label'
  dataset.test[,'label']<-as.factor(dataset.test[,'label'])
  colnames(dataset.test)[1:10]<-colnames(as.data.frame(data$FlowCAP_ND[[1]])[,1:10])

  
  rg.pred <- predict(rg.nd1, dat = dataset.test)
  tbl_lst[[length(tbl_lst)+1]]<-c(table(as.integer(rg.pred$predictions)), table(as.integer(dataset.test[,11])))
  #table(rg.pred$predictions)
  #table(as.integer(dataset.test[,11]))
  
  #AMI(as.integer(rg.pred$predictions), as.integer(dataset.test[,11]), mc.cores=1)
  f1[i]<-helper_match_evaluate_multiple(as.integer(rg.pred$predictions), as.integer(dataset.test[,11]))$mean_F1
  helper_match_evaluate_multiple_SweightedN(as.integer(rg.pred$predictions), as.integer(dataset.test[,11]))$total_SweightedN
  fg[i]<-helper_match_evaluate_greedy(as.integer(rg.pred$predictions), as.integer(dataset.test[,11]))$mean_F1
}
mean(f1)
sd(f1)
mean(fg)
sd(fg)




