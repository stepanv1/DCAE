# Deep belief network
# in flow cytometry
# the goal is to learn the netwok on the mixture
# of labeled and unlabeled data
# How to run PhenoGraph

# Not available in R, so follow steps below instead:

# Requires: Matlab, Statistics Toolbox

# 1. Open "cyt" GUI from Matlab by typing "cyt" in command window.
# 2. Load FCS file by clicking on green "plus" symbol under Gates.
# 3. Select all protein markers under Channels (note: in cyt version 3.0, marker names 
# may show as blank -- in this case, you will need to check the original column names in
# the FCS file, and select markers by counting).
# 4. Right-click and select PhenoGraph.
# 5. Select "Run on individual gates" ("Run on all gates together" may give an error).
# This requires data from all cell populations to be in the same FCS file. If you have
# multiple FCS files, concatenate them together first.
# 6. Set parameters and click "Cluster" to run.
# 7. Save output as a new FCS file by clicking on the save icon under Gates.

library(cytofkit) 
library(fpc)
library(cluster) 
library(Rtsne)
library(rgl)
library(gclus)

library(flowCore)
library(parallel)

seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_NAME<-'deepDBN'
kPnenoGraph=40
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
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
save(marker_names,  file=paste0(DATA_DIR, '/marker_names.RData'))


library(darch)
###################################################
### Run darch: deep belief method ####################
###################################################


#remove NaNs
dataset<-as.data.frame(data$FlowCAP_ND[[1]])
unassigned <- is.na(dataset[,'label'])
dataset <- dataset[!unassigned,]
dataset[, !(colnames(dataset) %in% 'label')]<-tanh(scale(dataset[, !(colnames(dataset) %in% 'label')], center=T))
dataset[,'label']<-as.factor(dataset[,'label'])

set.seed(12345)
IDX<-sample(1:nrow(dataset) , 40000)
dataset.train<-dataset[IDX,]
dataset.valid<-dataset[!(1:nrow(dataset) %in% IDX),]

darch <- darch(x0=dataset.train[,1:10, ], x=dataset.train[,1:10, ], y=dataset.train[,1:11, ], rbm.numEpochs=50 )  
darch=trainRBM(rbm, dataset[, 1:10], numEpochs = 1, numCD = 1, shuffleTrainData = T,
               ...)


system.time(model4 <- 
              h2o.deeplearning(x = 1:10,  # column numbers for predictors
                               y = 11,   # column number for label
                               training_frame = FlowCAP1.hex.Train, # data in H2O format
                               validation_frame = FlowCAP1.hex.Valid,
                               activation = 'RectifierWithDropout', # or 'Tanh'
                               input_dropout_ratio = 0, # % of inputs dropout
                               hidden_dropout_ratios = c(0.5, 0.5, 0.5), # % for nodes dropout
                               hidden = c(1000, 1000, 1000), # three layers of 50 nodes
                               epochs = 20000,
                               score_training_samples=1000, 
                               score_validation_samples=1000,
                               balance_classes=TRUE,
                               score_validation_sampling="Stratified",
                               #classification = T,
                               classification_stop=0)) # max. no. of epochs

system.time(model2 <- 
              h2o.deeplearning(x = 1:10,  # column numbers for predictors
                               y = 11,   # column number for label
                               training_frame = FlowCAP1.hex.Train, # data in H2O format
                               validation_frame = FlowCAP1.hex.Valid,
                               activation  = "Tanh",
                               input_dropout_ratio = 0.1, # % of inputs dropout
                               hidden = c(20, 20, 7), adaptive_rate =F,
                               momentum_start =  0.5, momentum_stable = 0.9, l1 = 1e-5,
                               epochs = 1800,   loss = "Quadratic")) 
plot(model2)
#momentum_start
#=  0.5, momentum_stable = 0.9, 

#prepare test set

dataset.test<-as.data.frame(data$FlowCAP_ND[[3]])
unassigned <- is.na(dataset.test[,'label'])
dataset.test <- dataset.test[!unassigned,]
dataset.test[,!(colnames(dataset.test) %in% 'label')]<-tanh(scale(dataset.test[,!(colnames(dataset.test) %in% 'label')], center=T))

FlowCAP1.hex.Test = as.h2o(dataset.test[,1:10])
system.time(predictions <- h2o.predict(model2, FlowCAP1.hex.Test))

predictions_df <- as.data.frame(predictions)

AMI(as.integer(predictions_df[,1]), as.integer(dataset.test[,11]), mc.cores=1)