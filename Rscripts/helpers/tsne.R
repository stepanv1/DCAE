# create tsne 2d and 3d visualisation for all data sets
library(fpc)
library(cluster) 
library(Rtsne)
library(rgl)
library(gclus)
library(parallel)

seed<-set.seed(12345)

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
    Mosmann_rare = file.path(DATA_DIR, "Mosmann_rare.fcs") 
    #FlowCAP_ND   = file.path(DATA_DIR, "FlowCAP_ND.fcs"), 
    #FlowCAP_WNV  = file.path(DATA_DIR, "FlowCAP_WNV.fcs")
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

sapply(data, length)

sapply(data[!is_FlowCAP], dim)
sapply(data[is_FlowCAP], function(d) {
    sapply(d, function(d2) {
        dim(d2)
    })
})

#Remove cells without labels from data
#For now not done: subsampling for data sets with excessive runtime (> 12 hrs on server)

#ix_subsample <- 1:6

#for (i in ix_subsample){
#        cat(dim(data[[i]]))    
#        data[[i]]<-data[[i]][data[[i]][,'label'] != 'NaN', ]
#}



# indices of protein marker columns

marker_cols <- list(
    Levine_32dim = 5:36, 
    Levine_13dim = 1:13, 
    Samusik_01   = 9:47, 
    Samusik_all  = 9:47, 
    Nilsson_rare = c(5:7, 9:18), 
    Mosmann_rare = c(7:9, 11:21), 
    FlowCAP_ND   = 3:12, 
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

#save preprocessed data set  
save(data, file=paste0(DATA_DIR, '/all_sets.RData'))

#run tsne
system.time(res_tsne<-mclapply(1:6, function(x){
    tsne_out <- Rtsne(data[[x]], check_duplicates = F, max_iter = 6000)
    tsne_out3D <- Rtsne(data[[x]], dims=3, check_duplicates=F, max_iter = 6000)
    return(list('tsne_out'=tsne_out, 'tsne_out3D'=tsne_out3D))
}, mc.cores=6))
save(res_tsne, file=paste0(DATA_DIR, '/tsne.RData'))
#load(file=paste0(DATA_DIR, '/tsneALL.RData'))

#visualise tsne results
library(rgl)
i=4
plot(res_tsne[[i]]$tsne_out$Y, pch='.')
plot3d(res_tsne[[i]]$tsne_out3D$Y,  pch='.')


