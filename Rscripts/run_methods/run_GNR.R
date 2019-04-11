# Run basic growing neural gas algorithm, GNR with utility function.
# Utility function makes algorith parametricless
#Wrning, installation of gmum.r might be tricky, github version did not wotk for me,
#used CRAN. Help and basic methods for evaluation and accessor methods are not found in CRAN package. Perhaps need to check github version

library(fpc)
library(cluster) 
library(rgl)
library(gclus)

library(flowCore)
library(parallel)
library(gmum.r)
library(FNN)
library(igraph)
chunks2<- function(n, nChunks) {split(1:n, ceiling(seq_along(1:n)/(n/nChunks)))}
seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_ID<-'1000units2MInter'


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

#marker_names<- lapply(data, colnames)
#save(marker_names,  file=paste0(DATA_DIR, '/marker_names.RData'))
###################################################
### Run GNR: automatic number of clusters ###
###################################################

# run phenoGraph with automatic selection of number of clusters

# note some of the FlowCAP_ND data sets give errors; skip these

out <- vector("list", length(data))
names(out) <- names(data)  
maxnodes=1000
#only run on Levin32 mass cytometry data
#system.time(for (i in 1) {
system.time(out<-mclapply(1:4, function(i){
  if (!is_FlowCAP[i]){
      scl_coord<-scale(data[[i]], center = TRUE, scale = TRUE)
      g  <- GNG(scl_coord, max.nodes=maxnodes,  max.iter = 2000000, verbosity=0, max.edge.age=200, lambda=200)
      ig<-convertToIGraph(g)
      cl_InfoMap<-cluster_infomap(ig, e.weights = NULL, v.weights = NULL, nb.trials = 100,
                                  modularity = TRUE)
      membershipInfoMap<-membership(cl_InfoMap)
      surv_nodes<-length(unique(clustering(g)))
      nodecoordsL<-lapply(1:surv_nodes, function(x) node(g, x)$pos)
      nodecoords<-do.call(rbind, nodecoordsL)
      nodeAssign<-unlist(mclapply(chunks2(nrow(scl_coord),60), function(x) get.knnx(nodecoords, scl_coord[x,], k=1, algorithm="kd_tree")$nn.index, mc.cores=1, mc.preschedule=F), recursive=T)
      names(membershipInfoMap) = 1:(length(membershipInfoMap)) 
      clus=unlist(mclapply(nodeAssign, function(i) membershipInfoMap[as.character(i)], mc.cores=6))
      nodecoordsL<-lapply(1:surv_nodes, function(x) node(g, x)$pos)
      nodecoords<-do.call(rbind, nodecoordsL)
      nodeError<-lapply(1:numberNodes(g), function(x) node(g,x)$error)
      nodePos<-lapply(1:numberNodes(g), function(x) node(g,x)$pos)
      nodeNeighbours<-lapply(1:numberNodes(g), function(x) node(g,x)$neighbours)
      out_i=list('clus'=clus, 'igraph'=ig, 'error'= errorStatistics(g), 'nodecoords'=nodecoords, 'nodeError' = nodeError, 'nodePos'=nodePos, 'nodeNeighbours'=nodeNeighbours, 'cl_InfoMap' = cl_InfoMap )  
  } else {
    # FlowCAP data sets: run clustering algorithm separately for each sample
    for (j in 1:length(data[[i]])) {
      # some of the FlowCAP_ND data sets give errors; skip these
      if ((i == 7) & (j %in% c(7, 9, 25))) next
      out_i[[j]] <- GNG(data[[i]][[j]], max.nodes=maxnodes,  max.iter = 1000, k=1.3, verbosity=0)
    }
  }
  return(out_i)
}
, mc.cores=3))

plot(out[[1]]$error)
summary(out[[4]]$ig)

names(out)=names(data)[1:4]

# extract cluster labels
#clus <- vector("list", length(data))
clus <- vector("list", 6)
names(clus) <- names(data)[1:6]

#for (i in 1) {
for (i in 1:4) {
  if (!is_FlowCAP[i]) {
    clus[[i]] <- out[[i]]$clus
    
  } else {
    # FlowCAP data sets
    clus_list_i <- vector("list", length(data[[i]]))
    names(clus_list_i) <- names(data[[i]])
    for (j in 1:length(data[[i]])) {
      if (!is.null(out[[i]][[j]])) {
        clus_list_i[[j]] <- out[[i]][[j]]$clus
      }
    }
    
    # convert FlowCAP cluster labels into format "sample_number"_"cluster_number"
    # e.g. sample 1, cluster 3 -> cluster label 1_3
    names_i <- rep(names(clus_list_i), times = sapply(clus_list_i, length))
    clus_collapse_i <- unlist(clus_list_i, use.names = FALSE)
    clus[[i]] <- paste(names_i, clus_collapse_i, sep = "_")
  }
}

sapply(clus, length)

# cluster sizes and number of clusters
# (for FlowCAP data sets, total no. of clusters = no. samples * no. clusters per sample)
table(clus[[1]])
sapply(clus, function(cl) length(table(cl)))

# save cluster labels
files_labels <- paste0(names(clus), ".txt")
dirname=paste0("../../results/auto/GNR/")
system(paste('mkdir ',dirname))
for (i in 1:length(files_labels)){
  res_i <- data.frame(label = clus[[i]])
  write.table(res_i, file = paste0(dirname,"/", CALC_ID, files_labels[i]), row.names = FALSE, quote = FALSE, sep = "\t")
}
save(out, file=paste0(dirname, "/", CALC_ID, "GNR_labels_graphs.RData"))

z=2
imr<-cluster_infomap(out[[z]]$igraph)
plot(imr, out[[z]]$igraph)

hist(unlist(out[[3]]$nodeError))
plot(out[[z]]$error)
length(out[[z]]$nodeError)
# save runtimes
#runtimes <- lapply(runtimes, function(r) r["elapsed"])
#runtimes <- t(as.data.frame(runtimes, row.names = "runtime"))

#write.table(runtimes, file = "../../results/auto/runtimes/runtime_phenoGraph.txt", 
#quote = FALSE, sep = "\t")

# save session information
#sink(file = "../../results/auto/session_info/session_info_phenoGraph.txt")
#print(sessionInfo())
#sink()



library(RANN)
t1 <- system.time(neighborMatrix <- find_neighbors(data[[4]], k=40+1)[,-1])
cat("DONE ~",t1[3],"s\n", " Compute jaccard coefficient between nearest-neighbor sets...")
t2 <- system.time(links <- cytofkit:::jaccard_coeff(neighborMatrix))

t1 <- system.time(neighborMatrix01 <- find_neighbors(data[[3]], k=40+1)[,-1])
cat("DONE ~",t1[3],"s\n", " Compute jaccard coefficient between nearest-neighbor sets...")
t2 <- system.time(links01 <- cytofkit:::jaccard_coeff(neighborMatrix01))

dirname=paste0("../../results/auto/phenoGraph/", CALC_NAME)
system(paste('mkdir ','dirname'))
save(neighborMatrix, neighborMatrix01, file=paste0(dirname, '/SamusikNeighborsK40.RData'))

hist(links[,3], 500)
hist(links01[,3], 500)







