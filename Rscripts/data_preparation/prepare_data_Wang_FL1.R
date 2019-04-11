#########################################################################################
# R script to prepare benchmark data set Wang Fl1
# 
# This is a 40-dimensional mass cytometry (CyTOF) data set
#########################################################################################


# load packages

library(flowCore)  # from Bioconductor
library(magrittr)  # from CRAN




#################
### LOAD DATA ###
#################

# see above for link to download files
# one FCS file per manually gated cluster, per individual (H1 and H2)
# 32 surface markers (dimensions), 14 manually gated populations, 2 individuals (H1 and H2)


# FCS filenames
# "unassigned" cells are those where cluster labels are unavailable
setwd('/mnt/f/Brinkman group/current/Stepan/WangData')
for (j in c(1:6)){
name=c('FL1', 'FL2', 'FL3', 'rLN4','rLN5','rLN6')[j]

#command<-paste0('mkdir ', './WangDataPatient/',name)
#system(command)
PATIENT_NAME<-name


files <- list.files(paste0("./WangDataPatient/",PATIENT_NAME), pattern = "\\.fcs$", full.names = TRUE)

#files_to_process<- files[-c(grep(c("NonT"), files),  grep(c("Tfh"), files), grep(c("Treg"), files),  grep(c("Tex"), files))]
#files_to_process<- files[-c(grep(c("NonT"), files))]

files_to_process<- files



# cell population names

gsub(paste0(".*",PATIENT_NAME,"…"),"",files_to_process) %>% 
  gsub("\\..*", "", .) ->
  pop_names

pop_names

df_pop_names <- data.frame(label = 1:length(pop_names), population = pop_names)
df_pop_names


# column names (protein markers and others)

read.FCS(files_to_process[1], transformation = FALSE, truncate_max_range = FALSE) %>% 
  exprs %>% 
  colnames %>% 
  unname %>% 
  gsub("\\(.*", "", .) -> 
  col_names

col_names


# vector of labels for individual FL1

#indiv <- rep(NA, length(files_assigned))
#indiv[grep("_H1\\.fcs$", files_assigned)] <- 1
#indiv[grep("_H2\\.fcs$", files_assigned)] <- 2
#indiv

#indiv_unassigned <- c(1, 2)
#indiv_unassigned



# load FCS files and add cluster labels ("assigned" cells only)

data <- matrix(nrow = 0, ncol = length(col_names) + 1)

for (i in 1:length(files_to_process)) {
  data_i <- flowCore::exprs(flowCore::read.FCS(files_to_process[i], 
                                               transformation = FALSE, 
                                               truncate_max_range = FALSE))
  print(name)
  print(dim(data_i))
  if (dim(data_i)[1]==0) next
  colnames(data_i) <- col_names
  
  # cluster labels
  data_i <- cbind(data_i, label = rep(i, nrow(data_i)))
  print(dim(data_i))
  # labels for each individual

  
  data <- rbind(data, data_i)
}

head(data)
dim(data)  # 104,184 assigned cells, 32 dimensions (plus 9 other columns)
table(data[, "label"])  # 14 manually gated clusters

#remove 'empty' channels
cols<-c("Y89Di", "Cd112Di", "In115Di",  "Pr141Di", "Nd142Di", "Nd143Di", "Nd144Di", "Nd145Di" , "Nd146Di",  "Sm147Di" ,   "Nd148Di"   ,   "Sm149Di"    ,  "Nd150Di"   ,   "Eu151Di" ,  "Sm152Di"   ,   "Eu153Di"   ,   "Sm154Di"   ,   "Gd155Di"   ,   "Gd156Di"    ,     "Gd158Di" , "Tb159Di"   ,   "Gd160Di"  ,"Dy161Di"   ,   "Dy162Di"   ,   "Dy163Di"  ,    "Dy164Di"   ,  "Ho165Di" , "Er166Di" , "Er167Di",   "Er168Di"  , "Tm169Di"  ,   "Er170Di"   ,   "Yb171Di"   ,  "Yb172Di",   "Yb173Di"    ,  "Yb174Di"  ,    "Lu175Di"  ,   "Yb176Di")
data<-data[, c(cols,'label')]

#########################
### ARCSINH TRANSFORM ###
#########################

# arcsinh transform
# using scale factor 5 for CyTOF data (see Bendall et al. 2011, Supp. Fig. S2)

data_notransform <- data
asinh_scale <- 5

cols_to_scale <- 1:38
data[, cols_to_scale] <- asinh(data[, cols_to_scale] / asinh_scale)

summary(data)
#check for duplicates
zzz=data[data[,'label']==4,][!duplicated(data[data[,'label']==4,]),]
dim(zzz)
dim(data[data[,'label']==4,])
rm(zzz)

table( data[,'label'])
#idx<-sort(sample(1:nrow(data[data[,'label']==1 | data[,'label']==2 | data[,'label']==3 | data[,'label']==4,]),1000))
#corData=cor(t(data[idx,]))
#heatmap.2((corData), dendrogram='none', Rowv=FALSE, Colv=FALSE,trace='none')
###################
### EXPORT DATA ###
###################

# combine data frames for assigned and unassigned cells
# export cell population names
write.table(df_pop_names, file = paste0("./WangDataPatient/",PATIENT_NAME,"/population_names_",PATIENT_NAME,".txt"), quote = FALSE, sep = "\t", row.names = FALSE)
# save data files in TXT format
write.table(data, file = paste0("./WangDataPatient/",PATIENT_NAME,".txt"), quote = FALSE, sep = "\t", row.names = FALSE)
# save data files in FCS format
flowCore::write.FCS(flowCore::flowFrame(data), filename = paste0("./WangDataPatient/",PATIENT_NAME,".fcs"))

#stack data matrices
if (name=='FL1'){
  dataAll<-data
  dataAll<-cbind(data, rep(j, nrow(data)))
} else {
  dataTmp<-cbind(data, rep(j, nrow(data)))
  dataAll<-rbind(dataAll,dataTmp)
}

}
#remove duplicated
#organize labels in hyerarchical manner
#remove nonT
colnames(dataAll)[ncol(data)+1]<-'patient'
dim(dataAll)
dataAll<-as.data.frame(dataAll)
zzz<-dataAll[!duplicated(dataAll[,label="HM"]),]
table(zzz$label)
temp<-dataAll[dataAll$label %in% c(1,2,3,5,6,7,8,10,12,13), ]
temp2<-dataAll[dataAll$label %in% c(4,9,15), ]
table(temp2$label)
zzz<-rbind(rbind(temp,temp2), dataAll[dataAll$label ==11, ])
zzz<-zzz[!duplicated(zzz[,1:38]),]
dim(zzz)
plot(as.numeric(zzz$label), pch='.')
table(zzz$label)
dataAll<-zzz




dim(dataAll)
write.table(dataAll, file = paste0("./WangDataPatient/","AllPatients",".txt"), quote = FALSE, sep = "\t", row.names = FALSE)

flowCore::write.FCS(flowCore::flowFrame(dataAll), filename = paste0("./WangDataPatient/","AllPatients",".fcs"))

####################################################################
#check degree of 'clusterization', how clusters correlate with patients
#outlier assignment using subspaces as reference set 
######################################################################
# S. Grinek 26.07.17
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
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
library(MASS)
seed<-set.seed(12345)
#CALC_NAME = 'DELETE'
CALC_NAME<-'Louvain_L2_k30'
k=30

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_calculate_AMI.R')
source("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_multiple.R")
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_create_snnk_graph.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_multiple.R')

# subdimensional helpers
######################################################################
#Script to calculate combined probabilistic outlier score
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_convBernully.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_global_outliers.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_topPercentile.R')
######################################################################
#clustering helpers
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')

library(scales)
library(gplots)
dim(dataAll)
table(dataAll[,'label'])
dataAll=as.data.frame(dataAll)
#All subtypes of Tcells
dim(dataAll[ dataAll$label %in% 1:10,])
#exclude all non T-cells
data0 = dataAll[ dataAll$label!=14,]
dim(data0)
lbls0=data0$label
#clustering
lbls0[lbls0==15]<-14 

#create graph
#nn.idx<-nnind105[[i]];k=30
#if(i %in% c(1,3)){
#  g<-create_snnk_graph_from_subset(nn.idx, k, clust_clean, mc.cores=5)$graph
#} else {
data0<-read.table( file = paste0("./WangDataPatient/","AllPatients_data0",".txt"), header=T)
k=30
 print(system.time(res<-create_snnk_graph_vptree(as.matrix(data0[patient==2,1:40]),k, metric='L2', mc.cores=10)))
g0<-res$graph
 
print(system.time(cl0<- louvain_multiple_runs_par(g0, num.run=5, mc.cores=5)))
table(cl0)
 AMI(cl0,lbls0[] )
helper_match_evaluate_multiple(cl0,lbls0[patient==2])
helper_match_evaluate_multiple_MoCu(cl0,lbls0)

AMI(cl0,data0$patient)
helper_match_evaluate_multiple(cl0,data0$patient)
helper_match_evaluate_multiple_MoCu(cl0,data0$patient)

write.table(data0, file = paste0("./WangDataPatient/","AllPatients_data0",".txt"), row.names = FALSE, quote = FALSE, sep = "\t")

data0<-read.table( file = paste0("./WangDataPatient/","AllPatients_data0",".txt"), header=T)

x_test_vae<-read.table( file = paste0("./WangDataPatient/","x_test_ae001.txt"), header=F, numerals = "no.loss",stringsAsFactors=F, colClasses="numeric")

dim(x_test_vae)
x_test_vae[1,]

k=30
print(system.time(res1<-create_snnk_graph_vptree(x_test_vae ,k, metric='L2', mc.cores=10)))
g1<-res1$graph

print(system.time(cl1<- louvain_multiple_runs_par(g1, num.run=5, mc.cores=5)))
table(cl1)
AMI(cl1,lbls0 )
helper_match_evaluate_multiple(cl1,lbls0)
helper_match_evaluate_multiple_MoCu(cl1,lbls0)

AMI(cl1,data0$patient)
helper_match_evaluate_multiple(cl1,data0$patient)
helper_match_evaluate_multiple_MoCu(cl1,data0$patient)

helper_match_evaluate_multiple(cl0[lbls0!=14],lbls0[lbls0!=14])
helper_match_evaluate_multiple(cl1[lbls0!=14],lbls0[lbls0!=14])

helper_match_evaluate_multiple(cl0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11])
helper_match_evaluate_multiple(cl1[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11])
helper_match_evaluate_multiple(cl2[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11])

helper_match_evaluate_multiple_MoCu(cl0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11])
helper_match_evaluate_multiple_MoCu(cl1[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11])
helper_match_evaluate_multiple_MoCu(cl2[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9 & lbls0!=11])


helper_match_evaluate_multiple_MoCu(cl0[lbls0!=14 & lbls0!=4 & lbls0!=9],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9])
helper_match_evaluate_multiple_MoCu

x_test_vae<-read.table( file = paste0("./WangDataPatient/","Wang0_x_test_vae001_2nd_run.txt"), header=F, numerals = "no.loss",stringsAsFactors=F, colClasses="numeric")

dim(x_test_vae)
resAll2<-vector('list',length=6)
resAll0<-vector('list',length=6)
patient = data0$patient
for (i in 1:length(unique(patient))){
x_test_vae1<-x_test_vae[patient==i,]
#x_test_vae[x_test_vae<=0.4]<-0
k=45
print(system.time(res2<-create_snnk_graph_vptree(x_test_vae1 ,k, metric='L2', mc.cores=10)))
g2<-res2$graph

print(system.time(cl2<- louvain_multiple_runs_par(g2, num.run=5, mc.cores=5)))
table(cl2)
AMI(cl2,lbls0[patient==i] )
helper_match_evaluate_multiple(cl2,lbls0[patient==i])
helper_match_evaluate_multiple_MoCu(cl2,lbls0[patient==i])
resAll2[[i]]<-cl2


k=45
print(system.time(res<-create_snnk_graph_vptree(as.matrix(data0[patient==i,1:40]),k, metric='L2', mc.cores=10)))
g0<-res$graph

print(system.time(cl0<- louvain_multiple_runs_par(g0, num.run=5, mc.cores=5)))
table(cl0)
AMI(cl0,lbls0[patient==i] )
helper_match_evaluate_multiple(cl0,lbls0[patient==i])
helper_match_evaluate_multiple_MoCu(cl0,lbls0[patient==i])
resAll0[[i]]<-cl0
}
lapply(3, function(x) {print(helper_match_evaluate_multiple(resAll0[[x]],lbls0[patient==x]))
  print(helper_match_evaluate_multiple(resAll2[[x]],lbls0[patient==x]))
  print(table(lbls0[patient==x]))} )

lapply(3, function(x) {print(helper_match_evaluate_multiple(resAll0[[x]],lbls0[patient==x]))
  print(helper_match_evaluate_multiple(resAll2[[x]],lbls0[patient==x]))
  print(table(lbls0[patient==x]))} )


lapply(6, function(x) {
  print(helper_match_evaluate_multiple_MoCu(resAll0[[x]],lbls0[patient==x])$total_MoCu)
  print(helper_match_evaluate_multiple_MoCu(resAll2[[x]],lbls0[patient==x])$total_MoCu)
  print(table(lbls0[patient==x]))
  table(cl2) } )

lapply(3, function(x) {
  print(helper_match_evaluate_multiple_MoCu(resAll0[[x]][lbls0[patient==x]!=11],lbls0[patient==x & lbls0!=11])$total_MoCu)
  print(helper_match_evaluate_multiple_MoCu(resAll2[[x]][lbls0[patient==x]!=11],lbls0[patient==x & lbls0!=11])$total_MoCu)
  print(table(lbls0[patient==x]))
  table(cl2) } )





per_match_evaluate_multiple(cl2,data0$patient)
helper_match_evaluate_multiple_MoCu(cl2,data0$patient)

helper_match_evaluate_multiple(cl2[lbls0!=14],lbls0[lbls0!=14])
helper_match_evaluate_multiple(cl2[lbls0!=14],lbls0[lbls0!=14])

helper_match_evaluate_multiple(cl2[lbls0!=14 & lbls0!=4 & lbls0!=9],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9])
helper_match_evaluate_multiple(cl2[lbls0!=14 & lbls0!=4 & lbls0!=9],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9])

helper_match_evaluate_multiple_MoCu(cl2[lbls0!=14 & lbls0!=4 & lbls0!=9],lbls0[lbls0!=14 & lbls0!=4 & lbls0!=9])
helper_match_evaluate_multiple_MoCu

library(subspace)
sbcl1<-SubClu(x_test_vae, epsilon=20, minSupport=80)
clq1<-CLIQUE(x_test_vae, xi = 10, tau = 0.2)
kNNdistplot(x_test_vae, k = 40)
library(dbscan)
dbs1<-dbscan(x_test_vae, eps=0.5, minPts = 40, weights = NULL, borderPoints = F)
dbs2<-dbscan(data0, eps=0.3, minPts = 40, weights = NULL, borderPoints = F)


sbcl0<-SubClu(data0, epsilon=0.5, minSupport=20)


library(reticulate)
path_to_python <- "/anaconda/bin/python"
use_python(path_to_python)





