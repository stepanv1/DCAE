#script to test outliers removal on Wang project data

#data, clus_assign, skewCut=1
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
#data, ref_subs, clus_assign, mc.cores=5
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_global_outliers.R')
#data,  ref_subset, k=30
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
#data, query, k, metric ='L2'
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_approx_distances.R')
#Helper based on igraph function to do Louvain hierarchical clusteringimport igraph
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_create_snnk_graph.R')

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

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_topPercentile.R')
#load Wang data
data0 = read.table('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/rln4Asin5Transform.txt', stringsAsFactors = F, header=T)
lblsT=data0[,39]
data=as.matrix(data0[,1:38])
data<-apply(data, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
N=nrow(data)

# 1. cluster
system.time(lblsC<-Louvain_multilevel(data, k=30))
levels=lblsC$levels
clus_assign<-unlist(lblsC$memberships[levels])
clus_assign <- clus_assign[-1]
num_clus<-length(unique(clus_assign))
helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))


#find subdimensions
rsClus<-helper_find_subdimensions(data, clus_assign, skewCut=1)
library(gplots)
heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))

LO <- vector("list", length = num_clus)  
system.time(
  for (i in 1:num_clus){
    print(i)
    rs<- clus_assign==i 
    LO[[i]]<-helper_local_outliersLOF2(data[,rsClus$subdim[i,]], ref_subset = rs, k=30)
  }
)


#GO<-helper_global_outliers_ApproxIID(data, rsClus$subdim, clus_assign, nbins=5, mc.cores=5)
#GO<-helper_global_outliers_Discrete(data, rsClus$subdim,  clus_assign, nbins=5, mc.cores=5)
#GO<-helper_global_outliers(data, rsClus$subdim,  clus_assign,  mc.cores=5)
#GO<-helper_global_outliers_Discrete_All_dims(data, rsClus$subdim, clus_assign, nbins=5, mc.cores=5)
GO<-helper_global_outliers_Uniform(data, rsClus$subdim, clus_assign, nbins=10, mc.cores=10)


GOm = Reduce(cbind, GO) 
LOm = Reduce(cbind, (lapply(LO, function(x) x$prob))) 
CO=as.matrix(GOm*LOm)
weights=unlist((table(clus_assign)/length(clus_assign)))
normCO<-apply(CO, 1, function(x) sum(weights*x))
score<-unlist(lapply(1:nrow(CO),  function(x) sum( (weights*CO[x,]/normCO[x])^2 ) ))
hist(score,500)

hist(score[clus_assign==4],500)
table(clus_assign)

cutoffProd=helper_topPercentile(0.05, score)$lim_cut


scoreLO<-unlist(lapply(1:N, function(x) LOm[x,clus_assign[x]]))
scoreGO<-unlist(lapply(1:N, function(x) GOm[x,clus_assign[x]]))

#compare to my plots
VAE<-read.table('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBatch10_VAE_noSigma.txt', 
                stringsAsFactors = F, header=F)

library(rgl)
rgl.open()
nbcol <- topo.colors(128)
nbcol <- nbcol[1:128]
plot3d(VAE[,1], VAE[,2], VAE[,3], col =nbcol[round(128*(score))], size=0.1)
plot3d(VAE[,1], VAE[,2], VAE[,3], col =nbcol[round(128*(scoreLO))], size=0.1)
plot3d(VAE[,1], VAE[,2], VAE[,3], col =nbcol[round(128*(scoreGO))], size=0.1)

#globally strongest outliers
cutoffGlob=helper_topPercentile(0.05, score)$lim_cut
#per cluster outliers
cutoffClus<-lapply(1:length(unique(clus_assign)), function(i) {
  helper_topPercentile(0.80,score[clus_assign==i])$lim_cut
  })
idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<cutoffClus[clus_assign[i]], F, T )))

outliers<- !idxOutlClus | score<cutoffGlob 
plot3d(VAE[,1], VAE[,2], VAE[,3], col = ifelse(outliers, "red", "black"), size=0.5)
plot3d(VAE[!outliers,1], VAE[!outliers,2], VAE[!outliers,3],, size=0.5)
open3d()
plot3d(VAE[,1], VAE[,2], VAE[,3], size=0.5)


#compare without outliers
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_approx_distances.R')
dataNew<-data[!outliers,]
system.time(lblsNew<-Louvain_multilevel(dataNew, k=30))
levelsNew=lblsNew$levels
clus_assignF<-unlist(lblsNew$memberships[levelsNew])[-1]
clus_assignNew<-unlist(lblsNew$memberships[levelsNew])
clus_assignNew <- clus_assignNew[-1]
num_clusNew<-length(unique(clus_assignNew))
tmp= (1:length(clus_assign))*NA
tmp[!outliers]<-clus_assignNew
clus_assignNew<-tmp
helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple(clus_assignNew, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple_SweightedN(clus_assign, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple_SweightedN(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])
helper_match_evaluate_multiple_MoCu(clus_assign, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple_MoCu(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])

#now compare identification of core clusters

core<- lblsT %in% c('CD4_CM', 'CD4_EM', 'CD4_EMRA', 'CD4_Naive',  'CD8_CM', 'CD8_EM',  'CD8_EMRA', 'CD8_Naive', 'NK',  'NKT ')
#cutoff=helper_topPercentile(0.50, score)$lim_cut
sum(core)
sum(core & !outliers)

helper_match_evaluate_multiple(clus_assign[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple(clus_assignNew[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple_SweightedN(clus_assign[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple_SweightedN(clus_assignNew[core & !outliers], as.numeric(as.factor(lblsT))[core & !outliers])
helper_match_evaluate_multiple_MoCu(clus_assign[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple_MoCu(clus_assignNew[core & !outliers], as.numeric(as.factor(lblsT))[core & !outliers])


#no reassign outliers
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
outAssign<-helper_assign_outliers(dataNew, data[outliers, ], clus_assignF)
allAssing<-(1:length(clus_assign))*NA
allAssing[outliers]<-outAssign
allAssing[!outliers]<-clus_assignF
table(allAssing)

helper_match_evaluate_multiple(clus_assign[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple(allAssing[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple(allAssing, as.numeric(as.factor(lblsT)))




helper_match_evaluate_multiple_SweightedN(clus_assign[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple_SweightedN(allAssing[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple_MoCu(clus_assign[core], as.numeric(as.factor(lblsT))[core])
helper_match_evaluate_multiple_MoCu(allAssing[core ], as.numeric(as.factor(lblsT))[core ])


open3d()
View(CO[score<cutoff,])
corProb=cor(clus_assign, CO, method='s')
nbcol <- topo.colors(15)
nbcol <- nbcol[1:15]
plot3d(VAE[,1], VAE[,2], VAE[,3], col =nbcol[as.numeric(as.factor(lblsT))], size=0.1, main='manual')
open3d()
nc=length(unique(clus_assign))
nbcol <- topo.colors(nc)
nbcol <- nbcol[1:nc]
plot3d(VAE[,1], VAE[,2], VAE[,3], col =nbcol[clus_assign], size=0.1,main='initial clustering')
open3d()
nc=length(unique(allAssing))
nbcol <- topo.colors(nc)
nbcol <- nbcol[1:nc]
plot3d(VAE[,1], VAE[,2], VAE[,3], col =nbcol[clus_assign], size=0.1,main='initial clustering')

open3d()
nc=length(unique(allAssing))

clus_assignNew[outliers]<-0
nbcol <- topo.colors(nc)
nbcol <- nbcol[1:nc]
plot3d(VAE[,1], VAE[,2], VAE[,3], col =ifelse(clus_assignNew!=0, nbcol[clus_assignNew], 'white'), size=0.1,main='initial clustering')
plot3d(VAE[!outliers,1], VAE[!outliers,2], VAE[!outliers,3], col =ifelse(clus_assignNew[!outliers]!=0, nbcol[clus_assignNew[!outliers]], 'white'), size=0.1,main='initial clustering')





