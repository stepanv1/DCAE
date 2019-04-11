#script to test outliers removal on Wang project data
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
library(gplots)
rr<-function(x){as.numeric(as.factor(x))}
#data, clus_assign, skewCut=1
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
#data, ref_subs, clus_assign, mc.cores=5
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_global_outliers.R')
#data,  ref_subset, k=30
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
#data, query, k, metric ='L2'
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_vptree_distances.R')
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
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_merge_signatures.R')

#source_dir = "/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient"
#file_list = glob.glob(source_dir + '/*.txt')


#load Art data
CALC_ID='Levin32realVignette'
#load Art data
DATA_DIR <- "../../benchmark_data_sets"
#data0 <- read.table('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/rln4Asin5Transform.txt', stringsAsFactors = F, header=T)
data0<-read.table(file = "/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/benchmark_data_sets/Levine_32dim.txt", 
                    sep = "\t", stringsAsFactors = F, header=T)
lblsT=data0[,40]
aFrame<-as.matrix(data0[,5:36])
aFrame[aFrame<0]<-0

# Small subset for the vignette
aFrame=aFrame[lblsT %in% 1:10, ]
lblsT=lblsT[lblsT %in% 1:10 ]

table(lblsT)
data<-apply(aFrame, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
dim(data)
N=nrow(data)


# 1. cluster

print(system.time({g<-create_snnk_graph_approx(data, k=30, metric='L2', mc.cores=10);
clus_assign<-louvain_multiple_runs_par(g$graph, num.run=5, mc.cores=5)}))

clus_assign0 <- clus_assign
#clus_assign <- clus_assign0
helper_match_evaluate_multiple(clus_assign, rr(lblsT))

# pick cells assigned to populations 
core<- !is.na(lblsT)
#cutoff=helper_topPercentile(0.50, score)$lim_cut
sum(core)


helper_match_evaluate_multiple(rr(clus_assign[core]), rr(lblsT[core]))
helper_match_evaluate_multiple_SweightedN(rr(clus_assign[core]),rr(lblsT[core]))
AMI(rr(rr(clus_assign[core])),rr(lblsT[core]))


num_clus<-length(unique(clus_assign))
# 2. find relevant subdimensions
rsClus<-helper_find_subdimensions(data, clus_assign, skewCut=1)
heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))
table(clus_assign)
table(lblsT, clus_assign)

# 3. merging identical signatures
system.time(clus_assignM<-helper_merge_signatures(rsClus$subdim, clus_assign,  data,  mc.cores=10))
rsClus<-helper_find_subdimensions(data, clus_assignM, skewCut=1)
heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))
table(clus_assignM)
num_clus<-length(table(clus_assignM))

helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple(rr(clus_assignM[core]),rr(lblsT[core]))


# 4. Assign Outlierness Scores
GO<-helper_global_outliers_Continuous_Sums(data, rsClus$subdim, clus_assignM, nbins=nbins, mc.cores=20)
print(system.time(LO<-mclapply(1:num_clus, function(i)  {rs<- clus_assignM==i; helper_local_outliersLOF2(data[,rsClus$subdim[i,]], ref_subset = rs, k=30)}, mc.cores=10)))
#save(GO,LO, file=paste0(DATA_DIR, '/artLevin32GOLO.RData'))
#load(file=paste0(DATA_DIR, '/artSamGOLO.RData'))

GOm = Reduce(cbind, GO) 
LOm = Reduce(cbind, (lapply(LO, function(x) x$prob))) 
# Create matrix with product of scores
CO=as.matrix(GOm*LOm)
weights=unlist((table(clus_assignM)/length(clus_assignM)))
#weights=1


COlist<-mclapply(1:length(clus_assignM), 
                 function(x)  unlist(lapply(1:num_clus, function(y) ifelse(CO[x,y]<0.5 & clus_assignM[x]!=y, 0, CO[x,y] ) )), mc.cores=10) 
CO <- do.call(rbind, COlist)

normCO<-apply(CO, 1, function(x) sum(weights*x))
# normalise score
score<-unlist(lapply(1:nrow(CO),  function(x) sum( (weights*CO[x,]/normCO[x])^2 ) ))
IDX<-apply(CO, 1, function(x) all(x==0))
score[IDX]<-0
hist(score,500)

# assign to outliers
idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<1, F, T )))
#idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<0.6, F, T )))

#idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<cutoffClus[clus_assign[i]], F, T )))


outliers<- !idxOutlClus 
#source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_approx_distances.R')

# 5. Cluster cleaned data set 
dataNew<-data[!outliers,]
dim(dataNew)
print(system.time({g<-create_snnk_graph_approx(dataNew, k=30, metric='L2', mc.cores=10);
lblsNew<-louvain_multiple_runs_par(g$graph, num.run=5, mc.cores=5)}))
rsClus<-helper_find_subdimensions(data[!outliers,], lblsNew, skewCut=1)
#merge clusters with identical signatures
lblsNew<-helper_merge_signatures(rsClus$subdim, lblsNew,data,  mc.cores=10)
clus_assignF<-unlist(lblsNew)
clus_assignNew<-unlist(clus_assignF)
num_clusNew<-length(unique(clus_assignNew))
tmp= (1:length(clus_assign))*NA
tmp[!outliers]<-clus_assignNew
clus_assignNew<-tmp


table(clus_assignNew)
#clus_assignNew<-lblsNew

helper_match_evaluate_multiple(rr(clus_assign[core]),rr(lblsT[core]))
helper_match_evaluate_multiple(rr(clus_assignM[core]),rr(lblsT[core]))
helper_match_evaluate_multiple(clus_assignNew[core],rr(lblsT[core]))



#6. now reassign outliers

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
system.time(outAssign<-helper_assign_outliers(dataNew, data[outliers, ], clus_assignF))
allAssing<-(1:length(clus_assign))*NA
allAssing[outliers]<-outAssign
allAssing[!outliers]<-clus_assignF

rsClus<-helper_find_subdimensions(data, allAssing, skewCut=1)
#merging identical signatures
allAssing<-helper_merge_signatures(rsClus$subdim, allAssing, data,  mc.cores=10)
table(allAssing)



helper_match_evaluate_multiple(rr(clus_assign[core]),rr(lblsT[core]))
helper_match_evaluate_multiple(rr(clus_assignM[core]),rr(lblsT[core]))
helper_match_evaluate_multiple(rr(allAssing[core]),rr(lblsT[core]))




helper_match_evaluate_multiple_SweightedN(rr(clus_assign[core]),rr(lblsT[core]))[9:10]
helper_match_evaluate_multiple_SweightedN(rr(clus_assignM[core]),rr(lblsT[core]))[9:10]
helper_match_evaluate_multiple_SweightedN(rr(allAssing[core]),rr(lblsT[core]))[9:10]
helper_match_evaluate_multiple_MoCu(rr(clus_assign[core]),rr(lblsT[core]))[8:9]
helper_match_evaluate_multiple_MoCu(rr(clus_assignM[core]),rr(lblsT[core]))[8:9]
helper_match_evaluate_multiple_MoCu(rr(allAssing[core]),rr(lblsT[core]))[8:9]
AMI(rr(clus_assign[core]),rr(lblsT[core]))
AMI(rr(clus_assignM[core]), rr(lblsT[core]))
AMI(rr(allAssing[core]),rr(lblsT[core]))



#Iterate
#################################################################################################################
#################################################################################################################
setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
iterations<-50
ScoreLst<-vector("list", iterations)
Assign<-vector("list", iterations)
ScoreLst[[1]]<-score
Assign[[1]]<-allAssing
print(system.time(for(iteration in 2:iterations){
  print(paste0('iteration ', iteration))
  clus_assign<-allAssing
  rsClus<-helper_find_subdimensions(data, clus_assign, skewCut=1)
  heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))
  print('Local outliers computation..')
  num_clus<-length(table(clus_assign))
  print(system.time(LO<-mclapply(1:num_clus, function(i)  {rs<- clus_assign==i; helper_local_outliersLOF2(data[,rsClus$subdim[i,]], ref_subset = rs, k=30)}, mc.cores=10)))
  #Problem p-values are inflated for bigger clusters
  #library(MASS)
  
  #GO<-helper_global_outliers_Discrete_Equidist_Bins(data, rsClus$subdim, clus_assign, nbins=nbins, mc.cores=5)
  GO<-helper_global_outliers_Continuous_Sums(data, rsClus$subdim, clus_assign, nbins=nbins, mc.cores=num_clus)
  resOut<-c(GO,LO)
  #save(GO,LO, file=paste0(DATA_DIR, '/artLevin32GOLO.RData'))
  #load(file=paste0(DATA_DIR, '/artLevin32GOLO.RData'))
  
  GOm = Reduce(cbind, GO) 
  LOm = Reduce(cbind, (lapply(LO, function(x) x$prob))) 
  CO=as.matrix(GOm*LOm)
  #weights=unlist((table(clus_assign)/length(clus_assign)))
  weights=unlist((table(clus_assign)/length(clus_assign)))
  #weights=1
  
  #not nulify 
  COlist<-mclapply(1:length(clus_assignM), 
                   function(x)  unlist(lapply(1:num_clus, function(y) ifelse(CO[x,y]<0.5 & clus_assign[x]!=y, 0, CO[x,y] ) )), mc.cores=10) 
  CO <- do.call(rbind, COlist)
  
  normCO<-apply(CO, 1, function(x) sum(weights*x))
  score<-unlist(lapply(1:nrow(CO),  function(x) sum( (weights*CO[x,]/normCO[x])^2 ) ))
  IDX<-apply(CO, 1, function(x) all(x==0))
  score[IDX]<-0
  #normLOm<-apply(LOm, 1, function(x) sum(weights*x))
  #score<-unlist(lapply(1:nrow(LOm),  function(x) sum( (weights*LOm[x,]/normLOm[x])^2 ) ))
  #CO_2<-t(apply(CO, 1, function(x) if(sum(x>0)==1){x}else{x*(x>0.05)}))
  #normCO_2<-apply(CO_2, 1, function(x) sum(weights*x))
  #score<-unlist(lapply(1:nrow(CO_2),  function(x) sum( (weights*CO_2[x,]/normCO_2[x])^2 ) ))
  #score[is.na(score)]<-0
  
  
  
  ScoreLst[[iteration]]<-score
  

  
  idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<1, F, T )))
  
  #idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<cutoffClus[clus_assign[i]], F, T )))
  
  
  outliers<- !idxOutlClus 
  #hist(score,500)
  dataNew<-data[!outliers,]
  dim(dataNew)
  print(system.time({g<-create_snnk_graph_approx(dataNew, k=30, metric='L2', mc.cores=10);
  lblsNew<-louvain_multiple_runs_par(g$graph, num.run=5, mc.cores=5)}))
  rsClus<-helper_find_subdimensions(data[!outliers,], lblsNew, skewCut=1)
  lblsNew<-helper_merge_signatures(rsClus$subdim, lblsNew, data,  mc.cores=10)
  clus_assignF<-unlist(lblsNew)
  clus_assignNew<-unlist(clus_assignF)
  num_clusNew<-length(unique(clus_assignNew))
  tmp= (1:length(clus_assign))*NA
  tmp[!outliers]<-clus_assignNew
  clus_assignNew<-tmp
  #merge again 
  table(clus_assignNew)
  #clus_assignNew<-lblsNew
  
  helper_match_evaluate_multiple(rr(clus_assign[core]),rr(lblsT[core]))
  helper_match_evaluate_multiple(clus_assignNew, as.numeric(as.factor(lblsT)))
  helper_match_evaluate_multiple_SweightedN(rr(clus_assign[core]),rr(lblsT[core]))[9:10]
  helper_match_evaluate_multiple_SweightedN(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])[9:10]
  helper_match_evaluate_multiple_MoCu(rr(clus_assign[core]),rr(lblsT[core]))[8:9]
  helper_match_evaluate_multiple_MoCu(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])[8:9]
  
  
  
  ########################################################################################################################################
  #now reassign outliers
  source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
  system.time(outAssign<-helper_assign_outliers(dataNew, data[outliers, ], clus_assignF))
  allAssing<-(1:length(clus_assign))*NA
  allAssing[outliers]<-outAssign
  allAssing[!outliers]<-clus_assignF
  
  rsClus<-helper_find_subdimensions(data, allAssing, skewCut=1)
  #merging identical signatures
  allAssing<-helper_merge_signatures(rsClus$subdim, allAssing, data,  mc.cores=10)
  heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))
  table(allAssing)
  
  Assign[[iteration]]<-allAssing
  
  
  print(helper_match_evaluate_multiple(rr(clus_assign[core]),rr(lblsT[core])))
  print(helper_match_evaluate_multiple(rr(clus_assignM[core]),rr(lblsT[core])))
  print(helper_match_evaluate_multiple(rr(allAssing[core]),rr(lblsT[core])))
  
  
  print(helper_match_evaluate_multiple_SweightedN(rr(clus_assign[core]),rr(lblsT[core]))[9:10])
  print(helper_match_evaluate_multiple_SweightedN(rr(clus_assignM[core]),rr(lblsT[core]))[9:10])
  print(helper_match_evaluate_multiple_SweightedN(rr(allAssing[core]),rr(lblsT[core]))[9:10])
  print(helper_match_evaluate_multiple_MoCu(rr(clus_assign[core]),rr(lblsT[core]))[8:9])
  print(helper_match_evaluate_multiple_MoCu(rr(clus_assignM[core]),rr(lblsT[core]))[8:9])
  print(helper_match_evaluate_multiple_MoCu(rr(allAssing[core]),rr(lblsT[core]))[8:9])
  print(AMI(rr(clus_assign[core]),rr(lblsT[core])))
  print(AMI(rr(clus_assignM[core]),rr(lblsT[core])))
  print(AMI(rr(allAssing[core]),rr(lblsT[core])))
  
  
  
  print(table(allAssing))
  print('Sum of scores, previous and current ')
  print(sum(ScoreLst[[iteration-1]]))
  print(sum(ScoreLst[[iteration]]))
}))

sumScores<-unlist(lapply(ScoreLst, function(x) sum(x)))
plot(sumScores)
top_ind<-which(sumScores==max(sumScores))-1
top_assign =Assign[[top_ind]]
#top_assign =Assign[[6]]


print(helper_match_evaluate_multiple(top_assign[core],rr(lblsT[core])))
print(helper_match_evaluate_multiple(clus_assign0[core] ,rr(lblsT[core])))
table(lblsT, clus_assign0)
table(lblsT, top_assign)

print(helper_match_evaluate_multiple_SweightedN(top_assign[core],rr(lblsT[core]))[9:10])
print(helper_match_evaluate_multiple_SweightedN(clus_assign0[core],rr(lblsT[core]))[9:10])
print(helper_match_evaluate_multiple_MoCu(top_assign[core],rr(lblsT[core]))[8:9])
print(helper_match_evaluate_multiple_MoCu(clus_assign0[core],rr(lblsT[core]))[8:9])
AMI(top_assign[core],rr(lblsT[core]))
AMI(clus_assign0[core],rr(lblsT[core]))
print(table(clus_assign0))
table(lblsT)  
save(Assign,ScoreLst, file=paste0(DATA_DIR, '/', CALC_ID, '.RData'))
#load(file=paste0(DATA_DIR, '/', CALC_ID, '.RData'))
AMIlist<-(unlist(lapply(Assign, function(x) AMI(x, as.numeric(as.factor(lblsT))))))
plot((sumScores/max(sumScores))[2:50], ylim=c(0,1))
points(AMIlist[1:49],  col='red')
cor((sumScores/max(sumScores))[2:50],AMIlist[1:49], method='spearman')
plot((sumScores/max(sumScores))[2:50], AMIlist[1:49])

