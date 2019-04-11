#script to test outliers removal on Wang project data
setwd("/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/run_methods")
library(gplots)
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

#load Art data
DATA_DIR <- "../../benchmark_data_sets"
load( file=paste0(DATA_DIR, '/artLevine32.RData'))
#aFrame=aFrame[lbls %in% 1:10, ]
#lbls=lbls[lbls %in% 1:10 ]


lblsT=lbls
table(lblsT)
data=as.matrix(aFrame)
dim(data)
data<-apply(data, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
N=nrow(data)
nbins =2

# 1. cluster

print(system.time({g<-create_snnk_graph_approx(data, k=30, metric='L2', mc.cores=10);
clus_assign<-louvain_multiple_runs_par(g$graph, num.run=5, mc.cores=5)}))

clus_assign0 <- clus_assign
helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))
#save(clus_assign, file=paste0(DATA_DIR, '/artLevin32Clus_assign.RData'))
#load( file=paste0(DATA_DIR, '/artLevin32Clus_assign.RData'))
num_clus<-length(unique(clus_assign))
table(clus_assign)

rsClusT<-helper_find_subdimensions(data, lblsT, skewCut=1)
heatmap.2(as.matrix(ifelse(rsClusT$subdim, 1,0)))
apply(rsClusT$subdim,1, sum)

library(beanplot)
rsClus<-helper_find_subdimensions(data, clus_assign, skewCut=1)
heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))
table(clus_assign)
cln=9
#beanplot(as.data.frame(data[clus_assign==cln,rsClus$subdim[cln, ]]), log="", what = c(T, T, F, F))
boxplot(as.data.frame(data[clus_assign==cln, rsClus$subdim[cln, ]]), log="", what = c(T, T, F, F))
boxplot(as.data.frame(data[clus_assign==cln, !rsClus$subdim[cln, ]]), log="", what = c(T, T, F, F))
#beanplot(as.data.frame(data[lblsT==8,]), log="", what = c(T, T, F, F))
boxplot(as.data.frame(data[clus_assign==cln,]), log="", what = c(T, T, F, F))
boxplot(as.data.frame(data[lblsT==8,]), log="", what = c(T, T, F, F))
#find subdimensions


#LO <- vector("list", length = num_clus)  
#system.time(
#  for (i in 1:num_clus){
#    print(i)
#    rs<- clus_assign==i 
#    LO[[i]]<-helper_local_outliersLOF2(data[,rsClus$subdim[i,]], ref_subset = rs, k=30)
#  }
#)

print(system.time(LO<-mclapply(1:num_clus, function(i)  {rs<- clus_assign==i; helper_local_outliersLOF2(data[,rsClus$subdim[i,]], ref_subset = rs, k=30)}, mc.cores=10)))


#Problem p-values are inflated for bigger clusters
#library(MASS)
#n=100000;d=15
#X<-mvrnorm(n = n, rep(100,d), diag(d), tol = 1e-6, empirical = FALSE, EISPACK = FALSE)
#Y<-mvrnorm(n = n, rep(101,d), diag(d), tol = 1e-6, empirical = FALSE, EISPACK = FALSE)
#Z=rbind(X,Y)
#system.time(OutLoc<-helper_local_outliersLOF2(Z, ref_subset = c(rep(T,n), rep(F,n)), k=30))
#hist(OutLoc$prob[1:n],50)
#hist(OutLoc$prob[(n+1):(2*n)],50)
#hist(OutLoc$prob,50)
#sum(OutLoc$prob[1:n])

#GO<-helper_global_outliers_ApproxIID(data, rsClus$subdim, clus_assign, nbins=5, mc.cores=5)
#GO<-helper_global_outliers_Discrete(data, rsClus$subdim,  clus_assign, nbins=5, mc.cores=5)
#GO<-helper_global_outliers(data, rsClus$subdim,  clus_assign,  mc.cores=5)
#GO<-helper_global_outliers_Discrete_All_dims(data, rsClus$subdim, clus_assign, nbins=5, mc.cores=5)
#GO<-helper_global_outliers_Uniform(data, rsClus$subdim, clus_assign, nbins=3, mc.cores=20)
#GO<-helper_global_outliers_Discrete_Equidist_Bins(data, rsClus$subdim, clus_assign, nbins=nbins, mc.cores=20)
#GO<-helper_global_outliers_Discrete_Sums(data, rsClus$subdim, clus_assign, nbins=nbins, mc.cores=20)
GO<-helper_global_outliers_Continuous_Sums(data, rsClus$subdim, clus_assign, nbins=nbins, mc.cores=20)
#save(GO,LO, file=paste0(DATA_DIR, '/artLevin32GOLO.RData'))
#load(file=paste0(DATA_DIR, '/artSamGOLO.RData'))

GOm = Reduce(cbind, GO) 
cln=5
hist(GOm[clus_assign==cln,clus_assign=20],15)
LOm = Reduce(cbind, (lapply(LO, function(x) x$prob))) 
CO=as.matrix(GOm*LOm)
hist(LOm[clus_assign==cln,clus_assign=9],50)
hist(CO[clus_assign==cln,clus_assign=1],50)
hist(CO[clus_assign==cln,clus_assign=cln],50)
#weights=unlist((table(clus_assign)/length(clus_assign)))
weights=1

#normCO<-apply(CO, 1, function(x) sum(weights*x))
#score<-unlist(lapply(1:nrow(CO),  function(x) sum( (weights*CO[x,]/normCO[x])^2 ) ))
#limit number of clusters adding to the probability measure
CO_2<-t(apply(CO, 1, function(x) if(sum(x>0)==1){x}else{x*(x>0.05)}))
normCO_2<-apply(CO_2, 1, function(x) sum(weights*x))
score<-unlist(lapply(1:nrow(CO_2),  function(x) sum( (weights*CO_2[x,]/normCO_2[x])^2 ) ))
score[is.na(score)]<-0
#normLOm<-apply(LOm, 1, function(x) sum(weights*x))
#score<-unlist(lapply(1:nrow(LOm),  function(x) sum( (weights*LOm[x,]/normLOm[x])^2 ) ))


hist(score,500)

hist(score[clus_assign==18],500)
#lapply(1:num_clus, function(x) hist(score[clus_assign==cln],500, main=cln))
table(clus_assign)
cln=7
hist(score[clus_assign==cln],500, main=cln)



cutoffProd=helper_topPercentile(0.05, score)$lim_cut


scoreLO<-unlist(lapply(1:N, function(x) LOm[x,clus_assign[x]]))
scoreGO<-unlist(lapply(1:N, function(x) GOm[x,clus_assign[x]]))


#globally strongest outliers
cutoffGlob=helper_topPercentile(0.5, score)$lim_cut
#per cluster outliers
#TODO use this to find flexing point
#https://github.com/RGLab/flowDensity/blob/trunk/R/helper_functions.R#L2
cutoffClus<-lapply(1:length(unique(clus_assign)), function(i) {
  helper_topPercentile(0.5,score[clus_assign==i])$lim_cut
})
idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<0.45, F, T )))
#idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<0.6, F, T )))

#idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<cutoffClus[clus_assign[i]], F, T )))


outliers<- !idxOutlClus 
hist(score,500)
hist(score,500, xlim=c(0,0.4))
hist(score[!outliers],500)

#outliers<- !idxOutlClus 
#outliers<-  score<cutoffGlob 
#outliers<-  score<0.3
hist(score,500)
hist(score[!outliers],500)
#lapply(1:10, function(x) hist(score[lblsT==x],500, main=x));
#table(clus_assign)
#table(lblsT)
#lapply(1:num_clus, function(cln) {X11();hist(score[clus_assign==cln], main=c(cln, sum(clus_assign==cln)), 50)})
#hist(score[lblsT==7 ], 50)
#hist(score[unlist(mclapply(clus_assign, function(x) sum(clus_assign==x)>900 , mc.cores=10))], 50)
#hist(score[unlist(mclapply(clus_assign, function(x) sum(clus_assign==x)<900 , mc.cores=10))], 50)
#hist(score[unlist(mclapply(lblsT, function(x) sum(lblsT==x)>6500 , mc.cores=10))], 50)
#hist(score[unlist(mclapply(lblsT, function(x) sum(lblsT==x)<6500 , mc.cores=10))], 50)
#lapply(1:10, function(cln) {X11();hist(score[lblsT==cln], main=c(cln, sum(lblsT==cln)), 500)})
#lapply(1:length(table(clus_assign)), function(cln) {X11();hist(score[clus_assign==cln], main=c(cln, sum(clus_assign==cln)), 500)})

cln=10
#hist(GOm[clus_assign==cln,cln],30)
#hist(LOm[clus_assign==cln,cln],50)
#plot(GOm[clus_assign==cln,cln],LOm[clus_assign==cln,cln], pch='.')
#lapply(1:num_clus, function(cln) cor(GOm[clus_assign==cln,cln],LOm[clus_assign==cln,cln], method='spearman'))

sum(outliers)
table(clus_assign)
table(clus_assign[outliers])
table(clus_assign[!outliers])
table(lblsT)
table(lblsT[!outliers])
#plot3d(VAE[,1], VAE[,2], VAE[,3], col = ifelse(outliers, "red", "black"), size=0.5)
#plot3d(VAE[!outliers,1], VAE[!outliers,2], VAE[!outliers,3],, size=0.5)
#open3d()
#plot3d(VAE[,1], VAE[,2], VAE[,3], size=0.5)


#save(lblsNew, file=paste0(DATA_DIR, '/artSamLblsNew.RData'))
#compare without outliers
#source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_approx_distances.R')
dataNew<-data[!outliers,]
dim(dataNew)
print(system.time({g<-create_snnk_graph_approx(dataNew, k=30, metric='L2', mc.cores=10);
lblsNew<-louvain_multiple_runs_par(g$graph, num.run=5, mc.cores=5)}))
clus_assignF<-unlist(lblsNew)
clus_assignNew<-unlist(clus_assignF)
num_clusNew<-length(unique(clus_assignNew))
tmp= (1:length(clus_assign))*NA
tmp[!outliers]<-clus_assignNew
clus_assignNew<-tmp
table(clus_assignNew)
#clus_assignNew<-lblsNew

helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple(clus_assignNew, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple_SweightedN(clus_assign, as.numeric(as.factor(lblsT)))[9:10]
helper_match_evaluate_multiple_SweightedN(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])[9:10]
helper_match_evaluate_multiple_MoCu(clus_assign, as.numeric(as.factor(lblsT)))[8:9]
helper_match_evaluate_multiple_MoCu(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])[8:9]


########################################################################################################################################
#now reassign outliers



source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
system.time(outAssign<-helper_assign_outliers(dataNew, data[outliers, ], clus_assignF))
allAssing<-(1:length(clus_assign))*NA
allAssing[outliers]<-outAssign
allAssing[!outliers]<-clus_assignF
table(allAssing)
length(allAssing)


helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))
helper_match_evaluate_multiple(allAssing, as.numeric(as.factor(lblsT)))




helper_match_evaluate_multiple_SweightedN(clus_assign, as.numeric(as.factor(lblsT)))[9:10]
helper_match_evaluate_multiple_SweightedN(allAssing, as.numeric(as.factor(lblsT)))[9:10]
helper_match_evaluate_multiple_MoCu(clus_assign, as.numeric(as.factor(lblsT)))[8:9]
helper_match_evaluate_multiple_MoCu(allAssing, as.numeric(as.factor(lblsT)))[8:9]
AMI(clus_assign, as.numeric(as.factor(lblsT)))
AMI(allAssing, as.numeric(as.factor(lblsT)))




#################################################################################################################
#################################################################################################################
setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
iterations<-15
ScoreLst<-vector("list", iterations)
Assign<-vector("list", iterations)
ScoreLst[[1]]<-score
Assign[[1]]<-allAssing
print(system.time(for(iteration in 2:iterations){
  print(paste0('iteration ', iteration))
  clus_assign<-allAssing
  num_clus<-length(table(allAssing))
  rsClus<-helper_find_subdimensions(data, clus_assign, skewCut=1)
  heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))
  table(clus_assign)
  cln=9
  #beanplot(as.data.frame(data[clus_assign==cln,rsClus$subdim[cln, ]]), log="", what = c(T, T, F, F))
  #boxplot(as.data.frame(data[clus_assign==cln, rsClus$subdim[cln, ]]), log="", what = c(T, T, F, F))
  #boxplot(as.data.frame(data[clus_assign==cln, !rsClus$subdim[cln, ]]), log="", what = c(T, T, F, F))
  #beanplot(as.data.frame(data[lblsT==8,]), log="", what = c(T, T, F, F))
  #boxplot(as.data.frame(data[clus_assign==cln,]), log="", what = c(T, T, F, F))
  #boxplot(as.data.frame(data[lblsT==8,]), log="", what = c(T, T, F, F))
  #find subdimensions
  
  
  #LO <- vector("list", length = num_clus)  
  #system.time(
  #  for (i in 1:num_clus){
  #    print(i)
  #    rs<- clus_assign==i 
  #    LO[[i]]<-helper_local_outliersLOF2(data[,rsClus$subdim[i,]], ref_subset = rs, k=30)
  #  }
  #)
  print('Local outliers computation..')
  print(system.time(LO<-mclapply(1:num_clus, function(i)  {rs<- clus_assign==i; helper_local_outliersLOF2(data[,rsClus$subdim[i,]], ref_subset = rs, k=30)}, mc.cores=10)))
  #Problem p-values are inflated for bigger clusters
  #library(MASS)
  #n=100000;d=15
  #X<-mvrnorm(n = n, rep(100,d), diag(d), tol = 1e-6, empirical = FALSE, EISPACK = FALSE)
  #Y<-mvrnorm(n = n, rep(101,d), diag(d), tol = 1e-6, empirical = FALSE, EISPACK = FALSE)
  #Z=rbind(X,Y)
  #system.time(OutLoc<-helper_local_outliersLOF2(Z, ref_subset = c(rep(T,n), rep(F,n)), k=30))
  #hist(OutLoc$prob[1:n],50)
  #hist(OutLoc$prob[(n+1):(2*n)],50)
  #hist(OutLoc$prob,50)
  #sum(OutLoc$prob[1:n])
  
  #GO<-helper_global_outliers_ApproxIID(data, rsClus$subdim, clus_assign, nbins=5, mc.cores=5)
  #GO<-helper_global_outliers_Discrete(data, rsClus$subdim,  clus_assign, nbins=5, mc.cores=5)
  #GO<-helper_global_outliers(data, rsClus$subdim,  clus_assign,  mc.cores=5)
  #GO<-helper_global_outliers_Discrete_All_dims(data, rsClus$subdim, clus_assign, nbins=5, mc.cores=5)
  #GO<-helper_global_outliers_Uniform(data, rsClus$subdim, clus_assign, nbins=10, mc.cores=1)
  #GO<-helper_global_outliers_Discrete_Equidist_Bins(data, rsClus$subdim, clus_assign, nbins=nbins, mc.cores=5)
  GO<-helper_global_outliers_Continuous_Sums(data, rsClus$subdim, clus_assign, nbins=nbins, mc.cores=20)
  resOut<-c(GO,LO)
  #save(GO,LO, file=paste0(DATA_DIR, '/artLevin32GOLO.RData'))
  #load(file=paste0(DATA_DIR, '/artLevin32GOLO.RData'))
  
  GOm = Reduce(cbind, GO) 
  LOm = Reduce(cbind, (lapply(LO, function(x) x$prob))) 
  CO=as.matrix(GOm*LOm)
  #weights=unlist((table(clus_assign)/length(clus_assign)))
  weights=1
  
  #normCO<-apply(CO, 1, function(x) sum(weights*x))
  #score<-unlist(lapply(1:nrow(CO),  function(x) sum( (weights*CO[x,]/normCO[x])^2 ) ))
  #IDX<-apply(CO, 1, function(x) all(x<10^(-10)))
  #score[IDX]<-0
  #normLOm<-apply(LOm, 1, function(x) sum(weights*x))
  #score<-unlist(lapply(1:nrow(LOm),  function(x) sum( (weights*LOm[x,]/normLOm[x])^2 ) ))
  CO_2<-t(apply(CO, 1, function(x) if(sum(x>0)==1){x}else{x*(x>0.05)}))
  normCO_2<-apply(CO_2, 1, function(x) sum(weights*x))
  score<-unlist(lapply(1:nrow(CO_2),  function(x) sum( (weights*CO_2[x,]/normCO_2[x])^2 ) ))
  score[is.na(score)]<-0
  
  
  
  ScoreLst[[iteration]]<-score
  
  hist(score,500)
  cln
  hist(score[clus_assign==cln],500)
  #lapply(1:num_clus, function(x) hist(score[clus_assign==cln],500, main=cln))
  table(clus_assign)
  cln=7
  #hist(score[clus_assign==cln],500, main=cln)
  #investigate peak at score=0.5
  cutoff=.5005
  View(cbind(lblsT[score>0.5 & score <cutoff], CO_2[score>0.5 & score <cutoff,])) 
  View(cbind(clus_assign[score>0.5 & score <cutoff] , cbind(lblsT[score>0.5 & score <cutoff], CO_2[score>0.5 & score <cutoff,])) )
  boxplot(cbind(lblsT[score>0.5 & score <cutoff]/14, CO_2[score>0.5 & score <cutoff,])) 
  table(clus_assign[score>0.5 & score <cutoff] , lblsT[score>0.5 & score <cutoff])
  heatmap.2(as.matrix(ifelse(rsClus$subdim, 1,0)))
  cutoffProd=helper_topPercentile(0.05, score)$lim_cut
  
  
  scoreLO<-unlist(lapply(1:N, function(x) LOm[x,clus_assign[x]]))
  scoreGO<-unlist(lapply(1:N, function(x) GOm[x,clus_assign[x]]))
  
  
  #globally strongest outliers
  cutoffGlob=helper_topPercentile(0.3, score)$lim_cut
  #per cluster outliers
  #TODO use this to find flexing point
  #https://github.com/RGLab/flowDensity/blob/trunk/R/helper_functions.R#L2
  cutoffClus<-lapply(1:length(unique(clus_assign)), function(i) {
    helper_topPercentile(0.5,score[clus_assign==i])$lim_cut
  })
  #idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<cutoffClus[clus_assign[i]], F, T )))
  idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<0.45, F, T )))
  
  #idxOutlClus<-unlist(lapply(1:N, function(i) ifelse(score[i]<cutoffClus[clus_assign[i]], F, T )))
  
  
  outliers<- !idxOutlClus 
  #hist(score,500)
  hist(score[!outliers],500)
  
  #outliers<- !idxOutlClus 
  #outliers<-  score<cutoffGlob 
  #outliers<-  score<0.3
  #hist(score,500)
  #hist(score[!outliers],500)
  table(clus_assign)
  table(lblsT)
  #lapply(1:num_clus, function(cln) {X11();hist(score[clus_assign==cln], main=c(cln, sum(clus_assign==cln)), 50)})
  #hist(score[lblsT==7 ], 50)
  #hist(score[unlist(mclapply(clus_assign, function(x) sum(clus_assign==x)>900 , mc.cores=10))], 50)
  #hist(score[unlist(mclapply(clus_assign, function(x) sum(clus_assign==x)<900 , mc.cores=10))], 50)
  #hist(score[unlist(mclapply(lblsT, function(x) sum(lblsT==x)>6500 , mc.cores=10))], 50)
  #hist(score[unlist(mclapply(lblsT, function(x) sum(lblsT==x)<6500 , mc.cores=10))], 50)
  #lapply(1:10, function(cln) {X11();hist(score[lblsT==cln], main=c(cln, sum(lblsT==cln)), 500)})
  #lapply(1:length(table(clus_assign)), function(cln) {X11();hist(score[clus_assign==cln], main=c(cln, sum(clus_assign==cln)), 500)})
  
  cln=10
  #hist(GOm[clus_assign==cln,cln],30)
  #hist(LOm[clus_assign==cln,cln],50)
  #plot(GOm[clus_assign==cln,cln],LOm[clus_assign==cln,cln], pch='.')
  #lapply(1:num_clus, function(cln) cor(GOm[clus_assign==cln,cln],LOm[clus_assign==cln,cln], method='spearman'))
  
  #sum(outliers)
  #table(clus_assign)
  table(clus_assign[outliers])
  table(clus_assign[!outliers])
  table(lblsT)
  table(lblsT[!outliers])
  #plot3d(VAE[,1], VAE[,2], VAE[,3], col = ifelse(outliers, "red", "black"), size=0.5)
  #plot3d(VAE[!outliers,1], VAE[!outliers,2], VAE[!outliers,3],, size=0.5)
  #open3d()
  #plot3d(VAE[,1], VAE[,2], VAE[,3], size=0.5)
  
  
  #save(lblsNew, file=paste0(DATA_DIR, '/artSamLblsNew.RData'))
  #compare without outliers
  #source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_approx_distances.R')
  dataNew<-data[!outliers,]
  dim(dataNew)
  print(system.time({g<-create_snnk_graph_approx(dataNew, k=30, metric='L2', mc.cores=10);
  lblsNew<-louvain_multiple_runs_par(g$graph, num.run=5, mc.cores=5)}))
  clus_assignF<-unlist(lblsNew)
  
  clus_assignNew<-unlist(clus_assignF)
  num_clusNew<-length(unique(clus_assignNew))
  num_clusNew<-length(unique(clus_assignNew))
  tmp= (1:length(clus_assign))*NA
  tmp[!outliers]<-clus_assignNew
  clus_assignNew<-tmp
  table(clus_assignNew)
  helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT)))
  helper_match_evaluate_multiple(clus_assignNew, as.numeric(as.factor(lblsT)))
  helper_match_evaluate_multiple_SweightedN(clus_assign, as.numeric(as.factor(lblsT)))[9:10]
  helper_match_evaluate_multiple_SweightedN(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])[9:10]
  helper_match_evaluate_multiple_MoCu(clus_assign, as.numeric(as.factor(lblsT)))[8:9]
  helper_match_evaluate_multiple_MoCu(clus_assignNew[!outliers], as.numeric(as.factor(lblsT))[!outliers])[8:9]
  
  
  
  #now reassign outliers
  source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
  system.time(outAssign<-helper_assign_outliers(dataNew, data[outliers, ], clus_assignF))
  allAssing<-(1:length(clus_assign))*NA
  allAssing[outliers]<-outAssign
  allAssing[!outliers]<-clus_assignF
  table(allAssing)
  length(allAssing)
  Assign[[iteration]]<-allAssing
  
  print(helper_match_evaluate_multiple(clus_assign, as.numeric(as.factor(lblsT))))
  print(helper_match_evaluate_multiple(allAssing, as.numeric(as.factor(lblsT))))
  
  
  
  
  print(helper_match_evaluate_multiple_SweightedN(clus_assign, as.numeric(as.factor(lblsT)))[9:10])
  print(helper_match_evaluate_multiple_SweightedN(allAssing, as.numeric(as.factor(lblsT)))[9:10])
  print(helper_match_evaluate_multiple_MoCu(clus_assign, as.numeric(as.factor(lblsT)))[8:9])
  print(helper_match_evaluate_multiple_MoCu(allAssing, as.numeric(as.factor(lblsT)))[8:9])
  AMI(clus_assign, as.numeric(as.factor(lblsT)))
  AMI(allAssing, as.numeric(as.factor(lblsT)))
  print(table(allAssing))
  print('Sum of scores, previous and current ')
  print(sum(ScoreLst[[iteration-1]]))
  print(sum(ScoreLst[[iteration]]))
}))

sumScores<-unlist(lapply(ScoreLst, function(x) sum(x)))
plot(sumScores)
top_ind<-which(sumScores==max(sumScores))-1
top_assign =Assign[[top_ind]]
#top_assign =Assign[[1]]


print(helper_match_evaluate_multiple(top_assign, as.numeric(as.factor(lblsT))))
print(helper_match_evaluate_multiple(clus_assign0 , as.numeric(as.factor(lblsT))))

print(helper_match_evaluate_multiple_SweightedN(top_assign, as.numeric(as.factor(lblsT)))[9:10])
print(helper_match_evaluate_multiple_SweightedN(clus_assign0, as.numeric(as.factor(lblsT)))[9:10])
print(helper_match_evaluate_multiple_MoCu(top_assign, as.numeric(as.factor(lblsT)))[8:9])
print(helper_match_evaluate_multiple_MoCu(clus_assign0, as.numeric(as.factor(lblsT)))[8:9])
AMI(top_assign, as.numeric(as.factor(lblsT)))
AMI(clus_assign0, as.numeric(as.factor(lblsT)))
print(table(clus_assign0))
table(lblsT)
save(Assign,ScoreLst, file=paste0(DATA_DIR, '/artLevin32Scores150.RData'))
AMIlist<-(unlist(lapply(Assign, function(x) AMI(x, as.numeric(as.factor(lblsT))))))
plot((sumScores/max(sumScores))[2:15], ylim=c(0,1))
points(AMIlist[1:14],  col='red')
cor((sumScores/max(sumScores))[2:15],AMIlist[1:14], method='spearman')
plot((sumScores/max(sumScores))[2:15], AMIlist[1:14])

cor(sumScores,AMIlist)
plot(sumScores,AMIlist)
