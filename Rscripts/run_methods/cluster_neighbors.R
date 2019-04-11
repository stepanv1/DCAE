# find the clusters to which 
# neighbours belong
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_louvain_multiple_runs.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_call_FLOCK.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_create_snnk_graph.R')

system.time(g_ob<-create_snnk_graph(cl_coord[,], k=90, metric='L1'))
            gc()
boxplot(cl_coord[lbls==5, ])

g<-simplify(g_ob$graph, remove.loops=T, edge.attr.comb=list(weight="sum"))

cl_resL<-louvain_multiple_runs(g, num.run = 5);gc()
#cl_resL<-helper_call_Flock(cl_coord, FLOCKDIR = '/home/sgrinek/bin/FLOCK')$Population
table(cl_resL)
helper_match_evaluate_multiple(cl_resL, lbls)
helper_match_evaluate_multiple(cl_resL[lbls!=0], lbls[lbls!=0])

relations <- g_ob$relations
asym_w <- sparseMatrix(i=relations$from,j=relations$to,x=relations$weight, symmetric = F, index1=T);gc()

asym_wn<-norm_mat(asym_w)
#sym_w<-(asym_w+t(asym_w))/2
#asym_wn<-norm_mat(sym_w)
IDXo<-sample((1:length(lbls))[lbls==0], 400); IDXi <- sample((1:length(lbls))[lbls!=0], 400)
Ptrace<-EnhanceDensityTrace(P=(asym_wn), V=rep(1,  dim(asym_wn)[1])/sum(rep(1,  dim(asym_wn)[1])), IDX=c(IDXo, IDXi), smooth=FALSE, debug=TRUE, maxit=5000, alpha=0.999, eps=1e-20)
helper_match_evaluate_multiple(ifelse(as.numeric(Ptrace$fin)==0, 1, 0), ifelse(lbls==0, 1, 0))
hist(log10(Ptrace$fin),200)
helper_match_evaluate_multiple(ifelse(as.numeric(log10(Ptrace$fin))< (-7.2), 1, 0), ifelse(lbls==0, 1, 0))
sum(as.numeric(log10(Ptrace$fin))< (-7.2))
table(lbls[log10(Ptrace$fin)> -7])

relationsWalkClus<-relations
relationsWalkClus$P<-Ptrace$fin[relationsWalkClus$to]
relationsWalkClus$clus<-cl_resL[relationsWalkClus$to]
relationsWalkClus$lblsT<-lbls[relationsWalkClus$to]
transN<-relationsWalkClus[,sum(weight), by=from]$V1
relationsWalkClus$trans <- relationsWalkClus$weight/transN[relationsWalkClus$from]

alphaK_ <- 1/((table(cl_resL))/length(cl_resL))
relationsWalkClus$alphaK <- alphaK_[cl_resL]
alphaK_norm<-relationsWalkClus[, sum(alphaK),by=from]$V1
relationsWalkClus$alphaK <-relationsWalkClus$alphaK / alphaK_norm[relationsWalkClus$from]


relationsWalkClus$transW <- relationsWalkClus$alphaK*relationsWalkClus$trans*relationsWalkClus$P
transW_norm<-relationsWalkClus[, sum(transW),by=from]$V1
relationsWalkClus$transW <-relationsWalkClus$transW / transW_norm[relationsWalkClus$from]

weightNeighb=relationsWalkClus[, .SD[, sum(transW)^2 , by=clus] , by=from]
#weightNeighb=relationsWalkClus[, .SD[, sum(trans*P)^2 , by=lblsT] , by=from]
scoreOut1<-weightNeighb[,sum(V1) ,by=from]$V1
hist((scoreOut1), 200)
hist(log10(Ptrace$fin*scoreOut1), 200)
hist(log10(Ptrace$fin),200)
helper_match_evaluate_multiple(ifelse(as.numeric(log10(Ptrace$fin*scoreOut1))< (-7.0) , 0, 1), ifelse(lbls==0, 1, 0))
helper_match_evaluate_multiple(ifelse(as.numeric(log10(Ptrace$fin*scoreOut1))< (-7.5) & as.numeric(log10(Ptrace$fin))< (-7), 0, 1), ifelse(lbls==0, 1, 0))

table(lbls[log10(Ptrace$fin*scoreOut1) > -7.2])
IDXcore <- log10(Ptrace$fin*scoreOut1) > -4
pairs(cl_coord[IDXcore,1:13][sample(sum(IDXcore),1500), ],pch='.')
table(lbls)
table(lbls[IDXcore])
table(cl_resL[IDXcore])
#IDXcore <- log10(Ptrace$fin)> -7
#IDXcore <- scoreOut1 > 0.2 ###looks like the best option
for(j in sort(unique(lbls)) ){
  cln=j
  boxplot(cl_coord[lbls==cln ,][sample(sum(lbls==cln), min(c(5000, sum(lbls==cln)) )) , ], main=j)
}
for(j in sort(unique(lbls[IDXcore])) ){
  cln=j
  boxplot(cl_coord[lbls[IDXcore]==cln ,][sample(sum(lbls[IDXcore]==cln), min(c(5000, sum(lbls[IDXcore]==cln)) )) , ], main=j)
}
for(j in sort(unique(cl_resL)) ){
  cln=j
  boxplot(cl_coord[cl_resL==cln ,][sample(sum(cl_resL==cln), min(c(5000, sum(cl_resL==cln)) )) , ], main=j)
}

cln=5
prc<-princomp(cl_coord[cl_resL==cln,])
plot(princomp(cl_coord[cl_resL==cln,]), npc=40)
prc$loading




#plot(cl_res, cl_coord[IDXcore, ], size=0.1)

#louvain
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_create_snnk_graph.R')
gCore<-create_snnk_graph(cl_coord[IDXcore, ], k=90, metric='L1')$graph; gc()
cl_resCore<-louvain_multiple_runs(gCore, num.run = 5);gc()
table(cl_resCore)
helper_match_evaluate_multiple(cl_resCore, lbls[IDXcore])

#apply random forest classification
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
assignPeriphery<-helper_assign_outliers(bulk_data=cl_coord[IDXcore, ], out_data= cl_coord[!IDXcore, ], bulk_labels=cl_resCore)
full_clus<-vector(mode='integer', length=dim(asym_w)[1])
full_clus[IDXcore]<-cl_resCore;full_clus[!IDXcore]<-assignPeriphery;
helper_match_evaluate_multiple(full_clus, lbls)
helper_match_evaluate_multiple(full_clus[lbls!=0], lbls[lbls!=0])

#dbscan
library("dbscan")
gCore<-graph_from_adjacency_matrix(((asym_w+t(asym_w))/2)[IDXcore,IDXcore], mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()
cl_resCore<-louvain_multiple_runs(gCore, num.run = 5);gc()
table(cl_resCore)
helper_match_evaluate_multiple(cl_resCore, lbls[IDXcore])

#apply random forest classification
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_random_forest.R')
assignPeriphery<-helper_assign_outliers(bulk_data=cl_coord[IDXcore, ], out_data= cl_coord[!IDXcore, ], bulk_labels=cl_resCore)
full_clus<-vector(mode='integer', length=dim(asym_w)[1])
full_clus[IDXcore]<-cl_resCore;full_clus[!IDXcore]<-assignPeriphery;
helper_match_evaluate_multiple(full_clus, lbls)











