#Script to calculate combined probabilistic outlier score
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_find_subdimensions.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_local_outliers.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_convBernully.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_global_outliers.R')
library(scales)
library(gplots)
#data <- cl_coord; lbls=lbls+1
#data <- data[[i]][!is.na(clus_truth[[i]]), ]; clus_assign<-clus_truth[[i]][!is.na(clus_truth[[i]])];

sub_res<-helper_find_subdimensions(data, clus_assign)$subdim
heatmap.2(ifelse(sub_res,1,0))
heatmap.2(cor(data[clus_assign==1,!sub_res[1,]]))
system.time({GO<-helper_global_outliers_Discrete_All_dims(data, sub_res, clus_assign, nbins=10, mc.cores=10);
LO<-mclapply(1:nrow(sub_res), function(x) helper_local_outliersLOF2(data[, sub_res[x,]],  clus_assign==x, k=30), mc.cores=10, mc.preschedule=F)})
#combo, LO*G0
GOm = Reduce(cbind, GO) 
LOm = Reduce(cbind, (lapply(LO, function(x) x$prob))) 
CO=as.matrix(GOm*LOm)
colnames(CO)=1:nrow(sub_res)
#correlation between cluster assignment and max score in CO
cor(clus_assign, as.numeric(unlist(apply(CO,1, function(x) which(max(x)==x)[1]))), method = 'spearman')
cor(clus_assign, apply(LOm,1, function(x) which(max(x)==x)[1]), method = 'spearman')
cor(clus_assign, apply(GOm,1, function(x) which(max(x)==x)[1]), method = 'spearman')
helper_match_evaluate_multiple(clus_assign, apply(CO,1, function(x) which(max(x)==x)[1]))
helper_match_evaluate_multiple(clus_assign, apply(GOm,1, function(x) which(max(x)==x)[1]))
helper_match_evaluate_multiple(clus_assign, apply(LOm,1, function(x) which(max(x)==x)[1]))

par(new = TRUE)
X11();plot(clus_assign+rnorm(length(clus_assign), sd=0.2), apply(CO,1, function(x) which(max(x)==x)[1])+rnorm(length(clus_assign), sd=0.2), pch = '.', col = alpha('black', 0.1))

#create for localization degree
LD <- apply(CO,1, function(x) ifelse(sum(x>0),sum((x/sum(x))^2), 0) )
MD <- apply(CO,1, function(x) max(x) )
inGO <- unlist(lapply(1:nrow(GOm), function(x) any(GOm[x,]>0.95 & LOm[x,]<0.05)))
sum(inGO)
table(clus_assign)
table(clus_assign[inGO])
hist(LD,200)
hist(LD[clus_assign==6],200)
#clust_clean<- LD>0.8;
clust_clean <- !((MD<0.05 & LD < 0.6)  | inGO) 
sum(clust_clean)
table(clus_assign[clust_clean])
  #par(new = TRUE)
points((clus_assign+rnorm(length(clus_assign), sd=0.2))[clust_clean], apply(CO[clust_clean,],1, function(x) which(max(x)==x) +rnorm(1, sd=0.2)), pch = '.', col = alpha('red', 0.1))
X11();plot((clus_assign+rnorm(length(clus_assign), sd=0.2))[clust_clean], apply(CO[clust_clean,],1, function(x) which(max(x)==x) +rnorm(1, sd=0.2)), pch = '.', col = alpha('red', 0.5))
helper_match_evaluate_multiple(clus_assign[clust_clean], apply(CO[clust_clean,],1, function(x) which(max(x)==x)[1]))
########################################################################################
#To remove
########################################################################################
GO<-helper_global_outliers(data, ref_subs, clus_assign, mc.cores=5)
p=helper_global_outliers_Discrete(data, ref_subs, clus_assign, nbins=10, mc.cores=5)
cor(clus_assign[clust_clean], apply(CO[clust_clean,],1, function(x) which(max(x)==x)), method = 'kendall')
library(beanplot)
cor(1-p, GO, method = 'spearman')
ns<-nrow(ref_subs)
ref_subs=sub_res
cln=10
lapply(1:ns, function(x) hist(p[clus_assign==x],1000, main=x))
X11();heatmap.2(as.matrix(ifelse(ref_subs, 1,0)))
X11();heatmap.2(cor(data[clus_assign==cln,sub_res[cln,]], method='spearman'))
X11();heatmap.2(cor(data[clus_assign==cln,!sub_res[cln,]], method='spearman'))
X11();heatmap.2(cor(data[clus_assign==cln,], method='spearman'))
X11();beanplot(as.data.frame(data[clus_assign==cln , !ref_subs[cln, ]]), main=cln, beanlines = 'median',  names= names(ref_subs[cln, !ref_subs[cln, ]]) , what = c(1,1,1,0), lab.cex=0.1);abline(a=0,b=0)
X11();beanplot(as.data.frame(data[clus_assign==cln , ref_subs[cln, ]]), main=cln, beanlines = 'median',  names= names(ref_subs[cln, ref_subs[cln, ]]) , what = c(1,1,1,0), lab.cex=0.1);abline(a=0,b=0)

table(clus_assign)
table(clus_truth[[i]], clus_assign)
cln=1
library(beanplot)
X11();beanplot(as.data.frame(data[clus_assign==cln ,]), main=cln, beanlines = 'median',  names= names(ref_subs[cln, ]) , what = c(1,1,1,0), lab.cex=0.1);abline(a=0,b=0)
X11();beanplot(as.data.frame(data[clus_assign==cln , ref_subs[cln, ]]), main=cln, beanlines = 'median',  names= names(ref_subs[cln, ref_subs[cln, ]]) , what = c(1,1,1,0), lab.cex=0.1);abline(a=0,b=0)
X11();beanplot(as.data.frame(data[clus_assign==cln , !ref_subs[cln, ]]), main=cln, beanlines = 'median',  names= names(ref_subs[cln, !ref_subs[cln, ]]) , what = c(1,1,1,0), lab.cex=0.1);abline(a=0,b=0)
plot(as.data.frame(data[clus_assign==cln , c('CXCR4','CD7')]) , pch='.')
plot(as.data.frame(data[clus_assign==cln , c('CXCR4','CD7', 'CD45RA', 'CD7')]) , pch='.')
library('rgl')
plot3d(as.data.frame(data[clus_assign==cln , c('CXCR4','CD7', 'CD45')]), pch='.', cex=0.1, size=0.05, col=10*(p[clus_assign==cln]+1))
plot3d(as.data.frame(data[clus_assign==cln , c('Flt3','CD49d', 'CD45RA')]), pch='.', cex=0.1, size=0.05, col=10*(p[clus_assign==cln]+1))
plot3d(as.data.frame(data[clus_assign==cln , c('Flt3','CD49d', 'CD45RA')]), pch='.', cex=0.1, size=0.05, col=10*(GO[clus_assign==cln]+1))

plot3d(as.data.frame(data[clus_assign==cln , c('CXCR4', 'CD7', 'CD4')]), pch='.', cex=0.1, size=0.05, col=10*(p[clus_assign==cln]+1))
plot3d(as.data.frame(data[clus_assign==cln , c('CXCR4', 'CD7', 'CD4')]), pch='.', cex=0.1, size=0.05, col=10*(GO[clus_assign==cln]+1))
plot((data[clus_assign==cln , c( 'CD49d')]), pch='.',  col=10*(p[clus_assign==cln]+1))
plot(p[clus_assign==cln], pch='.',  col=10*(p[clus_assign==cln]+1))
plot(p[clus_assign==cln], pch='.',  col=clus_truth[[i]][clus_assign==cln])
cor((data[clus_assign==cln , c( 'CD45')]), 10*(GO[clus_assign==cln]+1), method = 'spearman')
cor((data[clus_assign==cln , c( 'CD7')]), 10*(GO[clus_assign==cln]+1), method = 'spearman')
cor((data[clus_assign==cln , c( 'CD45')]), 10*(p[clus_assign==cln]+1), method = 'spearman')
cor((data[clus_assign==cln , c( 'CD7')]), 10*(p[clus_assign==cln]+1), method = 'spearman')
X11();heatmap.2(cor(scale(data[clus_assign==cln,]),method='spearman'), symm = FALSE,col=rainbow(8))
X11();heatmap.2(cor(scale(data[clus_assign==cln, ref_subs[cln, ]]),method='spearman'), symm = FALSE,col=rainbow(8))
X11();heatmap.2(cor(scale(data[clus_assign==cln, !ref_subs[cln, ]]),method='spearman'), symm = FALSE,col=rainbow(8))
eigen(cor(data[clus_assign==cln, ref_subs[cln, ]],method='spearman'))$values
eigen(cor(data[clus_assign==cln, !ref_subs[cln, ]],method='spearman'))$values
pchisq(sum(eigen(cor(scale(data[clus_assign==cln, ref_subs[cln, ]]),method='pearson'))$values), df=sum(ref_subs[cln, ]) )
pchisq(sum(eigen(cor(scale(data[clus_assign==cln, !ref_subs[cln, ]]),method='pearson'))$values), df=sum(!ref_subs[cln, ]) )
library(mclust)
x.gmm = Mclust((data[clus_assign==cln, c('CD16')])[sinh(data[clus_assign==cln, c('CD16')])>0], G=1:3)
summary(x.gmm)

library(silvermantest)
data[data<0]=0
X11();hist((data[clus_assign==cln, c('CD117')])[(data[clus_assign==cln, c('CD117')])>0],100)
silverman.test((data[clus_assign==cln, c('CD16')]), 1, adjust = T)
silverman.test((data[clus_assign==cln, c('CD117')])[sinh(data[clus_assign==cln, c('CD117')])>0], 1, adjust = T)
lapply(1:ncol(data), function(x) {print(silverman.test((data[clus_assign==cln, x])[sinh(data[clus_assign==cln, x])>0], 1, adjust = T)); print(colnames(data[clus_assign==cln, ])[x])})

table(clus_truth[[i]][clus_assign==cln][data[clus_assign==cln, c('CD117')]>0] )
cor(x=clus_truth[[i]][clus_assign==cln],  y=data[clus_assign==cln,ref_subs[cln, ]] , use='pairwise.complete.obs', method='kendall')
cor(x=clus_truth[[i]][clus_assign==cln],  y=data[clus_assign==cln,!ref_subs[cln, ]] , use='pairwise.complete.obs', method='kendall')

library(tclust)
km<-tclust (data[clus_assign==cln, c('CD16','CD38','CD7')], alpha = 0.2,k=2, iter.max = 200,restr.fact = 100)
table(clus_truth[[i]][clus_assign==cln], km$cluster)

library(rgl)
plot3d(data[clus_assign==cln, c('CD16','CD38','CD7')])
plot3d(data[clus_assign==cln, c('CD16','CD38','CD7')], col=clus_truth[[i]][clus_assign==cln])
plot3d(data[clus_assign==cln, c('CD16','CD38','CD7')], col=ifelse(km$cluster==0, 'black', km$cluster+1))
plot3d(data[clus_assign==cln, c('CD16','CD38','CD7')], col=ifelse(p<0.1, 'black', 100*(p+1)))
plot3d(data[clus_assign==cln, c('Flt3','CD123','HLA-DR')], col=ifelse(GO>1.0, 'black', 100*(GO+1)))
plot3d(data[clus_assign==cln, c('Flt3','CD123','HLA-DR')], col=ifelse(p<0.3, 'black', 100*(p+1)))

# To check the rate of clustering error depending on the absolute value of signal
# in noisy and not noisy dimensions 
clus_truthL<-lapply(clus_truth, function(x)   ifelse(is.na(x),1000,  x))
matchres<-helper_match_evaluate_multiple(clus_assign, clus_truth[[i]])

cor(matchres$F1, unlist(lapply(1:length(matchres$F1), function(x) mean(data[clus_truthL[[i]]==x, sub_resL$subdim[x, ]], na.rm = T))), method='kendall')
cor(matchres$F1, unlist(lapply(1:length(matchres$F1), function(x) mean(data[clus_truthL[[i]]==x, !sub_resL$subdim[x, ]], na.rm = T))), method='kendall')

cor(matchres$F1[-c(3,5,6,14)], unlist(lapply((1:length(matchres$F1))[-c(3,5,6,14)], function(x) median(scale(data[clus_truthL[[i]]==x, ]), na.rm = T))), method='spearman')
cor(matchres$F1[-c(3,5,6,14)], unlist(lapply((1:length(matchres$F1))[-c(3,5,6,14)], function(x) median(scale(data[clus_truthL[[i]]==x, sub_resL$subdim[x, ]]), na.rm = T))), method='spearman')
cor(matchres$F1[-c(3,5,6,14)], unlist(lapply((1:length(matchres$F1))[-c(3,5,6,14)], function(x) median(scale(data[clus_truthL[[i]]==x, !sub_resL$subdim[x, ]]), na.rm = T))), method='spearman')

cor(matchres$F1[-c(3,5,6,14)], unlist(lapply(1:length(matchres$F1[-c(3,5,6,14)]), function(x) abs(min(unlist(lapply(1:32, function(y) skewness(data[clus_truthL[[i]]==x, y], na.rm = T))))))))
cor(matchres$F1[-c(3,5,6,14)], unlist(lapply(1:length(matchres$F1[-c(3,5,6,14)]), function(x) min(abs(unlist(lapply(1:32, function(y) skewness(data[clus_truthL[[i]]==x, y], na.rm = T))))))))
cor(matchres$F1[-c(3,5,6,14)], unlist(lapply(1:length(matchres$F1[-c(3,5,6,14)]), function(x) min(abs(unlist(lapply((1:32)[sub_resL$subdim[x, ]], function(y) skewness(data[clus_truthL[[i]]==x, y], na.rm = T))))))))
X11();plot(matchres$F1[-c(3,5,6,14)], unlist(lapply(1:length(matchres$F1[-c(3,5,6,14)]), function(x) min(abs(unlist(lapply((1:32)[sub_resL$subdim[x, ]], function(y) skewness(data[clus_truthL[[i]]==x, y], na.rm = T))))))))
X11();cor(matchres$pr[-c(3,5,6,14)], unlist(lapply(1:length(matchres$F1[-c(3,5,6,14)]), function(x) max(abs(unlist(lapply((1:32)[sub_resL$subdim[x, ]], function(y) median(data[clus_truthL[[i]]==x, y], na.rm = T) /sd(data[clus_truthL[[i]]==x, y], na.rm = T) )))))))


library(energy)
dcor(data[clus_truthL[[i]]==9, ][sample(nrow(data[clus_truthL[[i]]==9, ]), 10000),] , data[clus_truthL[[i]]==10, ][sample(nrow(data[clus_truthL[[i]]==10, ]), 10000),])

#binary feature matrix
bfmL<-lapply(1:length(table(clus_assign)), function(x) apply(data[clus_truthL[[i]]==x, ], 2, function(y) ifelse(median(y)<0.3,0,1) ) )
bfmL<-lapply(1:length(table(clus_assign)), function(x) apply(data[clus_truthL[[i]]==x, ], 2, function(y) ifelse(median(y)<2,0,1) ) )
#metric based on + - and 'not expressed'
bfmL<-lapply(1:length(table(clus_assign)), function(x) apply(data[clus_truthL[[i]]==x, ], 2, function(y) ifelse(median(y)<0.3,0,ifelse(median(y)<2,1,2)) ) )
#distance is defined by low expressing markers
bfmL<-lapply(1:length(table(clus_assign)), function(x) apply(data[clus_truthL[[i]]==x, ], 2, function(y) ifelse(median(y)<0.3,0,ifelse(median(y)<2,3,0)) ) )
#distance is defined by low and high expressing markers
bfmL<-lapply(1:length(table(clus_assign)), function(x) apply(data[clus_assign==x, ], 2, function(y) ifelse(median(y)<0.3,0,ifelse(median(y)<2,3,3)) ) )


bfm<-Reduce(rbind, bfmL)
dists<-as.matrix(dist(bfm, method='euclidean'))
colnames(dists)=1:14;rownames(dists)=1:14
dists
View(dists)
X11();heatmap.2(dists)
X11();plot(rowSums(dists))
X11();plot(rowSums(dists),  matchres$F1)
#zzz=helper_global_outliers(dat, ref_sub, clus_assig)
#hist(zzz,200)
#for(x in 1:length(unique(clus_assig))) {
#  if (sum(!ref_sub[x, ])!=0)    
#  hist(zzz[clus_assig==x], 50, main=x)
#      }
table(clus_truth[[i]], clus_assign)

hist(rf(10000, 50, 2500, ncp=0.0),1000)
mu<-mean(rf(10000, 50, 2500, ncp=0.0))
vr<-var((rf(10000, 50, 2500, ncp=0.0)))
2*mu/(mu-1)
summary(fitdist(rf(10000, 50, 2500, ncp=0), "f", start=list(df1=50, df2=2500), method='mge', gof = "AD")  )

#we will fit doubly not central F distribution with d2=k*d1, and equal non-centrality parameters
summary(fitdist(rf(100000, 50, 2500, ncp=10), "f", start=list(df1=50, df2=2500, ncp=10), method='mge', gof = "AD",  lower = 5, upper = 3000)  )

