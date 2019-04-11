library(igraph)
library(parallel)
library(data.table)
library(scales)
library(Matrix)

source("../helpers/helper_match_evaluate_multiple.R")
source("../helpers/helper_match_evaluate_single.R")
source("../helpers/helper_match_evaluate_FlowCAP.R")
source("../helpers/helper_match_evaluate_FlowCAP_alternate.R")

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_MoC^u.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_H.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_evaluate_NMI.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^w_N.R')
source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_match_evaluate_S^u_N.R')








#triples=count_motifs(g, size = 2)
g_deg<-degree(g)
vert_weight<-strength(g)
vert_degree<-degree(g, v = V(g))
g_adj<-as_adjacency_matrix(g, type = c("both"), attr = NULL,
                           edges = FALSE, names = TRUE, sparse = igraph_opt("sparsematrices"))

g_w=as_adjacency_matrix(g, attr="weight", type = c("both"), 
                        edges = FALSE, names = TRUE, sparse = igraph_opt("sparsematrices"))


#gr_w <- g_w * rwc^3
#mc reweighting procedure
#############################################################################
#
#############################################################################
gr_res <- mcOut(g_w, addLoops = F, expansion = 2, inflation = 10,  max.iter = 2, ESM = TRUE); gc()
gr_res[[1]]
gr_w<-gr_res[[2]] 
head(sort(colSums(gr_w),decreasing = T))
hist(colSums(gr_w),  500000)
hist(colSums(gr_w), xlim=c(0,1), 5000)
table(lbls[rowSums(gr_w)==0])
table(lbls[colSums(gr_w)==0])
table(lbls[colSums(gr_w)>0])

re_strength<-colSums(gr_w)
str_den<-density(re_strength)
co_indx<-which(min(str_den$y[str_den$x>0  & str_den$x<1]) == str_den$y)
cut_off<-str_den$x[co_indx]; cut_off
hist(re_strength)
hist(g_w@x,500)


table(lbls[re_strengt<cut_off])
table(lbls[re_strengt<2.488284e-64])
table(lbls[colSums(gr_w)==1])
table(lbls)

nnzero(gr_w)
gr_w<-(gr_w+t(gr_w))/2; gc()
nnzero(gr_w);gc()




crit<-(vert_weight/colSums(gr_w))



lbls
hist(as(gr_w, "dgTMatrix")@x,50000000, xlim=c(0,.0000005));gc()
#hist(as(rwc^3, "dgTMatrix")@x,500)
hist(as(g_w, "dgTMatrix")@x,500)


g2<-graph_from_adjacency_matrix(gr_w, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()

g_rwc = graph_from_adjacency_matrix((rwc)^2, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA)

hist(strength(g_rwc),500)
hist(strength(g_rwc)*degree(g_rwc),5000)
hist(degree(g_rwc),5000, xlim=c(0,200))

hist(strength(g2),500)
hist(strength(g),500)
hist(strength(g2)*degree(g2),5000)

g_ww<-(gr_w * g_w); gc()
g3<-graph_from_adjacency_matrix(g_ww, mode =  "undirected", weighted = TRUE, diag = F,                            add.colnames = NULL, add.rownames = NA); gc()


cord_g<-length(V(g))
trian=count_triangles(g)
Ntrian=sum(trian)/3
g_deg<-degree(g)
deg_list= unlist(unique(g_deg))
#vertices of triangles
triVert<-as.numeric(triangles(g));gc()
#generate vertices ID's of triangles' edges
vert_ID<-c(triVert[c(T, T, F)], triVert[c(T, F, T)], triVert[c(F, T, T)]);gc()
#edge_ends<-unlist(mclapply(1:(length(triVert)/3), function(i) {a=3*(i-1); c(triVert[1+a], triVert[2+a], triVert[1+a], triVert[3+a], triVert[2+a], triVert[3+a]) }, mc.cores=3))
edgeIDs<- get.edge.ids(g, vp=vert_ID, directed = F, error = FALSE, multi = FALSE)
edgeWe<-edge_attr(g, 'weight', index = edgeIDs);gc()
cent_IDX<-c(triVert[c(F, F, T)], triVert[c(F, T, F)], triVert[c(T, F, F)]);gc()
list_cent<-unique(cent_IDX);gc() 
#weight in the outer edges of the triangles enclosing the vertex
#for fast access/search data.table is used
edgeWeTbl<-data.table(edgeWe, cent_IDX); setkey(edgeWeTbl, cent_IDX);gc()
edgeWeSum<-edgeWeTbl[, sum(edgeWe), by=cent_IDX];gc()

#emanating vertices
vert_weight<-strength(g)
vert_degree<-degree(g, v = V(g))
# Weight of neighboring vertices divided by degree
#for fast access/search data.table is used
g_neighbors=ego(g, order=1, nodes=V(g), mode = "all")
#list of ins:
from_g<-unlist(mclapply(1:(cord_g) , function(i) rep(i, vert_degree[i]), mc.cores=2));gc()
#list of outs:
inc_v<-vert_degree+1
to_g<-unlist(mclapply((1:cord_g), function(i) g_neighbors[[i]][2:inc_v[i]], mc.cores=2));gc()
#list of weights
edge_tbl<-data.table(from_g=from_g, to_g=to_g)

neighb_strength<-edge_tbl[  ,sum(vert_weight[to_g])/(.N) ,by=from_g]
smoothScatter(log(vert_weight), log(neighb_strength[,V1]))



#e_b=estimate_edge_betweenness(g, e=E(g), directed = F, cutoff=2)





pval<-unlist(mclapply(1:gorder(g), function(x){ 
  if (g_deg[x]!=0){
    n=g_deg[x]*(g_deg[x]-1)/2 
    k=trian[x]
    kp=0:(k-1)
    return(sum(exp(lchoose(n, kp)+(log(glob_ptri))*kp+(log(1-glob_ptri))*(n-kp))))
  } else return(1)
}, mc.cores=2))
hist(pval,100)


pv_st<-vector(mode='numeric', length=length(gorder(g)))
rank_tri<-vector(mode='numeric', length=length(gorder(g)))
p_tri_deg<-vector(mode='numeric', length=length(deg_list))
for (j in deg_list){
  p_tri_deg[j] <- sum(trian[g_deg==j])/sum(unlist(lapply(g_deg[g_deg==j], function(x) x*(x-1)/2)))
  rank_tri[g_deg==j] <- order(trian[g_deg==j]/unlist(lapply(g_deg[g_deg==j], function(x) x*(x-1)/2)), decreasing = T)
  for (i in 1:sum(g_deg==j)){
    n=g_deg[g_deg==j][i]*(g_deg[g_deg==j][i]-1)/2 
    k=trian[g_deg==j][i]
    kp=0:(k-1)
    pv_st[g_deg==j][i]<-sum(exp(lchoose(n, kp)+(log(p_tri_deg[j]))*kp+(log(1-p_tri_deg[j]))*(n-kp)))
  }
}

j=150
o<-order(trian[g_deg==j]/unlist(lapply(g_deg[g_deg==j], function(x) x*(x-1)/2)))
pv<-(trian[g_deg==j]/unlist(lapply(g_deg[g_deg==j], function(x) x*(x-1)/2)))
pv_st[g_deg==j]
rank_tri[g_deg==j]

#calculate the sum of weights in outer edges of triangles


#calculate sum of weights of vertices
#em_vertices<-gsub("\\--.*","",E(g)[ from(V(g)) ])


#plot vertex_veight vs neibr weights
yw<-(edgeWeSum[,V1]); xw<-(vert_weight[as.numeric((edgeWeSum[,cent_IDX]))])
yw<-log(edgeWeSum[,V1]); xw<-log(vert_weight[as.numeric((edgeWeSum[,cent_IDX]))])
yw<-(edgeWeSum[,V1]); xw<-(vert_weight[as.numeric((edgeWeSum[,cent_IDX]))]*(g_deg[as.numeric(edgeWeSum[,cent_IDX])]-1)/2)

library(msir)
# Calculates and plots a 1.96 * SD prediction band, that is,
# a 95% prediction band
plot(xw, yw, pch='.')
plot(log(xw), log(yw), pch='.')
plot(xw, yw, pch='.', col=ifelse(g_deg*(g_deg-1) + 23*trian < 2500, 'black', 'red'))
lsd <- loess.sd(yw ~ xw, nsigma = 1.96)
lines(lsd$x, lsd$y)
lines(lsd$x, lsd$upper, lty=2)
lines(lsd$x, lsd$lower, lty=2)

IDXloess <- (yw[order(xw)] < lsd$lower)
points(xw[order(xw)][IDXloess], yw[order(xw)][IDXloess], col='red', pch='.')
Nameloess<-names(xw[order(xw)][IDXloess])



#w_triangle
yw<-(edgeWeSum[,cent_IDX]/(g_deg[as.numeric(edgeWeSum[,cent_IDX])]*(g_deg[as.numeric(edgeWeSum[,cent_IDX])]-1)/2))
xw<-((g_deg[as.numeric(edgeWeSum[,cent_IDX])])^(-1)*vert_weight[as.numeric(edgeWeSum[,cent_IDX])])
#exclude the most on non-reciprocal connections
edgeWeSum2<-edgeWeSum[g_deg[as.numeric(names(edgeWeSum))]<=30]
yw<-(edgeWeSum2)
xw<-((g_deg[as.numeric(names(edgeWeSum2))]-1)/2 * vert_weight[as.numeric(names(edgeWeSum2))])



plot(xw, yw, pch='.')

lsd <- loess.sd(yw ~ xw, nsigma = 1.96)
lines(lsd$x, lsd$y)
lines(lsd$x, lsd$upper, lty=2)
lines(lsd$x, lsd$lower, lty=2)

IDXloess <- (yw[order(xw)] < lsd$lower)
points(xw[order(xw)][IDXloess], yw[order(xw)][IDXloess], col='red', pch='.')
Nameloess<-names(xw[order(xw)][IDXloess])





pv_st<-vector(mode='numeric', length=length(gorder(g)))
rank_tri<-vector(mode='numeric', length=length(gorder(g)))
p_tri_deg<-vector(mode='numeric', length=length(deg_list))
for (j in deg_list){
  p_tri_deg[j] <- sum(trian[g_deg==j])/sum(unlist(lapply(g_deg[g_deg==j], function(x) x*(x-1)/2)))
  rank_tri[g_deg==j] <- order(trian[g_deg==j]/unlist(lapply(g_deg[g_deg==j], function(x) x*(x-1)/2)), decreasing = T)
  for (i in 1:sum(g_deg==j)){
    n=g_deg[g_deg==j][i]*(g_deg[g_deg==j][i]-1)/2 
    k=trian[g_deg==j][i]
    kp=0:(k-1)
    pv_st[g_deg==j][i]<-sum(exp(lchoose(n, kp)+(log(p_tri_deg[j]))*kp+(log(1-p_tri_deg[j]))*(n-kp)))
  }
}
hist(pv_st,500)


sum(IDX!=T & lbls==0)
sum(IDX!=T)
sum(lbls==0)

hist(strength(g)[IDX],500)
hist(strength(g)[!IDX],500)



sum(pv_st>10^(-2))
IDX <- pv_st>10^(-4)#global probability estimate -based ranking
IDX <- !(rank_tri %in% c(1,2))#vertices with the two lowest numbers of triangles for each k
IDX<- !((trian<(min(trian)+50)) & (g_deg<=30) )#all vertices included at least in one triangle
IDXo<-!(((1:gorder(g)) %in% setdiff(V(g), list_cent)) & (g_deg<=30) )
IDXi<-!(((1:gorder(g)) %in% setdiff(V(g), list_cent)) & (g_deg>30) )
#hist(strength(g)[!IDX]/g_deg[!IDX],500)
hist(strength(g)[!IDXi]*(g_deg[!IDXi]-1),100, xlim=c(0,100))
hist(strength(g)[!IDXo]*(g_deg[!IDXo]-1), col='red', 100, add=T)

#strength per triangle edge
hist(strength(g)[!IDXi]/(g_deg[!IDXi]),100)
hist(strength(g)[!IDXo]/(g_deg[!IDXo]), col='red', 100, add=T)

IDX<-!(g_deg*(g_deg-1) + 23*trian < 2500)
 sum(!IDX)
IDX<-!(xw+10.85*yw<40)
sum(!IDX)
IDX<-!((xw+10.85*yw)/yw<)
sum(!IDX)

IDX<-!(xw<20)
sum(!IDX)

IDX<-!((yw-1.4299366)/xw>2/10.85)
sum(!IDX)
smoothScatter(xw, yw-1.4299366, pch='.', col=ifelse(IDX, 'black', 'red'))
points(xw[!IDX], yw[!IDX], pch='.', col='red')
smoothScatter(xw, (yw-1.4299366)/xw, pch='.', col=ifelse(IDX, 'black', 'red'))
abline(h=0.08701)
IDX<-!((yw-1.4299366)/xw>0.08701)
sum(!IDX)
points(xw[!IDX], (yw[!IDX]-1.4299366)/xw[!IDX], pch='.', col='red')

#logged weights
#################################################################
yw<-(edgeWeSum[,V1]); xw<-(vert_weight[as.numeric((edgeWeSum[,cent_IDX]))]*(g_deg[as.numeric(edgeWeSum[,cent_IDX])]-1)/2)
smoothScatter(log(xw), log(yw), bandwidth=0.1, nrpoints = 11510)
summary(lm(log(yw)~log(xw)))
ic<-(summary(lm(log(yw)~log(xw))))$coefficients[1]
c<-(summary(lm(log(yw)~log(xw))))$coefficients[2]
abline(ic,c, col='red')
IDX<-!((log(yw)-c)/log(xw)>c)
smoothScatter(log(xw), (log(yw)-ic)/log(xw), pch='.', col=ifelse(IDX, 'black', 'red'), bandwidth=0.1, nrpoints = 11510)

plot(log(xw), log(yw)-ic -c*log(xw), pch='.')
#smoothScatter(log(xw), (log(yw)-ic)/log(xw), pch='.',  bandwidth=0.1, nrpoints = 11510)
smoothScatter(log(xw), log(yw)-ic -c*log(xw), pch='.',  bandwidth=0.21, nrpoints = 1151)
abline(h=0)
abline(h=mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)))
IDX<-!(log(yw)-ic -c*log(xw)>mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)))
sum(!IDX)
abline(v=mean(log(xw))-3*sd(log(xw)))
IDX=!(log(yw)-ic -c*log(xw)>mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)) | (log(xw) < mean(log(xw))-3*sd(log(xw))))
#IDX=!(log(yw)-ic -c*log(xw)>mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)) | (log(xw) < log(15)))
points(log(xw[!IDX]),log(yw[!IDX])-ic -c*log(xw[!IDX]) , pch='.', col='red')
sum(!IDX)

lsd <- loess.sd((log(yw)-ic)/log(xw)~ log(xw), nsigma = 2)
smoothScatter(log(xw), (log(yw)-ic)/log(xw), pch='.', col=ifelse(IDX, 'black', 'red'))
points(log(xw), (log(yw)-ic)/log(xw), pch='.', col=ifelse(IDX, 'black', 'red'))
lines(lsd$x, lsd$upper, lty=2)
lines(lsd$x, lsd$lower, lty=2)
lines(lsd$x, lsd$y, lty=2)

IDXloess <- ((log(yw)[order(xw)]-ic)/log(xw)[order(xw)] > lsd$upper)
points(log(xw)[order(xw)][IDXloess], (log(yw)[order(xw)][IDXloess]-ic)/log(xw)[order(xw)][IDXloess], col='red', pch='.')
Nameloess<-names(xw[order(xw)][IDXloess])
#######################################################################################
#no log
plot(xw, yw, col=alpha('black', alpha = 0.5), pch='.', xlim=c(0, 100))
plot(log(xw), log(yw), col=alpha('black', alpha = 0.5), pch='.', xlim=c(0, log(1000)))
summary(lm(yw~xw))
ic<-(summary(lm(yw~xw)))$coefficients[1]
c<-(summary(lm(yw~xw)))$coefficients[2]

smoothScatter(xw, yw-ic -c*xw, pch='.',  bandwidth=0.1, nrpoints = 11510)
abline(h=0)
abline(h=mean(yw-ic -c*xw)+3*sd(yw-ic -c*xw))
IDX<-!(yw-ic -c*xw>mean(yw-ic -c*xw)+3*sd(yw-ic -c*xw))
sum(!IDX)
abline(v=mean(xw)-2*sd(xw))
IDX=!(log(xw) < 1.5)
IDX=!(yw-ic -c*xw>mean(yw-ic -c*xw)+3*sd(yw-ic -c*xw) | (xw < (mean(xw)-2.5*sd(xw))  & yw-ic - c*xw> 0))
points(log(xw[!IDX]),log(yw[!IDX])-ic -c*log(xw[!IDX]) , pch='.', col='red')
sum(!IDX)


#robust resgression
#################################################################
yw<-(edgeWeSum[,V1]); xw<-(vert_weight[as.numeric((edgeWeSum[,cent_IDX]))]*(g_deg[as.numeric(edgeWeSum[,cent_IDX])]-1)/2)
smoothScatter((xw), (yw), bandwidth=0.1, nrpoints = 1151)
smoothScatter((xw), (yw), bandwidth=0.1, nrpoints = 1151, xlim=c(0,100))
rr <-lm((yw)~(xw))
rrs<-summary(rr)
out.thresh = rrs$control$eps.outlier
out.liers = rrs$rweights[which(rrs$rweights <= out.thresh)]
IDX<-!(1:cord_g %in% names(out.liers))
points(xw[!IDX], yw[!IDX], col='red', pch='.')

o<-segmented(rr,seg.Z=~xx)

ic<-rr$coefficients[1]
c<-rr$coefficients[2]
smoothScatter(xw,(yw-ic -c*xw)/xw, pch='.',  bandwidth=0.1, nrpoints = 11510)
smoothScatter(xw,(yw-c*xw)/xw, pch='.',  bandwidth=0.1, nrpoints = 1151, xlim=c(0,100))
smoothScatter(xw,yw, pch='.',  bandwidth=0.5, nrpoints = 1151, xlim=c(0,100))
abline(ic,c, col='red')
#IDX<-!(1:cord_g %in% names(out.liers) & yw/xw-ic -c*xw >0)
IDX<-!((yw-ic -c*xw)/xw>0.25 )
points((xw[!IDX]), (yw[!IDX]-ic -c*(xw[!IDX]))/(yw[!IDX]) , pch='.', col='red')
plot((xw[!IDX]), (yw[!IDX]-ic -c*(xw[!IDX]))/(yw[!IDX]) ,  col='red')
sum(!IDX)

IDX<-!(xw<20 & yw - ic -c*xw >0)
points((xw[!IDX]), yw[!IDX] , pch='.', col='red')
sum(!IDX)


rr <-lmrob((yw[!IDX])~(xw[!IDX]))
rrs<-summary(rr)
ic<-rr$coefficients[1]
c<-rr$coefficients[2]
abline(ic,c, col='red')

plot(log(xw), log(yw)-ic -c*log(xw), pch='.')
#smoothScatter(log(xw), (log(yw)-ic)/log(xw), pch='.',  bandwidth=0.1, nrpoints = 11510)
smoothScatter(log(xw), log(yw)-ic -c*log(xw), pch='.',  bandwidth=0.21, nrpoints = 1151)
abline(h=0)
abline(h=mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)))
abline(v=mean(log(xw))-3*sd(log(xw)))
IDX=!(rr$w<1 & (log(yw)-ic -c*log(xw)>0))
#IDX=!(log(yw)-ic -c*log(xw)>mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)) | (log(xw) < log(15)))
points(log(xw[!IDX]), log(yw[!IDX])-ic -c*log(xw[!IDX]) , pch='.', col='red')
hist(xw, 1000, xlim=c(0,50))
IDX<-!(xw<15)
sum(!IDX)

#################################################################
# new coordinates
#################################################################
gd_s<-g_deg[as.numeric(edgeWeSum[,cent_IDX])]
yw<-(edgeWeSum[,V1])/((gd_s-1)/2); xw<-vert_weight[as.numeric((edgeWeSum[,cent_IDX]))]
hist((edgeWeSum[,V1])/((gd_s-1)/2)/xw, 500, xlim=c(0,0.3))
smoothScatter((xw), (edgeWeSum[,V1])/((gd_s-1)/2)/xw, xlim=c(0, 3), main='alpha')
smoothScatter((xw), (yw))
smoothScatter((xw), (yw), bandwidth=0.1, nrpoints = 1151, xlim=c(0,2))
rr <-lm((yw)~(xw))
rrs<-summary(rr)
out.thresh = rrs$control$eps.outlier
out.liers = rrs$rweights[which(rrs$rweights <= out.thresh)]
IDX<-!(1:cord_g %in% names(out.liers))
points(xw[!IDX], yw[!IDX], col='red', pch='.')

o<-segmented(rr,seg.Z=~xx)

ic<-rr$coefficients[1]
c<-rr$coefficients[2]
smoothScatter(xw,(yw-ic -c*xw)/xw, pch='.',  bandwidth=0.1, nrpoints = 11510)
smoothScatter(xw,(yw-ic -c*xw)/xw, pch='.',  bandwidth=0.1, nrpoints = 1151, xlim=c(0,100))
smoothScatter(xw,yw, pch='.', nrpoints = 1151)
abline(ic,c, col='red')
#IDX<-!(1:cord_g %in% names(out.liers) & yw/xw-ic -c*xw >0)
IDX<-!((yw-ic -c*xw)/xw>0.25 )
points((xw[!IDX]), (yw[!IDX]-ic -c*(xw[!IDX]))/(yw[!IDX]) , pch='.', col='red')
plot((xw[!IDX]), (yw[!IDX]-ic -c*(xw[!IDX]))/(yw[!IDX]) ,  col='red')
sum(!IDX)

IDX<-!(xw<20 & yw - ic -c*xw >0)
points((xw[!IDX]), yw[!IDX] , pch='.', col='red')
sum(!IDX)


rr <-lmrob((yw[!IDX])~(xw[!IDX]))
rrs<-summary(rr)
ic<-rr$coefficients[1]
c<-rr$coefficients[2]
abline(ic,c, col='red')

plot(log(xw), log(yw)-ic -c*log(xw), pch='.')
#smoothScatter(log(xw), (log(yw)-ic)/log(xw), pch='.',  bandwidth=0.1, nrpoints = 11510)
smoothScatter(log(xw), log(yw)-ic -c*log(xw), pch='.',  bandwidth=0.21, nrpoints = 1151)
abline(h=0)
abline(h=mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)))
abline(v=mean(log(xw))-3*sd(log(xw)))
IDX=!(rr$w<1 & (log(yw)-ic -c*log(xw)>0))
#IDX=!(log(yw)-ic -c*log(xw)>mean(log(yw)-ic -c*log(xw))+3*sd(log(yw)-ic -c*log(xw)) | (log(xw) < log(15)))
points(log(xw[!IDX]), log(yw[!IDX])-ic -c*log(xw[!IDX]) , pch='.', col='red')
sum(!IDX)












hist(strength(g)[((1:gorder(g)) %in% setdiff(V(g), list_cent))]*(g_deg[((1:gorder(g)) %in% setdiff(V(g), list_cent))]-1), 100, xlim=c(0,200))

plot(strength(g)*g_deg)

############################################################################################
#IDX<-(!(1:gorder(g))  %in% as.numeric(Nameloess))
# remove edges with high betwenness 
indg<-subgraph.edges(g, eids=E(g)[(1:length(E(g)))[log(e_b)<10]], delete.vertices = TRUE)

IDX<- !(colSums(gr_w)<0.5)
indg<-induced_subgraph(g, (1:gorder(g))[IDX])
comi<-cluster_louvain(indg)
com<-cluster_louvain(g)
system.time(com2<-cluster_louvain(g2))
com3<-cluster_louvain(g3)


#res_mcl<-mcl(x=g_w, addLoops = F, expansion = 2, inflation = 2, allow1 = FALSE, max.iter = 3, ESM = T)

#comi<-cluster_infomap(indg)
#com<-cluster_infomap(g)

comn=membership(com)

comf<-rep(NA,length(lbls))
comf[IDX]<-membership(comi)
comf[!IDX]<-100

table(membership(comi))
table(membership(com))
table(membership(com2))
table(lbls)
table(lbls[!IDX])

lblsD <- lbls[lbls!=0]#remove outliers from evaluation 
comfD <- comf[lbls!=0]
comnD <- comn[lbls!=0]

helper_match_evaluate_multiple(comfD, lblsD)
helper_match_evaluate_multiple(comnD, lblsD)

helper_match_evaluate_multiple_SweightedN(comfD, lblsD)
helper_match_evaluate_multiple_SweightedN(comfD, lblsD)

resS_Nu <- vector("list", length(clus))
names(resS_Nu) <- names(clus)

#with assignments of outliers:
helper_match_evaluate_multiple(comf, lbls)
helper_match_evaluate_multiple(membership(com), lbls)
helper_match_evaluate_multiple(membership(com2), lbls)
helper_match_evaluate_multiple(membership(com3), lbls)

table(lbls[!IDX])

#tsne visualisation
library(Rtsne)
#tsne3D <- Rtsne(X=cl_coord, dims = 3, perplexity = 30, verbose = T, max_iter = 1000) 
#tsne <- Rtsne(X=cl_coord, dims = 2, perplexity = 30, verbose = T, max_iter = 1000) 
ncolors=length(unique(lbls))
col_true=rainbow(ncolors)
colors<-unlist(lapply(lbls, function(x) col_true[x]))
colors[lbls==0]='black'
color2=ifelse(IDX, 'green', 'red'); color2[lbls==0 & IDX]='black' #check hits and misses
#data<- cbind(f, clusters)
#plot(res_tsne$tsne_out$Y,col=colors, pch='.', cex=1)

plot(tsne$Y,  col=colors, pch='.')
plot(tsne$Y,  col=ifelse(IDX, 'green', 'red'), pch='.')
plot(tsne$Y,  col=ifelse(lbls==0, 'red', 'green'), pch='.')
plot(tsne$Y,  col=color2, pch='.')

resi<-helper_match_evaluate_multiple(membership(comi), lbls)



open3d()
plot3d(tsne3D$Y,  col=colors, pch='.') 

open3d()
plot3d(tsne3D$Y,  col=ifelse(IDX, 'green', 'red'), pch='.') 

open3d()
plot3d(tsne3D$Y,  col=ifelse(lbls==0, 'red', 'green'), pch='.') 

open3d()#check hits and misses
plot3d(tsne3D$Y,  col=color2, pch='.') 



