#identify subdimensions where flowclusters live

#dat <- data[[i]]; clus_assig <- clus_assign[[i]]

library(moments)
helper_find_subdimensions<-function(data, clus_assign, skewCut=1){
  subdim<-matrix(NA, nrow=length(unique(clus_assign)), ncol=dim(data)[2])
  pvalue<-matrix(NA, nrow=length(unique(clus_assign)), ncol=dim(data)[2])
  skew  <-matrix(NA, nrow=length(unique(clus_assign)), ncol=dim(data)[2])
  data[data<0]<-0
  colnames(subdim) <- colnames(data)
  colnames(pvalue) <- colnames(data)
  colnames(skew) <- colnames(data)
  
  for(j in sort(unique(clus_assign)) ){
    clust<-as.data.frame(data[clus_assign==j ,])
    y=data[clus_assign==j ,];
    y=y[sample(sum(clus_assign==j), min(c(46340, sum(clus_assign==j)) )),]
    
    res <- lapply(as.data.frame(y), function(x) { 
      sk<-skewness(x);
      #browser()
      if (all(x==0)){
        return(list('sk'=sk, 'pv'= 0, 'assign'=FALSE))
      }
      if(length(x)<8){
        return(list('sk'=sk, 'pv'= 1000, 'assign'=FALSE)) 
      }
      test<-agostino.test(x, alternative = 'less');
      
      assign<-!(skewness(x)>skewCut & test$p.value<10^(-5) & sum(x==0)/length(x)>0.2);
      if(nrow(y)<50){
        assign<-!(skewness(x)>skewCut & test$p.value<10^(-2) & sum(x==0)/length(x)>0.1)
      }
      
      return(list('sk'=sk, 'pv'= test$p.value, 'assign'=assign))
      })
    #print(res)
    #print(unlist(lapply(res, function(x)  x['assign'])))
    subdim[j, ]<- unlist(lapply(res, function(x)  x['assign']))
    skew[j, ]<- unlist(lapply(res, function(x)  x['sk']))
    #print(unlist(lapply(res, function(x)  x['pv'])))
    pvalue[j, ]<- unlist(lapply(res, function(x)  x['pv']))
  }
  return(list('subdim'=subdim, 'pv'=pvalue, 'sk'=skew))
}

#subdi<-helper_find_subdimensions(dat, clus_assig)

#library(beanplot)
#for(j in sort(unique(clus_assig)) ){
#  cln=j
#  beanplot(as.data.frame(dat[clus_assig==cln ,][sample(sum(clus_assig==cln), min(c(5000, #sum(clus_assig==cln)) )), ]), main=j, beanlines = 'median',  names= unlist(lapply(as.character(subdi[cln, ]), function(x) substr(x, 1,1))) , what = c(1,1,1,0))
#}

#for(j in sort(unique(clus_assig)) ){
#     cln=j
#     beanplot(as.data.frame(dat[clus_assig==cln ,][sample(sum(clus_assig==cln), min(c(5000, sum(clus_assig==cln)) )), ]), main=j, beanlines = 'median',  names= unlist(lapply(as.character(subdi[cln, ]), function(x) substr(x, 1,1))) , what = c(1,1,1,0))
#   }

#cln=10
#beanplot(as.data.frame(dat[clus_assig==cln ,][sample(sum(clus_assig==cln), min(c(5000, sum(clus_assig==cln)) )), ]), main=j, beanlines = 'median',  names= unlist(lapply(as.character(subdi[cln, ]), function(x) substr(x, 1,1))) , what = c(1,1,1,0))
#cln=10
#beanplot(as.data.frame(dat[clus_assig==cln ,][sample(sum(clus_assig==cln), min(c(5000, sum(clus_assig==cln)) )), ]), main=j, beanlines = 'median',  names= unlist(lapply(as.character(subdi[cln, ]), function(x) substr(x, 1,1))) , what = c(1,1,1,0))

#cln=10
#beanplot(as.data.frame(dat[clus_assig==cln ,][sample(sum(clus_assig==cln), min(c(5000, sum(clus_assig==cln)) )), ]), main=cln, beanlines = 'median',  names= names(subdi[cln, ]) , what = c(1,1,1,0), lab.cex=0.1)



#heatmap(as.matrix(ifelse(subdi, 1,0)))

#(lapply(as.data.frame(data[clus_assign==1 ,]), function(x)   agostino.test(x, alternative #= 'less')))
