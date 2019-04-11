library(gmp)
library(parallel)
chunk <- function(x,n){
  if (n==1) {
  return(x)} else {
  return(split(x, cut(seq_along(x), n, labels = FALSE)))
  }
}

chooze2 <- function(n,k) (as.bigz(factorialZ(n) / (factorialZ(k) * factorialZ(n-k))))

pCardIntersection<-function(U, G, N, c, log10=TRUE, mc.cores=10){
  #if (G>N){tmp=G; G=N; N=tmp}
  nom <- sum.bigz(chooseZ(U,(c:G)) * chooseZ(U-(c:G), G-(c:G))* chooseZ(U-G, N-(c:G)))
  denom<- sum.bigz(chooseZ(U,(0:G)) * chooseZ(U-(0:G), G-(0:G)) * chooseZ(U-G, N-(0:G)))
  if (log10==TRUE){
    return(log10(nom) - log10(denom))
    } else {
    return(nom/denom)  
    }
}

pCardIntersection2<-function(U, G, N, c, log10=TRUE, mc.cores=10){
  #if (G>N){tmp=G; G=N; N=tmp}
  nom<-sum.bigz(unlist(mclapply(chunk(c:G, mc.cores), function(x) chooze2(U,x)*chooze2(U-x, G-x)*chooze2(U-G, N-x), mc.cores=mc.cores)));gc()
  denom<-sum.bigz(unlist(mclapply(chunk(0:G, mc.cores), function(x) chooze2(U,x)*chooze2(U-x, G-x)*chooze2(U-G, N-x), mc.cores=mc.cores)));gc()
  if (log10==TRUE){
    return(log10(nom) - log10(denom))
  } else {
    return(nom/denom)  
  }
}

#plot(unlist(lapply(0:15, function(x) 1-as.numeric(pCardIntersection(30, 15, 15, x, log10=F)))))

pdfIntersection<-function(U, G, N, c, log10=TRUE, mc.cores=10){
  #if (G>N){tmp=G; G=N; N=tmp}
  nom <- chooseZ(U,(c)) * chooseZ(U-(c), G-(c))* chooseZ(U-G, N-(c))
  denom<- sum.bigz(chooseZ(U,(0:G)) * chooseZ(U-(0:G), G-(0:G)) * chooseZ(U-G, N-(0:G)))
  if (log10==TRUE){
    return(log10(nom) - log10(denom))
  } else {
    return(nom/denom)  
  }
}  
#plot(unlist(lapply(0:15, function(x) as.numeric(pdfIntersection(30, 15, 15, x, log10=F)))))




