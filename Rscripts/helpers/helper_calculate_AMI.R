############################################################
#Calculate pairwise AMI
#based on http://stackoverflow.com/questions/21831953/r-package-available-for-adjusted-mutual-information
library(parallel, warn.conflicts=F, quietly=T);
library(entropy)
is.naturalnumber <-
  function(x, tol = .Machine$double.eps^0.5)  x > tol & abs(x - round(x)) < tol

library(gmp)
library(parallel)
chunk <- function(x,n){
  if (n==1) {
    return(x)} else {
      return(split(x, cut(seq_along(x), n, labels = FALSE)))
    }
}

chooze2 <- function(n,k) (as.bigz(factorialZ(n) / (factorialZ(k) * factorialZ(n-k))))
#All data - U,   N,G - clusters, c-intersection
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
    return(c(log10(nom) - log10(denom), nom, denom))
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




#based on https://github.com/defleury/adjusted_mutual_information/blob/master/calculate.adjusted_mutual_information.R
AMI<-function(clus_algorithm, clus_truth, mc.cores=4){
  
  f_emi <- function(s1,s2,l1,l2,n, mc.cores=mc.cores){    #expected mutual information
    res<-mclapply(1:l1, function(i){
      s_emi <- 0
      for (j in 1:l2){
        #cat(i,j, '\n')
        min_nij <- max(1,s1[i]+s2[j]-n)
        max_nij <- min(s1[i],s2[j])
        n.ij <- seq(min_nij, max_nij)   #sequence of consecutive numbers
        t1<- (n.ij / n) * log((as.numeric(n.ij) * as.numeric(n)) / (as.numeric(s1[i])*as.numeric(s2[j])))
        t2 <- exp(lfactorial(s1[i]) + lfactorial(s2[j]) + lfactorial(n - s1[i]) + lfactorial(n - s2[j]) - lfactorial(n) - lfactorial(n.ij) - lfactorial(s1[i] - n.ij) - lfactorial(s2[j] - n.ij) - lfactorial(n - s1[i] - s2[j] + n.ij))
        emi <- sum(t1*t2)
        s_emi=s_emi + emi
        #cat(s_emi + emi, '\n')
      }
      return(s_emi)}, 
      mc.cores=mc.cores)
    #cat(unlist(res))
    return(sum(unlist(res)))
  }
  
  f_rez <- function(v1,v2, mc.cores=mc.cores){
    s1 <- tabulate(v1);
    s2 <- tabulate(v2);
    l1 <- length(s1)
    l2 <- length(s2)
    N <- length(v1)
    tij <- table(v1,v2)  #contingency table n(i,j)=t(i,j). this would be equivalent with table(v1,v2)
    mi <- mi.empirical(tij) #function for Mutual Information from package entropy
    h1 <- entropy(s1)
    h2 <- entropy(s2)
    nmi <- mi/max(h1,h2)        # NMI Normalized MI
    emi <- f_emi(s1,s2,l1,l2,N, mc.cores=mc.cores) # EMI Expected MI
    ami <- (mi-emi)/(max(h1,h2)-emi)  #AMI Adjusted MI
    cat('mi= ', mi, 'nmi= ', nmi, 'emi= ', emi, 'h1= ', h1, 'h2= ', h2, '\n')
    cat('ami= ', ami, '\n')
    return(ami)   
  }
  
  unassigned_T <- is.na(clus_truth)
  unassigned_A <- is.na(clus_algorithm)
  clus_algorithm <- clus_algorithm[!(unassigned_T | unassigned_A)]
  clus_truth <- clus_truth[!(unassigned_T | unassigned_A)]
  if (length(clus_algorithm) != length(clus_truth)) warning("vector lengths are not equal")
  
  #if there are cluster assignments not belonging to naturals, rename
  top_clus <- max(clus_algorithm)
  clus_algorithm<-unlist(lapply(clus_algorithm, function(x) if (!is.naturalnumber(x))
  return(abs(x) + top_clus+1) else return(x)))
  #rename labels to be sequence of natural numbers
  clus_algorithm<-match(clus_algorithm, unique(clus_algorithm))
  #cat(clus_algorithm, '\n')
  top_clus <- max(clus_truth)
  clus_truth<-unlist(lapply(clus_truth, function(x) if (!is.naturalnumber(x))
    return(abs(x) + top_clus+1) else return(x)))
  clus_truth<-match(clus_truth, unique(clus_truth))
  #cat(clus_truth, '\n')
  
  
  return(f_rez(clus_algorithm, clus_truth, mc.cores=mc.cores))
}

