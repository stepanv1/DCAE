#calculate convolution of n
#independent Bernoulli densities 
#to estimate density of sum
library(distr)
convBernoulli <-function(pvec){
  n<-length(pvec)
  B<-Binom(prob=pvec[1], size=1)
  for(i in 2:n){
    Add<-Binom(prob=pvec[n], size=1)
    B <- convpow(B+Add,1)
  }
  pdf  <- d(B) #distr function
  return(list('pdfvec'=pdf(0:n), 'pdf'=pdf))
}

#g=c(0.2,0.1,0.05,0.05, 0.05,0.05, 0.05,0.05, 0.05,0.05,0.05,0.05, 0.05,0.05, 0.05,0.05)
#X11();plot(0:length(g),convBernoulli(g)$pdfvec)
#sum(convBernoulli(g)$pdfvec[1:5])

#pvec<-rep(c(0.1,0.2,0.7, 0.25,0.1,0.65, 0.2,0.33,0.47),1)
#supp<-c(0,1,2)
#distroptions(DefaultNrFFTGridPointsExponent=14,TruncQuantile=1e-6,DistrCollapse=F, DistrResolution=1e-7)
convDiscrete <-function(pvec, supp){
  #browser()
  distroptions(DefaultNrFFTGridPointsExponent=14,TruncQuantile=1e-6,DistrCollapse=F, DistrResolution=1e-7)
  ls <- length(supp)
  n <-length(pvec)/ls
  
  D <- DiscreteDistribution(supp = supp, prob = pvec[1:ls])
  #X11();plot(D)
  for(i in 2:n){
    Add<-DiscreteDistribution(supp = supp, prob = pvec[ls*(i-1)+(1:ls)])
    D <- convpow(D+Add,1)
  }
  pdf  <- d(D) #distr function
  #print('done')
  return(list('pdfvec'=pdf(0:((ls-1)*n) ), 'pdf'=pdf))
}
#zzz=convDiscrete(pvec, supp)
#ls <- length(supp)
#n <-length(pvec)/ls
#X11(); plot(0:((ls-1)*n), zzz$pdfvec)

