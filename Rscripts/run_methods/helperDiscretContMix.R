#creates a mixture of discrete \delta_{plus}(0)
#distribution plus kde estimated distribution
#p - probabilities of zeros
helper_DiscretContMix<-function(n, p, kde){
  dc <- rbinom(n,1, p)
  zer <- rep(0, sum(dc))
  nonzer <- rkde(kde, n=sum(!dc), positive=T)
  return(sample(c(zer, nonzer), n))
}