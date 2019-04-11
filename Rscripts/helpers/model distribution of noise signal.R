#model distribution of noise signal
Ncells=5000
Nmes=20


channelInt<-lapply(1:Ncells, function(x){
  unlist(lapply(1:(rchisq(1, 5)^3+17), function(y){
    z=rnorm(1)
    (ifelse(z>3,z,0))
  }))
})

MeanchannelInt<-asinh(unlist(lapply(channelInt,sum))/5)
X11();hist(MeanchannelInt,300)
skewness(MeanchannelInt,300)
X11();hist(rnorm(500)^8,300)

X11();hist(rchisq(10000,20),300)
skewness(rchisq(10000,3))

X11();hist(rchisq(10000, 5)+17,200)
