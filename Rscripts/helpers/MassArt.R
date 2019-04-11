#Artificiall data set mimicking 
#Mass Cytometry
n_cl=15
ndim=40
#Form Signal Dimensions matrix
#'Average' number of signal (relevant) dimensions 
SignalDimLocation <- 5
SignalDimMin <- 3
SignalDimMax <- 15
prob <- SignalDimLocation/ndim
SigDims<-rbinom(n_cl, ndim, prob)
SigDims<-ifelse(SigDims<3, SigDims+1, SigDims)
SigDims<-ifelse(SigDims>SignalDimMax, SigDims-1, SigDims)
SignalDim <- matrix(unlist(lapply(SigDims, function(x) sample(c(rep(1,x), rep(0, ndim-x)) ))), nrow = n_cl,byrow=T)
library(sna)
plot.sociomatrix(SignalDim, diaglab=F)

#Generate cluster populations
ClusterPopulationsLocation <- 2000
ClusPop<- ClusterPopulationsLocation*rbinom(n_cl, 7, 0.2)
ClusPop<-ifelse(ClusPop<100, ClusPop+100, ClusPop)
ClusPop<-ifelse(ClusPop>15000, 15000, ClusPop)
sum(ClusPop)

#Generate cluster means, taking into account SignalDim

#form Noisy Dimensions per cluster using Samusik data for noisy dimensions
#read RData file, generated in the script ...










#generate clusters



#unify clusters
