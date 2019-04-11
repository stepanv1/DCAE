#Aszalinin skewd distros
library(sn)
library(beanplot)
cp <- list(mean=c(1,1), var.cov=matrix(c(3,2,2,3)/3, 2, 2), gamma1=c(0.7, 0.7))
dp <- cp2dp(cp, "SN")
rnd <- rmsn(10000, dp=dp)
beanplot(rnd[,2])
plot(x=rnd[,1], y=rnd[,2], pch='.')

betacor=
rnd <- rmsn(10000,xi=c(1,1), Omega=matrix(c(3,2,2,3)/3, 2, 2), alpha = 1*c(1,1), tau=0)
rnd<-rmst(10000,xi=c(1,1), Omega=matrix(c(3,2,2,3)/3, 2, 2), alpha = 1*c(1,1), nu=10)
beanplot(rnd[,2])
plot(x=rnd[,1], y=rnd[,2], pch='.')

library(MASS)

# We will use the command mvrnorm to draw a matrix of variables
#http://www.econometricsbysimulation.com/2014/02/easily-generate-correlated-variables.html
# Let's keep it simple, 
mu <- rep(0,4)
Sigma <- matrix(.7, nrow=4, ncol=4) + diag(4)*.3

rawvars <- mvrnorm(n=10000, mu=mu, Sigma=Sigma)

cov(rawvars); cor(rawvars)
# We can see our normal sample produces results very similar to our 
#specified covariance levels.

# No lets transform some variables
pvars <- pnorm(rawvars)

# Through this process we already have 
cov(pvars); cor(pvars)
# We can see that while the covariances have dropped significantly, 
# the simply correlations are largely the same.

plot(rawvars[,1], pvars[,2], main="Normal of Var 1 with probabilities of Var 2")

# To generate correlated poisson
poisvars <- qlnorm(pvars, meanlog=log(2))
mean(poisvars[,1])
cor(poisvars, rawvars) 
# This matrix presents the correlation between the original values generated
# and the tranformed poisson variables.  We can see that the correlation matrix
# is very similar though somewhat "downward biased".  This is because any
# transformation away from the original will reduce the correlation between the
# variables.

plot(poisvars,rawvars, main="Poisson Transformation Against Normal Values")

library('MethylCapSig')
X <- mvlognormal(n = 10000, Mu = c(3, 3), 
                 Sigma = c(1,1), R=matrix(c(3,2,2,3)/3, 2, 2));
beanplot(X[,1])
plot(x=X[,2], y=X[,3], pch='.')










