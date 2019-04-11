#calculation of the probability of the sequence
#of cosines beetwin D-dimensional vectors

betas <- pbeta((1:100)/100, shape1=4.50, shape2=4.50, ncp = 0, lower.tail = TRUE, log.p = FALSE)
plot(betas); plot(diff(betas))
mean(betas)

pval=2*(1-pbeta(0.95, shape1=4.50, shape2=4.50, ncp = 0, lower.tail = TRUE, log.p = FALSE)) #coefficient 2 
#since the distribution is two-tailed
#Apply this by taking a product of angles between clusters to the section where cluster is suspected
#cpmpare to the area where cluster is not suspectefd