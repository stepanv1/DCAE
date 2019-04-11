library(MASS)
library(RColorBrewer)
#
hist_2d <- function(x,y, n=25){
  # Color housekeeping
  rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))
  r <- rf(32)
  
  h1 <- hist(x, breaks=n, plot=F)
  h2 <- hist(y, breaks=n, plot=F)
  top <- max(h1$counts, h2$counts)
  k <- kde2d(x, y, n=n)
  
  # margins
  oldpar <- par()
  par(mar=c(3,3,1,1))
  layout(matrix(c(2,0,1,3),2,2,byrow=T),c(3,1), c(1,3))
  image(k, col=r) #plot the image
  par(mar=c(0,2,1,0))
  barplot(h1$counts, axes=F, ylim=c(0, top), space=0, col='red')
  par(mar=c(2,0,0.5,1))
  barplot(h2$counts, axes=F, xlim=c(0, top), space=0, col='red', horiz=T)
}