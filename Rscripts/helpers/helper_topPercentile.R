#returns indices above (below) xth percentile in vector v

helper_topPercentile <- function(x, v, direction = 'above'){
  ranks<-frank(v, na.last=TRUE, ties.method="first")
  lim_rank <- x * length(v)
  positions <- 1:length(v)
  if (direction=='above') {
    positions <- positions[ifelse(ranks > lim_rank, TRUE, FALSE)]
      } else {
    positions <- positions[ifelse(ranks < lim_rank, TRUE, FALSE)]  
      }
  lim_cut<-v[which(ranks==floor(lim_rank))][1]
  if(is.na(lim_cut))(lim_cut=0)
  return(list('positions'=positions, 'lim_cut'=lim_cut))
}