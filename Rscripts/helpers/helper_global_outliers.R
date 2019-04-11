#identify global outliers based on
#the distance in noisy dimensions
#i=1
#ref_subs<-sub_res
#data <- data[[i]][!is.na(clus_truth[[i]]), ]; cl<-clus_truth[[i]][!is.na(clus_truth[[i]])]; ref_subs<-ref_subs[[i]]$subdim
#data <- data[[i]][individual==1, ]; clus_assign <- clus_assign[[i]][individual==1 ]; ref_subs<-ref_subs[[i]]
#data <- dataL[[i]]; clus_assign <- clus_truthL[[i]]; 
#ref_subs<-helper_find_subdimensions(data, clus_assign)$subdim
library(parallel)
helper_global_outliers <- function(data, ref_subs, clus_assign, mc.cores=5){
  mat_min<-matrix(NA, nrow=length(unique(clus_assign)), ncol=dim(data)[2])
  mat_sd<-matrix(NA, nrow=length(unique(clus_assign)), ncol=dim(data)[2])
  colnames(mat_min) <- colnames(data); colnames(mat_sd) <- colnames(data);
  
  for(j in sort(unique(clus_assign))){
    mat_min[j, ] <- unlist(lapply(as.data.frame(data[clus_assign==j ,]), function(x)  median(x)))
    mat_sd[j, ] <- unlist(lapply(as.data.frame(data[clus_assign==j ,]), function(x)  sd(x)))
    }
  
  GO <- mclapply(1:nrow(data), function(x) {
    noise_dim <- !ref_subs[clus_assign[x], ]
    if(sum(noise_dim)==0){return(NaN)}
    y<-data[x, noise_dim]
    if (sum(noise_dim)>1){
      norm_y<-sqrt( sum((y- mat_min[clus_assign[x], noise_dim])^2 / (mat_sd[clus_assign[x], noise_dim])^2) )  / sqrt(sum(noise_dim))} else {
      norm_y<-sqrt( sum((y- mat_min[clus_assign[x], noise_dim])^2 / (mat_sd[clus_assign[x], noise_dim])^2) )  
      }
    return(norm_y)
      }, mc.cores=mc.cores)
  
  return(unlist(GO))
}

source('/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/Rscripts/helpers/helper_convBernully.R')

helper_global_outliers_Bernoulli <- function(data, ref_subs, clus_assign, mc.cores=5){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  ns<-nrow(ref_subs)
  
  bin_data <- data>0
  
  freq_list <-lapply(1:ns, function(x) apply(bin_data[clus_assign==x, !ref_subs[x, ]], 2,  function(y) 
    sum(y)/length(y)))
  
  pdf_list <- lapply(freq_list,   convBernoulli) 
  cdf_list <- lapply(pdf_list, function(x) cumsum(x$pdfvec))
  p_list<-lapply(cdf_list, function(x) 1-x)
  
  p_mat<-mclapply(1:nrow(data), function(x) {
    lapply(1:ns, function(y){
      #sum(bin_data[x,!ref_subs[clus_assign[x], ] ])+1
      p_list[[y]][sum(bin_data[x,!ref_subs[y, ] ])+1]    
               })
                                             }, mc.cores=5) 
  p_mat<- do.call("rbind", p_mat)
  
  
  return(p_mat)
}


helper_global_outliers_Discrete <- function(data, ref_subs, clus_assign, nbins=10, mc.cores=5){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  ns<-nrow(ref_subs)
  
  clus_noise_dim <- lapply(1:ns, function(x) data[clus_assign==x, !ref_subs[x, ]])
  clus_noise_dim <- lapply(clus_noise_dim, function(x) ifelse(x<0,0,x))
  not_zero_idx <- lapply(clus_noise_dim, function(x) x>0)
  #assign to quantile per each data point in each noisy dimensions
  clus_quant_n<-lapply(1:ns, function(y) {
    res<-lapply(1:ncol(clus_noise_dim[[y]]), function(x){ 
      qn<-cut(clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]], breaks=quantile(clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]], seq(0,1, length.out=nbins)), include.lowest=T, labels=F);
      clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]] <- qn;
      return(clus_noise_dim[[y]][,x])
                                                         });
    return(Reduce(cbind,res))
                                          })
  
  freq_list <-lapply(1:ns, function(x) unlist(apply(as.matrix(clus_quant_n[[x]]), 2,  function(y){ 
    unlist(as.vector(table(factor(y, levels = c(0:(nbins-1)))))/length(y), recursive = TRUE)})))
  freq_list <- lapply(freq_list, function(x) as.vector(x))
  
  pdf_list <- mclapply(freq_list, function(x) convDiscrete(pvec=x, supp=0:(nbins-1)), mc.cores=5)    
  
  cdf_list <- lapply(pdf_list, function(x) cumsum(x$pdfvec))
  p_list<-lapply(cdf_list, function(x) 1-x)
  
  idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[clus_assign==x])
  #calculate probabilities per data point using cdf
  p_l<-mclapply(1:ns, function(x) {
      pl<-p_list[[x]]
      cq<-clus_quant_n[[x]]
      #sum(bin_data[x,!ref_subs[clus_assign[x], ] ])+1
      pl[apply(cq, 1, function(y) sum(y))]    
                                   }, mc.cores=mc.cores)
                                            
  #reorder result to match order in data matrix
  idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[clus_assign==x])
  p<-vector('numeric', length=nrow(data))
  p[unlist(idx_per_clus)]<-unlist(p_l)
  
  return(p)
}
#d=data;ref_subs<-ref_subs[[3]]
helper_global_outliers_Discrete_All_dims <- function(data, ref_subs, cl, nbins=3, mc.cores=5){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  ns<-nrow(ref_subs)
  
  clus_noise_dim <- lapply(1:ns, function(x) data[cl==x, !ref_subs[x, ]])
  clus_noise_dim <- lapply(clus_noise_dim, function(x) ifelse(x<0,0,x))
  not_zero_idx <- lapply(clus_noise_dim, function(x) x>0)
  #assign to quantile per each data point in each noisy dimensions
  clus_quant_n<-lapply(1:ns, function(y) {
    res<-lapply(1:ncol(clus_noise_dim[[y]]), function(x){ 
      noise <- clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]]
      breaks <- quantile(noise, seq(0,1, length.out=nbins))
      qn<-cut(noise, breaks=breaks, include.lowest=T, labels=F);
      noise <- qn
      signal<-clus_noise_dim[[y]][, x]
      signal[not_zero_idx[[y]][,x]]<-noise
      return(signal)
    });
    return(res)
  })
  
  freq_list <-lapply(1:ns, function(x) unlist(lapply(clus_quant_n[[x]],   function(y){ 
    unlist(as.vector(table(factor(y, levels = c(0:(nbins-1)))))/length(y), recursive = TRUE)})))
  freq_list <- lapply(freq_list, function(x) as.vector(x))
  
  pdf_list <- mclapply(freq_list, function(x) convDiscrete(pvec=x, supp=0:(nbins-1)), mc.cores=5) 
  cdf_list <- lapply(pdf_list, function(x) cumsum(x$pdfvec))
  p_list<-lapply(cdf_list, function(x) 1-x)
  
  data[data<0]<-0
  #calculate quantiles in each subdimension set for all data points
  clus_quant_nALL<-mclapply(1:ns, function(y) {
    res<-lapply(1:sum(!ref_subs[y,]), function(x){ 
      noise<-clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]]
      nz_i<-data[ ,!ref_subs[y,] ][, x]>0 
      nz_d<-data[ ,!ref_subs[y,] ][, x][nz_i]
      breaks<-c( quantile(noise, seq(0,1, length.out=nbins)))
      qn<-cut(nz_d, breaks=breaks,  include.lowest=T, labels=F);
      #dealing with the values ouside breaks range
      qn[nz_d<breaks[1]]<-1
      qn[nz_d>breaks[10]]<-nbins-1
      signal<-rep(0,length(data[ ,!ref_subs[y,] ][, x]))
      signal[nz_i]<-qn
      return(signal)
    });
    return(rowSums(Reduce(cbind,res)))
  },
  mc.cores=mc.cores, mc.preschedule=F)
  
  
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #calculate probabilities per data point using cdf
  p_l<-mclapply(1:ns, function(x) {
    pl<-p_list[[x]]
    cq<-clus_quant_nALL[[x]]
    #sum(bin_data[x,!ref_subs[cl[x], ] ])+1
    return(pl[cq+1])    
  }, mc.cores=1)
  
  #reorder result to match order in data matrix
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #p<-vector('numeric', length=nrow(data))
  #p[unlist(idx_per_clus)]<-unlist(p_l)
  
  return(p_l)
}

helper_global_outliers_Uniform <- function(data, ref_subs, cl, nbins=10, mc.cores=5){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  ns<-nrow(ref_subs)
  ndim<-ncol(ref_subs)
  data[data<0]<-0
  noisedims<-(1:ndim)[!!colSums(!ref_subs)]
  colnames(ref_subs)<-1:ncol(ref_subs)
  
  #assign to bin (bins are equidistanse in intensity) per each data point in each noisy dimensions
  bins_noise_dims<-mclapply(noisedims, function(y) {
    res <- unlist(lapply(1:ns, function(x){ 
      data[cl==x, ifelse(!ref_subs[x,y], y, FALSE)]}))
    breaks=c(-1, 0, max(res)/((nbins-1):1)) 
    return(list('n'=cut(res, breaks=breaks, include.lowest=T, labels=F)-1, 'breaks'=breaks));
    }, mc.cores=1)
  freq_list <-lapply(bins_noise_dims, function(x) as.vector(table(factor(x$n, levels = c(0:(nbins-1)))))/length(x$n)) 
  names(freq_list)<-noisedims
  
  #create a distribution of sum of bin numbers in each cluster 
  D1pdf_list <- mclapply(1:ns, function(x) unlist(freq_list[as.character((1:ndim)[!ref_subs[x,]])]), 
                       mc.cores=mc.cores)
  pdf_list <- mclapply(D1pdf_list, function(x) convDiscrete(pvec=unlist(x), supp=0:(nbins-1)), mc.cores=mc.cores) 
  
  cdf_list <- lapply(pdf_list, function(x) cumsum(x$pdfvec))
  p_list<-lapply(cdf_list, function(x) 1-x)
  
  #calculate quantiles in each subdimension set for all data points
  br<-lapply(bins_noise_dims, function(x) x$breaks)
  names(br)<-noisedims
  clus_sums_ALL<-mclapply(1:ns, function(y) {
    res<-lapply((1:ndim)[!ref_subs[y,]], function(x){ 
      breaks<-br[[as.character(x)]]
      n<-cut(data[, x], breaks=breaks,  include.lowest=T, labels=F)-1;
      #dealing with the values ouside breaks range
      n[data[, x]>breaks[nbins+1]]<-nbins-1
      return(n)
    });
    return(rowSums(Reduce(cbind,res)))
  },
  mc.cores=mc.cores, mc.preschedule=F)
  
  
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #calculate probabilities per data point using cdf
  p_l<-mclapply(1:ns, function(x) {
    pl<-p_list[[x]]
    cq<-clus_sums_ALL[[x]]
    #sum(bin_data[x,!ref_subs[cl[x], ] ])+1
    return(pl[cq+1])    
  }, mc.cores=5)
  
  #reorder result to match order in data matrix
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #p<-vector('numeric', length=nrow(data))
  #p[unlist(idx_per_clus)]<-unlist(p_l)
  
  return(p_l)
}

#forcing noise dimensions bee approximately identical
helper_global_outliers_ApproxIID <- function(data, ref_subs, cl, nbins=5, mc.cores=5){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  ns<-nrow(ref_subs)
  ndim<-ncol(ref_subs)
  data[data<0]<-0
  noisedims<-(1:ndim)[!!colSums(!ref_subs)]
  colnames(ref_subs)<-1:ncol(ref_subs)
  
  #assign to bin (bins are equidistanse in intensity) per each data point in each noisy dimensions
  bins_zero_cut_off<-mclapply(noisedims, function(y) {
    res <- unlist(lapply(1:ns, function(x){ 
      data[cl==x, ifelse(!ref_subs[x,y], y, FALSE)]}))
    zerocut<-sum(res==0)/length(res)  
    return(list('res'=res, 'zerocut'=zerocut))
  }, mc.cores=1)
  
  max_cut_off <- max(unlist(lapply(bins_zero_cut_off, function(x) x$zerocut)))
  
  zero_break <- mclapply(bins_zero_cut_off, function(x) {
    quantile(x$res,max_cut_off)
  }, mc.cores=1)
  zero_break<-unlist(zero_break)
  names(zero_break)<-noisedims
  
  bins_noise_dims<-mclapply(noisedims, function(y) {
    res <- unlist(lapply(1:ns, function(x){ 
      data[cl==x, ifelse(!ref_subs[x,y], y, FALSE)]}))
    zb<-zero_break[as.character(y)]
    breaks=unlist(c(-1, zb, (max(res)-zb)/((nbins-1))*(1:(nbins-1))+zb)) 
    
    return(list('n'=cut(res, breaks=breaks, include.lowest=T, labels=F)-1, 'breaks'=breaks));
  }, mc.cores=1)
  
  freq_list <-lapply(bins_noise_dims, function(x) as.vector(table(factor(x$n, levels = c(0:(nbins-1)))))/length(x$n)) 
  names(freq_list)<-noisedims
  
  #zzz=Reduce(rbind, freq_list)
  #matplot(log10(t(zzz)), type= 'l')
  
  
  #create a distribution of sum of bin numbers in each cluster 
  D1pdf_list <- mclapply(1:ns, function(x) unlist(freq_list[as.character((1:ndim)[!ref_subs[x,]])]), mc.cores=mc.cores)
  
  pdf_list <- mclapply(D1pdf_list, function(x) convDiscrete(pvec=unlist(x), supp=(0:(nbins-1))^2), mc.cores=1) 
  
  cdf_list <- lapply(pdf_list, function(x) cumsum(x$pdfvec))
  p_list<-lapply(cdf_list, function(x) 1-x)
  
  #calculate quantiles in each subdimension set for all data points
  br<-lapply(bins_noise_dims, function(x) x$breaks)
  names(br)<-noisedims
  clus_sums_ALL<-mclapply(1:ns, function(y) {
    res<-lapply((1:ndim)[!ref_subs[y,]], function(x){ 
      breaks<-br[[as.character(x)]]
      n<-(cut(data[, x], breaks=breaks,  include.lowest=T, labels=F)-1)^2;
      #dealing with the values ouside breaks range
      n[data[, x]>breaks[nbins+1]]<-(nbins-1)^2
      return(n)
    });
    return(rowSums(Reduce(cbind,res)))
  },
  mc.cores=mc.cores, mc.preschedule=F)
  
  
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #calculate probabilities per data point using cdf
  p_l<-mclapply(1:ns, function(x) {
    pl<-p_list[[x]]
    cq<-clus_sums_ALL[[x]]
    #sum(bin_data[x,!ref_subs[cl[x], ] ])+1
    return(pl[cq+1])    
  }, mc.cores=5)
  
  #reorder result to match order in data matrix
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #p<-vector('numeric', length=nrow(data))
  #p[unlist(idx_per_clus)]<-unlist(p_l)
  
  return(p_l)
}

#create distributionon identical quantiles in ach cluster
helper_global_outliers_Discrete_Quantiles <- function(data, ref_subs, cl, nbins=3, mc.cores=10){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  ns<-nrow(ref_subs)
  #find first quatile corresponding highest number of zeroes in dimensions
  clus_noise_dim <- lapply(1:ns, function(x) data[cl==x, !ref_subs[x, ]])
  clus_noise_dim <- lapply(clus_noise_dim, function(x) ifelse(x<0,0,x))
  not_zero_idx <- lapply(clus_noise_dim, function(x) x>0)
  #assign to quantile per each data point in each noisy dimensions
  #View(clus_noise_dim[[1]])
  clus_quant_n<-lapply(1:ns, function(y) {
    #create unified quantile set for all dimensions in
    #cluster number y
    cl_zeroes<-sum(clus_noise_dim[[y]]==0)
    zerofreq <- cl_zeroes / length(clus_noise_dim[[y]])
    noisevec<- (clus_noise_dim[[y]]); dim(noisevec)<-NULL
    ln<-length(noisevec)
    breaks <- seq(0,max(noisevec), length.out=nbins)
    freqs1Dnonzero <- unlist(lapply(1:(nbins-1), function(i) sum(noisevec<breaks[i+1] & noisevec>breaks[i])/ln))
    freqs1D<-c(zerofreq, freqs1Dnonzero)
    #print(freqs1D)
    
    res<-lapply(1:ncol(clus_noise_dim[[y]]), function(x){ 
      noise <- clus_noise_dim[[y]][, x]
      #create equidistant breaks
      breaks <-  c(-1,quantile(noise, probs = cumsum(freqs1D)))
      #print(breaks)
      qn<-cut(noise, breaks=breaks, labels = F, include.lowest=T);
      qn[which(is.na(qn))] <- nbins
      return(qn)
    });
    return(res)
  })
  
  freq_list <-lapply(1:ns, function(x) unlist(lapply(clus_quant_n[[x]],   function(y){ 
    unlist(as.vector(table(factor(y, levels = c(1:(nbins)))))/length(y), recursive = TRUE)})))
  freq_list <- lapply(freq_list, function(x) as.vector(x))
  
  pdf_list <- mclapply(freq_list, function(x) convDiscrete(pvec=x, supp=1:(nbins)), mc.cores=mc.cores) 
  cdf_list <- lapply(pdf_list, function(x) cumsum(x$pdfvec))
  p_list<-lapply(cdf_list, function(x) 1-x)
  
  data[data<0]<-0
  #calculate quantiles in each subdimension set for all data points
  clus_quant_nALL<-mclapply(1:ns, function(y) {
    res<-lapply(1:sum(!ref_subs[y,]), function(x){ 
      noise<-clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]]
      nz_i<-data[ ,!ref_subs[y,] ][, x]>0 
      nz_d<-data[ ,!ref_subs[y,] ][, x][nz_i]
      breaks<-c( seq(0,1, length.out=nbins))
      qn<-cut(nz_d, breaks=breaks,  include.lowest=T, labels=F);
      #dealing with the values ouside breaks range
      qn[nz_d<breaks[1]]<-1
      qn[nz_d>breaks[10]]<-nbins-1
      signal<-rep(0,length(data[ ,!ref_subs[y,] ][, x]))
      signal[nz_i]<-qn
      return(signal)
    });
    return(rowSums(Reduce(cbind,res)))
  },
  mc.cores=mc.cores, mc.preschedule=F)
  
  
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #calculate probabilities per data point using cdf
  p_l<-mclapply(1:ns, function(x) {
    pl<-p_list[[x]]
    cq<-clus_quant_nALL[[x]]
    #sum(bin_data[x,!ref_subs[cl[x], ] ])+1
    return(pl[cq+1])    
  }, mc.cores=1)
  
  #reorder result to match order in data matrix
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #p<-vector('numeric', length=nrow(data))
  #p[unlist(idx_per_clus)]<-unlist(p_l)
  
  return(p_l)
}


helper_global_outliers_Discrete_Equidist_Bins <- function(data, ref_subs, cl, nbins=3, mc.cores=5){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  ns<-nrow(ref_subs)
  #find first quatile corresponding highest number of zeroes in dimensions
  
  clus_noise_dim <- lapply(1:ns, function(x) data[cl==x, !ref_subs[x, ]])
  clus_noise_dim <- lapply(clus_noise_dim, function(x) ifelse(x<0,0,x))
  not_zero_idx <- lapply(clus_noise_dim, function(x) x>0)
  #assign to quantile per each data point in each noisy dimensions
  #View(clus_noise_dim[[1]])
  clus_quant_n<-lapply(1:ns, function(y) {
    #create unified quantile for all dimensions in
    #cluster number y
    res<-lapply(1:ncol(clus_noise_dim[[y]]), function(x){ 
      noise <- clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]]
      #create equidistant breaks
      breaks <- seq(0,1, length.out=nbins)
      qn<-cut(noise, breaks=breaks, include.lowest=T, labels=F);
      noise <- qn
      signal<-clus_noise_dim[[y]][, x]
      signal[not_zero_idx[[y]][,x]]<-noise
      return(signal)
    });
    return(res)
  })
  
  freq_list <-lapply(1:ns, function(x) unlist(lapply(clus_quant_n[[x]],   function(y){ 
    unlist(as.vector(table(factor(y, levels = c(0:(nbins-1)))))/length(y), recursive = TRUE)})))
  freq_list <- lapply(freq_list, function(x) as.vector(x))
  
  pdf_list <- mclapply(freq_list, function(x) convDiscrete(pvec=x, supp=0:(nbins-1)), mc.cores=5) 
  cdf_list <- lapply(pdf_list, function(x) cumsum(x$pdfvec))
  p_list<-lapply(cdf_list, function(x) 1-x)
  
  data[data<0]<-0
  #calculate quantiles in each subdimension set for all data points
  clus_quant_nALL<-mclapply(1:ns, function(y) {
    res<-lapply(1:sum(!ref_subs[y,]), function(x){ 
      noise<-clus_noise_dim[[y]][, x][not_zero_idx[[y]][,x]]
      nz_i<-data[ ,!ref_subs[y,] ][, x]>0 
      nz_d<-data[ ,!ref_subs[y,] ][, x][nz_i]
      breaks<-c( seq(0,1, length.out=nbins))
      qn<-cut(nz_d, breaks=breaks,  include.lowest=T, labels=F);
      #dealing with the values ouside breaks range
      qn[nz_d<breaks[1]]<-1
      qn[nz_d>breaks[10]]<-nbins-1
      signal<-rep(0,length(data[ ,!ref_subs[y,] ][, x]))
      signal[nz_i]<-qn
      return(signal)
    });
    return(rowSums(Reduce(cbind,res)))
  },
  mc.cores=mc.cores, mc.preschedule=F)
  
  
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #calculate probabilities per data point using cdf
  p_l<-mclapply(1:ns, function(x) {
    pl<-p_list[[x]]
    cq<-clus_quant_nALL[[x]]
    #sum(bin_data[x,!ref_subs[cl[x], ] ])+1
    return(pl[cq+1])    
  }, mc.cores=1)
  
  #reorder result to match order in data matrix
  #idx_per_clus<-lapply(1:ns, function(x) (1:nrow(data))[cl==x])
  #p<-vector('numeric', length=nrow(data))
  #p[unlist(idx_per_clus)]<-unlist(p_l)
  
  return(p_l)
}

helper_global_outliers_Discrete_Sums <- function(data, ref_subs, cl, nbins=20, mc.cores=20){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  data[data<0]<-0
  ns<-nrow(ref_subs)
  #find first quatile corresponding highest number of zeroes in dimensions
  
  clus_noise_dim <- lapply(1:ns, function(x) data[cl==x, !ref_subs[x, ]])
  clus_noise_dim <- lapply(clus_noise_dim, function(x) ifelse(x<0,0,x))
  not_zero_idx <- lapply(clus_noise_dim, function(x) x>0)
  #breaks <- seq(0,1, length.out=nbins+1)
  clus_sums<-lapply(1:ns, function(y) {
    #create unified quantile for all dimensions in
    #cluster number y
    breaks <- seq(0,max(noise), length.out=nbins+1)
    noise <- clus_noise_dim[[y]]
    noise_quant<-apply(noise, 2,function(x)  cut(x, breaks=breaks, include.lowest=T, labels=F))
    sum_quant <- apply(noise_quant, 1, function(x)  sum(x))
    quant_ecdf<-ecdf(sum_quant)
    return(list('noise_quant' = noise_quant, 'sum_quant'=sum_quant, 'quant_ecdf'=quant_ecdf, 'breaks'=breaks ))
  })
  
 
  #calculate sums in each subdimension set for all data points
  p_l<-mclapply(1:ns, function(y) {
    quant_ecdf<-clus_sums[[y]][['quant_ecdf']]
    #create unified quantile for all dimensions in
    #cluster number y
    noiseProjection <- data[, !ref_subs[y,]]
    breaks = clus_sums[[y]]$breaks 
    noiseProjection_quant<-apply(noiseProjection, 2, function(x)  cut(x, breaks=c(breaks,1), include.lowest=T, labels=F))
    sumProjection_quant <- apply(noiseProjection_quant, 1, function(x)  sum(x))
    pval<-1-quant_ecdf(sumProjection_quant)
    return(pval)
  }, mc.cores=mc.cores)
 
   return(p_l)
}

helper_global_outliers_Continuous_Sums <- function(data, ref_subs, cl, nbins=20, mc.cores=20){
  #pre-compute multivariate Bernulli pdfs for each subdimension
  data[data<0]<-0
  ns<-nrow(ref_subs)
  #find first quatile corresponding highest number of zeroes in dimensions
  
  clus_noise_dim <- lapply(1:ns, function(x) data[cl==x, !ref_subs[x, ]])
  clus_noise_dim <- lapply(clus_noise_dim, function(x) ifelse(x<0,0,x))
  not_zero_idx <- lapply(clus_noise_dim, function(x) x>0)
  #breaks <- seq(0,1, length.out=nbins+1)
  clus_sums<-lapply(1:ns, function(y) {
    #create unified quantile for all dimensions in
    #cluster number y
    #print(y)
    noise <- clus_noise_dim[[y]]
    sum_n <- apply(noise, 1, function(x)  sum(x^2))
    c_ecdf<-ecdf(sum_n)
    return(list('c_ecdf'=c_ecdf))
  })
  
  
  #calculate sums in each subdimension set for all data points
  p_l<-mclapply(1:ns, function(y) {
    c_ecdf<-clus_sums[[y]][['c_ecdf']]
    #create unified quantile for all dimensions in
    #cluster number y
    noiseProjection <- data[, !ref_subs[y,]]
    sumProjection_c <- apply(noiseProjection, 1, function(x)  sum(x^2))
    pval<-1-c_ecdf(sumProjection_c)
    return(pval)
  }, mc.cores=mc.cores)
  
  return(p_l)
}





#zzz=helper_global_outliers_Discrete(data, ref_subs, clus_assign, nbins=10, mc.cores=5)
#qqplot(zzz, runif(length(zzz)), pch='.')

#pl_perclus<-unlist(lapply(1:nrow(ref_subs), function(x) p_l[[x]][clus_assign==x]))
#qqplot(pl_perclus, runif(length(pl_perclus)), pch='.', col='red', add=T)
