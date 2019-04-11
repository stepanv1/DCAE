#A helper function to visualise high dimensional clusters in the 
#coordinates of most important markers or in terms of first
#principal components
library(rgl)


helper_cluster_plot<-function(cells=c(1,2), cl=dataset[,11],   data=dataset[,1:10],
  popnames=as.character(1:7), Nmark=9, markers=NULL){


  ncolors=length(unique(cl))
  col=colorRampPalette(c("violet","blue",  "yellow", "orange", "green","red"))(ncolors)
  names(col) = unique(cl)
  colors<-unlist(lapply(cl, function(x) col[as.character(x)]))
  
  
  IDXsset<-cl %in% cells
  sset<-data[IDXsset, ]
  #plot3d(sset[,c('IgD', 'IgM', 'MHCII')])
  pc_set<-princomp(sset)
  pcLoad<-as.data.frame(pc_set$loadings[,1])
  cat('Top nine markers', '\n')
  colnames(sset)[order(abs(pcLoad),decreasing=T)][1:9]
  top_comp<-colnames(sset)[order(abs(pcLoad),decreasing=T)]
  if (is.null(markers) ){
    plot3d(sset[,top_comp[c(1,2,3)]], pch='.', col=colors[IDXsset])
    plot(sset[,top_comp[1:Nmark]], pch='.', col=colors[IDXsset])
  } else {
    plot3d(sset[, markers], pch='.', col=colors[IDXsset])
    plot(sset[, markers], pch='.', col=colors[IDXsset])
  }
}

