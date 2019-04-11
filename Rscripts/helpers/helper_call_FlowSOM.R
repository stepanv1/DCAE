#call flowSOM

helper_call_FlowSOM <- function(data, subset=NULL, marker_cols){
  subdata<-data
  if (!is.null(subset)){
    subdata@exprs<-data@exprs[subset, ]
  }
  
  fSOM <- FlowSOM::ReadInput(subdata, transform = FALSE, scale = FALSE)
  fSOM <- FlowSOM::BuildSOM(fSOM, colsToUse = marker_cols)
  fSOM <- FlowSOM::BuildMST(fSOM)
  
  meta <- FlowSOM::MetaClustering(fSOM$map$codes, method = "metaClustering_consensus")
  clus <- meta[fSOM$map$mapping[, 1]]
  return(clus)
}

helper_call_FlowSOM40 <- function(data, subset = NULL, marker_cols, numC=40){
  subdata<-data
  if (!is.null(subset)){
    subdata@exprs<-data@exprs[subset, ]
  }
  
  fSOM <- FlowSOM::ReadInput(subdata, transform = FALSE, scale = FALSE)
  fSOM <- FlowSOM::BuildSOM(fSOM, colsToUse = marker_cols, 
                            xdim = 10, ydim = 10)
  fSOM <- FlowSOM::BuildMST(fSOM)
  
  meta <- suppressMessages(ConsensusClusterPlus::ConsensusClusterPlus(t(fSOM$map$codes), maxK = numC, seed = 12345))
  meta <- meta[[numC]]$consensusClass
  
  clus <- meta[fSOM$map$mapping[, 1]]
  return(clus)
}




    