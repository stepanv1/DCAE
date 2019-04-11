## Developed by Albina Rahim
## Date: December 13, 2016
## This function creates GlobalFrame by storing 1000 random cells from each FCS file. 
## As input it requires the store.allFCS matrix and the number of CPUs to use while parallelization
## The store.allFCS matrix is created as an output from the preProcessingFunc.R function, which contains
## information on the Paths, Name of the Files, Genotypes, Barcodes, Assay Dates, Gender, Number of Channels, Number of Cells for each FCS file


globalFrameFunc <- function(store.allFCS, no_cores){
    library("flowCore")
    library("flowBin")
    firstIteration <- TRUE
  
    print("Start creating the Global Frame")
    file.names <- data.frame(store.allFCS, stringsAsFactors = F)
  
    gFrame <- ddply(file.names, "FCS.files", function(x){
      
      f <- read.FCS(filename = paste0(x$Path, "/", x$FCS.files))
      f <- compensate(f, f@description$SPILL)
  
      if ( firstIteration == TRUE){
        firstIteration <- FALSE
        g <- f
        g@exprs <- g@exprs[sample(1:length(f@exprs[,1]), 1000), ]
      }else {
        g@exprs <- rbind( g@exprs, f@exprs[sample(1:length(f@exprs[,1]), 1000), ])
      }
      data.frame(g@exprs)
    }, .parallel = TRUE) # end ddply
    
    ## Arranging the output from ddply() in order of the file.names
    gFrame <- join(file.names, gFrame)
    
    # ## Coverting the gFrame returned by ddply from a data frame to a matrix
    # gFrame <- as.matrix(gFrame)
    
    ## Reading the first FCS file in the storage matrix as a template for creating the global frame
    g <- read.FCS(filename = paste0(store.allFCS[1,c('Path')], "/", store.allFCS[1,c('FCS files')]))
    g.temp.colnames <- colnames(g)
    ## Replacing the expression matrix in the template FCS file with the global frame matrix
    g@exprs <- as.matrix(gFrame[,10:ncol(gFrame)])
    colnames(g) <- g.temp.colnames
    
    return(list(globalFrame = g, globalFrame.Matrix = gFrame))
}
