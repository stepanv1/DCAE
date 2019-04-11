## Developed by Albina Rahim
## Date: December 09, 2016
## This function does the Pre-Processing of the datasets and can be applied on files of all Panels and Centres. 
## As input it requires the paths of the raw FCS files and paths of the metadata spreadsheets and the number of CPUs to use while parallelization
## It will remove FCS files:
## 1. with no Barcodes
## 2. whose Barcodes are not listed in the metadata spreadsheets
## 3. which are Corrupted
## 4. with less than 20,000 cells
## 5. which are Duplicates
## As output it returns a large matrix- store.allFCS, which contains information of the paths, Panel/Organ,
## Genotype of the file,  Names of the FCS files, Label Barcode, Assay Date, Gender, Number of Channels, and Number of Cells.
## It also returns a long list of Genotypes, unique Genotypes, details of FCS files which were sent to us but 
## whose information was missing in the spreadsheet, information of those Barcodes in the spreadsheet for which the corresponding 
## FCS files were not given to us for automated analysis, information on all the Corrupted FCS files so that we can 
## send them to Centre for resending the files through flowRepository, details of files which has less than 20,000 cells,
## and information to determine if all the files have the same number of channels

#############################################################################################################

## preProcessingFunc <- function(inputPath, inputCSV, no_cores){
##            ....
## }
## inputPath is a list which contains paths of all the datasets sent at various times
## inputCSV contains information for all the metadata spreadsheets sent at various times
## no_cores to determine how many CPUs to use while implenting the parallelization

##############################################################################################################

preProcessingFunc <- function(inputPath, inputCSV, no_cores){
  library("flowCore")
  library("flowBin")
  library("stringr")
 
  
  #source("Codes/3iTcellfunctions.R")
  store.allFCS <- NULL
  NObarcodes.FCS <- 0
  notListed.FCS <- NULL
  corrupted.FCS <- NULL
  lessCells.FCS <- NULL
  duplicate.FCS.temp <- NULL
  Barcodes.NoFCS.temp <- NULL
  Genotype <- c()
  Mouse_Label <- c()
  numPaths <- length(inputPath) # Length of inputPath to determine the number of paths from where the files need to be retrieved

  
  for(i in 1:numPaths){
    # Path to the FCS files
    pathFCS <- unlist(inputPath[i]) 
    # Reads all folders and files in current path folder and makes a list of all of their paths
    allFCS <- dir(pathFCS, full.names=T, recursive=T, pattern = "*.fcs") 
    
    store.allFCS.temp <- sapply(1:length(allFCS), function(x){pathFCS})
    store.allFCS.temp <- cbind(store.allFCS.temp, sapply(1:length(allFCS), function(x){unlist(strsplit(allFCS[x], split = "/"))[length(unlist(strsplit(allFCS[x], split = "/")))-1]}))
    store.allFCS.temp <- cbind(store.allFCS.temp, NA) # Column for Genotype
    store.allFCS.temp <- cbind(store.allFCS.temp, sapply(1:length(allFCS), function(x){unlist(strsplit(allFCS[x], split = "/"))[length(unlist(strsplit(allFCS[x], split = "/")))]}))
    store.allFCS.temp <- cbind(store.allFCS.temp, str_extract(store.allFCS.temp[,4],"L[0-9]+"))
    store.allFCS.temp <- cbind(store.allFCS.temp, NA) # Column for Assay date
    store.allFCS.temp <- cbind(store.allFCS.temp, NA) # Column for Gender
    
    ###########################################################################################################
    # 1. Checking for files with NO Barcodes and remove them

    index.Remove <- which(is.na(store.allFCS.temp[,5]))
    
    if(length(index.Remove) != 0){
      NObarcodes.FCS <- NObarcodes.FCS+length(index.Remove)
      store.allFCS.temp <- store.allFCS.temp[-index.Remove,]
    }
    
    colnames(store.allFCS.temp) <- c("Path", "Panel/Organ", "Genotype", "FCS files", "Barcodes", "Assay Date", "Gender")
    
    ################################################################################
    # 2. Files whose Barcodes are not listed in the metadata spreadsheets
    ## Reading the metadata spreadsheet
    CSVfile <- read.csv(unlist(inputCSV[i]))
    CSVfile <- as.matrix(CSVfile)
    Genotype.temp <- CSVfile[,c('Genotype')]
    Genotype.temp <- sub("/","_", Genotype.temp)
    Mouse_Label.temp <- CSVfile[,c('Label.Barcode')]
    Assay_Date <- CSVfile[,c('Assay.Date')]
    Gender <- CSVfile[,c('Gender')]
    
    ## Checking for files whose Barcodes are not listed in the spreadsheet
    countX <-0
    index.Remove <- 0
    print("Start checking for files whose Barcodes are NOT listed in the spreadsheet")  
    for(x in 1:nrow(store.allFCS.temp)){
      temp <- grep(store.allFCS.temp[x,5], Mouse_Label.temp)
      if(length(temp) == 1){
        store.allFCS.temp[x,3] <- Genotype.temp[temp]
        store.allFCS.temp[x,6] <- Assay_Date[temp]
        store.allFCS.temp[x,7] <- Gender[temp]
      } else{
        index.Remove <- c(index.Remove,x)
        countX <- countX+1}
    }
    index.Remove <-index.Remove[index.Remove !=0]
       
    ## Storing information of all the FCS files which were sent to us but whose information were not listed in the metadata spreadsheet
    # Removing information of the FCS files which were sent to use but whose information were not listed in the metadata spreadsheet from the main storage matrix
    if(length(index.Remove) != 0){
      notListed.FCS <- rbind(notListed.FCS, store.allFCS.temp[index.Remove,])
      store.allFCS.temp <- store.allFCS.temp[-index.Remove,]
    }
    
    ########################################################################################################
    
    ## 3. Code for removing duplicate FCS files based on their Barcodes 
    Barcodes <- str_extract(store.allFCS.temp[,c('Barcodes')],"L[0-9]+")
    duplicate.index <- which(duplicated(Barcodes)==TRUE)
    duplicate.FCS.temp <- rbind(duplicate.FCS.temp, store.allFCS.temp[duplicate.index,])
    store.allFCS.temp <- store.allFCS.temp[!duplicated(store.allFCS.temp[,c('Barcodes')]),]
    #rownames(store.allFCS.temp) <- 1:length(store.allFCS.temp[,1])
    
    ###############################################################################################################
    
    ## 4. This part of the script extracts those Barcodes in the spreadsheet for which the corresponding FCS files were not given to us for automated analysis
    allBarcodes.metadata <- unlist(strsplit(Mouse_Label.temp, ",", perl = TRUE))
    # To determine if there are any duplicate Barcode entries in the metadata spreadsheet
    duplicate.index <- which(duplicated(allBarcodes.metadata)==TRUE)
    if(length(duplicate.index) != 0){
      allBarcodes.metadata <- allBarcodes.metadata[-duplicate.index]
    }
    
    Barcodes.NoFCS.temp <- c(Barcodes.NoFCS.temp, setdiff(allBarcodes.metadata, store.allFCS.temp[,c('Barcodes')]))
    
    ################################################################################################################
    # 5. Checking for files which are Corrupted and removing them. We parallelize this part of the code using ddply()
    # We also record the number of Channels for all the files and the number of Cells.
    
    print("Start finding the Corrupted Files")
    file.names <- data.frame(store.allFCS.temp, stringsAsFactors = F)

    corrupted.cols.cells <- ddply(file.names, "FCS.files", function(x){
      index.Corrupted <- matrix(nrow = 1, ncol = 1, data = NA)
      NumberOfCols <- matrix(nrow = 1, ncol = 1, data = NA)
      NumberOfCells <- matrix(nrow = 1, ncol = 1, data = NA)
      
      f <- try(read.FCS(filename = paste0(x$Path, "/", x$FCS.files)), silent = TRUE)
      if(class(f)=="try-error"){
        index.Corrupted[1] <- "Corrupted"
      }else{
        NumberOfCols[1] <- ncol(f@exprs)
        NumberOfCells[1] <- nrow(f@exprs)
      }
      data.frame(index.Corrupted, NumberOfCols, NumberOfCells)
    }, .parallel = TRUE) # end ddply
    
    ## Arranging the output from ddply() in order of the file.names
    corrupted.cols.cells <- join(file.names, corrupted.cols.cells)
    
    ## Combinging the Number of Channels and Number of Cells with the temporary storage matrix.
    store.allFCS.temp <- cbind(store.allFCS.temp, corrupted.cols.cells[,9], corrupted.cols.cells[,10])
    colnames(store.allFCS.temp) <- c("Path", "Panel/Organ", "Genotype", "FCS files", "Barcodes", "Assay Date", "Gender", "Number of Channels", "Number of Cells")
    
    # Locating the indices of the Corrupted files
    index.Corrupted <- which(!is.na(corrupted.cols.cells[,8]))
    
    
    ## Storing the information for the Corrupted files, so we can send the information to Centre for resending these files through flowRepository
    ## Removing the Corrupted FCS files from store.allFCS.temp
    if(length(index.Corrupted) != 0){
      corrupted.FCS <- rbind(corrupted.FCS, store.allFCS.temp[index.Corrupted,])
      store.allFCS.temp <- store.allFCS.temp[-index.Corrupted,]
    }
    
    print("End of finding the Corrupted Files and removing them from the stored matrix")
    
    
    ##########################################################################################################
    ## 6. Checking for files which has less than 20,000 cells and storing information for such files separately and then removing them from the main storage matrix
    index.lessCells <- 0
    index.lessCells <- which(as.numeric(store.allFCS.temp[,c('Number of Cells')]) < 20000)
    index.lessCells <- index.lessCells[index.lessCells !=0]
  
    if(length(index.lessCells) != 0){
      lessCells.FCS <- rbind(lessCells.FCS, store.allFCS.temp[index.lessCells,])
      store.allFCS.temp <- store.allFCS.temp[-index.lessCells,]
    }

    
    ########################################################################################################
    
    Genotype <- c(Genotype, Genotype.temp) # Combining all the Genotypes together
    Mouse_Label <- c(Mouse_Label, Mouse_Label.temp) # Combining all the Barcodes together
    
    ## Combining the Paths, Name of the Files, Genotypes, Barcodes, Assay Dates, Gender, Number of Channels, Number of Cells with each FCS file in a large matrix store.allFCS
    store.allFCS <- rbind(store.allFCS, store.allFCS.temp) 
    
  } # end of outer for-loop
  
  colnames(store.allFCS) <- c("Path", "Panel/Organ", "Genotype", "FCS files", "Barcodes", "Assay Date", "Gender", "Number of Channels", "Number of Cells")
 
  
  ########################################################################################################
  
  ## 7. Code for removing duplicate FCS files based on their barcodes, if there are duplicates between the different batches of dataset that were sent to us
  Barcodes <- str_extract(store.allFCS[,c('Barcodes')],"L[0-9]+")
  duplicate.index <- which(duplicated(Barcodes)==TRUE)
  if(length(duplicate.index) != 0){
    duplicate.FCS <- rbind(duplicate.FCS.temp, store.allFCS[duplicate.index,c('Path', 'Panel/Organ', 'Genotype', 'FCS files', 'Barcodes', 'Assay Date', 'Gender')])
    store.allFCS <- store.allFCS[!duplicated(store.allFCS[,c('Barcodes')]),]
    rownames(store.allFCS) <- 1:length(store.allFCS[,1])
  }else{
    duplicate.FCS <- duplicate.FCS.temp
  }
 
  ## Finding duplicates among the Barcodes listed for ALL the matadata spreadsheet but whose FCS files were not sent to us
  duplicate.index <- which(duplicated(Barcodes.NoFCS.temp)==TRUE)
  if(length(duplicate.index) != 0){
    Barcodes.NoFCS.temp <- Barcodes.NoFCS.temp[-duplicate.index]
  }
  ## Cross checking between the different batches to determine the Barcodes listed in ALL the metadata spreadsheet but whose FCS files were not sent to us
  Barcodes.NoFCS <- setdiff(Barcodes.NoFCS.temp, store.allFCS[,c('Barcodes')])
  
  # Finding the unique Genotypes (KOs + WTs)
  uniqueGT <- unique(Genotype) 
  
  ## Checking if all the FCS files have the same number of Channels
  numChannels <- unique(store.allFCS[,c('Number of Channels')])
  
  return(list(store.allFCS = store.allFCS, NObarcodes.FCS = NObarcodes.FCS, Genotype = Genotype, uniqueGT = uniqueGT, notListed.FCS = notListed.FCS, Barcodes.NoFCS = Barcodes.NoFCS, corrupted.FCS = corrupted.FCS, lessCells.FCS = lessCells.FCS, duplicate.FCS = duplicate.FCS, numChannels = numChannels))

}
