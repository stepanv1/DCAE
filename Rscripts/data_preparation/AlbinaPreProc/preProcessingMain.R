## Originally written by Albina Rahim 
## Date: December 09, 2016
## This is the Main Script which calls the two functions:
## preProcessingFunc.R : for pre-Processing of the datasets
## globalFrameFunc.R: for creating the Global Frame

remove(list=ls())

setwd("/code/Projects/3i/Panel_T-cell/")

library("plyr")
library("doMC")

## Function for the pre-Processing of the datasets
source("Codes/preProcessingFunc.R")
## Function for creating a Global Frame
source("Codes/globalFrameFunc.R")
## Function for TimeOutput in order to determine the execution time 
source("Codes/3iTcellfunctions.R")

## This part of the script was taken from Sibyl for the purpose of parallelizing the execution of this script
## no_cores to determine how many CPUs to use while implenting the parallelization
no_cores <- detectCores() - 1
registerDoMC(no_cores)

start <- Sys.time()

## User needs to set the Output path in the Bioinformatics Drive
outputPath <- "/mnt/f/FCS data/IMPC/IMPC-Results/3i/Panel_T-cell/SPLEEN/"

## Paths to the FCS files sent on Spleen Organ of August 2015, MArch 2016, and October 2016
inPath1 <- "/mnt/f/FCS data/IMPC/IMPC_2016_Pilot_Data/KCL_WTSI_2015-08-27/SPLN T Labelled"
inPath2 <- "/mnt/f/FCS data/IMPC/IMPC_2016_Pilot_Data/KCL_WTSI_2016-03-08/SPLN T 169+"
inPath3 <- "/mnt/f/FCS data/IMPC/IMPC_2016_Pilot_Data/KCL_WTSI_2016-10-11/SPLN T Labelled"

# Creating a list of all the paths from where we will retrieve the data files from
inputPath <- list(inPath1, inPath2, inPath3)


# Reading the spreadsheets containing the metadata information
inCSVpath1 <- "/mnt/f/FCS data/IMPC/IMPC_2016_Pilot_Data/KCL_WTSI_2015-08-27/attachments/3i_IMPC_Data_Genotypes.csv"
inCSVpath2 <- "/mnt/f/FCS data/IMPC/IMPC_2016_Pilot_Data/KCL_WTSI_2016-03-08/attachments/SPLN data RB.csv"
inCSVpath3 <- "/mnt/f/FCS data/IMPC/IMPC_2016_Pilot_Data/KCL_WTSI_2016-10-11/attachments/download_phnSpleenImmuno.csv"

# Creating a list of all the CSV paths from where we will retrieve the metadata information from
inputCSV <- list(inCSVpath1, inCSVpath2, inCSVpath3)

preProcessing.Output <- preProcessingFunc(inputPath, inputCSV, no_cores)
store.allFCS <- preProcessing.Output$store.allFCS
NObarcodes.FCS <- preProcessing.Output$NObarcodes.FCS
Genotype <- preProcessing.Output$Genotype
uniqueGT <- preProcessing.Output$uniqueGT
notListed.FCS <- preProcessing.Output$notListed.FCS
Barcodes.NoFCS <- preProcessing.Output$Barcodes.NoFCS
corrupted.FCS <- preProcessing.Output$corrupted.FCS
lessCells.FCS <- preProcessing.Output$lessCells.FCS
duplicate.FCS <- preProcessing.Output$duplicate.FCS
numChannels <- preProcessing.Output$numChannels

## Writing the Summary of the Pre-Processing output in a text file
print("Printing the Summary of the Pre-Processing Output: ")
suppressWarnings(dir.create (paste0(outputPath,"Results/")))
sink(file = paste0(outputPath,"Results/preProcessing-Summary.txt"), split = TRUE)
print(paste0("There are in total ", nrow(store.allFCS), " files for analysis."))
print(paste0("Number of FCS files with NO Barcodes: ", NObarcodes.FCS))
print(paste0("Number of FCS files which were sent to us but whose information was missing in the metadata spreadsheet: ", nrow(notListed.FCS)))
print(paste0("Number of Barcodes which were there in the metadata spreadsheet but for which we received no FCS files: ", length(Barcodes.NoFCS)))
print(paste0("Number of Corrupted files: ", nrow(corrupted.FCS)))            
print(paste0("Number of files with < 20,000 cells: ", nrow(lessCells.FCS)))            
print(paste0("Number of Duplicate FCS files: ", nrow(duplicate.FCS)))                   
if(length(numChannels == 1)){
  print(paste0("All files have the same number of channels: ", numChannels))
} else{
  print("All files doesnot have the same number of channels. Needs further check.")
}
sink()

## Calling the Function for creating the Global Frame
globalFrame.Output <- globalFrameFunc(store.allFCS, no_cores)
## Saving the Global Frame
globalFrame <- globalFrame.Output$globalFrame
## Saving the Expression Matrix of the Global Frame with the information of each FCS file
## This matrix can later be used if we need to remove any FCS files and its corresponding expression matrix values from the global frame
globalFrame.Matrix <- globalFrame.Output$globalFrame.Matrix
print("End of creating the Global Frame")


save (store.allFCS, file =  paste0(outputPath,"Results/store.allFCS.Rdata") )
save(Genotype, file = paste0(outputPath,"Results/Genotype.Rdata"))
save(uniqueGT, file = paste0(outputPath,"Results/uniqueGT.Rdata"))
save(notListed.FCS, file = paste0(outputPath, "Results/notListed.FCS.Rdata"))
save(Barcodes.NoFCS, file = paste0(outputPath, "Results/Barcodes.NoFCS.Rdata"))
save(corrupted.FCS , file = paste0(outputPath, "Results/corrupted.FCS.Rdata"))
save(lessCells.FCS, file = paste0(outputPath,"Results/lessCells.FCS.Rdata"))
save(duplicate.FCS, file = paste0(outputPath,"Results/duplicate.FCS.Rdata"))
save(globalFrame, file = paste0(outputPath,"Results/globalFrame.Rdata"))
save(globalFrame.Matrix, file = paste0(outputPath, "Results/globalFrame.Matrix.Rdata"))

cat("Total time is: ",TimeOutput(start),sep="")
