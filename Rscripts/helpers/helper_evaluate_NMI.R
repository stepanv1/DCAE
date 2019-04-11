#########################################################################################
# Function to match cluster labels with manual gating (reference standard) population 
# labels and calculate precision, recall, and S^w_H split-merge based score,
# w means that we use weighted version of the score, H- entropy based similarity measure,  s(C,L) 
#
# Matching criterion: Hungarian algorithm
#
# Use this function for data sets with multiple populations of interest
#
# Based on Lukas Weber, August 2016
#########################################################################################
library(clue)
library(entropy)
# arguments:
# - clus_algorithm: cluster labels from algorithm
# - clus_truth: true cluster labels
# (for both arguments: length = number of cells; names = cluster labels (integers))
helper_evaluate_NMI <- function(clus_algorithm, clus_truth){
  
  # number of detected clusters
  n_clus <- length(table(clus_algorithm))
  
  # remove unassigned cells (NA's in clus_truth)
  unassigned <- is.na(clus_truth)
  clus_algorithm <- clus_algorithm[!unassigned]
  clus_truth <- clus_truth[!unassigned]
  if (length(clus_algorithm) != length(clus_truth)) warning("vector lengths are not equal")
  n_points<-length(clus_truth)
  
  #if there are cluster assignments not belonging to naturals, rename
  #top_clus <- max(clus_algorithm)
  #clus_algorithm<-unlist(lapply(clus_algorithm, function(x) if (!is.naturalnumber(x))
  #  return(abs(x) + top_clus+1) else return(x)))
  conf_mat<-table(clus_truth, clus_algorithm)
  NMI <- mi.empirical(conf_mat)/sqrt(entropy(table(clus_truth))*entropy(table(clus_algorithm)))
  return(list(n_clus = n_clus, NMI = NMI))
}


