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
#library(flexclust)

#entropy function
H_clus<-function(clustering){
  freqs <- table(clustering)/length(clustering)
  return(-sum(freqs * log2(freqs)))
}

is.naturalnumber <-
  function(x, tol = .Machine$double.eps^0.5)  x > tol & abs(x - round(x)) < tol

# arguments:
# - clus_algorithm: cluster labels from algorithm
# - clus_truth: true cluster labels
# (for both arguments: length = number of cells; names = cluster labels (integers))
helper_match_evaluate_multiple_SunweightedH <- function(clus_algorithm, clus_truth) {
  
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
  
  tbl_algorithm <- table(clus_algorithm)
  tbl_truth <- table(clus_truth)
  
  
  # detected clusters in rows, true populations in columns
  pr_mat <- re_mat <- SunweightedH_mat <- matrix(NA, nrow = length(tbl_algorithm), ncol = length(tbl_truth))
  
  for (i in 1:length(tbl_algorithm)) {
    for (j in 1:length(tbl_truth)) {
      i_int <- as.integer(names(tbl_algorithm))[i]  # cluster number from algorithm
      j_int <- as.integer(names(tbl_truth))[j]  # cluster number from true labels
      
      true_positives <- sum(clus_algorithm == i_int & clus_truth == j_int, na.rm = TRUE)
      true_negatives <- sum(clus_algorithm != i_int & clus_truth != j_int, na.rm = TRUE)  
      detected <- sum(clus_algorithm == i_int, na.rm = TRUE)
      truth <- sum(clus_truth == j_int, na.rm = TRUE)
      negative <- sum(clus_truth != j_int, na.rm = TRUE)
      
      # calculate precision, recall, and RandInd score
      precision_ij <- true_positives / detected
      recall_ij <- true_positives / truth
      
      #calculate induced clustering L on C:
      i_j <- clus_algorithm[clus_truth==j_int]
      #calculate induced clustering C on L:
      j_i <- clus_truth[clus_algorithm==i_int]
      #calculate entropies
      H_i_j <- H_clus(i_j)
      H_j_i <- H_clus(j_i)
      #calculate 'meet' and its weight
      j_card <- sum(clus_truth==j_int)
      i_card <- sum(clus_algorithm==i_int)
      #weight_meet_i_j <- sum(clus_truth==j_int & clus_algorithm==i_int)/n_points
      
      SunweightedH_ij <- (1 - H_i_j/log2(j_card))*(1 - H_j_i/log2(i_card))
      #cat(c(SweightedH_ij, '\n'))
      if (SunweightedH_ij == "NaN") SweightedH_ij <- 0
      
      pr_mat[i, j] <- precision_ij
      re_mat[i, j] <- recall_ij
      SunweightedH_mat[i, j] <- SunweightedH_ij
      
      SunweightedH_ij <- (1-H_i_j/log2(j_card))*(1-H_j_i/log2(i_card))
      #cat(c(SweightedH_ij, '\n'))
      if (SunweightedH_ij == "NaN") SunweightedH_ij <- 0
      
      pr_mat[i, j] <- precision_ij
      re_mat[i, j] <- recall_ij
      SunweightedH_mat[i, j] <- SunweightedH_ij
    }
  }
  
  # put back cluster labels (note some row names may be missing due to removal of unassigned cells)
  rownames(pr_mat) <- rownames(re_mat) <- rownames(SunweightedH_mat) <- names(tbl_algorithm)
  colnames(pr_mat) <- colnames(re_mat) <- colnames(SunweightedH_mat) <- names(tbl_truth)
  
  # match labels using Hungarian algorithm applied to matrix of RandInd scores (Hungarian
  # algorithm calculates an optimal one-to-one assignment)
  
  # use transpose matrix (Hungarian algorithm assumes n_rows <= n_cols)
  SunweightedH_mat_trans <- t(SunweightedH_mat)
  
  if (nrow(SunweightedH_mat_trans) <= ncol(SunweightedH_mat_trans)) {
    # if fewer (or equal no.) true populations than detected clusters, can match all true populations
    labels_matched <- clue::solve_LSAP(SunweightedH_mat_trans, maximum = TRUE)
    # use row and column names since some labels may have been removed due to unassigned cells
    labels_matched <- as.numeric(colnames(SunweightedH_mat_trans)[as.numeric(labels_matched)])
    names(labels_matched) <- rownames(SunweightedH_mat_trans)
    
  } else {
    # if fewer detected clusters than true populations, use transpose matrix and assign
    # NAs for true populations without any matching clusters
    labels_matched_flipped <- clue::solve_LSAP(SunweightedH_mat, maximum = TRUE)
    # use row and column names since some labels may have been removed due to unassigned cells
    labels_matched_flipped <- as.numeric(rownames(SunweightedH_mat_trans)[as.numeric(labels_matched_flipped)])
    names(labels_matched_flipped) <- rownames(SunweightedH_mat)
    
    labels_matched <- rep(NA, ncol(SunweightedH_mat))
    names(labels_matched) <- rownames(SunweightedH_mat_trans)
    labels_matched[as.character(labels_matched_flipped)] <- as.numeric(names(labels_matched_flipped))
  }
  
  # precision, recall, RandInd score, and number of cells for each matched cluster
  pr <- re <- SunweightedH <- n_cells_matched <- rep(NA, ncol(SunweightedH_mat))
  names(pr) <- names(re) <- names(SunweightedH) <- names(n_cells_matched) <- names(labels_matched)
  
  for (i in 1:ncol(SunweightedH_mat)) {
    # set to 0 if no matching cluster (too few detected clusters); use character names 
    # for row and column indices in case subsampling completely removes some clusters
    pr[i] <- ifelse(is.na(labels_matched[i]), 0, pr_mat[as.character(labels_matched[i]), names(labels_matched)[i]])
    re[i] <- ifelse(is.na(labels_matched[i]), 0, re_mat[as.character(labels_matched[i]), names(labels_matched)[i]])
    SunweightedH[i] <- ifelse(is.na(labels_matched[i]), 0, SunweightedH_mat[as.character(labels_matched[i]), names(labels_matched)[i]])
    
    n_cells_matched[i] <- sum(clus_algorithm == labels_matched[i], na.rm = TRUE)
  }
  
  # means across populations
  mean_pr <- mean(pr)
  mean_re <- mean(re)
  mean_SunweightedH <- mean(SunweightedH)
  total_SunweightedH <- sum(SunweightedH_mat, na.rm = TRUE)
  
  
  return(list(n_clus = n_clus, 
              pr = pr, 
              re = re, 
              SunweightedH = SunweightedH, 
              labels_matched = labels_matched, 
              n_cells_matched = n_cells_matched, 
              mean_pr = mean_pr, 
              mean_re = mean_re, 
              mean_SunweightedH = mean_SunweightedH,
              total_SunweightedH = total_SunweightedH))
}


