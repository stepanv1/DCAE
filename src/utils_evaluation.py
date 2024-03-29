'''
utility functions, mostly for the performance evaluation
'''
import rpy2
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from sklearn.neighbors import NearestNeighbors
from skhubness.neighbors import FalconnLSH #comment out then running SAUCIE and UMAP scripts
from sklearn import metrics, datasets
from plotly.graph_objs import Scatter3d, Scatter
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
import seaborn as sns
from joblib import Parallel, delayed
import ot
import faiss
import math

import random

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


import ctypes
from numpy.ctypeslib import ndpointer
lib = ctypes.cdll.LoadLibrary("/media/grinek/Seagate/DCAE/Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #Sigma
                ctypes.c_size_t]

def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

def compute_f1(lblsT, lblsP):
    string = """
#########################################################################################
# Function to match cluster labels with manual gating (reference standard) population 
# labels and calculate precision, recall, and F1 score
#
# Matching criterion: Hungarian algorithm
#
# Use this function for data sets with multiple populations of interest
#
# Lukas Weber, August 2016
#########################################################################################

# arguments:
# - clus_algorithm: cluster labels from algorithm
# - clus_truth: true cluster labels
# (for both arguments: length = number of cells; names = cluster labels (integers))
helper_match_evaluate_multiple <- function(clus_algorithm, clus_truth) {
  library(clue)
  # number of detected clusters
  n_clus <- length(table(clus_algorithm))
  
  # remove unassigned cells (NA's in clus_truth)
  unassigned <- is.na(clus_truth)
  clus_algorithm <- clus_algorithm[!unassigned]
  clus_truth <- clus_truth[!unassigned]
  if (length(clus_algorithm) != length(clus_truth)) warning("vector lengths are not equal")
  
  tbl_algorithm <- table(clus_algorithm)
  tbl_truth <- table(clus_truth)
  
  # detected clusters in rows, true populations in columns
  pr_mat <- re_mat <- F1_mat <- matrix(NA, nrow = length(tbl_algorithm), ncol = length(tbl_truth))
  
  for (i in 1:length(tbl_algorithm)) {
    for (j in 1:length(tbl_truth)) {
      i_int <- as.integer(names(tbl_algorithm))[i]  # cluster number from algorithm
      j_int <- as.integer(names(tbl_truth))[j]  # cluster number from true labels
      
      true_positives <- sum(clus_algorithm == i_int & clus_truth == j_int, na.rm = TRUE)
      detected <- sum(clus_algorithm == i_int, na.rm = TRUE)
      truth <- sum(clus_truth == j_int, na.rm = TRUE)
      
      # calculate precision, recall, and F1 score
      precision_ij <- true_positives / detected
      recall_ij <- true_positives / truth
      F1_ij <- 2 * (precision_ij * recall_ij) / (precision_ij + recall_ij)
      
      if (F1_ij == "NaN") F1_ij <- 0
      
      pr_mat[i, j] <- precision_ij
      re_mat[i, j] <- recall_ij
      F1_mat[i, j] <- F1_ij
    }
  }
  
  # put back cluster labels (note some row names may be missing due to removal of unassigned cells)
  rownames(pr_mat) <- rownames(re_mat) <- rownames(F1_mat) <- names(tbl_algorithm)
  colnames(pr_mat) <- colnames(re_mat) <- colnames(F1_mat) <- names(tbl_truth)
  
  # match labels using Hungarian algorithm applied to matrix of F1 scores (Hungarian
  # algorithm calculates an optimal one-to-one assignment)
  
  # use transpose matrix (Hungarian algorithm assumes n_rows <= n_cols)
  F1_mat_trans <- t(F1_mat)
  
  if (nrow(F1_mat_trans) <= ncol(F1_mat_trans)) {
    # if fewer (or equal no.) true populations than detected clusters, can match all true populations
    labels_matched <- clue::solve_LSAP(F1_mat_trans, maximum = TRUE)
    # use row and column names since some labels may have been removed due to unassigned cells
    labels_matched <- as.numeric(colnames(F1_mat_trans)[as.numeric(labels_matched)])
    names(labels_matched) <- rownames(F1_mat_trans)
    
  } else {
    # if fewer detected clusters than true populations, use transpose matrix and assign
    # NAs for true populations without any matching clusters
    labels_matched_flipped <- clue::solve_LSAP(F1_mat, maximum = TRUE)
    # use row and column names since some labels may have been removed due to unassigned cells
    labels_matched_flipped <- as.numeric(rownames(F1_mat_trans)[as.numeric(labels_matched_flipped)])
    names(labels_matched_flipped) <- rownames(F1_mat)
    
    labels_matched <- rep(NA, ncol(F1_mat))
    names(labels_matched) <- rownames(F1_mat_trans)
    labels_matched[as.character(labels_matched_flipped)] <- as.numeric(names(labels_matched_flipped))
  }
  
  # precision, recall, F1 score, and number of cells for each matched cluster
  pr <- re <- F1 <- n_cells_matched <- rep(NA, ncol(F1_mat))
  names(pr) <- names(re) <- names(F1) <- names(n_cells_matched) <- names(labels_matched)
  
  for (i in 1:ncol(F1_mat)) {
    # set to 0 if no matching cluster (too few detected clusters); use character names 
    # for row and column indices in case subsampling completely removes some clusters
    pr[i] <- ifelse(is.na(labels_matched[i]), 0, pr_mat[as.character(labels_matched[i]), names(labels_matched)[i]])
    re[i] <- ifelse(is.na(labels_matched[i]), 0, re_mat[as.character(labels_matched[i]), names(labels_matched)[i]])
    F1[i] <- ifelse(is.na(labels_matched[i]), 0, F1_mat[as.character(labels_matched[i]), names(labels_matched)[i]])
    
    n_cells_matched[i] <- sum(clus_algorithm == labels_matched[i], na.rm = TRUE)
  }
  
  # means across populations
  mean_pr <- mean(pr)
  mean_re <- mean(re)
  mean_F1 <- mean(F1)
  
  return(list(n_clus = n_clus, 
              pr = pr, 
              re = re, 
              F1 = F1, 
              labels_matched = labels_matched, 
              n_cells_matched = n_cells_matched, 
              mean_pr = mean_pr, 
              mean_re = mean_re, 
              mean_F1 = mean_F1))
}
"""
    # import function from R
    rfuncs = SignatureTranslatedAnonymousPackage(string, "rfuncs")
    match_evaluate_multiple=rfuncs.helper_match_evaluate_multiple
    res= match_evaluate_multiple(rpy2.robjects.vectors.IntVector(lblsT), rpy2.robjects.vectors.IntVector(lblsP))
    return res.rx('mean_F1')[0][0]

# from here: https://gist.github.com/mikebirdgeneau/a61459aede3a29743c780548753b7fde
def compute_MVpert(n, min, mode, max, sigma):
    #install R packages  mvtnorm and freedom
    # for this to run
    # R code is taken from https: //gist.github.com/mikebirdgeneau/a61459aede3a29743c780548753b7fde
    string = """
#' Multivariate Beta PERT distributions
#'
#' @description Generates random deviates from correlated (modified) pert distributions.
#'    this is performed by remapping correlated normal distributions to the beta pert
#'    distributions using quantiles.
#'
#' @param n Number of observations. If length(n) > 1, the length is taken to be the number required.
#' @param min Vector of minima.
#' @param mode Vector of modes.
#' @param max Vector of maxima.
#' @param sigma covariance matrix, default is diag(ncol(x)).
#'
#' @return Returns a matrix of values of dimensions n x length(mode).
#' @export
#'
#' @examples
mvtbetapert <- function(n, min, mode, max, sigma = diag(length(mode))){
  require(freedom)
  # Generate correlated normal distributions:
  mvn <- mvtnorm::rmvnorm(n, mean = rep(0, length(mode)), sigma = sigma)
    
  # Convert to quantiles to be re-mapped to the betaPERT distribution:
  for (j in 1:ncol(mvn)){
    mvn[, j] <- rank(mvn[, j]) / length(mvn[, j])
  }
  # Convert quantiles desired betaPERT distributions
  for (j in 1:ncol(mvn)){
    prt <- rpert(n, x.min = min[j],  x.max = max[j], x.mode = mode[j],)
    mvn[, j] <- quantile(prt, probs = mvn[, j])
  }
  return(mvn)
}
"""
    # import function from R
    rfuncs = SignatureTranslatedAnonymousPackage(string, "rfuncs")
    mvtbetapert = rfuncs.mvtbetapert
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    res = mvtbetapert(n, min, mode, max, sigma)
    np.res=np.array(res)
    np.res
    return np.res

#generate fibbonachi grid for surface area estimate
# from here: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012

def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

# if nearest neighbiur of fibbonaci grid is another memmber of the grid area
# around the grid is called empty. Surface are is then estimates as n_empty/n_nonempty*4*pi*r^2
def estimate_surface_area_spehere(data, n_fib=10000 ):
    pass
#same but with a simple grid
def estimate_surface_area_plane(data, n_fib=10000):
    pass


def compute_cluster_performance(lblsT, lblsP):
    Adjusted_Rand_Score = metrics.adjusted_rand_score(lblsT, lblsP)
    adjusted_MI_Score = metrics.adjusted_mutual_info_score(lblsT, lblsP)
    #find cluster matching and mean weighted f1_score
    lblsTint, lblsPint =  np.unique(lblsT, return_inverse=True)[1], np.unique(lblsP, return_inverse=True)[1]
    F1_score = compute_f1(lblsTint, lblsPint)
    return {'Adjusted_Rand_Score': Adjusted_Rand_Score, 'adjusted_MI_Score': adjusted_MI_Score, 'F1_score': F1_score}

# comment out then running SAUCIE and UMAP scripts
def find_neighborsLSH(data, k_, cores=16):
    lshf = FalconnLSH(n_candidates = k_+1, radius= 1.0, metric = 'euclidean', num_probes = 50, n_jobs = cores, verbose = 0)
    lshf.fit(data)
    dist, ind = lshf.kneighbors(data, n_candidates=k_+1, return_distance =True)
    return {'dist': np.array(dist[:, 1:]), 'idx': np.array(ind[:, 1: ])}

def find_neighbors(data, k_, metric='euclidean', cores=16, algorithm = "ball_tree"):
    tree = NearestNeighbors(n_neighbors=k_, algorithm=algorithm, leaf_size=30, metric=metric, metric_params=None,
                            n_jobs=cores)
    tree.fit(data)
    dist, ind = tree.kneighbors(return_distance=True)
    return {'dist': np.array(dist), 'idx': np.array(ind)}

#compare neighbourhoof assignments
def compare_neighbours(idx1, idx2, kmax=90):
    nrow = idx1.shape[0]
    ncol = idx1.shape[1]
    match =  np.arange(kmax, dtype = 'float')
    for i in range(kmax):
        print(i)
        match_k = sum([ len(set(idx1[j,0:i]).intersection(idx2[j,0:i]))/(i+1) for j in range(nrow)])
        match[i] =match_k/nrow
        print(match[i])
    return match

#compare neighbourhoof assignments
def neighbour_marker_similarity_score(z, data, kmax=30, num_cores=12):
    neib_z = find_neighbors(z, kmax, metric='euclidean')['idx']
    nrow = data.shape[0]
    match =  np.zeros(kmax, dtype = 'float')
    def score_per_i(i):
        print(i)
        match_k = sum([np.sqrt(np.sum((np.mean(data[neib_z[j, :(i+1)], :], axis=0) - data[j, :]) ** 2)) for j in range(nrow)])
        return match_k / nrow
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i)(i) for i in range(0,kmax))
    match = results
    return match

def neighbour_onetomany_score(z, idx, kmax=30, num_cores=12):
    nrow = z.shape[0]
    match =  np.zeros(kmax, dtype = 'float')
    per_cell_match = np.zeros((kmax, nrow), dtype = 'float')
    def score_per_i(i):
        print(i)
        per_cell= np.array([np.sqrt(np.sum((np.mean(z[idx[j, :(i+1)], :], axis=0) - z[j, :]) ** 2)) for j in range(nrow)])
        match_k = np.sum(per_cell)
        return [match_k / nrow, per_cell]
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i)(i) for i in range(0,kmax))
    for i in range(0,kmax):
        match[i] = results[i][0]
    for i in range(0,kmax):
        per_cell_match[i,:] = results[i][1]
    return [match, per_cell_match]

def neighbour_onetomany_score_normalized(z, idx, kmax=30, num_cores=16):
    #divide MSS by max distance in the neighbourhood in y-space
    #kmax=kmax + 1
    neib_z = find_neighbors(z, kmax, metric='euclidean')['dist']
    nrow = z.shape[0]
    match =  np.zeros(kmax, dtype = 'float')
    per_cell_match = np.zeros((kmax, nrow), dtype = 'float')
    def score_per_i(i):
        print(i)
        per_cell= np.array([np.sqrt(np.sum((np.mean(z[idx[j, :(i+1)], :], axis=0) - z[j, :]) ** 2))/neib_z[j,i] for j in range(nrow)])
        match_k = np.sum(per_cell)
        return [match_k / nrow, per_cell]
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i)(i) for i in range(0,kmax))
    for i in range(0,kmax):
        match[i] = results[i][0]
    for i in range(0,kmax):
        per_cell_match[i,:] = results[i][1]
    return [match, per_cell_match]

def neighbour_marker_similarity_score_per_cell(z, data, kmax=30, num_cores=12):
    kmax=kmax + 1
    nrow = z.shape[0]
    neib_z = find_neighbors(z, kmax, metric='euclidean')['idx']
    match =  np.zeros(kmax, dtype = 'float')
    per_cell_match = np.zeros((kmax, nrow), dtype = 'float')
    def score_per_i(i):
        print(i)
        per_cell= np.array([np.sqrt(np.sum((np.mean(data[neib_z[j, :(i+1)], :], axis=0) - data[j, :]) ** 2)) for j in range(nrow)])
        match_k = np.sum(per_cell)
        return [match_k / nrow, per_cell]
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i)(i) for i in range(0,kmax))
    for i in range(0,kmax):
        match[i] = results[i][0]
    for i in range(0,kmax):
        per_cell_match[i,:] = results[i][1]
    return [match, per_cell_match]

def generate_clusters_pentagon(num_noisy = 25, branches_loc = [3,4], sep=3/2, pent_size= 2, k=2):
    """ function to generate artificial clusters with branches and different
    number of noisy dimensions, branching
    Creates a cluster with noisy dimensions and branches at the clusters
    which number is passed as an argument
    branches can be between 0 and 4
    :param num_noisy: number of non-informative dimensions
           branches_loc a number, (0 to 4) to which 'core clusters' to attach a branch
    :return: a numpy array with clusters nad labels
    """
    d= 5
    # subspace clusters centers
    original_dim = d + num_noisy

    # generate pentagon for dims 0 and 3 in clusters 0 to 4
    pentagon = []
    R = pent_size
    for n in range(0, 5):
        x = R * math.cos(math.radians(90 + n * 72))
        y = R * math.sin(math.radians(90 + n * 72))
        pentagon.append([x, y])

    pnt=  np.array(pentagon)
    pnt[:,0]=pnt[:,0] - np.min(pnt[:,0])
    pnt[:,1]=pnt[:,1] - np.min(pnt[:,1])

    center_list0 = np.array([np.zeros(original_dim),
                             np.zeros(original_dim),
                   np.zeros(original_dim),
                             np.zeros(original_dim),
                   np.zeros(original_dim), # pentagon seed
                   np.zeros(original_dim), np.zeros(original_dim), #branches
                   np.concatenate((np.zeros(4), 0.5*sep*np.ones(1),
                                   np.zeros((num_noisy-4)), np.ones(4)), axis=0)]) #big one

    #pentagonalizing:
    for i in range(0,5):
        center_list0[i][[0,3]] = pnt[i,:]


    #attaching branches to there positions in linear squence
    import copy
    center_list = copy.deepcopy(center_list0)

    center_list[5,:] = center_list0[branches_loc[0],:]
    center_list[5,1] = sep
    center_list[6,:] = center_list0[branches_loc[1],:]
    center_list[6,2] = sep

    # cluster populatiosn
    ncl0 = ncl1 = ncl2 = ncl3  = ncl4 = ncl5 = ncl6 = int(k*6000)
    ncl7 = int(k*20000)
    # cluster labels
    lbls = np.concatenate((np.zeros(ncl0), np.ones(ncl1), 2*np.ones(ncl2), 3*np.ones(ncl3), 4*np.ones(ncl4),
                           5*np.ones(ncl5), 6*np.ones(ncl6), -7*np.ones(ncl7)), axis=0)
    #introduce correlation

    r = datasets.make_spd_matrix(d, random_state=12346)
    r5 = datasets.make_spd_matrix(d,  random_state=12348)
    r6 = datasets.make_spd_matrix(d, random_state=12349)
    r7 = datasets.make_spd_matrix(d, random_state=12347)

    #compute_MVpert(n, min, mode, max, r)
    minb = np.zeros(d)
    modeb =sep/2 * np.ones(d)
    maxb =sep*np.ones(d)
    # Generate the samples.
    y0 = center_list[0,:][:d]+compute_MVpert(ncl0, minb, modeb, maxb, r)
    y1 = center_list[1,:][:d]+compute_MVpert(ncl1, minb, modeb, maxb, r)
    y2 = center_list[2,:][:d]+compute_MVpert(ncl2, minb, modeb, maxb, r)
    y3 = center_list[3,:][:d]+compute_MVpert(ncl3, minb, modeb, maxb, r)
    y4 = center_list[4,:][:d]+compute_MVpert(ncl4, minb, modeb, maxb, r)
    y5 = center_list[5,:][:d]+compute_MVpert(ncl5, minb, modeb, maxb, r5)
    y6 = center_list[6,:][:d]+compute_MVpert(ncl6, minb, modeb, maxb, r6)
    y7 = center_list[7,:][np.concatenate((np.zeros(4), np.ones(1), np.zeros(num_noisy-4),
                                    np.ones(4))).astype('bool')]+compute_MVpert(ncl7, minb, modeb, maxb, r7)


    cl0 = np.concatenate([y0, np.zeros((ncl0,original_dim - d))], axis=1 )
    cl1 = np.concatenate([y1, np.zeros((ncl1,original_dim - d))], axis=1 )
    cl2= np.concatenate([y2, np.zeros((ncl2,original_dim - d))], axis=1 )
    cl3 = np.concatenate([y3, np.zeros((ncl3,original_dim - d))], axis=1 )
    cl4 = np.concatenate([y4, np.zeros((ncl4,original_dim - d))], axis=1 )
    cl5 = np.concatenate([y5, np.zeros((ncl5,original_dim - d))], axis=1 )
    cl6 = np.concatenate([y6, np.zeros((ncl6,original_dim - d))], axis=1 )
    cl7 =  np.zeros((ncl7,original_dim ))
    cl7[:,np.concatenate((np.zeros(4), np.ones(1), np.zeros(num_noisy-4), np.ones(4))).astype('bool')] = y7

    # add noise to orthogonal dimensions
    noise_scale =0.2
    diagM=  np.diag(np.ones(original_dim - d))
    minb = np.zeros(original_dim - d)
    modeb = noise_scale * np.ones(original_dim - d)
    maxb = 3*sep * np.ones(original_dim - d)
    # Generate the random samples.
    cl0_noisy = cl0 + np.concatenate([np.zeros((ncl0,d)), compute_MVpert(ncl0, minb, modeb, maxb, diagM)], axis=1 )
    cl1_noisy = cl1 + np.concatenate([np.zeros((ncl1,d)), compute_MVpert(ncl1, minb, modeb, maxb, diagM)], axis=1 )
    cl2_noisy = cl2 + np.concatenate([np.zeros((ncl2,d)), compute_MVpert(ncl2, minb, modeb, maxb, diagM)], axis=1 )
    cl3_noisy = cl3 + np.concatenate([np.zeros((ncl3,d)), compute_MVpert(ncl3, minb, modeb, maxb, diagM)], axis=1 )
    cl4_noisy = cl4 + np.concatenate([np.zeros((ncl4,d)), compute_MVpert(ncl4, minb, modeb, maxb, diagM)], axis=1 )
    cl5_noisy = cl5 + np.concatenate([np.zeros((ncl5,d)), compute_MVpert(ncl5, minb, modeb, maxb, diagM)], axis=1 )
    cl6_noisy = cl6 + np.concatenate([np.zeros((ncl6,d)), compute_MVpert(ncl6, minb, modeb, maxb, diagM)], axis=1 )
    cl7_noisy = cl7
    cl7_noisy[:, ~np.concatenate((np.zeros(4), np.ones(1), np.zeros(num_noisy-4), np.ones(4))).astype('bool')] = \
        compute_MVpert(ncl7, minb, modeb, maxb, diagM)

    noisy_clus = np.concatenate((cl0_noisy, cl1_noisy, cl2_noisy, cl3_noisy, cl4_noisy, cl5_noisy, cl6_noisy,
                                 cl7_noisy), axis=0)

    return noisy_clus, lbls

def neighbour_marker_similarity_score_per_cell(z, data, kmax=30, num_cores=12):
    nrow = z.shape[0]
    neib_z = find_neighbors(z, kmax, metric='euclidean')['idx']
    match =  np.zeros(kmax, dtype = 'float')
    per_cell_match = np.zeros((kmax, nrow), dtype = 'float')
    def score_per_i(i):
        print(i)
        per_cell= np.array([np.sqrt(np.sum((np.mean(data[neib_z[j, :(i+1)], :], axis=0) - data[j, :]) ** 2)) for j in range(nrow)])
        match_k = np.sum(per_cell)
        return [match_k / nrow, per_cell]
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i)(i) for i in range(0,kmax))
    for i in range(0,kmax):
        match[i] = results[i][0]
    for i in range(0,kmax):
        per_cell_match[i,:] = results[i][1]
    return [match, per_cell_match]



#plotly 3D plotting functions
def plot3D_cluster_colors(z, lbls, camera=None, legend=True, msize=1):
    x = z[:, 0]
    y = z[:, 1]
    z1 = z[:, 2]

    lbls_list = np.unique(lbls)
    nM = len(np.unique(lbls))

    palette = sns.color_palette("husl", nM)
    colors = np.array([rgb2hex(palette[i]) for i in range(len(palette))])

    fig = go.Figure()
    for m in range(nM):
        IDX = [x == lbls_list[m] for x in lbls]
        xs = x[IDX];
        ys = y[IDX];
        zs = z1[IDX];
        fig.add_trace(Scatter3d(x=xs, y=ys, z=zs,
                                name=lbls_list[m],
                                mode='markers',
                                marker=dict(
                                    size=msize,
                                    color=colors[m],  # set color to an array/list of desired values
                                    opacity=0.5,
                                ),
                                text=lbls[IDX],
                                # hoverinfo='text')], filename='tmp.html')
                                hoverinfo='text'))
        fig.update_layout(yaxis=dict(range=[-3, 3]),
                          margin=dict(l=0, r=0, b=0, t=10))
        if camera == None:
            camera = dict(
                up=dict(x=0, y=0., z=1),
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        fig.update_layout(scene_camera=camera, showlegend=legend)
        # set colour to white
        fig.update_layout(dict(xaxis=dict( showgrid=True, gridwidth=1, gridcolor="#eee")))
        fig.update_layout(scene=dict(
            xaxis=dict(visible=True, backgroundcolor='rgba(0,0,0,0)', showgrid=True, gridcolor="#eee", gridwidth=1,
                       showline=True, zeroline=True),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showgrid=True, gridcolor="#eee", gridwidth=1, showline=True,
                       zeroline=True),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showgrid=True, gridcolor="#eee", gridwidth=1, showline=True,
                       zeroline=True)
        ))

    fig.update_layout(legend=dict(font=dict(family="Courier", size=20, color="black"),itemsizing='constant' ),
                      legend_title=dict(font=dict(family="Courier", size=30, color="blue" )) )
    return fig

def plot2D_cluster_colors(z, lbls, legend=True, msize=1):
    x = z[:, 0]
    y = z[:, 1]

    lbls_list = np.unique(lbls)
    nM = len(np.unique(lbls))

    palette = sns.color_palette('husl', nM)
    colors = np.array([rgb2hex(palette[i]) for i in range(len(palette))])

    fig = go.Figure()
    for m in range(nM):
        IDX = [x == lbls_list[m] for x in lbls]
        xs = x[IDX];
        ys = y[IDX];
        fig.add_trace(Scatter(x=xs, y=ys,
                                name=lbls_list[m],
                                mode='markers',
                                marker=dict(
                                    size=msize,
                                    color=colors[m],  # set color to an array/list of desired values
                                    opacity=0.5,
                                ),
                                text=lbls[IDX],
                                # hoverinfo='text')], filename='tmp.html')
                                hoverinfo='text'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=10))
        fig.update_layout(showlegend=legend)
        fig.update_layout(dict(paper_bgcolor='white', plot_bgcolor='white'),
                          xaxis=dict(showgrid=True, gridcolor="#eee", gridwidth=1, showline=True, zeroline=True),
                          yaxis=dict(showgrid=True, gridcolor="#eee", gridwidth=1, showline=True, zeroline=True)
                          )
    fig.update_layout(legend=dict(font=dict(family="Courier", size=20, color="black"), itemsizing='constant'),
                      legend_title=dict(font=dict(family="Courier", size=30, color="blue")))
    return fig
#overlap with markers
def plot3D_marker_colors(z, data, markers, sub_s = 50000, lbls=None, msize=1):
    nrows = z.shape[0]
    sub_idx = np.random.choice(range(nrows), sub_s, replace=False)
    x = z[sub_idx, 0]
    y = z[sub_idx, 1]
    zz = z[sub_idx, 2]
    lbls_s = lbls[sub_idx]
    sFrame = data[sub_idx, :]
    result = [markers.index(i) for i in markers]
    sFrame = sFrame[:, result]
    nM = len(markers)
    m = 0
    fig = go.Figure()
    fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                            mode='markers',
                            marker=dict(
                                size=msize,
                                color=sFrame[:, m],  # set color to an array/list of desired values
                                colorscale='Viridis',  # choose a colorscale
                                opacity=0.5,
                                colorbar=dict(xanchor='left', x=-0.05, len=0.5),
                                showscale=True
                            ),
                            text=lbls_s,
                            hoverinfo='text',
                            ))
    for m in range(1, nM):
        fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                                mode='markers',
                                visible="legendonly",
                                marker=dict(
                                    symbol="circle",
                                    size=msize,
                                    color=sFrame[:, m],  # set color to an array/list of desired values
                                    colorscale='Viridis',  # choose a colorscale
                                    opacity=0.5,
                                    colorbar=dict(xanchor='left', x=-0.05, len=0.5),
                                    showscale=True
                                ),
                                text=lbls_s,
                                hoverinfo='text'
                                ))

    vis_mat = np.zeros((nM, nM), dtype=bool)
    np.fill_diagonal(vis_mat, True)

    button_list = list([dict(label=markers[m],
                             method='update',
                             args=[{'visible': vis_mat[m, :]},
                                   # {'title': markers[m],
                                   {'showlegend': False}]) for m in range(len(markers))])
    fig.update_layout(
        showlegend=False,
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=button_list
        )
        ])
    return fig

def plot2D_marker_colors(z, data, markers, sub_s = 50000, lbls=None, msize=1):
    nrows = z.shape[0]
    sub_idx = np.random.choice(range(nrows), sub_s, replace=False)
    x = z[sub_idx, 0]
    y = z[sub_idx, 1]
    lbls_s = lbls[sub_idx]
    sFrame = data[sub_idx, :]
    result = [markers.index(i) for i in markers]
    sFrame = sFrame[:, result]
    nM = len(markers)
    m = 0
    fig = go.Figure()
    fig.add_trace(Scatter(x=x, y=y,
                            mode='markers',
                            marker=dict(
                                size=msize,
                                color=sFrame[:, m],  # set color to an array/list of desired values
                                colorscale='Viridis',  # choose a colorscale
                                opacity=0.5,
                                colorbar=dict(xanchor='left', x=-0.05, len=0.5),
                                showscale=True
                            ),
                            text=lbls_s,
                            hoverinfo='text',
                            ))
    for m in range(1, nM):
        fig.add_trace(Scatter(x=x, y=y,
                                mode='markers',
                                visible="legendonly",
                                marker=dict(
                                    size=msize,
                                    color=sFrame[:, m],  # set color to an array/list of desired values
                                    colorscale='Viridis',  # choose a colorscale
                                    opacity=0.5,
                                    colorbar=dict(xanchor='left', x=-0.05, len=0.5),
                                    showscale=True
                                ),
                                text=lbls_s,
                                hoverinfo='text'
                                ))

    vis_mat = np.zeros((nM, nM), dtype=bool)
    np.fill_diagonal(vis_mat, True)

    button_list = list([dict(label=markers[m],
                             method='update',
                             args=[{'visible': vis_mat[m, :]},
                                   # {'title': markers[m],
                                   {'showlegend': False}]) for m in range(len(markers))])
    fig.update_layout(
        showlegend=False,
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=button_list
        )
        ])
    return fig

def plot3D_performance_colors(z, perf, lbls=None, msize=1):
    x = z[:, 0]
    y = z[:, 1]
    zz = z[:, 2]

    fig = go.Figure()
    fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                            mode='markers',
                            marker=dict(
                                size=msize,
                                color=perf,  # set color to an array/list of desired values
                                colorscale='Viridis',  # choose a colorscale
                                opacity=0.5,
                                colorbar=dict(xanchor='left', x=-0.05, len=0.5),
                                showscale=True
                            ),
                            text=lbls,
                            hoverinfo='text',
                            ))
    fig.update_layout(scene = dict(
                      xaxis = dict(visible=True, backgroundcolor='rgba(0,0,0,0)', showgrid = True, gridcolor = "#eee", gridwidth = 1, showline=True, zeroline=True),
                      yaxis=dict(backgroundcolor='rgba(0,0,0,0)',showgrid=True, gridcolor= "#eee", gridwidth = 1, showline=True, zeroline=True),
                      zaxis=dict(backgroundcolor='rgba(0,0,0,0)',showgrid=True, gridcolor="#eee", gridwidth=1, showline=True, zeroline=True)
                      ))
    return fig

def plot2D_performance_colors(z, perf, lbls=None, msize=1):
    x = z[:, 0]
    y = z[:, 1]
    fig = go.Figure()
    fig.add_trace(Scatter(x=x, y=y,
                            mode='markers',
                            marker=dict(
                                size=msize,
                                color=perf,  # set color to an array/list of desired values
                                colorscale='Viridis',  # choose a colorscale
                                opacity=0.5,
                                colorbar=dict(xanchor='left', x=-0.1, len=0.5),
                                showscale=True
                            ),
                            text=lbls,
                            hoverinfo='text',
                            ))

    fig.update_layout(dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'),
                      xaxis = dict(showgrid = True, gridcolor = "#eee", gridwidth = 1, showline=True, zeroline=True),
                      yaxis=dict(showgrid=True, gridcolor= "#eee", gridwidth = 1, showline=True, zeroline=True)
                      )

    return fig




#project on mean radius
def projZ(x):
    def radius(a):
        return np.sqrt(np.sum(a**2))
    r = np.mean(np.apply_along_axis(radius, 1, x))
    return(x/r)

####################################
# functions from https://github.com/jlmelville/quadra
# Area under the RNX curve.!
# http://jlmelville.github.io/sneer/analysis.html  - also R code
# for precision recall measurements
# functions from pltutils:
# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# slightly modified by SG to allow knn-matrix as input
#https://github.com/BorealisAI/eval_dr_by_wsd
#https://www.borealisai.com/en/blog/dimensionality-reduction-finally-has-quantifiable-imperfections/
#https://www.borealisai.com/media/filer_public/97/35/9735bcc1-635f-40ab-a7e0-3873ced5b1d3/nips_2018.pdf
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def tickoff(ax=None):
    if ax is not None:
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        try:
            ax.zaxis.set_major_formatter(NullFormatter())
        except:
            pass
    else:
        plt.tick_params(
            axis='both',        # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom='off',       # ticks along the bottom edge are off
            top='off',          # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off

def show3d(data, t, ax, view_init=None, cmap=plt.cm.Spectral, linewidth=0.,
           markersize=10.):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.scatter(x, y, z, c=t, cmap=cmap, lw=linewidth, s=markersize)
    if view_init is not None:
        ax.view_init(*view_init)
    tickoff(ax)
# from wsd.py

def get_self_knn_idx(data, k):
    dim = data.shape[1]
    data = np.ascontiguousarray(data.astype('float32'))
    try:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 1
        ngpu = 2
        res = [faiss.StandardGpuResources() for i in range(ngpu)]
        index = faiss.GpuIndexFlatL2(res[1], dim, cfg)
    except:
        index = faiss.IndexFlatL2(dim)
    index.add(data)

    zzz, kidx = index.search(data, k + 1)

    return kidx[:, 1:]

def get_mean_pdist(x, num=None):
    if num is None:
        num = len(x)
    rand_idx = random.sample(range(len(x)), min(len(x), num))
    distmat = squareform(pdist(x[rand_idx]))# creates 1000x1000: TODO change this
    return distmat[np.triu_indices(distmat.shape[0])].mean()

def normalize_data_by_mean_pdist(x, num=None):
    if num is None:
        num = len(x)
    return x / get_mean_pdist(x, num)

def get_emd2(pts1, pts2):
    distmat = ot.dist(pts1, pts2)
    a = ot.unif(len(pts1))
    b = ot.unif(len(pts2))
    return ot.emd2(a, b, distmat)

def get_wsd_scores(x, y, k, num_meandist=None, compute_knn_x=False, x_knn=None):
    if compute_knn_x:
        kidx_x = get_self_knn_idx(x, k)
    else:
        kidx_x = x_knn
    kidx_y = get_self_knn_idx(y, k)
    x = normalize_data_by_mean_pdist(x, num_meandist)
    y = normalize_data_by_mean_pdist(y, num_meandist)

    assert len(kidx_x) == len(kidx_x) == len(x) == len(y)

    discontiuity = np.array(
        [get_emd2(y[kidx_x[i]], y[kidx_y[i]]) for i in range(len(x))]
    )
    manytoone = np.array(
        [get_emd2(x[kidx_x[i]], x[kidx_y[i]]) for i in range(len(x))]
    )

    return discontiuity, manytoone


def get_emd2_normalized(pts1, pts2):
    distmat = ot.dist(pts1, pts2, metric = 'euclidean') #by deafault squared euclidean
    a = ot.unif(len(pts1))
    b = ot.unif(len(pts2))
    return ot.emd2(a, b, distmat)

def get_wsd_scores_normalized(x, y, k, num_meandist=None, compute_knn_x=False, x_knn=None, nc=12):
    if compute_knn_x:
        kidx_x = get_self_knn_idx(x, k)
    else:
        kidx_x = x_knn
    #get distanceas and knn in y-space
    x = normalize_data_by_mean_pdist(x, num_meandist)
    y = normalize_data_by_mean_pdist(y, num_meandist)

    neib = find_neighbors(y, k, metric='euclidean', cores=nc)
    dist_y, kidx_y = neib['dist'], neib['idx']

    assert len(kidx_x) == len(kidx_x) == len(x) == len(y)

    discontiuity = np.array(
        [get_emd2_normalized(y[kidx_x[i]], y[kidx_y[i]])/(dist_y[i,k-1]) for i in range(len(x))]# distance to knn
    )
    manytoone = np.array(
        [get_emd2(x[kidx_x[i]], x[kidx_y[i]]) for i in range(len(x))]
    )

    return discontiuity, manytoone

def delta_wsd_scores(x, y, idx,  kmax=30, num_cores=12):
    #wsd scores with optimal transportation to delta function
    def d_dis_score(y=y, idx=idx, kmax=kmax, num_cores=num_cores):
        kmax = kmax + 1
        nrow = y.shape[0]
        match = np.zeros(kmax, dtype='float')
        per_cell_match = np.zeros((kmax, nrow), dtype='float')

        def score_per_i(i):
            print(i)
            per_cell = np.array(
                [np.mean(np.sqrt(np.sum( (y[idx[j, :i], :] - y[j, :]) **2, axis=1))) for j in range(nrow)])
            match_k = np.sum(per_cell)
            return [match_k / nrow, per_cell]

        results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
            delayed(score_per_i)(i) for i in range(1, kmax))
        for i in range(1, kmax):
            match[i] = results[i - 1][0]
        for i in range(1, kmax):
            per_cell_match[i, :] = results[i - 1][1]
        return [match, per_cell_match]

    def d_ms_score(y=y, x=x, kmax=kmax, num_cores=num_cores):
        kmax = kmax + 1
        nrow = y.shape[0]
        neib_y = find_neighbors(y, kmax, metric='euclidean')['idx']
        match = np.zeros(kmax, dtype='float')
        per_cell_match = np.zeros((kmax, nrow), dtype='float')

        def score_per_i(i):
            print(i)
            per_cell = np.array(
                [np.mean(np.sqrt(np.sum((x[neib_y[j, :i], :] - x[j, :]) ** 2, axis=1))) for j in range(nrow)])
            match_k = np.sum(per_cell)
            return [match_k / nrow, per_cell]

        results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
            delayed(score_per_i)(i) for i in range(1, kmax))
        for i in range(1, kmax):
            match[i] = results[i - 1][0]
        for i in range(1, kmax):
            per_cell_match[i, :] = results[i - 1][1]
        return [match, per_cell_match]

    # call both functions
    d_dis = d_dis_score(y=y, idx=idx, kmax=kmax, num_cores=num_cores)
    d_ms = d_ms_score(y=y, x=x, kmax=kmax, num_cores=num_cores)
    return d_dis, d_ms

# function to generate artificial clusters with branches and different
# number of noisy dimensions
def preprocess_artificial_clusters(aFrame, lbls, k=30, num_cores=12, outfile='test'):
    """ function to generate
    :param aFrame, lbls:np.array with clusters and thir lbls
    :return: NULL
    """
    k3 = k * 3

    original_dim=aFrame.shape[1]

    nb=find_neighbors(aFrame, k3, metric='euclidean', cores=num_cores)
    Idx = nb['idx']; Dist = nb['dist']

    # find nearest neighbours
    rk=range(k3)
    def singleInput(i):
         nei =  aFrame[Idx[i,:],:]
         di = [np.sqrt(sum(np.square(aFrame[i] - nei[k_i,]))) for k_i in rk]
         return [nei, di, i]
    nrow = len(lbls)
    inputs = range(nrow)

    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)

    neibALL = np.zeros((nrow, k3, original_dim))
    Distances = np.zeros((nrow, k3))
    neib_weight = np.zeros((nrow, k3))
    Sigma = np.zeros(nrow, dtype=float)
    for i in range(nrow):
        neibALL[i,] = results[i][0]
    for i in range(nrow):
        Distances[i,] = results[i][1]
    #Compute perpelexities
    nn=30
    perp((Distances[:,0:k3]),       nrow,     original_dim,   neib_weight,          nn,          k3,   Sigma,    12)
          #(     double* dist,      int N,    int D,       double* P,     double perplexity,    int K, int num_threads)

    topk = np.argsort(neib_weight, axis=1)[:,-nn:]
    topk= np.apply_along_axis(np.flip, 1, topk,0)
    neibALL=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])

    np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
             neibALL=neibALL,  Sigma=Sigma)
    return aFrame, Idx, Dist, Sigma, lbls, neibALL
#comment out then running SAUCIE and UMAP scripts
def preprocess_artificial_clusters_LSH(aFrame, lbls, k=30, num_cores=16, outfile='test'):
    """ function to generate
    :param aFrame, lbls:np.array with clusters and thir lbls
    :return: NULL
    """
    k3 = k * 3

    original_dim=aFrame.shape[1]

    nb=find_neighborsLSH(aFrame, k3, cores=num_cores)
    Idx = nb['idx']; Dist = nb['dist']

    # find nearest neighbours
    nn=k
    rk=range(k3)
    def singleInput(i):
         nei =  aFrame[Idx[i,:],:]
         di = [np.sqrt(sum(np.square(aFrame[i] - nei[k_i,]))) for k_i in rk]
         return [nei, di, i]
    nrow = len(lbls)
    inputs = range(nrow)

    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)

    neibALL = np.zeros((nrow, k3, original_dim))
    Distances = np.zeros((nrow, k3))
    neib_weight = np.zeros((nrow, k3))
    Sigma = np.zeros(nrow, dtype=float)
    for i in range(nrow):
        neibALL[i,] = results[i][0]
    for i in range(nrow):
        Distances[i,] = results[i][1]
    #Compute perpelexities
    nn=30
    perp((Distances[:,0:k3]),       nrow,     original_dim,   neib_weight,          nn,          k3,   Sigma,    12)
          #(     double* dist,      int N,    int D,       double* P,     double perplexity,    int K, int num_threads)

    topk = np.argsort(neib_weight, axis=1)[:,-nn:]
    topk= np.apply_along_axis(np.flip, 1, topk,0)
    neibALL=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])

    np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
             neibALL=neibALL,  Sigma=Sigma)
    return aFrame, Idx, Dist, Sigma, lbls, neibALL
