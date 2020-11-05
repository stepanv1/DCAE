#wrapper requires rpy2 installed (use pip install and restart pyhton)
#https://github.com/lmweber/cytometry-clustering-comparison/blob/master/helpers/helper_match_evaluate_multiple.R
#labels should be incoded as naturals 1,2, 3
import rpy2
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
import seaborn as sns

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
    rfuncs = SignatureTranslatedAnonymousPackage(string, "rfuncs")

    match_evaluate_multiple=rfuncs.helper_match_evaluate_multiple
    res= match_evaluate_multiple(rpy2.robjects.vectors.IntVector(lblsT), rpy2.robjects.vectors.IntVector(lblsP))
    return res.rx('mean_F1')[0][0]

def compute_cluster_performance(lblsT, lblsP):
    Adjusted_Rand_Score = metrics.adjusted_rand_score(lblsT, lblsP)
    adjusted_MI_Score = metrics.adjusted_mutual_info_score(lblsT, lblsP)
    #find cluster matching and mean weighted f1_score
    lblsTint, lblsPint =  np.unique(lblsT, return_inverse=True)[1], np.unique(lblsP, return_inverse=True)[1]
    F1_score = compute_f1(lblsTint, lblsPint)
    return {'Adjusted_Rand_Score': Adjusted_Rand_Score, 'adjusted_MI_Score': adjusted_MI_Score, 'F1_score': F1_score}

def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

def find_neighbors(data, k_, metric='manhattan', cores=12):
    tree = NearestNeighbors(n_neighbors=k_, algorithm="ball_tree", leaf_size=30, metric=metric, metric_params=None,
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

'''
data=aFrame
idx = find_neighbors(z, k_ = kmax, metric=metric, cores=12)['idx']
idx = find_neighbors(embedUMAP, k_ = kmax, metric=metric, cores=12)['idx']
def compute_homogenuity(data, z, idx, kmax=30, metric= 'euclidean'):
    nrow = idx.shape[0]
    ncol = idx.shape[1]
    match =  np.arange(kmax, dtype = 'float')
    hom_k= np.zeros(kmax)
    for i in range(1, kmax):
        hom = sum([sum(np.std(data[idx[j,:(i+1)],:], axis=0)) for j in range(nrow)])/nrow
        hom_k[i] = hom
    return hom_k
hom_kAE = hom_k
hom_UMAP = hom_k
'''

#plotly 3D plotting functions
def plot3D_cluster_colors(x, y, z ,lbls):
    #nrow = len(x)
    # subsIdx=np.random.choice(nrow,  500000)
    num_lbls = (np.unique(lbls, return_inverse=True)[1])
    # analog of tsne plot fig15 from Nowizka 2015, also see fig21

    lbls_list = np.unique(lbls)
    nM = len(np.unique(lbls))

    palette = sns.color_palette(None, nM)
    colors = np.array([rgb2hex(palette[i]) for i in range(len(palette))])

    fig = go.Figure()
    for m in range(nM):
        IDX = [x == lbls_list[m] for x in lbls]
        xs = x[IDX];
        ys = y[IDX];
        zs = z[IDX];
        fig.add_trace(Scatter3d(x=xs, y=ys, z=zs,
                                name=lbls_list[m],
                                mode='markers',
                                marker=dict(
                                    size=1,
                                    color=colors[m],  # set color to an array/list of desired values
                                    opacity=0.5,
                                ),
                                text=lbls[IDX],
                                # hoverinfo='text')], filename='tmp.html')
                                hoverinfo='text'))
        fig.update_layout(yaxis=dict(range=[-3, 3]),
                          margin=dict(l=0, r=0, b=0, t=10))

    return fig

#overlap with markers
def plot3D_marker_colors(z, data, markers, sub_s = 50000, lbls=None):
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
                                size=0.5,
                                color=data[:, m],  # set color to an array/list of desired values
                                colorscale='Viridis',  # choose a colorscale
                                opacity=0.5,
                                colorbar=dict(xanchor='left', x=-0.05, len=0.5),
                                showscale=True
                            ),
                            text=lbls_s,
                            hoverinfo='text',
                            ))
    for m in range(1, nM):
        # for m in range(1,3):
        fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                                mode='markers',
                                visible="legendonly",
                                marker=dict(
                                    size=0.5,
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


def plot2D_cluster_colors(x, y, lbls):
    #nrow = len(x)
    # subsIdx=np.random.choice(nrow,  500000)
    num_lbls = (np.unique(lbls, return_inverse=True)[1])
    # analog of tsne plot fig15 from Nowizka 2015, also see fig21

    lbls_list = np.unique(lbls)
    nM = len(np.unique(lbls))

    palette = sns.color_palette(None, nM)
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
                                    size=1,
                                    color=colors[m],  # set color to an array/list of desired values
                                    opacity=0.5,
                                ),
                                text=lbls[IDX],
                                # hoverinfo='text')], filename='tmp.html')
                                hoverinfo='text'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=10))
    return fig

def plot2D_marker_colors(z, data, markers, sub_s = 50000, lbls=None):
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
                                size=0.5,
                                color=data[:, m],  # set color to an array/list of desired values
                                colorscale='Viridis',  # choose a colorscale
                                opacity=0.5,
                                colorbar=dict(xanchor='left', x=-0.05, len=0.5),
                                showscale=True
                            ),
                            text=lbls_s,
                            hoverinfo='text',
                            ))
    for m in range(1, nM):
        # for m in range(1,3):
        fig.add_trace(Scatter3d(x=x, y=y,
                                mode='markers',
                                visible="legendonly",
                                marker=dict(
                                    size=0.5,
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

#project on mean radius
def projZ(x):
    def radius(a):
        return np.sqrt(np.sum(a**2))
    r = np.mean(np.apply_along_axis(radius, 1, x))
    return(x/r)

