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
from joblib import Parallel, delayed
from pathos import multiprocessing


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

#generate fibbonachi grid for surface area estimate
# from here: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012
import math
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

# if nearest neighbiur of fibbonaci gris is another memmber of the grid area
# around the grid is called empty. Surface are is then estimaptes as n_empty/n_nonempty*4*pi*r^2
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

#compare neighbourhoof assignments
def neighbour_marker_similarity_score(z, data, kmax=30, num_cores=12):
    kmax = kmax + 1
    neib_z = find_neighbors(z, kmax, metric='euclidean')['idx']
    nrow = data.shape[0]
    match =  np.zeros(kmax, dtype = 'float')
    def score_per_i(i):
        print(i)
        match_k = sum([np.sqrt(np.sum((np.mean(data[neib_z[j, :i], :], axis=0) - data[j, :]) ** 2)) for j in range(nrow)])
        return match_k / nrow
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i, check_pickle=False)(i) for i in range(1,kmax))
    match[1:]  = results
    return match

def neighbour_onetomany_score(z, idx, kmax=30, num_cores=12):
    kmax=kmax + 1
    nrow = z.shape[0]
    match =  np.zeros(kmax, dtype = 'float')
    per_cell_match = np.zeros((kmax, nrow), dtype = 'float')
    def score_per_i(i):
        print(i)
        per_cell= np.array([np.sqrt(np.sum((np.mean(z[idx[j, :i], :], axis=0) - z[j, :]) ** 2)) for j in range(nrow)])
        match_k = np.sum(per_cell)
        return [match_k / nrow, per_cell]
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i, check_pickle=False)(i) for i in range(1,kmax))
    for i in range(1,kmax):
        match[i] = results[i-1][0]
    for i in range(1,kmax):
        per_cell_match[i,:] = results[i-1][1]
    return [match, per_cell_match]



def neighbour_marker_similarity_score_per_cell(z, data, kmax=30, num_cores=12):
    kmax=kmax + 1
    nrow = z.shape[0]
    neib_z = find_neighbors(z, kmax, metric='euclidean')['idx']
    match =  np.zeros(kmax, dtype = 'float')
    per_cell_match = np.zeros((kmax, nrow), dtype = 'float')
    def score_per_i(i):
        print(i)
        per_cell= np.array([np.sqrt(np.sum((np.mean(data[neib_z[j, :i], :], axis=0) - data[j, :]) ** 2)) for j in range(nrow)])
        match_k = np.sum(per_cell)
        return [match_k / nrow, per_cell]
    results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(
          delayed(score_per_i, check_pickle=False)(i) for i in range(1,kmax))
    for i in range(1,kmax):
        match[i] = results[i-1][0]
    for i in range(1,kmax):
        per_cell_match[i,:] = results[i-1][1]
    return [match, per_cell_match]



#data = aFrame
#zzz=neighbour_marker_similarity_score(z, data, kmax=90)

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
def plot3D_cluster_colors(z ,lbls):
    x = z[:, 0]
    y = z[:, 1]
    z1 = z[:, 2]
    #nrow = len(x)
    # subsIdx=np.random.choice(nrow,  500000)

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
        zs = z1[IDX];
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


def plot2D_cluster_colors(z, lbls):
    x = z[:, 0]
    y = z[:, 1]
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
#TODO: add above
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
import random

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import ot
import faiss


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

    _, kidx = index.search(data, k + 1)

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

'''
import pandas as pd
#get a subsample of Levine data and create artificial data with it
source_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess'
outfile = source_dir + '/Levine32euclid_scaled.npz'
npzfile = np.load(outfile)
data = npzfile['aFrame'];
color = npzfile['lbls'];
data= data[color!='"unassigned"']
color = color[color!='"unassigned"']
#subsample
nrows = data.shape[0]
sub_idx = np.random.choice(range(nrows), 20000, replace=False)
data= data[sub_idx]
color = color[sub_idx]
color=pd.Categorical(pd.factorize(color)[0])
# example from github
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
np.random.seed(0)
#data, color = datasets.make_blobs(n_samples=10000, centers=20, n_features=30,
#                   random_state=0)

data, color = datasets.make_classification(n_samples=20000, n_features=15,  n_informative=5, n_redundant=0, n_repeated=0,
                n_classes=3, n_clusters_per_class=1, weights=None, flip_y=0.5, class_sep=2.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=12345)

#data, color = datasets.make_swiss_roll(
#    1000, random_state=0)
#y = PCA(n_components=2).fit_transform(data)
import umap.umap_ as umap
mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(data)
y0 =  mapper.transform(data)
import copy
y=copy.deepcopy(y0)
#cdict = {0: 'red', 1: 'blue', 2: 'green'}
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
dataset = pd.DataFrame()
dataset['x'] = y[:, 0]
dataset['y'] = y[:, 1]
dataset['color'] = [str(x) for x in color]
plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
#sns.scatterplot(data=dataset, x="x", y="y", hue="color")
#lets overlap and split our data
y[(y[:,0]>=6) & (y[:,1]>=5), 1]=y[(y[:,0]>=6) & (y[:,1]>=5), 1]+1
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
y[(y[:,0]<6) & (y[:,1]>5), 1]=y[(y[:,0]<6) & (y[:,1]>5), 1]-10
plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
x_idx = find_neighbors(data, 30, metric= 'euclidean', cores = 12)['idx']
discontinuity, manytoone = get_wsd_scores(data, y, 30, num_meandist=10000, compute_knn_x=False, x_knn=x_idx)

onetomany_score = neighbour_onetomany_score(y, x_idx, kmax=30, num_cores=12)[1]
onetomany_score[29,:]
marker_similarity_score = neighbour_marker_similarity_score_per_cell(y, data, kmax=30, num_cores=12)[1]

vmax1 = np.percentile(discontinuity,95)
vmax2 = np.percentile(manytoone,95)
vmax3=np.percentile(onetomany_score[29,:],95)
vmax4 = np.percentile(marker_similarity_score[29,:],95)
vmin1 = np.percentile(discontinuity,5)
vmin2 = np.percentile(manytoone,5)
vmin3=np.percentile(onetomany_score[29,:],5)
vmin4 = np.percentile(marker_similarity_score[29,:],5)




sz=0.1
sns.set_style("white")
fig = plt.figure(figsize=(10, 10))
sbpl1= plt.subplot(3, 2, 1)
plt.title("UMAP")
plt.scatter(y0[:, 0], y0[:, 1], c=color, s=sz, cmap=plt.cm.Spectral)

sbpl2= plt.subplot(3, 2, 2)
plt.title("Broken UMAP")
plt.scatter(y[:, 0], y[:, 1], c=color, s=sz, cmap=plt.cm.Spectral)
#plt.axis("off")

sbpl3 = plt.subplot(3, 2, 3)
plt.title("discontinuity")
img= plt.scatter(y[:, 0], y[:, 1], c=discontinuity,
            vmax=vmax1, vmin=vmin1,
            s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax1)

sbpl4 = plt.subplot(3, 2, 4)
plt.title("many-to-one")
img = plt.scatter(y[:, 0], y[:, 1], c=manytoone,
                  vmax=vmax2, vmin=vmin2,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax2)

plt.subplot(3, 2, 5)
sbpl5 = plt.title("one-to-many")
img = plt.scatter(y[:, 0], y[:, 1], c=onetomany_score[29,:]
, vmax=vmax3, vmin=vmin3,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax3)

sbpl6 = plt.subplot(3, 2, 6)
plt.title("marker similarity")
img = plt.scatter(y[:, 0], y[:, 1], c=marker_similarity_score[29,:]
, vmax=vmax4,  vmin=vmin4,
                  s=sz, cmap=plt.cm.rainbow)
plt.colorbar(img)
plt.clim(vmin1, vmax4)

#cbax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
#cb = plt.colorbar(img, cax=cbax)
#cb.outline.set_linewidth(0.)
#plt.clim(0., vmax)


plt.show()
'''

#TODO: realize global structure measure
# General idea for paper: on tyeh sphere there can be N fartherst points,
# while inside circle ~sqrt(M) - -beter preservation of topology
def k_farthest_neib_sphere():
    '''
    build knn tree (kd or ball)
    find point opposite to given point and
    quire tree for its knn, they will be fartherst points on the sphere

    '''
    pass
def k_farthest_neib_plane():
    '''
    similar idea to farthestspoints_sphere
    instead of convex hull find knn and colsest to
    the opposite point on the eclosing circle will be the furthers one
    :return:
    '''

    pass
def global_structure_preservation_score():
    '''
    find average distance (in x-space) of n-farhest neighbours (in y space) per point
    average over all points in data set
    '''
    pass
