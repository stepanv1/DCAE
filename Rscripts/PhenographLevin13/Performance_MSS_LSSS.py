'''
Compute MSS and LSS performance measures on DCAE, UMAP and
'''
import math

import numpy as np
import os
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors


os.chdir('/home/stepan/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/stepan/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])

# Compute performance for DCAE
z_dir  = DATA_ROOT + "Artificial_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/DCAE_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    #read data
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res=np.load(z_dir + '/' + str(bl) + '_latent_rep_3D.npz')
    z= npz_res['z']

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS=MSS, LSSS= LSSS)

# Compute performance for UMAP
z_dir  = DATA_ROOT + "Artificial_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Artificial_sets/UMAP_output/Performance"
#bl = list_of_branches[1]
for bl in list_of_branches:
    # read data
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/' + str(bl) + '_latent_rep_3D.npz')
    z = npz_res['z']
    #divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr= (np.max(z[:,0])-np.min(z[:,0]))*(np.max(z[:,1])-np.min(z[:,1]))
    z=z / S_pr * 4 * math.pi

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS=MSS, LSSS=LSSS)

# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Artificial_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance"
#bl = list_of_branches[1]
for bl in list_of_branches:
    # read data
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Dist = npzfile['Dist']
    Idx = npzfile['Idx']
    neibALL = npzfile['neibALL']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/' + str(bl) + '_latent_rep_3D.npz')
    z = npz_res['z']
    # divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr = (np.max(z[:, 0]) - np.min(z[:, 0])) * (np.max(z[:, 1]) - np.min(z[:, 1]))
    z = z / S_pr * 4 * math.pi

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS=MSS, LSSS=LSSS)

