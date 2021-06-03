'''
Compute tolpological perormance measures in artificiial sets on DCAE, UMAP and SAUCIE
'''
import math
import pandas as pd
import numpy as np
import os
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors
def get_topology_list(bl):
    """ Gets a touple defining branches and creates a list of nearest neighbour clusters"""
    # basic topology shared by 5 clusters in pentagon
    topolist = [[1,4], [0,2], [1,3], [2,4], [0,3]]
    #correct by adding the negbour from bl touple
    topolist[bl[0]].append(5)
    topolist[bl[1]].append(6)
    return topolist
import random
from scipy.spatial import distance
def get_representation_topology(z, lbls):
    """compute actual , returning 3 nearest 3 neighbours per each cluster in pentagon"""
    #sample each cluster
    l_list= [-7.,  0.,  1.,  2.,  3.,  4.,  5.,  6.]
    indx = random.sample(range(len(lbls)), 5000)
    lbls_s = lbls[indx]
    z_s = z[indx,:]
    topolist_estimate = [[], [], [],[], []]
    for i in range(5):
        dist = [np.mean(distance.cdist(z_s[lbls_s==i,:], z_s[lbls_s==label,:])) for label in l_list]
        #get indexes of  closest clusters, and exclude itself, exclude i th cluster
        seq = sorted(dist)
        rank = [seq.index(v) for v in dist]
        #get top 3
        rank2 = np.array(rank)[np.array(l_list) != np.array(i)]
        l_list2 = np.array(l_list)[np.array(l_list) != np.array(i)]
        nn_list =l_list2[rank2.argsort()][:4]
        topolist_estimate[i] = nn_list
    return topolist_estimate

def get_topology_match score(topolist, topolist_estimate):
    pass




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

    discontinuity, manytoone = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

# Compute performance for UMAP
z_dir  = DATA_ROOT + "Artificial_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Artificial_sets/UMAP_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    # read data
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_2D.npz')
    z = npz_res['z']
    #divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr= (np.max(z[:,0])-np.min(z[:,0]))*(np.max(z[:,1])-np.min(z[:,1]))
    z=z / S_pr * 4 * math.pi

    discontinuity, manytoone = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Artificial_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"
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
    npz_res = np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz')
    z = npz_res['z']
    # divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr = (np.max(z[:, 0]) - np.min(z[:, 0])) * (np.max(z[:, 1]) - np.min(z[:, 1]))
    z = z / S_pr * 4 * math.pi

    discontinuity, manytoone = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

#create Borealis graphs
PLOTS = DATA_ROOT + "Artificial_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Artificial_sets/DCAE_output/Performance/", DATA_ROOT + "Artificial_sets/UMAP_output/Performance/",DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
dir = bor_res_dirs[0]
bl  = list_of_branches[0]
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_branches:
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
        npz_res =  np.load(outfile)
        discontinuity = npz_res['discontinuity']
        manytoone = npz_res['manytoone']
        discontinuity =np.median(discontinuity)
        manytoone= np.median(manytoone)
        line = pd.DataFrame([[methods[i], str(bl), discontinuity, manytoone]],   columns =['method','branch','discontinuity','manytoone'])
        df=  df.append(line)


import seaborn as sns
import matplotlib.pyplot as plt


sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='branch', y='discontinuity', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Discontinuity.png")

g = sns.barplot(x='branch', y='manytoone', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0.34, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Manytoone.png")