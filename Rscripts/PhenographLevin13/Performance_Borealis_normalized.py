'''
Compute mmd-based paerformance scores performance measures on DCAE, UMAP and SAUCIE
'''
import math
import pandas as pd
import numpy as np
import os
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores_normalized, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors


os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])

# Compute performance for DCAE
z_dir  = DATA_ROOT + "Artificial_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/DCAE_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    #read data
    print(output_dir)
    print(bl)
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res=np.load(z_dir + '/' + str(bl) + '_latent_rep_3D.npz')
    z= npz_res['z']

    discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

# Compute performance for UMAP
z_dir  = DATA_ROOT + "Artificial_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Artificial_sets/UMAP_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    # read data
    print(output_dir)
    print(bl)
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_2D.npz')
    z = npz_res['z']


    discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Artificial_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    # read data
    print(output_dir)
    print(bl)
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

    discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized.npz'
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
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized.npz'
        npz_res =  np.load(outfile)
        discontinuity = npz_res['discontinuity']
        manytoone = npz_res['manytoone']
        discontinuity =np.median(discontinuity)
        manytoone= np.median(manytoone)
        line = pd.DataFrame([[methods[i], str(bl), discontinuity, manytoone]],   columns =['method','branch','discontinuity','manytoone'])
        df=  df.append(line)


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('PS')

sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='branch', y='discontinuity', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Discontinuity_normalized.eps", format='eps', dpi = 350)
plt.close()

g = sns.barplot(x='branch', y='manytoone', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0.34, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Manytoone_normalized.eps", format='eps', dpi = 350)
plt.close()