'''
Compute MSS and LSS performance measures on DCAE, UMAP and
'''
import math
import pandas as pd
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
    np.savez(outfile, MSS0=MSS[0], LSSS0= LSSS[0], MSS1=MSS[1], LSSS1= LSSS[1])

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
    npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_2D.npz')
    z = npz_res['z']
    #divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr= (np.max(z[:,0])-np.min(z[:,0]))*(np.max(z[:,1])-np.min(z[:,1]))
    z=z / S_pr * 4 * math.pi

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

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
    npz_res = np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz')
    z = npz_res['z']
    # divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr = (np.max(z[:, 0]) - np.min(z[:, 0])) * (np.max(z[:, 1]) - np.min(z[:, 1]))
    z = z / S_pr * 4 * math.pi

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

#create MSS_LSSS graphs
PLOTS = DATA_ROOT + "Artificial_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Artificial_sets/DCAE_output/Performance/", DATA_ROOT + "Artificial_sets/UMAP_output/Performance/",DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
dir = bor_res_dirs[0]
bl  = list_of_branches[0]

k=30
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_branches:
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
        npz_res =  np.load(outfile)
        #MSS0 = npz_res['MSS0'][k]
        MSS1 = npz_res['MSS1']
        #LSSS0 = npz_res['LSSS0'][k]
        MSS0 = np.median(MSS1[k,:])
        LSSS1 = npz_res['LSSS1']
        LSSS0 = np.median(LSSS1[k, :])
        #discontinuity =np.median(discontinuity)
        #manytoone= np.median(manytoone)
        line = pd.DataFrame([[methods[i], str(bl), MSS0, LSSS0]],   columns =['method','branch','MSS','LSSS'])
        df=  df.append(line)


import seaborn as sns
import matplotlib.pyplot as plt



sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='branch', y='MSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "MSS.png")

g = sns.barplot(x='branch', y='LSSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "LSSS.png")