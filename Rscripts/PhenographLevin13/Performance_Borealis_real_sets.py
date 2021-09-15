'''
Compute emd performance measures on DCAE, UMAP and SAUCIE
'''
import math
import pandas as pd
import numpy as np
import os
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors

epochs = 500

os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
DATA_DIR = DATA_ROOT + 'CyTOFdataPreprocess/'
source_dir = DATA_ROOT + 'Real_sets/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']

# Compute performance for DCAE
z_dir  = DATA_ROOT + "Real_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Real_sets/DCAE_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_inputs:
    #read data
    infile = DATA_DIR  + bl
    npzfile = np.load(infile,  allow_pickle=True)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/' + str(bl) + 'epochs' + str(epochs) + '_latent_rep_wider_3D.npz', allow_pickle=True)
    z = npz_res['z']

    discontinuity, manytoone = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_wider.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

# Compute performance for UMAP
z_dir  = DATA_ROOT + "Real_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Real_sets/UMAP_output/Performance"
#bl = list_of_branches[1]
for bl in list_of_inputs:
    #read data
    infile = DATA_DIR  + bl
    npzfile = np.load(infile,  allow_pickle=True)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read UMAP output
    npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_2D.npz',  allow_pickle=True)
    z = npz_res['z']
    #divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr= (np.max(z[:,0])-np.min(z[:,0]))*(np.max(z[:,1])-np.min(z[:,1]))
    z=z / S_pr * 4 * math.pi

    discontinuity, manytoone = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Real_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Real_sets/SAUCIE_output/Performance"
#bl = list_of_branches[1]
for bl in list_of_inputs:
    #read data
    infile = DATA_DIR  + bl
    npzfile = np.load(infile,  allow_pickle=True)
    aFrame = npzfile['aFrame'];
    Dist = npzfile['Dist']
    Idx = npzfile['Idx']
    neibALL = npzfile['neibALL']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz',  allow_pickle=True)
    z = npz_res['z']
    # divide by max_r and multiply by 4 pi to level field with DCAE
    S_pr = (np.max(z[:, 0]) - np.min(z[:, 0])) * (np.max(z[:, 1]) - np.min(z[:, 1]))
    z = z / S_pr * 4 * math.pi

    discontinuity, manytoone = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

#create Borealis graphs
PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
dir = bor_res_dirs[0]
#bl  = list_of_branches[0]
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_inputs:
        if i == 0:
            outfile = bor_res_dirs[i] + '/' + str(
                bl) + '_BOREALIS_PerformanceMeasures_wider_.npz'  # STOPPED Here
        else:
            outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
        npz_res =  np.load(outfile,  allow_pickle=True)
        discontinuity = npz_res['discontinuity']
        manytoone = npz_res['manytoone']
        discontinuity =np.median(discontinuity)
        manytoone= np.median(manytoone)
        line = pd.DataFrame([[methods[i], str(bl), discontinuity, manytoone]],   columns =['method','Set','discontinuity','manytoone'])
        df=  df.append(line)


import seaborn as sns
import matplotlib.pyplot as plt

#rename sets for plot

di = {'Levine32euclid_scaled_no_negative_removed.npz':'Levine32',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz':'Pregnancy',  'Shenkareuclid_shifted.npz':'Shenkar'}
df =  df.replace({"Set": di})
import matplotlib
matplotlib.use('PS')

sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='Set', y='discontinuity', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Discontinuity_wider.eps", format='eps', dpi = 350)
plt.close()

g2 = sns.barplot(x='Set', y='manytoone', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g2.set(ylim=(0.05, None))
g2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Manytoone_wider.eps", format='eps', dpi = 350)
plt.close()

# as tables

# tables move to Borealis measures file
df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.105856, 0.141267, 0.118188], 'discontinuity': [0.002820, 0.005547, 0.000888]})
df_BORAI.to_csv(PLOTS  + 'Levine32_' + 'Borealis_measures_wider.csv', index=False)

df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.165165, 0.188827, 0.183314], 'discontinuity': [0.093898, 0.012045, 0.005639]})
df_BORAI.to_csv(PLOTS + 'Pregnancy_' + 'Borealis_measures_wider.csv', index=False)



df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.070870, 0.003969, 0.009664 ], 'discontinuity': [0.307553, 0.307055, 0.315236]})
df_BORAI.to_csv(PLOTS  + 'Shenkar_' + 'Borealis_measures_wider.csv', index=False)



