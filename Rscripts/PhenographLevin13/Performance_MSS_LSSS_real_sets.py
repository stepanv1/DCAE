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
#bl = list_of_inputs[0]
for bl in list_of_inputs:
    #read data
    infile = DATA_DIR  + bl
    npzfile = np.load(infile,  allow_pickle=True)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res=np.load(z_dir + '/' + str(bl) + 'epochs' +str(epochs) + '_latent_rep_3D.npz',  allow_pickle=True)
    z= npz_res['z']

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0= LSSS[0], MSS1=MSS[1], LSSS1= LSSS[1])

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

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

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

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

#create MSS_LSSS graphs######STPPED HERE
PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
i= 0
bl  =list_of_inputs[0]


k=30
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_inputs:
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'# STOPPED Here
        npz_res =  np.load(outfile,  allow_pickle=True)
        #MSS0 = npz_res['MSS0'][k]
        MSS1 = npz_res['MSS1']
        #LSSS0 = npz_res['LSSS0'][k]
        MSS0 = np.median(MSS1[k,:])
        LSSS1 = npz_res['LSSS1']
        LSSS0 = np.median(LSSS1[k, :])
        #discontinuity =np.median(discontinuity)
        #manytoone= np.median(manytoone)
        line = pd.DataFrame([[methods[i], str(bl), MSS0, LSSS0]],   columns =['method','Set','MSS','LSSS'])
        df =  df.append(line)




import seaborn as sns
import matplotlib.pyplot as plt

#rename sets for plot

di = {'Levine32euclid_scaled_no_negative_removed.npz':'Levine32',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz':'Pregnancy',  'Shenkareuclid_shifted.npz':'Shenkar'}
df =  df.replace({"Set": di})
import matplotlib
matplotlib.use('PS')

sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='Set', y='MSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0.25, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
g.figure.savefig(PLOTS +'k_'+str(k)+'_'+ "MSS.eps", format='eps', dpi = 350)
plt.close()

g2 = sns.barplot(x='Set', y='LSSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g2.set(ylim=(0, None))
g2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
g2.figure.savefig(PLOTS +'k_'+str(k)+'_'+ "LSSS.eps", format='eps', dpi = 350)
plt.close()

# plots at each k
PAPERPLOTS  = './PAPERPLOTS/'
bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']

df = pd.DataFrame()
bl = list_of_inputs[0]

for bl in list_of_inputs:
    measures = {key: [] for key in methods}
    for i in range(3):
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures.npz'  # STOPPED Here
        npz_res = np.load(outfile, allow_pickle=True)
        # MSS0 = npz_res['MSS0'][k]
        MSS1 = npz_res['MSS1']
        # LSSS0 = npz_res['LSSS0'][k]
        MSS0 = np.median(MSS1, axis=1)
        LSSS1 = npz_res['LSSS1']
        LSSS0 = np.median(LSSS1, axis=1)
        measures[methods[i]] = [MSS0, LSSS0]

    df_simMSS = pd.DataFrame({'k': range(0, 91)[1:], 'DCAE': measures['DCAE'][0][1:],
                           'SAUCIE':  measures['SAUCIE'][0][1:],
                           'UMAP':  measures['UMAP'][0][1:]})
    plt.plot('k', 'DCAE', data=df_simMSS, marker='o', markersize=5, color='skyblue', linewidth=3)
    plt.plot('k', 'SAUCIE', data=df_simMSS, marker='v', color='orange', linewidth=2)
    plt.plot('k', 'UMAP', data=df_simMSS, marker='x', color='olive', linewidth=2)
    plt.legend()
    plt.savefig(PLOTS + bl + 'performance_marker_similarity_score.png')
    plt.show()
    plt.clf()

    df_simMSS = pd.DataFrame({'k': range(0, 91)[1:], 'DCAE': measures['DCAE'][1][1:],
                              'SAUCIE': measures['SAUCIE'][1][1:],
                              'UMAP': measures['UMAP'][1][1:]})
    plt.plot('k', 'DCAE', data=df_simMSS, marker='o', markersize=5, color='skyblue', linewidth=3)
    plt.plot('k', 'SAUCIE', data=df_simMSS, marker='v', color='orange', linewidth=2)
    plt.plot('k', 'UMAP', data=df_simMSS, marker='x', color='olive', linewidth=2)
    plt.legend()
    plt.savefig(PLOTS + bl + 'performance_marker_onetomany_score.png')
    plt.show()
    plt.clf()



# tables move to Borealis measures file
df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.1160, 0.1667, 0.1321], 'discontinuity': [0.0414, 0.0113, 0.0052]})
df_BORAI.to_csv(PAPERPLOTS  + 'Pregnancy_' + 'Borealis_measures.csv', index=False)









