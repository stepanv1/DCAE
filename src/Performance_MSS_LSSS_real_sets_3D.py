'''
Compute MSS and LSSS performance measures on DCAE, UMAP and SAUCIE 3D
using normalized measures
Real sets
'''
import pandas as pd
import numpy as np
import os
from utils_evaluation import neighbour_onetomany_score_normalized, \
    neighbour_marker_similarity_score_per_cell

k = 30
epochs_list = [1000]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 1
alp = 0.5
m = 10
patience = 1000
min_delta = 1e-4

ID = 'DCAE' + '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)

#epoch_list =  [750]
os.chdir('/media/grinek/Seagate/DCAE/')
DATA_ROOT = '/media/grinek/Seagate/'
DATA_DIR = DATA_ROOT + 'CyTOFdataPreprocess/'
source_dir = DATA_ROOT + 'Real_sets/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz', 'Samusik_01.npz']
z_dir  = DATA_ROOT + "Real_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Real_sets/DCAE_output/Performance/"

for epochs in epochs_list:
    for bl in list_of_inputs:
        #read data
        infile = DATA_DIR  + bl
        npzfile = np.load(infile,  allow_pickle=True)
        aFrame = npzfile['aFrame'];
        Idx = npzfile['Idx']
        lbls = npzfile['lbls']
        
        # read DCAE output
        npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz',
                          allow_pickle=True)
        z= npz_res['z']

        MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
        LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

        outfile = output_dir + '/'  + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_MSS_LSSS_PerformanceMeasures_normalized_3D.npz'
        np.savez(outfile, MSS0=MSS[0], LSSS0= LSSS[0], MSS1=MSS[1], LSSS1= LSSS[1])
    
    # Compute performance for UMAP
    z_dir  = DATA_ROOT + "Real_sets/UMAP_output/"
    output_dir =  DATA_ROOT + "Real_sets/UMAP_output/Performance"
    for bl in list_of_inputs:
        #read data
        infile = DATA_DIR  + bl
        npzfile = np.load(infile,  allow_pickle=True)
        aFrame = npzfile['aFrame'];
        Idx = npzfile['Idx']
        lbls = npzfile['lbls']

        # read UMAP output
        npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_3D.npz',  allow_pickle=True)
        z = npz_res['z']

        MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
        LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

        outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized_3D.npz'
        np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

    # Compute performance for SAUCIE
    z_dir = DATA_ROOT + "Real_sets/SAUCIE_output/"
    output_dir =  DATA_ROOT + "Real_sets/SAUCIE_output/Performance"
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
        npz_res = np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_3D.npz',  allow_pickle=True)
        z = npz_res['z']

        MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
        LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

        outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized_3D.npz'
        np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

    #created MSS_LSSS graphs######
    PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
    bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
    methods = ['DCAE', 'UMAP', 'SAUCIE']
    k=30
    df = pd.DataFrame()
    for i in range(3):
        for bl in list_of_inputs:
            if i == 0:
                outfile = bor_res_dirs[i] + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_MSS_LSSS_PerformanceMeasures_normalized_3D.npz'  # STOPPED Here
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized_3D.npz'
            npz_res =  np.load(outfile,  allow_pickle=True)
            MSS1 = npz_res['MSS1']
            MSS0 = np.median(MSS1[k,:])
            LSSS1 = npz_res['LSSS1']
            LSSS0 = np.median(LSSS1[k, :])

            line = pd.DataFrame([[methods[i], str(bl), MSS0, LSSS0]],   columns =['method','Set','MSS','LSSS'])
            df =  df.append(line)

    import seaborn as sns
    import matplotlib.pyplot as plt

    #rename sets for plot
    di = {'Levine32euclid_scaled_no_negative_removed.npz':'Levine32',
    'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz':'Pregnancy',  'Shenkareuclid_shifted.npz':'Shenkar', 'Samusik_01.npz':'Samusik_01'}
    df =  df.replace({"Set": di})
    import matplotlib
    matplotlib.use('PS')

    sns.set(rc={'figure.figsize':(14, 4)})
    g = sns.barplot(x='Set', y='MSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
    g.set(ylim=(0.25, None))
    g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    g.figure.savefig(PLOTS + ID + "_" + 'k_'+str(k)+ '_epochs' +str(epochs) +'_'+ "MSS_normalized_3D.eps", format='eps', dpi = 350,
                      bbox_inches="tight")
    plt.close()

    g2 = sns.barplot(x='Set', y='LSSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
    g2.set(ylim=(0, None))
    g2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    g2.figure.savefig(PLOTS + ID + "_" + 'k_'+str(k)+'_epochs' +str(epochs)+'_'+ "LSSS_normalized_3D.eps", format='eps', dpi = 350,
                      bbox_inches="tight")
    plt.close()

    # plots at each k
    PAPERPLOTS  = './PAPERPLOTS/'
    bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
    list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
    'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz', 'Samusik_01.npz']
    names = ['Levine32','Pregnancy', 'Shekhar', 'Samusik_01']
    df = pd.DataFrame()
    bl = list_of_inputs[0]

    plt.rcParams["figure.figsize"] = (10,3)

    k_start = 9
    for n_set in range(3):
        bl = list_of_inputs[n_set]
        measures = {key: [] for key in methods}
        for i in range(3):
            if i==0:
                outfile = bor_res_dirs[i] + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_MSS_LSSS_PerformanceMeasures_normalized_3D.npz'
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized_3D.npz'
            npz_res = np.load(outfile, allow_pickle=True)
            MSS1 = npz_res['MSS1']
            MSS0 = np.median(MSS1, axis=1)
            LSSS1 = npz_res['LSSS1']
            LSSS0 = np.median(LSSS1, axis=1)
            measures[methods[i]] = [MSS0, LSSS0]

        df_simMSS = pd.DataFrame({'k': range(k_start  , 90)[:], 'DCAE': measures['DCAE'][0][k_start :],
                               'SAUCIE':  measures['SAUCIE'][0][k_start :],
                               'UMAP':  measures['UMAP'][0][k_start :]})
        plt.plot('k', 'DCAE', data=df_simMSS, marker='o', markersize=5, color='skyblue', linewidth=3)
        plt.plot('k', 'SAUCIE', data=df_simMSS, marker='v', color='orange', linewidth=2)
        plt.plot('k', 'UMAP', data=df_simMSS, marker='x', color='olive', linewidth=2)
        plt.legend()
        plt.savefig(PLOTS + ID + "_" + names[n_set ] +'_epochs' +str(epochs) + '_' + 'MSS_3D.eps', format='eps', dpi = 350)
        plt.show()
        plt.clf()

        df_simMSS = pd.DataFrame({'k': range(k_start, 90)[:], 'DCAE': measures['DCAE'][1][k_start:],
                                  'SAUCIE': measures['SAUCIE'][1][k_start:],
                                  'UMAP': measures['UMAP'][1][k_start:]})
        plt.plot('k', 'DCAE', data=df_simMSS, marker='o', markersize=5, color='skyblue', linewidth=3)
        plt.plot('k', 'SAUCIE', data=df_simMSS, marker='v', color='orange', linewidth=2)
        plt.plot('k', 'UMAP', data=df_simMSS, marker='x', color='olive', linewidth=2)
        plt.legend()
        plt.savefig(PLOTS + ID + "_" + names[n_set ] + '_epochs' +str(epochs) + '_' + 'LSSS_3D.eps', format='eps', dpi = 350)
        plt.show()
        plt.clf()