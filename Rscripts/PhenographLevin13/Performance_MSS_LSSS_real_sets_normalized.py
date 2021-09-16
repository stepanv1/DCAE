'''
Compute MSS and LSS performance measures on DCAE, UMAP and SAUCIE
using normalized measures
'''
import math
import pandas as pd
import numpy as np
import os
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score_normalized, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors

epoch_list =  [50, 100, 200, 300, 400, 500, 1000]
#epoch_list =  [1000]
for epochs in epoch_list:
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
        npz_res=np.load(z_dir + '/' + str(bl) + 'epochs' +str(epochs) + '_latent_rep_wider_3D.npz',  allow_pickle=True)
        z= npz_res['z']

        MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
        LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

        outfile = output_dir + '/' + str(bl) + 'epochs' + str(epochs) + '_MSS_LSSS_PerformanceMeasures_normalized_wider.npz'
        np.savez(outfile, MSS0=MSS[0], LSSS0= LSSS[0], MSS1=MSS[1], LSSS1= LSSS[1])
    '''
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

        MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
        LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

        outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
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

        MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
        LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

        outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
        np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])
    '''
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
            if i == 0:
                outfile = bor_res_dirs[i] + '/' + str(
                    bl) + 'epochs' +str(epochs)+ '_MSS_LSSS_PerformanceMeasures_normalized_wider.npz'  # STOPPED Here
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
            npz_res =  np.load(outfile,  allow_pickle=True)
            #MSS0 = npz_res['MSS0'][k]
            MSS1 = npz_res['MSS1']
            #LSSS0 = npz_res['LSSS0'][k]
            MSS0 = np.median(MSS1[k,:])
            LSSS1 = npz_res['LSSS1']
            LSSS0 = np.median(LSSS1[k, :])

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
    g.figure.savefig(PLOTS +'k_'+str(k)+ '_epochs' +str(epochs) +'_'+ "MSS_normalized_wider.eps", format='eps', dpi = 350)
    plt.close()
    # the bug is herew

    g2 = sns.barplot(x='Set', y='LSSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
    g2.set(ylim=(0, None))
    g2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    g2.figure.savefig(PLOTS +'k_'+str(k)+'_epochs' +str(epochs)+'_'+ "LSSS_normalized_wider.eps", format='eps', dpi = 350)
    plt.close()

    # plots at each k
    PAPERPLOTS  = './PAPERPLOTS/'
    bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
    list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
    'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']
    names = ['Levine32','Pregnancy', 'Shekhar']
    df = pd.DataFrame()
    bl = list_of_inputs[0]

    plt.rcParams["figure.figsize"] = (3,1)

    k_start = 9
    for n_set in range(3):
        bl = list_of_inputs[n_set]
        measures = {key: [] for key in methods}
        for i in range(3):
            if i==0:
                outfile = bor_res_dirs[i] + '/' + str(bl) +'epochs' +str(epochs)+ '_MSS_LSSS_PerformanceMeasures_normalized_wider.npz'  # STOPPED Here
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
            npz_res = np.load(outfile, allow_pickle=True)
            # MSS0 = npz_res['MSS0'][k]
            MSS1 = npz_res['MSS1']
            # LSSS0 = npz_res['LSSS0'][k]
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
        plt.savefig(PLOTS + names[n_set ] +'_epochs' +str(epochs) + '_' + 'MSS_wider.eps', format='eps', dpi = 350)
        plt.show()
        plt.clf()

        df_simMSS = pd.DataFrame({'k': range(k_start, 90)[:], 'DCAE': measures['DCAE'][1][k_start:],
                                  'SAUCIE': measures['SAUCIE'][1][k_start:],
                                  'UMAP': measures['UMAP'][1][k_start:]})
        plt.plot('k', 'DCAE', data=df_simMSS, marker='o', markersize=5, color='skyblue', linewidth=3)
        plt.plot('k', 'SAUCIE', data=df_simMSS, marker='v', color='orange', linewidth=2)
        plt.plot('k', 'UMAP', data=df_simMSS, marker='x', color='olive', linewidth=2)
        plt.legend()
        plt.savefig(PLOTS + names[n_set ] + '_epochs' +str(epochs) + '_' + 'LSSS_wider.eps', format='eps', dpi = 350)
        plt.show()
        plt.clf()


