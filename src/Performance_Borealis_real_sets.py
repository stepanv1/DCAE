'''
Compute emd-based performance measures for DCAE, UMAP and SAUCIE
Dimensionality Reduction has Quantifiable Imperfections: Two Geometric Bounds,
https://arxiv.org/abs/1811.00115
Discontinuity measure was modified to allow cross-method comparisons
'''
import pandas as pd
import numpy as np
import os
from utils_evaluation import  get_wsd_scores_normalized
from utils_evaluation import  plot2D_marker_colors, plot3D_marker_colors
from plotly.io import to_html

k = 30
epoch_list = [1000]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 1
alp = 0.2
m = 10
patience = 1000
min_delta = 1e-4
g=0
ID = 'DCAE_lam_1_batch_128_alp_0.5_m_10'

os.chdir('/media/grinek/Seagate/DCAE/')
DATA_ROOT = '/media/grinek/Seagate/'
DATA_DIR = DATA_ROOT + 'CyTOFdataPreprocess/'
source_dir = DATA_ROOT + 'Real_sets/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz', 'Samusik_01.npz']
z_dir = DATA_ROOT + "Real_sets/DCAE_output/"
output_dir = DATA_ROOT + "Real_sets/DCAE_output/Performance/"

for epochs in epoch_list:
    for bl in list_of_inputs:
        print(output_dir)
        print(bl)
        #read data
        infile = DATA_DIR  + bl
        npzfile = np.load(infile,  allow_pickle=True)
        aFrame = npzfile['aFrame'];
        Idx = npzfile['Idx'][:,:30]
        lbls = npzfile['lbls']

        # read DCAE output
        npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', allow_pickle=True)
        z = npz_res['z']

        discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 30, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)

        outfile = output_dir + '/'  + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_BOREALIS_PerformanceMeasures.npz'
        np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

    # Compute performance for UMAP
    z_dir  = DATA_ROOT + "Real_sets/UMAP_output/"
    output_dir =  DATA_ROOT + "Real_sets/UMAP_output/Performance"
    #bl = list_of_branches[1]
    for bl in list_of_inputs:
        print(output_dir)
        print(bl)
        #read data
        infile = DATA_DIR  + bl
        npzfile = np.load(infile,  allow_pickle=True)
        aFrame = npzfile['aFrame'];
        Idx = npzfile['Idx'][:,:30]
        lbls = npzfile['lbls']
    
        # read UMAP output
        npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_2D.npz',  allow_pickle=True)
        z = npz_res['z']
    
        discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 30, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)
    
        outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
        np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)
    
    # Compute performance for SAUCIE
    z_dir = DATA_ROOT + "Real_sets/SAUCIE_output/"
    output_dir =  DATA_ROOT + "Real_sets/SAUCIE_output/Performance"
    for bl in list_of_inputs:
        print(output_dir)
        print(bl)
        #read data
        infile = DATA_DIR  + bl
        npzfile = np.load(infile,  allow_pickle=True)
        aFrame = npzfile['aFrame'];
        Dist = npzfile['Dist']
        Idx = npzfile['Idx'][:,:30]
        neibALL = npzfile['neibALL']
        lbls = npzfile['lbls']
    
        # read DCAE output
        npz_res = np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz',  allow_pickle=True)
        z = npz_res['z']
    
        discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 30, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)
    
        outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
        np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

    #create Borealis graphs
    import seaborn as sns
    import matplotlib.pyplot as plt
    PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
    bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
    methods = ['DCAE', 'UMAP', 'SAUCIE']
    df = pd.DataFrame()

    for i in range(3):
        fig_disc = plt.figure()
        ax = fig_disc.add_subplot(111)
        for bl in list_of_inputs:
            if i == 0:
                outfile = DATA_ROOT + "Real_sets/DCAE_output/Performance/"   + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_BOREALIS_PerformanceMeasures.npz'
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
            npz_res =  np.load(outfile,  allow_pickle=True)
            discontinuity = npz_res['discontinuity']
            manytoone = npz_res['manytoone']
            discontinuity =np.median(discontinuity)
            manytoone= np.median(manytoone)
            line = pd.DataFrame([[methods[i], str(bl), discontinuity, manytoone]],   columns =['method','Set','discontinuity','manytoone'])
            df=  df.append(line)
            #plot discontinuity and manytone
            discontinuity = npz_res['discontinuity']
            manytoone = npz_res['manytoone']

            npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz',
                              allow_pickle=True)
            z = npz_res['z']

            infile = DATA_DIR + bl
            npzfile = np.load(infile, allow_pickle=True)
            lbls = npzfile['lbls'];

            data= np.column_stack((discontinuity, manytoone))
            if bl == list_of_inputs[2]:
                sub_s=z.shape[0]
            else:
                sub_s = 50000
            if i==0:
                fig = plot3D_marker_colors(z, data, (list(['discontinuity', 'manytoone'])), sub_s=sub_s, lbls=lbls, msize=1)
            else:
                fig = plot2D_marker_colors(z, data, (list(['discontinuity', 'manytoone'])), sub_s=sub_s, lbls=lbls, msize=1)
            html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                               include_mathjax=False, post_script=None, full_html=True,
                               animation_opts=None, default_width='100%', default_height='100%', validate=True)
            html_dir = output_dir
            Html_file = open(
                html_dir + "/" + str(methods[i]) + "_" + 'DiscontinuityManytoone' + "_" + str(bl) + '_epochs_' + str(epochs) + ".html", "w")
            Html_file.write(html_str)
            Html_file.close()

    #plot overlapping histograms per data set
    #Disconrtinuity
    mecol = {'DCAE': 'r',
             'UMAP': 'b', 'SAUCIE': 'y'}
    di = {'Levine32euclid_scaled_no_negative_removed.npz':'Levine32',
    'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz':'Pregnancy',  'Shenkareuclid_shifted.npz':'Shenkar', 'Samusik_01.npz': 'Samusik_01'}
    for bl in list_of_inputs:
        fig_disc = plt.figure()
        ax = fig_disc.add_subplot(111)
        for i in range(3):
            if i == 0:
                outfile = DATA_ROOT + "Real_sets/DCAE_output/Performance/" + ID + "_" + str(bl) + 'epochs' + str(
                    epochs) + '_BOREALIS_PerformanceMeasures.npz'
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
            npz_res = np.load(outfile, allow_pickle=True)
            discontinuity = npz_res['discontinuity']
            manytoone = npz_res['manytoone']
            discontinuity = np.median(discontinuity)
            manytoone = np.median(manytoone)
            line = pd.DataFrame([[methods[i], str(bl), discontinuity, manytoone]],
                                columns=['method', 'Set', 'discontinuity', 'manytoone'])
            df = df.append(line)
            # plot discontinuity and manytone
            discontinuity = npz_res['discontinuity']
            manytoone = npz_res['manytoone']

            npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz',
                              allow_pickle=True)
            z = npz_res['z']

            infile = DATA_DIR + bl
            npzfile = np.load(infile, allow_pickle=True)
            lbls = npzfile['lbls'];

            data = np.column_stack((discontinuity, manytoone))

            # plot overlapping histograms per data set

            ax.hist(np.log(data[:, 0]), ls='dashed', bins=50, alpha=0.3, lw=3, color=mecol[methods[i]])
        fig_disc.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        plt.title(di[bl])
        plt.show()

    # Manytoone
    mecol = {'DCAE': 'r',
             'UMAP': 'b', 'SAUCIE': 'y'}
    di = {'Levine32euclid_scaled_no_negative_removed.npz': 'Levine32',
          'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz': 'Pregnancy', 'Shenkareuclid_shifted.npz': 'Shenkar',
          'Samusik_01.npz': 'Samusik_01'}
    for bl in list_of_inputs:
        fig_disc = plt.figure()
        ax = fig_disc.add_subplot(111)
        for i in range(3):
            if i == 0:
                outfile = DATA_ROOT + "Real_sets/DCAE_output/Performance/" + ID + "_" + str(bl) + 'epochs' + str(
                    epochs) + '_BOREALIS_PerformanceMeasures.npz'
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
            npz_res = np.load(outfile, allow_pickle=True)
            discontinuity = npz_res['discontinuity']
            manytoone = npz_res['manytoone']
            discontinuity = np.median(discontinuity)
            manytoone = np.median(manytoone)
            line = pd.DataFrame([[methods[i], str(bl), discontinuity, manytoone]],
                                columns=['method', 'Set', 'discontinuity', 'manytoone'])
            df = df.append(line)
            # plot discontinuity and manytone
            discontinuity = npz_res['discontinuity']
            manytoone = npz_res['manytoone']

            npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz',
                              allow_pickle=True)
            z = npz_res['z']

            infile = DATA_DIR + bl
            npzfile = np.load(infile, allow_pickle=True)
            lbls = npzfile['lbls'];

            data = np.column_stack((discontinuity, manytoone))

            # plot overlapping histograms per data set

            ax.hist((data[:, 1]), ls='dashed', bins=100, alpha=0.3, lw=3, color=mecol[methods[i]])
        fig_disc.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        plt.title(di[bl])
        plt.show()

    #rename sets for plot

    di = {'Levine32euclid_scaled_no_negative_removed.npz':'Levine32',
    'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz':'Pregnancy',  'Shenkareuclid_shifted.npz':'Shenkar', 'Samusik_01.npz': 'Samusik_01'}
    df =  df.replace({"Set": di})
    import matplotlib
    matplotlib.use('PS')

    sns.set(rc={'figure.figsize':(14, 4)})
    g = sns.barplot(x='Set', y='discontinuity', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
    g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.savefig(PLOTS + "Discontinuity_SAMUSIK" +  ID + '_epochs' +str(epochs)+ ".eps", format='eps', dpi = 350)
    plt.close()

    g2 = sns.barplot(x='Set', y='manytoone', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
    g2.set(ylim=(0.05, None))
    g2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.savefig(PLOTS + "Manytoone_SAMUSIK" + ID + '_epochs' +str(epochs)+ ".eps", format='eps', dpi = 350)
    plt.close()


for epochs in epoch_list:

    PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
    bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",
                    DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
    methods = ['DCAE', 'UMAP', 'SAUCIE']
    df = pd.DataFrame()
    for i in range(3):
        for bl in list_of_inputs:
            if i == 0:
                outfile = bor_res_dirs[i] + '/' + ID + "_" + str(bl) + 'epochs' + str(
                    epochs) + '_BOREALIS_PerformanceMeasures.npz'
            else:
                outfile = bor_res_dirs[i] + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
            npz_res = np.load(outfile, allow_pickle=True)
            discontinuity = npz_res['discontinuity']
            manytoone = npz_res['manytoone']
            discontinuity = np.median(discontinuity)
            manytoone = np.median(manytoone)
            line = pd.DataFrame([[methods[i], str(bl), discontinuity, manytoone]],
                                columns=['method', 'Set', 'discontinuity', 'manytoone'])
            df = df.append(line)

    import seaborn as sns
    import matplotlib.pyplot as plt

    df = df.replace('Shenkar', 'Shekhar')
    # rename sets for plot

    di = {'Levine32euclid_scaled_no_negative_removed.npz': 'Levine32',
          'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz': 'Pregnancy', 'Shenkareuclid_shifted.npz': 'Shenkar', 'Samusik_01.npz': 'Samusik_01'}
    df = df.replace({"Set": di})
    import matplotlib

    matplotlib.use('PS')

    sns.set(rc={'figure.figsize': (14, 4)})
    g = sns.barplot(x='Set', y='discontinuity', hue='method', data=df.reset_index(),
                    palette=['tomato', 'yellow', 'limegreen'])
    g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.savefig(PLOTS + "Discontinuity_SAMUSIK" + ID + '_epochs' + str(epochs) + ".eps", format='eps', dpi=350)
    plt.close()

    g2 = sns.barplot(x='Set', y='manytoone', hue='method', data=df.reset_index(), palette=['tomato', 'yellow', 'limegreen'])
    g2.set(ylim=(0.05, None))
    g2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.savefig(PLOTS + "Manytoone_SAMUSIK" + ID + '_epochs' + str(epochs) + ".eps", format='eps', dpi=350)
    plt.close()

    # as tables

    # tables move to Borealis measures file
    df_BORAI  =df[df['Set']=='Levine32'][['method','manytoone','discontinuity']]
    df_BORAI.round(3).to_csv(PLOTS  + 'Levine32_'  +ID + '_' + 'epochs' + str(
                    epochs)+ 'Borealis_measures.csv', index=False)
    #
    df_BORAI  =df[df['Set']=='Pregnancy'][['method','manytoone','discontinuity']]
    df_BORAI.round(3).to_csv(PLOTS + 'Pregnancy_' +ID + '_' + 'epochs' + str(
                    epochs)+ 'Borealis_measures.csv', index=False)
    #
    df_BORAI  =df[df['Set']=='Shenkar'][['method','manytoone','discontinuity']]
    df_BORAI.round(3).to_csv(PLOTS + 'Shekhar_' +ID + '_' + 'epochs' + str(
                    epochs)+ 'Borealis_measures.csv', index=False)

    df_BORAI = df[df['Set'] == 'Samusik_01'][['method', 'manytoone', 'discontinuity']]
    df_BORAI.round(3).to_csv(PLOTS + 'Samusik_01_' + ID + '_' + 'epochs' + str(
        epochs) + 'Borealis_measures.csv', index=False)



