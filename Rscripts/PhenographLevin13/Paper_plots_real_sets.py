'''
create 2D projections of DCAE, UMAP and SAUCIE outputs for the manuscript
'''
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import os
from utils_evaluation import get_wsd_scores_normalized

k = 30
epoch_list = [250, 500, 1000]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 1
alp = 0.2
m = 10
patience = 1000
min_delta = 1e-4
g = 0.1

ID = 'clip_grad_exp_MDS' + '_g_' + str(g) + '_lam_' + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(
    alp) + '_m_' + str(m)

# epoch_list =  [750]
os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
DATA_DIR = DATA_ROOT + 'CyTOFdataPreprocess/'
source_dir = DATA_ROOT + 'Real_sets/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
                  'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz', 'Shenkareuclid_shifted.npz']
z_dir = DATA_ROOT + "Real_sets/DCAE_output/"
output_dir = DATA_ROOT + "Real_sets/DCAE_output/Performance/"

bl_index  = [0,1,2]
camera_positions = [[0,0,0], [0,0,0], [0,0,0]]
epochs = 1000

idx = bl_index[0]
unassigned_lbls = ['"unassigned"', '"Unassgined"', '-1']
for idx in bl_index:
    print(output_dir)
    bl = list_of_inputs [idx]
    print(bl)
    # read data
    infile = DATA_DIR + bl
    npzfile = np.load(infile, allow_pickle=True)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx'][:, :30]
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', allow_pickle=True)
    z = npz_res['z']
    lb = lbls[lbls!=unassigned_lbls[0]]
    z = z[lbls!=unassigned_lbls[0],:]

    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns

    fig = plt.figure()
    ax = Axes3D(fig)
    cl=np.unique(lb)
    sns.reset_orig()
    colors = sns.color_palette("husl", n_colors=len(cl))
    groups = []
    for i in range(len(cl)):
        groups.append(ax.scatter(xs=z[:,0][lb==cl[i]], ys=z[:,1][lb==cl[i]], zs=z[:,2][lb==cl[i]], c = colors[i],  s=0.1))
        #ax.legend()
    ax.view_init(elev=100., azim=5)
    #ax.legend(groups, cl, loc=4)
    fig.subplots_adjust(right=0.8)
    ax.legend(groups, cl,loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=14,  markerscale=14)
    fig.show()

    print('ax.azim {}'.format(ax.azim))
    print('ax.elev {}'.format(ax.elev))


    outfile = output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(
        epochs) + '_BOREALIS_PerformanceMeasures_wider.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity=discontinuity)
'''
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
#bl = list_of_branches[1]
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
'''
# create Borealis graphs
PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",
                DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
# dir = bor_res_dirs[0]
# bl  = list_of_inputs[0]
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_inputs:
        if i == 0:
            outfile = output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(
                epochs) + '_BOREALIS_PerformanceMeasures_wider.npz'
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

# rename sets for plot

di = {'Levine32euclid_scaled_no_negative_removed.npz': 'Levine32',
      'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz': 'Pregnancy', 'Shenkareuclid_shifted.npz': 'Shenkar'}
df = df.replace({"Set": di})
import matplotlib

matplotlib.use('PS')

sns.set(rc={'figure.figsize': (14, 4)})
g = sns.barplot(x='Set', y='discontinuity', hue='method', data=df.reset_index(),
                palette=['tomato', 'yellow', 'limegreen'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Discontinuity" + ID + '_epochs' + str(epochs) + "_wider.eps", format='eps', dpi=350)
plt.close()

g2 = sns.barplot(x='Set', y='manytoone', hue='method', data=df.reset_index(),
                 palette=['tomato', 'yellow', 'limegreen'])
g2.set(ylim=(0.05, None))
g2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "Manytoone" + ID + '_epochs' + str(epochs) + "_wider.eps", format='eps', dpi=350)
plt.close()

# as tables

# tables move to Borealis measures file
# df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.118989,  0.158191, 0.132919], 'discontinuity': [2.288722, 5.971887, 5.429483]})
# df_BORAI.to_csv(PLOTS  + 'Levine32_'  +ID + '_' + 'Borealis_measures_wider.csv', index=False)
#
# df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.184362, 0.209887, 0.201161], 'discontinuity': [12.233907, 16.489910, 17.582431 ]})
# df_BORAI.to_csv(PLOTS + 'Pregnancy_' +ID + '_' + 'Borealis_measures_wider.csv', index=False)
#
# df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.338864, 0.348597, 0.337997], 'discontinuity': [3.267888, 3.548169, 5.955621]})
# df_BORAI.to_csv(PLOTS  + 'Shenkar_' +ID + '_' + 'Borealis_measures_wider.csv', index=False)



