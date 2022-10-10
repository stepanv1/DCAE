'''
Compute emd-based paerformance scores performance measures for DCAE, UMAP and SAUCIE
Dimensionality Reduction has Quantifiable Imperfections: Two Geometric Bounds,
https://arxiv.org/abs/1811.00115
'''
import pandas as pd
import numpy as np
import os
from utils_evaluation import get_wsd_scores_normalized

os.chdir('/media/grinek/Seagate/DCAE/')
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])

#parameters of run
k = 30
epochs_list = [500]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 0.1
alp = 0.5
m = 10
patience = 500
min_delta = 1e-4

ID = 'DCAE_lam_0.1_batch_128_alp_0.5_m_10'

epochs = 500
# Compute performance for DCAE
z_dir  = DATA_ROOT + "Artificial_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/DCAE_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    print('bl =', bl)
    # read data
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/'  + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz')
    z = npz_res['z']

    discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 30, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)

    outfile = output_dir + '/'  + ID + "_"  + str(bl) + '_BOREALIS_PerformanceMeasures_normalized.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)


# Compute performance for UMAP
z_dir  = DATA_ROOT + "Artificial_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Artificial_sets/UMAP_output/Performance/"
for bl in list_of_branches:
    # read data
    print(output_dir)
    print(bl)
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx'][:,:30]
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_3D.npz')
    z = npz_res['z']


    discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 30, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized_3D.npz'
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
    Idx = npzfile['Idx'][:,:30]
    neibALL = npzfile['neibALL']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_3D.npz')
    z = npz_res['z']
    discontinuity, manytoone = get_wsd_scores_normalized(aFrame, z, 30, num_meandist=10000, compute_knn_x=False, x_knn=Idx, nc=16)

    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized_3D.npz'
    np.savez(outfile, manytoone=manytoone, discontinuity= discontinuity)

#create Borealis graphs
PLOTS = DATA_ROOT + "Artificial_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Artificial_sets/DCAE_output/Performance/", DATA_ROOT + "Artificial_sets/UMAP_output/Performance/",DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
df = pd.DataFrame()

for i in range(3):
    for bl in list_of_branches:
        if bor_res_dirs[i] != bor_res_dirs[0]:
            outfile = bor_res_dirs[i] + '/' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized_3D.npz'
        else:
            outfile = bor_res_dirs[i] + ID + '_' + str(bl) + '_BOREALIS_PerformanceMeasures_normalized.npz'

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
plt.savefig(PLOTS +   ID + "_" +  'epochs' + str(epochs)+ "Discontinuity_3D.eps", format='eps', dpi = 350)
plt.close()

g = sns.barplot(x='branch', y='manytoone', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0.34, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +   ID + "_" + 'epochs' + str(epochs)+  "Manytoone_3D.eps", format='eps', dpi = 350)
plt.close()








