'''
Compute MSS and LSS performance measures on DCAE, UMAP and Compute MSS and LSS performance measures on DCAE, UMAP and SAUCIE
for artificial clusters
tied weight implementation is from here
https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6Ijk5MWIwNjM2YWFkYTM0MWM1YTA4ZTBkOGYyNDA2OTcyMDY0ZGM4ZWQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MzExODM4MTksImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExNjI5OTE0NzA0NDIzMzQ5MDIxNSIsImVtYWlsIjoic3RlcGFudjFAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF6cCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsIm5hbWUiOiJTdGVwYW4gR3JpbnlvayIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQVRYQUp6S0FfcHpVMXJDR1NmWW1YYWNXZTZJbkhxWllMTl9NeTdqcDhVPXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6IlN0ZXBhbiIsImZhbWlseV9uYW1lIjoiR3JpbnlvayIsImlhdCI6MTYzMTE4NDExOSwiZXhwIjoxNjMxMTg3NzE5LCJqdGkiOiIzNTEwNmU4ZjhkMTIyNDQ4MzEzYjRlMGFkYTVkOWFiZjFlYTE2Nzg1In0.m8geKRaotPU7k0WjujEuY3BS97Z1v6RU7K_0vu8zxLLyWHoM9_XbeRauY_0ArXk9xmrHG47Dp3AYT9swzIG9-NL4Aqvs2-AYVloHOmaG1VfBX5slI3XyHv0gg80f4XvlCpzNJDwFmOSUaonB4l164_dCprVk4B2A-5x_I5mUdUQcB8bslUi_cIxewf4FUzKmbvkiVzA-HetHexiUiTgZrEwQCaO24Q6dbmnXtzc7cdV3lwxoFCJjs95mXiJGAPZFwnx4WgjNwQwbEfvcNsgr-1W1RvI80AtfasAk2eHtNLMSvPznsJB73hI73xBCusxjSPOJFuEWd2_hz_iJUVwlJg
argument for transposed instead true inverse: https://stats.stackexchange.com/questions/489429/why-are-the-tied-weights-in-autoencoders-transposed-and-not
'''
import math
import pandas as pd
import numpy as np
import os
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score_normalized, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors


os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])

# Compute performance for DCAE
z_dir  = DATA_ROOT + "Experiments/Artificial_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Experiments/Artificial_sets/DCAE_output/Performance/"
#bl = list_of_branches[0]
for bl in list_of_branches:
    print(z_dir)
    print('bl=',bl)
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
    LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0= LSSS[0], MSS1=MSS[1], LSSS1= LSSS[1])

# Compute performance for UMAP
z_dir  = DATA_ROOT + "Experiments/Artificial_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Experiments/Artificial_sets/UMAP_output/Performance"
#bl = list_of_branches[1]
for bl in list_of_branches:
    print(z_dir)
    print('bl=', bl)
    # read data
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']

    # read DCAE output
    npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_2D.npz')
    z = npz_res['z']

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Experiments/Artificial_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Experiments/Artificial_sets/SAUCIE_output/Performance"
#bl = list_of_branches[1]
for bl in list_of_branches:
    print(z_dir)
    print('bl=', bl)
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

    MSS = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=16)
    LSSS = neighbour_onetomany_score_normalized(z, Idx, kmax=90, num_cores=16)

    outfile = output_dir + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
    np.savez(outfile, MSS0=MSS[0], LSSS0=LSSS[0], MSS1=MSS[1], LSSS1=LSSS[1])

#create MSS_LSSS graphs
PLOTS = DATA_ROOT + "Experiments/Artificial_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Experiments/Artificial_sets/DCAE_output/Performance/",
                DATA_ROOT + "Experiments/Artificial_sets/UMAP_output/Performance/",
                DATA_ROOT + "Experiments/Artificial_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
dir = bor_res_dirs[0]
bl  = list_of_branches[0]

k=30
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_branches:
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
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
import matplotlib
matplotlib.use('PS')


sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='branch', y='MSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "MSS_normalized.eps", format='eps', dpi = 350)
plt.close()


g = sns.barplot(x='branch', y='LSSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "LSSS_normalized.eps", format='eps', dpi = 350)
plt.close()



k=20
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_branches:
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
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

sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='branch', y='MSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "MSS_k_20_normalized.eps", format='eps', dpi = 350)
plt.close()


g = sns.barplot(x='branch', y='LSSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "LSSS_k_20_normalized.eps", format='eps', dpi = 350)
plt.close()


k=60
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_branches:
        outfile = bor_res_dirs[i] + '/' + str(bl) + '_MSS_LSSS_PerformanceMeasures_normalized.npz'
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

sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='branch', y='MSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "MSS_k_60_normalized.eps", format='eps', dpi = 350)
plt.close()


g = sns.barplot(x='branch', y='LSSS', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.set(ylim=(0, None))
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS +'k_'+str(k)+'_'+ "LSSS_k_60_normalized.eps", format='eps', dpi = 350)
plt.close()
