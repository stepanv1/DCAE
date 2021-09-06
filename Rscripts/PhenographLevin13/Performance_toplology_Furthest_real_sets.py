'''
Compute topological perormance measures in real sets on DCAE, UMAP and SAUCIE, grounds truth is computed as farthest
cluster
'''
import math
import pandas as pd
import numpy as np
import os
import random
from scipy.spatial import distance
from utils_evaluation import table

#number of nearest neighbour clustersa

def decision(probability):
    return random.random() < probability
ns_sample = 5000 #size of cluster subsample for distance calculation

def get_real_topology_list_f(lbls, aFrame):
    """ Gets a labels and coordinates in HD and creates a list of nearest neighbour clusters"""
    # k_nn_cl - parameter setting how many nearest neghbout clusters we are tracking
    # exclude ungated
    excl = lbls != -1
    lbls = lbls[excl]
    aFrame = aFrame[excl]

    l_list = np.unique(lbls)
    size_list = [sum(lbls == l) for l in l_list]
    zip_iterator = zip(l_list, size_list)
    size_dict = dict(zip_iterator)

    # indx = random.sample(range(len(lbls)), 10000)  # TODO: create samling which takes small clusters without subsampling
    indx = [True if size_dict[l] <= ns_sample else decision(ns_sample / size_dict[l]) for l in lbls]
    lbls_s = lbls[indx]
    a_s = aFrame[indx, :]
    topolist_estimateHD = [[] for _ in range(len(l_list))]
    #ncl = len(l_list)
    #dist_estimateHD = [[] for _ in range(len(l_list))] # unassigned cluster is labelled at -1 and we do not compute
    # nearest neigbours for it
    for i in range(len(l_list)):
        print(i)
        dist = [np.mean(distance.cdist(a_s[lbls_s == i, :], a_s[lbls_s == label, :], metric='minkowski', p=1)) for label
                in l_list]
        #dist_estimateHD[i] = dist
        # dist = [np.sqrt(np.sum((z_s[lbls_s == i, :].mean(0) - z_s[lbls_s == label, :].mean(0))**2)) for label in l_list]
        # get indexes of  closest clusters, and exclude itself, exclude i th cluster
        seq = sorted(dist)
        rank = [seq.index(v) for v in dist]
        # get top 4
        rank2 = np.array(rank)[np.array(l_list) != np.array(i)]
        l_list2 = np.array(l_list)[np.array(l_list) != np.array(i)]
        nn_list = l_list2[rank2.argsort()][-5:-1]
        topolist_estimateHD[i] = nn_list
    return topolist_estimateHD


def get_representation_topology_f(z, lbls):
    """ Gets a labels and coordinates in LD and creates a list of nearest neighbour clusters"""
    # k_nn_cl - parameter setting how many nearest neghbout clusters we are tracking
    excl = lbls != -1
    lbls = lbls[excl]
    z = z[excl]

    l_list = np.unique(lbls)
    size_list = [sum(lbls == l) for l in l_list]
    zip_iterator = zip(l_list, size_list)
    size_dict = dict(zip_iterator)

    # indx = random.sample(range(len(lbls)), 10000)  # TODO: create samling which takes small clusters without subsampling
    indx = [True if size_dict[l] <= ns_sample else decision(ns_sample / size_dict[l]) for l in lbls]
    lbls_s = lbls[indx]
    z_s = z[indx, :]
    topolist_estimate = [[] for _ in range(
        len(l_list))]  # unassigned cluster is labelled at -1 and we do not compute  # unassigned cluster is labelled at -1 and we do not compute
    # nearest neigbours for it
    for i in range(len(l_list)):
        print(i)
        dist = [np.mean(distance.cdist(z_s[lbls_s == i, :], z_s[lbls_s == label, :])) for label in l_list]
        # dist = [np.sqrt(np.sum((z_s[lbls_s == i, :].mean(0) - z_s[lbls_s == label, :].mean(0))**2)) for label in l_list]
        # get indexes of  closest clusters, and exclude itself, exclude i th cluster
        seq = sorted(dist)
        rank = [seq.index(v) for v in dist]
        # get top 4
        rank2 = np.array(rank)[np.array(l_list) != np.array(i)]
        l_list2 = np.array(l_list)[np.array(l_list) != np.array(i)]
        nn_list = l_list2[rank2.argsort()][-5:-1]
        topolist_estimate[i] = nn_list
    return topolist_estimate


def get_topology_match_score(topolist, topolist_estimate):
    """compare match computed and prescribed"""
    topolist = [np.array(x) for x in topolist]
    # leave only closest neighbours
    topolist_estimate = [
        topolist_estimate[i][:len(topolist[i])] if len(topolist[i].shape) != 0 else topolist_estimate[i][0] for i in
        range(len(topolist))]
    # match_score = sum([ len(topolist[i]) - len(np.intersect1d(topolist_estimate[i], topolist[i])) for i in range(len(topolist))])
    match_score = sum(
        [len(topolist[i]) - len(np.intersect1d(topolist_estimate[i], topolist[i])) if len(topolist[i].shape) != 0
         else 1 - len(np.intersect1d(topolist_estimate[i], topolist[i])) for i in range(len(topolist))])
    return match_score

epochs = 500

os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
DATA_DIR = DATA_ROOT + 'CyTOFdataPreprocess/'
source_dir = DATA_ROOT + 'Real_sets/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

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
    # re-label data making labels numerical, unassigned cells are'-1'
    lbls[[(l == '"unassigned"') or (l == '"Unassgined"')  for l in lbls]]  = -1
    le.fit(lbls)
    y = le.transform(lbls)
    y[lbls=='-1'] = '-1'
    lbls=y
    # read DCAE output
    npz_res = np.load(z_dir + '/' + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', allow_pickle=True)
    z = npz_res['z']

    topolist_estimateHD = get_real_topology_list_f(lbls, aFrame)
    topolist_estimate = get_representation_topology_f(z, lbls)
    top_score = get_topology_match_score(topolist_estimateHD, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + str(bl) + "_" + 'Farthest' + '_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score= top_score)

# Compute performance for UMAP
z_dir  = DATA_ROOT + "Real_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Real_sets/UMAP_output/Performance/"
#bl = list_of_inputs[1]
for bl in list_of_inputs:
    infile = DATA_DIR + bl
    npzfile = np.load(infile, allow_pickle=True)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']
    # re-label data making labels numerical, unassigned cells are'-1'
    lbls[[(l == '"unassigned"') or (l == '"Unassgined"') for l in lbls]] = -1
    le.fit(lbls)
    y = le.transform(lbls)
    y[lbls == '-1'] = '-1'
    lbls = y

    # read UMAP output
    npz_res = np.load(z_dir + str(bl) + '_UMAP_rep_2D.npz',  allow_pickle=True)
    z = npz_res['z']

    topolist_estimateHD = get_real_topology_list_f(lbls, aFrame)
    topolist_estimate = get_representation_topology_f(z, lbls)
    top_score = get_topology_match_score(topolist_estimateHD, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + str(bl) + "_" + 'Farthest' +'_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score=top_score)


# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Real_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"
#bl = list_of_branches[1]
for  bl in list_of_inputs:
    infile = DATA_DIR + bl
    npzfile = np.load(infile, allow_pickle=True)
    aFrame = npzfile['aFrame'];
    Idx = npzfile['Idx']
    lbls = npzfile['lbls']
    # re-label data making labels numerical, unassigned cells are'-1'
    lbls[[(l == '"unassigned"') or (l == '"Unassgined"') for l in lbls]] = -1
    le.fit(lbls)
    y = le.transform(lbls)
    y[lbls == '-1'] = '-1'
    lbls = y

    #read SAUCIE output
    npz_res = np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz',  allow_pickle=True)
    z = npz_res['z']

    topolist_estimateHD = get_real_topology_list_f(lbls, aFrame)
    topolist_estimate = get_representation_topology_f(z, lbls)
    top_score = get_topology_match_score(topolist_estimateHD, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + str(bl) + "_" + 'Farthest'+'_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score=top_score)

#create  graphs
PLOTS = DATA_ROOT + "Real_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Real_sets/DCAE_output/Performance/", DATA_ROOT + "Real_sets/UMAP_output/Performance/",DATA_ROOT + "Real_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
dir = bor_res_dirs[0]
bl  = list_of_inputs[0]
df = pd.DataFrame()

for i in range(3):
    for bl in list_of_inputs:
        outfile = bor_res_dirs[i] + '/' + str(bl) + "_" + 'Farthest'+'_Topological_PerformanceMeasures.npz'
        npz_res =  np.load(outfile)
        score = int(npz_res['top_score'])

        line = pd.DataFrame([[methods[i], str(bl), score]],   columns =['method','Set','score'])
        df=  df.append(line)

#rename 'branch' by shorter line
di = {'Levine32euclid_scaled_no_negative_removed.npz':'Levine32',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz':'Pregnancy',  'Shenkareuclid_shifted.npz':'Shenkar'}
df =  df.replace({"Set": di})
import matplotlib
matplotlib.use('PS')


outfile = DATA_ROOT + 'Real_sets/' + 'Farthest_'+ 'Topological_PerformanceMeasures.pkl'
df.to_pickle(outfile)

df = pd.read_pickle(outfile)



import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('PS')

sns.set(rc={'figure.figsize':(14, 4)})
g=sns.lineplot(data=df[df["Set"]=='Levine32'],x='k_nn_cl', y='score', hue='method',)
g.set_xticks(range(1,4)) # <--- set the ticks first
g.set_xticklabels(['1','2','3'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "TopoScore"+ "_" + 'Levine32' + ".eps", format='eps', dpi = 350)
plt.close()

sns.set(rc={'figure.figsize':(14, 4)})
g=sns.lineplot(data=df[df["Set"]=='Pregnancy'],x='k_nn_cl', y='score', hue='method',)
g.set_xticks(range(1,4)) # <--- set the ticks first
g.set_xticklabels(['1','2','3'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "TopoScore"+ "_" + 'Pregnancy' + ".eps", format='eps', dpi = 350)
plt.close()


sns.set(rc={'figure.figsize':(14, 4)})
g=sns.lineplot(data=df[df["Set"]=='Shenkar'],x='k_nn_cl', y='score', hue='method',)
g.set_xticks(range(1,4)) # <--- set the ticks first
g.set_xticklabels(['1','2','3'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + "TopoScore"+ "_" + 'Shenkar' + ".eps", format='eps', dpi = 350)
plt.close()

# create table with just first nearest neighbour
df_knn_1  =df[df['k_nn_cl']==1]
