'''
Compute topological performance measures in artificial sets on DCAE, UMAP and SAUCIE
'''
import pandas as pd
import numpy as np
import os
from scipy.spatial import distance


def get_topology_list(bl):
    """ Gets a touple defining branches and creates a list of nearest neighbour clusters"""
    # basic topology shared by 5 clusters in pentagon, nearest neighbours in pentagon and its branches for clusters 0 to 6
    # for branches, 5 and 6 closest neighbour should be the  cluster in pentagon to they are attached to
    topolist = [[1,4], [0,2], [1,3], [2,4], [0,3], bl[0], bl[1]]
    #correct by adding the negbour from bl touple
    topolist[bl[0]].append(5)
    topolist[bl[1]].append(6)
    return topolist

def generate_true_toplogy(aFrame, lbls):
    """ Gets a touple defining branches and creates a list of nearest neighbour clusters
    based on coordinates of central clusters with noise excluded
    """
    clus = aFrame
    original_dim = aFrame.shape[1]
    d = 5; k=2


    l_list = [-7., 0., 1., 2., 3., 4., 5., 6.]
    lbls_s = lbls  # [indx]
    z_s = clus  # [indx,:]
    true_topology_list = [[], [], [], [], [], [], []]
    true_dist_list = [[], [], [], [], [], [], []]
    for i in range(7):
        dist = [np.mean(distance.cdist(z_s[lbls_s == i, :], z_s[lbls_s == label, :])) for label in l_list]
        seq = sorted(dist)
        true_dist_list[i] = seq
        rank = [seq.index(v) for v in dist]
        # get top 4
        rank2 = np.array(rank)[np.array(l_list) != np.array(i)]
        l_list2 = np.array(l_list)[np.array(l_list) != np.array(i)]
        nn_list = l_list2[rank2.argsort()][:7]
        true_topology_list[i] = nn_list

    return true_topology_list, true_dist_list

def get_representation_topology(z, lbls):
    """compute actual, returning 3 nearest neighbours per each cluster in pentagon"""
    #sample each cluster
    l_list= [-7.,  0.,  1.,  2.,  3.,  4.,  5.,  6.]
    lbls_s = lbls
    z_s = z
    topolist_estimate = [[], [], [], [], [], [], []]
    for i in range(7):
        dist = [np.mean(distance.cdist(z_s[lbls_s==i,:], z_s[lbls_s==label,:])) for label in l_list]
        seq = sorted(dist)
        rank = [seq.index(v) for v in dist]
        #get top 4
        rank2 = np.array(rank)[np.array(l_list) != np.array(i)]
        l_list2 = np.array(l_list)[np.array(l_list) != np.array(i)]
        nn_list =l_list2[rank2.argsort()][:4]
        topolist_estimate[i] = nn_list
    return topolist_estimate

def get_topology_match_score(topolist, topolist_estimate):
    """compare match computed and prescribed"""
    topolist = [np.array(x) for x in topolist]
    # leave only closest neighbours
    topolist_estimate = [topolist_estimate[i][:len(topolist[i])] if len(topolist[i].shape)!=0  else topolist_estimate[i][0] for i in range(len(topolist))]
    match_score = sum(
        [len(topolist[i]) - len(np.intersect1d(topolist_estimate[i], topolist[i])) if len(topolist[i].shape)!=0
         else  1- len(np.intersect1d(topolist_estimate[i], topolist[i]))  for i in range(len(topolist))])
    return match_score


os.chdir('/media/grinek/Seagate/DCAE/')
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
z_dir  = DATA_ROOT + "Artificial_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/DCAE_output/Performance/"
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
ID = 'DCAE_lam_0.1_batch_128_alp_0.5_m_10'
epochs =500
# Compute performance for DCAE

for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    lbls = npzfile['lbls']
    # read DCAE output
    npz_res=np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz')
    z= npz_res['z']
    topolist = get_topology_list(bl)
    topolist_estimate = get_representation_topology(z, lbls)
    top_score = get_topology_match_score(topolist, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + ID +  str(bl) + 'epochs' + str(epochs) + '_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score= top_score)

# Compute performance for UMAP
z_dir  = DATA_ROOT + "Artificial_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Artificial_sets/UMAP_output/Performance/"

for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    lbls = npzfile['lbls']
    # read UMAP output
    npz_res=np.load(z_dir + '/' + str(bl) + '_UMAP_rep_2D.npz')
    z= npz_res['z']
    topolist = get_topology_list(bl)
    topolist_estimate = get_representation_topology(z, lbls)
    top_score = get_topology_match_score(topolist, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + str(bl) + '_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score= top_score)
# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Artificial_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"

for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    lbls = npzfile['lbls']
    # read SAUCIE output
    npz_res=np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz')
    z= npz_res['z']
    topolist = get_topology_list(bl)
    topolist_estimate = get_representation_topology(z, lbls)
    top_score = get_topology_match_score(topolist, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + str(bl) + '_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score= top_score)

#create  graphs
PLOTS = DATA_ROOT + "Artificial_sets/PLOTS/"
bor_res_dirs = [DATA_ROOT + "Artificial_sets/DCAE_output/Performance/", DATA_ROOT + "Artificial_sets/UMAP_output/Performance/",DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"]
methods = ['DCAE', 'UMAP', 'SAUCIE']
dir = bor_res_dirs[0]
bl  = list_of_branches[0]
df = pd.DataFrame()
for i in range(3):
    for bl in list_of_branches:
        if  bor_res_dirs[i]!=bor_res_dirs[0]:
            outfile = bor_res_dirs[i] + '/' +  str(bl) + '_Topological_PerformanceMeasures.npz'
        else:
            outfile = bor_res_dirs[i] + '/'   + ID +   str(bl) + 'epochs' + str(epochs) + '_Topological_PerformanceMeasures.npz'
        print(outfile)
        npz_res =  np.load(outfile)
        score = int(npz_res['top_score'])

        line = pd.DataFrame([[methods[i], str(bl), score]],   columns =['method','branch','score'])
        df=  df.append(line)


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('PS')

sns.set(rc={'figure.figsize':(14, 4)})
g = sns.barplot(x='branch', y='score', hue='method', data=df.reset_index(), palette=['tomato','yellow','limegreen'])
g.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.savefig(PLOTS + ID +"_" + 'epochs' + str(epochs) + "TopoScore.eps", format='eps', dpi = 350)
plt.close()

df2 = df.groupby('method').sum()
print(df2)
df2.to_csv(output_dir + '/' + ID + 'epochs' + str(epochs) +'_Summary_Performance_topology.csv', index=True)


# check true topolgies
t_list =[None] * 25
for i in range(25):
    print(i)
    infile = source_dir + 'set_' + str(list_of_branches[i]) + '.npz'
    npzfile = np.load(infile)
    aFrame = npzfile['aFrame'];
    lbls = npzfile['lbls'];
    t_list[i] = generate_true_toplogy(aFrame, lbls)
    print(t_list[i])
