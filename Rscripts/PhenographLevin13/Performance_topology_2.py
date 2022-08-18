'''
Compute tolpological perormance measures in artificiial sets on DCAE, UMAP and SAUCIE
'''
import math
import pandas as pd
import numpy as np
import os
import random
from scipy.spatial import distance


from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors
def get_topology_list(bl, top_list):
    """ Gets a touple defining branches and creates a list of nearest neighbour clusters"""
    # basic topology shared by 5 clusters in pentagon, nearest neighbours in pentagon and its branches for clusters 0 to 6
    # for branches, 5 and 6 closest neighbour should bethe  clustr in pentagon to they are attached to
    topolist = [[1,4], [0,2], [1,3], [2,4], [0,3], bl[0], bl[1]]
    #correct by adding the negbour from bl touple
    topolist[bl[0]].append(5)
    topolist[bl[1]].append(6)
    return topolist

def generate_true_toplogy(aFrame, lbls):
    """ Gets a touple defining branches and creates a list of nearest neighbour clusters
    based on coordinates of central clusters with noise excluded
    """
    #TODO: rewrite other functions here to use true_topology_list as input, this function should replace
    # get_topology_list
    # nullify noisy dimensions
    clus = aFrame
    original_dim = aFrame.shape[1]
    d = 5; k=2
    num_noisy = original_dim - d
    ncl0 = ncl1 = ncl2 = ncl3 = ncl4 = ncl5 = ncl6 = int(k * 6000)
    ncl7 = int(k * 20000)


    # clus[lbls==0, d:] =  np.zeros((ncl0, original_dim - d))
    # clus[lbls == 1, d:] = np.zeros((ncl1, original_dim - d))
    # clus[lbls == 2, d:] = np.zeros((ncl2, original_dim - d))
    # clus[lbls == 3, d:] = np.zeros((ncl3, original_dim - d))
    # clus[lbls == 4, d:] = np.zeros((ncl4, original_dim - d))
    # clus[lbls == 5, d:] = np.zeros((ncl5, original_dim - d))
    # clus[lbls == 6, d:] = np.zeros((ncl6, original_dim - d))
    # clus[np.ix_(lbls == -7,  np.logical_not(np.concatenate((np.zeros(4), np.ones(1), np.zeros(num_noisy - 4), np.ones(4))).astype('bool')))] = 0

    l_list = [-7., 0., 1., 2., 3., 4., 5., 6.]
    # indx = random.sample(range(len(lbls)), 12000)
    lbls_s = lbls  # [indx]
    z_s = clus  # [indx,:]
    true_topology_list = [[], [], [], [], [], [], []]
    true_dist_list = [[], [], [], [], [], [], []]
    for i in range(7):
        dist = [np.mean(distance.cdist(z_s[lbls_s == i, :], z_s[lbls_s == label, :])) for label in l_list]
        # dist = [np.sqrt(np.sum((z_s[lbls_s == i, :].mean(0) - z_s[lbls_s == label, :].mean(0))**2)) for label in l_list]
        # get indexes of  closest clusters, and exclude itself, exclude i th cluster
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
    #indx = random.sample(range(len(lbls)), 12000)
    lbls_s = lbls#[indx]
    z_s = z#[indx,:]
    topolist_estimate = [[], [], [], [], [], [], []]
    for i in range(7):
        dist = [np.mean(distance.cdist(z_s[lbls_s==i,:], z_s[lbls_s==label,:])) for label in l_list]
        #dist = [np.sqrt(np.sum((z_s[lbls_s == i, :].mean(0) - z_s[lbls_s == label, :].mean(0))**2)) for label in l_list]
        #get indexes of  closest clusters, and exclude itself, exclude i th cluster
        seq = sorted(dist)
        rank = [seq.index(v) for v in dist]
        #get top 4
        rank2 = np.array(rank)[np.array(l_list) != np.array(i)]
        l_list2 = np.array(l_list)[np.array(l_list) != np.array(i)]
        nn_list =l_list2[rank2.argsort()]#[:4]
        topolist_estimate[i] = nn_list
    return topolist_estimate

def get_topology_match_score(topolist, topolist_estimate):
    """compare match computed and prescribed"""
    topolist = [np.array(x) for x in topolist]
    # create dictionary from toplolist
    #topo_reord  =[]#    for i in range(len(topodict)):
    #for i in range(7):
        #pos = [0, 1, 2, 3, 4, 5, 6]
        #topodict = dict(zip(topolist[i], pos))
        #topo_reord.append([topodict[y] for y in topolist_estimate[i]])
    #match_score =  sum([np.sum(np.abs(np.asarray(y)-np.asarray(pos))) for y in topo_reord])
    topo_1cl = [(np.intersect1d(topolist_estimate[i][:2], topolist[i][:2]).shape[0]) for i in range(7)]
    match_score =  sum(topo_1cl)
    return match_score


os.chdir('/home/grinek/PycharmProjects/BIOIBFO25L/')
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
ID = 'DCAE_lam_0.1_batch_128_alp_0.5_m_10'
#ID ='Decreasing_MMD_zero_g_0_lam_0.1_batch_128_alp_0.2_m_10'
#ID ='DICSCONT_DELU_0.2_repulsive_MMD_0.05_experiment_g_10_lam_0.1_batch_128_alp_0.2_m_10'
 #'clip_grad_exp_MDS_g_0.1_lam_0.1_batch_128_alp_0.2_m_10' #'DICSCONT_DELU_0.2_g_0.1_lam_0.1_batch_128_alp_0.2_m_10'        #'zero_MDS_g_0_lam_0.1_batch_128_alp_0.2_m_10' #'clip_grad_exp_MDS_g_0.1_lam_0.1_batch_128_alp_0.2_m_10'
epochs = 500
# Compute performance for DCAE
z_dir  = DATA_ROOT + "Artificial_sets/DCAE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/DCAE_output/Performance/"

#compute true topolgies
# t_list = [None] * 25
# for i in range(25):
#     print(i)
#     infile = source_dir + 'set_' + str(list_of_branches[i]) + '.npz'
#     npzfile = np.load(infile)
#     aFrame = npzfile['aFrame'];
#     lbls = npzfile['lbls'];
#     t_list[i] = generate_true_toplogy(aFrame, lbls)
#     print(t_list[i])
#
# top_list = [x[0] for x in t_list]
# outfile = output_dir + '/' 'True_closest_neighbours.npy'
# np.save(outfile, top_list, allow_pickle=True)
# b = np.load(outfile, allow_pickle=True)
#
# top_list = [x[1] for x in t_list]
# outfile = output_dir + '/' 'True_closest_distances.npy'
# np.save(outfile, top_list, allow_pickle=True)
# e = np.load(outfile, allow_pickle=True)

outfile = output_dir + '/' 'True_closest_neighbours.npy'
top_list = np.load(outfile, allow_pickle=True)
top_list = dict(zip(list_of_branches, top_list))


#bl = list_of_branches[0]
for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    lbls = npzfile['lbls']
    # read DCAE output
    npz_res=np.load(z_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz')
    z= npz_res['z']
    topolist = np.abs(top_list[bl])
    topolist_estimate = (get_representation_topology(z, lbls))
    topolist_estimate = [np.abs(x) for x in topolist_estimate]
    top_score = get_topology_match_score(topolist, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + ID +  str(bl) + 'epochs' + str(epochs) + '_2_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score= top_score)
'''
# Compute performance for UMAP
z_dir  = DATA_ROOT + "Artificial_sets/UMAP_output/"
output_dir =  DATA_ROOT + "Artificial_sets/UMAP_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    lbls = npzfile['lbls']
    # read DCAE output
    npz_res=np.load(z_dir + '/' + str(bl) + '_UMAP_rep_2D.npz')
    z= npz_res['z']
    topolist = np.abs(top_list[bl])
    topolist_estimate = (get_representation_topology(z, lbls))
    topolist_estimate = [np.abs(x) for x in topolist_estimate]
    top_score = get_topology_match_score(topolist, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + str(bl) + '_2_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score= top_score)
# Compute performance for SAUCIE
z_dir = DATA_ROOT + "Artificial_sets/SAUCIE_output/"
output_dir =  DATA_ROOT + "Artificial_sets/SAUCIE_output/Performance/"
#bl = list_of_branches[1]
for bl in list_of_branches:
    infile = source_dir + 'set_' + str(bl) + '.npz'
    npzfile = np.load(infile)
    lbls = npzfile['lbls']
    # read DCAE output
    npz_res=np.load(z_dir + '/' + str(bl) + '_SAUCIE_rep_2D.npz')
    z= npz_res['z']
    topolist = np.abs(top_list[bl])
    topolist_estimate = (get_representation_topology(z, lbls))
    topolist_estimate = [np.abs(x) for x in topolist_estimate]
    top_score = get_topology_match_score(topolist, topolist_estimate)
    print(top_score)
    outfile = output_dir + '/' + str(bl) + '_2_Topological_PerformanceMeasures.npz'
    np.savez(outfile, top_score= top_score)
'''
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
            outfile = bor_res_dirs[i] + '/' +  str(bl) + '_2_Topological_PerformanceMeasures.npz'
        else:
            outfile = bor_res_dirs[i] + '/'   + ID +   str(bl) + 'epochs' + str(epochs) + '_2_Topological_PerformanceMeasures.npz'
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
plt.savefig(PLOTS + ID +"_" + 'epochs' + str(epochs) + "_2_TopoScore.eps", format='eps', dpi = 350)
plt.close()

df2 = df.groupby('method').sum()
print(df2)
df2.to_csv(output_dir + '/' + ID + 'epochs' + str(epochs) +'_2_Summary_Performance_topology.csv', index=True)


