# script generates n sibdimensional clusters
# of 5 and 10 dims in 30 dimensional space
import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing
from sklearn.model_selection import train_test_split
from numpy.ctypeslib import ndpointer
import ctypes
import sklearn
import seaborn as sns
import os
import h5py

from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from pathos import multiprocessing

num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)


def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}


lib = ctypes.cdll.LoadLibrary("./Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                 ctypes.c_size_t, ctypes.c_size_t,
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                 ctypes.c_double, ctypes.c_size_t,
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # Sigma
                 ctypes.c_size_t]

from mdcgenpy import clusters
from mdcgenpy.clusters import distributions

unif_d = clusters.distributions.get_dist_function('uniform')
n_cl = 2
dims = 30
# generate signatures, 'ones' are denoting 'meaninful dimensions', 'zeroes' - noise
sig1 = np.concatenate((np.zeros(25), np.ones(5)), axis=0)
sig2 = np.concatenate((np.ones(5), np.zeros(25)), axis=0)
sig3 = np.concatenate((np.zeros(10), np.ones(5), np.zeros(15)), axis=0)
sig4 = np.concatenate((np.zeros(15), np.ones(5), np.zeros(10)), axis=0)
sig5 = np.concatenate((np.zeros(20), np.ones(5), np.zeros(5)), axis=0)
sig6 = np.concatenate((np.zeros(5), np.ones(5), np.zeros(20)), axis=0)

cl1_center = np.zeros(dims)
cl2_center = np.concatenate((np.ones(20), np.zeros(10)), axis=0)

# generate functions for input of data-generating function
ncl1 = ncl2 = 1


def cl1(shape, param):
    return cl1_center + np.concatenate([np.zeros((shape, 20)), np.random.uniform(low=-param, high=param, size=(shape, 10))],
                                       axis=1)


def cl2(shape, param):
    return cl2_center + np.concatenate([np.random.uniform(low=-param, high=param, size=(shape, 10)), np.zeros((ncl2, 20))],
                                       axis=1)


distributions_list = [cl1, cl2]
compactness_factor_list = [1, 1]
cluster_gen = clusters.ClusterGenerator(pseed=10, n_samples=2000, n_feats=dims, k=n_cl, min_samples=100,
                                        possible_distributions=None, distributions=distributions_list, mv=True,
                                        corr=0.0, compactness_factor=compactness_factor_list, alpha_n=1, scale=None,
                                        outliers=0, rotate=False, add_noise=0, n_noise=None,
                                        ki_coeff=3.0)
# Get tuple with a numpy array with samples and another with labels
data, labels = cluster_gen.generate_data()

# two subspace clusters centers
original_dim = 30
cl1_center = np.zeros(original_dim)
cl2_center = np.concatenate((np.ones(20), np.zeros(10)), axis=0)
ncl1 = ncl2 = 100000
'''
cl1_center = np.zeros(original_dim)
cl2_center = np.concatenate((np.ones(20),  np.zeros(10)), axis=0 )

lbls = np.concatenate((np.zeros(ncl1), np.ones(ncl2)), axis=0)

cl1 = cl1_center +  np.concatenate([np.zeros((ncl1,20)),  np.random.uniform(low=-1, high=1, size=(ncl1,10))], axis=1 )
cl2 = cl2_center +  np.concatenate([np.random.uniform(low=-1, high=1, size=(ncl2,10)),  np.zeros((ncl2,20))], axis=1 )
sns.violinplot(data= cl1, bw = 0.1);sns.violinplot(data= cl2, bw = 0.1);
#noisy or not:
noise_sig1 =  np.concatenate((np.zeros(20),  np.ones(10)), axis=0 )
noise_sig2 = np.concatenate((np.ones(10), np.zeros(20)), axis=0 )
noise_scale =1
# add noise to orthogonal dimensions
cl1_noisy = cl1 + np.concatenate([np.random.normal(loc=0, scale = noise_scale, size=(ncl1,20)), np.zeros((ncl1,10))], axis=1 )
cl2_noisy = cl2 + np.concatenate([ np.zeros((ncl2, 10)), np.random.normal(loc=0, scale = noise_scale, size=(ncl2,20))], axis=1 )
sns.violinplot(data= cl1_noisy, bw = 0.1);sns.violinplot(data= cl2_noisy, bw = 0.1);

# create noisy neighbours, 30 per each initial point,  neighbours live in parallel dims
# and add orthogonal noise
def Perturbation(i, cl, noise_sig, nn):
    ncol = np.shape(cl)[1]
    nrow = np.shape(cl)[0]
    sample= cl[i,:] + (noise_sig-1)* np.random.normal(loc=0, scale = noise_scale, size=(nn,30)) +\
            noise_sig * np.random.normal(loc=0, scale = noise_scale, size=(nn,30))
    return sample
nn=30
resSample1 = Parallel(n_jobs=48, verbose=0, backend="threading")(delayed(Perturbation,
                check_pickle=False)(i, cl1, noise_sig1, nn) for i in range(np.shape(cl1_noisy)[0]))
#cl1_noisy_nn = np.vstack(resSample1)
cl1_noisy_nn = np.zeros((ncl1, nn, 30))
for i in range(ncl1):
    cl1_noisy_nn[i,:,:] = resSample1[i]

resSample2 = Parallel(n_jobs=48, verbose=0, backend="threading")(delayed(Perturbation,
                check_pickle=False)(i, cl2, noise_sig2, nn) for i in range(np.shape(cl2_noisy)[0]))
cl2_noisy_nn = np.zeros((ncl2, nn, 30))
for i in range(ncl2):
    cl2_noisy_nn[i,:,:] = resSample2[i]
del resSample1, resSample2

# find neighbours an define clusters consiting solely from perturbed data
def find_neighbors(data, k_, metric='euclidean', cores=12):
    tree = NearestNeighbors(n_neighbors=k_, algorithm="ball_tree", leaf_size=30, metric=metric,  metric_params=None, n_jobs=cores)
    tree.fit(data)
    dist, ind = tree.kneighbors(return_distance=True)
    return {'dist': np.array(dist), 'idx': np.array(ind)}

noisy_clus = np.concatenate(np.vstack((cl1_noisy_nn[:, np.random.choice(cl1_noisy_nn.shape[1], 1, replace=False), :],
                            cl2_noisy_nn[:, np.random.choice(cl2_noisy_nn.shape[1], 1, replace=False), :] )), axis=0)
# visualise with umap

# find orthogonal distances
# zero's in noise_sig are noisy dims
def ort_dist(i, noise_sig, cl, cl_center):
    dist= np.sum(np.square((cl[i,:] - cl_center) * (noise_sig-1)))
    return dist
resSample1 = Parallel(n_jobs=48, verbose=0, backend="threading")(delayed(ort_dist,
                check_pickle=False)(i, noise_sig1, noisy_clus[lbls==0,:], cl1_center)
                                                                 for i in range(np.shape(noisy_clus[lbls==0,:])[0]))
cl1_ort_dist = np.array(resSample1)
resSample2 = Parallel(n_jobs=48, verbose=0, backend="threading")(delayed(ort_dist,
                check_pickle=False)(i, noise_sig2, noisy_clus[lbls==1,:], cl2_center)
                                                                 for i in range(np.shape(noisy_clus[lbls==1,:])[0]))
cl2_ort_dist = np.array(resSample2)


nrow=noisy_clus.shape[0]
# find nearest neighbours
nb=find_neighbors(noisy_clus, nn, metric='manhattan', cores=48)
Idx = nb['idx']; Dist = nb['dist']


def singleInput(i):
    nei = noisy_clus[Idx[i, :], :]
    return [nei, i]


nrow=noisy_clus.shape[0]
# find nearest neighbours
nn=30

rk=range(nn)
def singleInput(i):
     nei =  noisy_clus[Idx[i,:],:]
     di = [np.sqrt(sum(np.square(noisy_clus[i] - nei[k_i,]))) for k_i in rk]
     return [nei, di, i]

inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=48, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
neibALL = np.zeros((nrow, nn, original_dim))
Distances = np.zeros((nrow, nn))
neib_weight = np.zeros((nrow, nn))
Sigma = np.zeros(nrow, dtype=float)
for i in range(nrow):
    neibALL[i,] = results[i][0]
for i in range(nrow):
    Distances[i,] = results[i][1]
#Compute perpelexities
perp((Distances[:,0:nn]),       nrow,     original_dim,   neib_weight,          nn,          nn,   Sigma,    48)
      #(     double* dist,      int N,    int D,       double* P,     double perplexity,    int K, int num_threads)
np.shape(neib_weight)
plt.plot(neib_weight[1,])
#sort and normalise weights
topk = np.argsort(neib_weight, axis=1)[:,-nn:]
topk= np.apply_along_axis(np.flip, 1, topk,0)
neib_weight=np.array([ neib_weight[i, topk[i]] for i in range(len(topk))])
neib_weight=sklearn.preprocessing.normalize(neib_weight, axis=1, norm='l1')
neibALL=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])

plt.plot(neib_weight[1,:]);plt.show()

outfile = './data/ArtBulbz.npz'
np.savez(outfile, Idx=Idx, cl1=cl1, cl2=cl2, noisy_clus=noisy_clus,
         lbls=lbls,  Dist=Dist, cl1_noisy_nn =cl1_noisy_nn,
         cl2_noisy_nn=cl2_noisy_nn, cl1_noisy =cl1_noisy,
         cl2_noisy=cl2_noisy,cl1_ort_dist=cl2_ort_dist, cl2_ort_dist=cl2_ort_dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)

import dill                            
filepath = 'session_ArtdataGeneration.pkl'
dill.dump_session(filepath) # Save the session
dill.load_session(filepath)

'''

outfile = './data/ArtBulbz.npz'
npzfile = np.load(outfile)
lbls = npzfile['lbls'];
Idx = npzfile['Idx'];
cl1 = npzfile['cl1'];
cl2 = npzfile['cl2'];
noisy_clus = npzfile['noisy_clus'];
lbls = npzfile['lbls'];
Dist = npzfile['Dist'];
cl1_noisy_nn = npzfile['cl1_noisy_nn'];
cl2_noisy_nn = npzfile['cl2_noisy_nn'];
cl1_noisy = npzfile['cl1_noisy'];
cl2_noisy = npzfile['cl2_noisy'];
cl1_ort_dist = npzfile['cl2_ort_dist'];
cl2_ort_dist = npzfile['cl2_ort_dist'];
neibALL = npzfile['neibALL']
neib_weight = npzfile['neib_weight']
Sigma = npzfile['Sigma']
