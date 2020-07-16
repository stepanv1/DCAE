'''This script to visualise cytoff data using deep variational autoencoder with MMD  with neighbourhood denoising and
contracting knnCVAE, neighbourhood cleaning removed
TODO ?? pretraining
Now with entropic weighting of neibourhoods
Innate‐like CD8+ T‐cells and NK cells: converging functions and phenotypes
Ayako Kurioka,corresponding author 1 , 2 Paul Klenerman, 1 , 2 and Christian B. Willbergcorresponding author 1 , 2
CD8+ T Cells and NK Cells: Parallel and Complementary Soldiers of Immunotherapy
Jillian Rosenberg1 and Jun Huang1,2
'''

import keras
import tensorflow as tf
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob
from keras.layers import Input, Dense, Lambda, Layer, Dropout, BatchNormalization
from keras.layers.noise import AlphaDropout
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K
from keras import metrics
import random
# from keras.utils import multi_gpu_model
import timeit
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.constraints import maxnorm
# import readline
# mport rpy2
# from rpy2.robjects.packages import importr
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()
# stsne = importr('stsne')
# subspace = importr('subspace')
import sklearn
from sklearn.preprocessing import MinMaxScaler
# from kerasvis import DBLogger
# Start the keras visualization server with
# export FLASK_APP=kerasvis.runserver
# flask run
import seaborn as sns
import warnings
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, l1
from keras.models import load_model
from keras import regularizers
# import champ
# import ig
# import metric
# import dill
from keras.callbacks import TensorBoard

# from GetBest import GetBest

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)


def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}


markers = ["CD3", "CD45", "pNFkB", "pp38", "CD4", "CD20", "CD33", "pStat5", "CD123", "pAkt", "pStat1", "pSHP2",
           "pZap70",
           "pStat3", "CD14", "pSlp76", "pBtk", "pPlcg2", "pErk", "pLat", "IgM", "pS6", "HLA-DR", "CD7"]


def naive_power(m, n):
    m = np.asarray(m)
    res = m.copy()
    for i in range(1, n):
        res *= m
    return res


class CustomMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('mse'):
                print
                logs[k]
            if k.endswith('mean_squared_error_weighted'):
                print
                logs[k]
            if k.endswith('pen_zero'):
                print
                logs[k]
            if k.endswith('kl_l'):
                print
                logs[k]


class CustomEarlyStopping(Callback):
    def __init__(self, criterion=0.001,
                 patience=0, verbose=0):
        super(EarlyStopping, self).__init__()

        self.criterion = criterion
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.lesser

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_train = logs.get('val_loss')
        # current_train = logs.get('loss')
        if current_train is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(current_train, self.criterion):
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))

class EarlyStoppingByValLoss(Callback):
    def __init__(self, monitor='val_loss', min_delta=0.01, verbose=1, patience=1,
                 restore_best_weights=False):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value

    def on_epoch_end(self, epoch, logs=None):
        current = np.mean(self.get_monitor_value(logs))
        print(current)
        if current is None:
            return

        if current + self.min_delta < self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

class AnnealingCallback(Callback):
    def __init__(self, weight, kl_weight_lst):
        self.weight = weight
        self.kl_weight_lst = kl_weight_lst
    def on_epoch_end (self, epoch, logs={}):
        new_weight = K.eval(self.kl_weight_lst[epoch])
        K.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.weight)))




# import vpsearch as vp
# def find_neighbors(data, k_, metric='manhattan', cores=12):
#   res = vp.find_nearest_neighbors(data, k_, cores, metric)
#   return {'dist':np.array(res[1]), 'idx': np.int32(np.array(res[0]))}


# from libKMCUDA import kmeans_cuda, knn_cuda
# def find_neighbors(data, k_, metric='euclidean', cores=12):
#    ca = kmeans_cuda(np.float32(data), 25, metric="euclidean", verbosity=1, seed=3, device=0)
#    neighbors = knn_cuda(k_, np.float32(data), *ca, metric=metric, verbosity=1, device=0)
#    return {'dist':0, 'idx': np.int32(neighbors)}

num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)
def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

import ctypes
from numpy.ctypeslib import ndpointer

lib = ctypes.cdll.LoadLibrary("./Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #Sigma
                ctypes.c_size_t]


from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from pathos import multiprocessing


# results = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in range(100))

def find_neighbors(data, k_, metric='manhattan', cores=12):
    tree = NearestNeighbors(n_neighbors=k_, algorithm="ball_tree", leaf_size=30, metric=metric, metric_params=None,
                            n_jobs=cores)
    tree.fit(data)
    dist, ind = tree.kneighbors(return_distance=True)
    return {'dist': np.array(dist), 'idx': np.array(ind)}


# load data
k = 30
perp = k
k3 = k * 3
ID = 'Nowicka2017'
'''
data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816
'''
source_dir = '/media/FCS_local/Stepan/data/WeberLabels/'
'''

#file_list = glob.glob(source_dir + '/*.txt')
data0 = np.genfromtxt(source_dir + "d_matrix.txt"
, names=None, dtype=float, skip_header=1)
aFrame = data0[:,1:] 
# set negative values to zero
aFrame[aFrame < 0] = 0

patient_table = np.genfromtxt(source_dir + "label_patient.txt", names=None, dtype='str', skip_header=1, delimiter=" ", usecols = (1, 2, 3))
lbls=patient_table[:,0]
IDX = np.random.choice(patient_table.shape[0], patient_table.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
aFrame= aFrame[IDX,:]
lbls = lbls[IDX]



len(lbls)
#scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
#scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=48)
Idx = nb['idx']; Dist = nb['dist']


#Dist = Dist[IDX]
#Idx = Idx[IDX]

nrow=Idx.shape[0]
# find nearest neighbours
def singleInput(i):
    nei = noisy_clus[Idx[i, :], :]
    return [nei, i]

# find nearest neighbours
nn=30

rk=range(k3)
def singleInput(i):
     nei =  aFrame[Idx[i,:],:]
     di = [np.sqrt(sum(np.square(aFrame[i] - nei[k_i,]))) for k_i in rk]
     return [nei, di, i]

nrow = len(lbls)
inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = 48
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=48, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
neibALL = np.zeros((nrow, k3, original_dim))
Distances = np.zeros((nrow, k3))
neib_weight = np.zeros((nrow, k3))
Sigma = np.zeros(nrow, dtype=float)
for i in range(nrow):
    neibALL[i,] = results[i][0]
for i in range(nrow):
    Distances[i,] = results[i][1]
#Compute perpelexities
nn=30
perp((Distances[:,0:k3]),       nrow,     original_dim,   neib_weight,          nn,          k3,   Sigma,    48)
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

outfile = source_dir + '/Nowicka2017euclid.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)



#outfile = source_dir + '/Nowicka2017euclid.npz'
#np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls,  Dist=Dist)
'''
outfile = source_dir + '/Nowicka2017euclid.npz'

npzfile = np.load(outfile)
lbls = npzfile['lbls'];
Idx = npzfile['Idx'];
aFrame = npzfile['aFrame'];
Dist = npzfile['Dist']
neibALL = npzfile['neibALL']
neib_weight = npzfile['neib_weight']
Sigma = npzfile['Sigma']

from sklearn.decomposition import PCA
principalComponents  = PCA(n_components = 2).fit_transform(aFrame)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pd.DataFrame(data=lbls, columns=['lbls'])], axis = 1)
sns.pairplot(x_vars = ['principal component 1'], y_vars=['principal component 2'], data=finalDf, hue="lbls", size=5)

import umap
#TODO: compare distances and sigma-weightred distances with local graph weights, especially
# in how they preserve cluster state (i.e weights of the edges inside the same cluster are less)
# https://umap-learn.readthedocs.io/en/latest/api.html
fss , _ ,_ = umap.umap_.fuzzy_simplicial_set(aFrame, n_neighbors=90, random_state=1,
                            metric = 'euclidean', metric_kwds={},
                            knn_indices=Idx[:,0:90], knn_dists=Dist[:,0:90],
            angular=False, set_op_mix_ratio=1.0, local_connectivity=1.0, apply_set_operations=True, verbose=True)
#TODO try nearest neighbour comp using umap functions
#umap.umap_.nearest_neighbors(X, n_neighbors, metric, metric_kwds, angular, random_state, low_memory=False, use_pynndescent=True, verbose=False)
# crete a metric to demonstrate cluster preservation by the nearest neighbour
# create a matrices containing indices and weights
fss_lil =  fss.tolil()
fss_rows = fss_lil.rows
fss_vals = fss_lil.data
max_len  = len(max(fss_rows, key=len))
# sort indices and weigths by value of weights
from operator import itemgetter
val_sort , row_sort = [], [];
for i in range(len(fss_vals)):
    res = [list(x) for x in zip(*sorted(zip(fss_vals[i], fss_rows[i]), key=itemgetter(0), reverse=True))]
    val_sort.append(res[0][0:30])
    row_sort.append(res[1][0:30])
# take top 30 elemetes to create index and value matrices
weight_tab=np.stack(val_sort)
idx_tab =np.stack(row_sort)
# create table for cluster distribution in neighbours
num_lbls = (np.unique(lbls, return_inverse=True)[1])
lbls_tab=(np.stack([num_lbls[idx_tab[k,:]]  for k in range(len(lbls))]))
nei_lbls = np.c_[num_lbls, lbls_tab]
# Same nn-based neighbours
lbls_tab_nn=(np.stack([num_lbls[Idx[:,0:30][k,:]]  for k in range(len(lbls))]))
nei_lbls_nn = np.c_[num_lbls, lbls_tab_nn]
# purity vectors
#error count
error_rate=[30 - sum(np.equal(lbls_tab[k,:], num_lbls[k]))   for k in range(len(lbls))]
error_rate_nn=[30 - sum(np.equal(lbls_tab_nn[k,:], num_lbls[k]))   for k in range(len(lbls))]
plt.hist(error_rate,30)
plt.hist(error_rate_nn,30, color='red')
np.mean(error_rate)
np.mean(error_rate_nn)
#plot marker distributions
# transform labels into natural numbers
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(lbls)
lbls = le.transform(lbls)
table(lbls)

bw=0.2
cols = 8
rows = 1
plt.figure()
gs = plt.GridSpec(rows, cols)
for cl in range(0, 8):
    bw=0.2
    i = cl
    plt.subplot(gs[0,i])
    plt.title(cl)
    sns.violinplot(data= aFrame[num_lbls==cl , :], bw = bw);
    plt.tight_layout(pad=0.1)
plt.show();

#correlation plot
bw=0.2
cols = 3
rows = 3
plt.figure()
gs = plt.GridSpec(rows, cols)
for cl in range(0, 8):
    bw=0.2
    i = cl//rows
    j = cl - i*cols
    plt.subplot(gs[i, j])
    plt.title(cl)
    dt=aFrame[num_lbls==cl , :]
    df = pd.DataFrame(dt)
    corr = df.corr()
    df.corr().unstack().sort_values().drop_duplicates().tail()
    corr = corr[abs(corr)>0.2]
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    plt.tight_layout(pad=0.1)
plt.show();

# lbls2=npzfile['lbls'];Idx2=npzfile['Idx'];aFrame2=npzfile['aFrame'];
# cutoff2=npzfile['cutoff']; Dist2 =npzfile['Dist']
cutoff = np.repeat(0.1, 24)
batch_size = 256
original_dim = 24
latent_dim = 2
intermediate_dim = 12
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

#annealing schedule
#kl_weight = K.variable(value=0)
#kl_weight_lst = K.variable(np.array(frange_cycle_linear(0.0, 1.0, epochs, n_cycle=10, ratio=0.75)))


epsilon_std = 1.0
U = 10  # energy barier
szr = 0  # number of replicas in training data set
szv = 10000  # number of replicas in validation data set

# nearest neighbours

# generate neighbours data
features = aFrame
IdxF = Idx[:, ]
nrow = np.shape(features)[0]
b = 0
neibALL = np.zeros((nrow, k3, original_dim))
cnz = np.zeros((original_dim))
cut_neibF = np.zeros((nrow, original_dim))
weight_distALL = np.zeros((nrow, k3))
weight_neibALL = np.zeros((nrow, k3))
rk = range(k3)

print('compute training sources and targets...')
from scipy.stats import binom

sigmaBer = np.sqrt(cutoff * (1 - cutoff) / k)
# precompute pmf for all cutoffs at given kfrange_cycle_linear
probs = 1 - cutoff
pmf = np.zeros((original_dim, k + 1))
for j in range(original_dim):
    rb = binom(k, probs[j])
    pmf[j, :] = (1 - rb.cdf(range(k + 1))) ** 10


def singleInput(i):
    nei = features[IdxF[i, :], :]
    cnz = [np.sum(np.where(nei[:k, j] == 0, 0, 1)) for j in range(original_dim)]
    # cut_nei= np.array([0 if (cnz[j] >= cutoff[j] or cutoff[j]>0.5) else
    #                   (U/(cutoff[j]**2)) * ( (cutoff[j] - cnz[j]) / sigmaBer[j] )**2 for j in range(original_dim)])
    cut_nei = np.array([U * pmf[j, :][cnz[j]] for j in range(original_dim)])
    # weighted distances computed in L2 metric
    weight_di = [np.sqrt(sum((np.square(features[i] - nei[k_i,]) / (1 + cut_nei)))) for k_i in rk]
    return [nei, cut_nei, weight_di, i]


'''
inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
'''

'''
for i in range(nrow):
 neibALL[i,] = results[i][0]
for i in range(nrow):
    cut_neibF[i,] = results[i][1]
for i in range(nrow):
    weight_distALL[i,] = results[i][2]
del results
'''
# plt.hist(cut_neibF[cut_neibF!=0],50)

# [aFrame, neibF, cut_neibF, weight_neibF]
# training set
# targetTr = np.repeat(aFrame, r, axis=0)
targetTr = aFrame
neibF_Tr = neibALL
weight_neibF_Tr = neib_weight
sourceTr = aFrame

# Model-------------------------------------------------------------------------

SigmaTsq = Input(shape=(1,))
neib = Input(shape=(k, original_dim,))
# var_dims = Input(shape = (original_dim,))
weight_neib = Input(shape=(k,))
x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu')(x)
# h.set_weights(ae.layers[1].get_weights())
z_mean =  Dense(latent_dim, name='z_mean')(h)

encoder = Model([x, neib, SigmaTsq, weight_neib], z_mean, name='encoder')

# we instantiate these layers separately so as to reuse them later
#decoder_input = Input(shape=(latent_dim,))

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='relu')
h_decoded = decoder_h(z_mean)
x_decoded_mean = decoder_mean(h_decoded)
#decoder = Model(decoder_input, x_decoded_mean, name='decoder')

#train_z = encoder([x, neib, SigmaTsq, weight_neib])
#train_xr = decoder(train_z)
autoencoder = Model(inputs=[x, neib, SigmaTsq, weight_neib], outputs=x_decoded_mean)

# Loss and optimizer ------------------------------------------------------



normSigma = nrow / sum(1 / Sigma)
weight_neibF = np.full((nrow, k), 1/k)
def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1))), axis=-1)
    weightedN = k * original_dim * K.dot(dst,
                                         K.transpose(weight_neib))  # not really a mean square error after we done this
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 * normSigma * (1/SigmaTsq) )
    #return tf.multiply(weightedN, 0.5)

lam=1e-4
def contractive():
        W = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = encoder.get_layer('z_mean').output
        dh = h * (1 - h)  # N_batch x N_hidden
        return 1/normSigma * (SigmaTsq) * lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)
        #return lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

def custom_loss(x, x_decoded_mean, z_mean):
    msew = mean_square_error_NN(x, x_decoded_mean)
    #print('msew done', K.eval(msew))
    #mmd loss
    #loss_nll = K.mean(K.square(train_xr - x))
    #batch_size = batch_size #K.shape(train_z)[0]
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    #print('batch_size')
    #latent_dim = latent_dim
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    loss_mmd = compute_mmd(true_samples, z_mean)
    #print(K.shape(loss_mmd))
    #return msew +  1 * contractive()
    return msew + loss_mmd + 1*contractive()

loss = custom_loss(x, x_decoded_mean, z_mean)
autoencoder.add_loss(loss)
autoencoder.compile(optimizer='adam', metrics=[mean_square_error_NN, contractive,  custom_loss])
print(autoencoder.summary())
print(encoder.summary())
#print(decoder.summary())



#callbacks = [EarlyStoppingByLossVal( monitor='loss', value=0.01, verbose=0

earlyStopping=EarlyStoppingByValLoss( monitor='val_loss', min_delta=0.0001, verbose=1, patience=10, restore_best_weights=True)
#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss')
epochs = 500
history = autoencoder.fit([targetTr[2000:172600,:], neibF_Tr[2000:172600,:],  Sigma[2000:172600], weight_neibF[2000:172600,:]],
                epochs=epochs, batch_size=batch_size,
                validation_data=([targetTr[0:2000,:], neibF_Tr[0:2000,:],  Sigma[0:2000], weight_neibF[0:2000,:]], None) , #shuffle=True,
                callbacks=[CustomMetrics(), earlyStopping])#, validation_data=([targetTr, neibF_Tr,  Sigma, weight_neibF], None))
z = encoder.predict([aFrame, neibF,  Sigma, weight_neibF])

#encoder.save_weights('encoder_weightsWEBERCELLS_2D_MMD_CONTRACTIVEk30_2000epochs_LAM_0.0001.h5')

#encoder.load_weights('encoder_weightsWEBERCELLS_2D_MMD_CONTRACTIVEk30_200epochs_LAM_0.0001.h5')

#z = encoder.predict([aFrame, neibF,  Sigma, weight_neibF])

#- visualisation -----------------------------------------------------------------------------------------------

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter

# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
# x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt')

nrow = np.shape(z)[0]
# subsIdx=np.random.choice(nrow,  500000)

x = z[:, 0]
y = z[:, 1]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=1,
                    color=lbls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=clust,
                hoverinfo='text')], filename='2000epochs_LAM_0.0001.html')




# analog of tsne plot fig15 from Nowizka 2015, also see fig21
for m in range(len(markers)):
    plot([Scatter(x=x, y=y,
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=aFrame[:, m],  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(title = markers[m])
                        ),
                        text=clust,
                        hoverinfo='text'
                        )], filename='Balanced2dPerplexityk30weighted2000epochs_LAM_0.00001_' + markers[m] + '.html')


le.fit(patient_table[:,1])
pt = le.transform(patient_table[:,1])
treat = [1 if 'BCR-XL' in str else 0 for str in patient_table]
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=5,
                    color=treat,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=patient_table,
                hoverinfo='text')], filename='2000epochs_Treatment_LAM_0.001.html')


le.fit(patient_table[:,1])
pt = le.transform(patient_table[:,1])
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=2,
                    color=pt,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=patient_table[:,1],
                hoverinfo='text')], filename='2D_Patient_Treatment_LAM_0.001.html')


le.fit(patient_table[:,2])
p = le.transform(patient_table[:,2])
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=1,
                    color=p,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=patient_table[:,2],
                hoverinfo='text')], filename='2dPerplexityk30weighted200epochs_Patient_LAM_0.001.html')

le.fit(patient_table[:,1])
pt = le.transform(patient_table[:,1])
treat = [1 if 'BCR-XL' in str else 0 for str in patient_table[:,1]]
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=1,
                    color=treat,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=patient_table[:,1],
hoverinfo='text')], filename='2dPerplexityk30weighted200epochs_Treatment_LAM_0.001.html')

#neighbours in z space
centr = 75015
z_neighb = z[Idx[centr,:30], ]
plot_url = plotly.offline.plot([Scatter(x=x, y=y,
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=lbls,  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(title = markers[m])
                        ),
                        text=clust,
                        hoverinfo='text'
                        ),
                 Scatter(x=z_neighb[:,0], y=z_neighb[:,1],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='red',  # set color to an array/list of desired values
                            # choose a colorscale
                            opacity=1
                            ),
                        text=clust,
                        hoverinfo='text'
                         ),
                 Scatter(x=[z[centr, 0]], y=[z[centr, 1]],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='blue',  # set color to an array/list of desired values
                            # choose a colorscale
                            opacity=1
                            ),
                        text=clust,
                        hoverinfo='text'
                         )], filename='2dneighbPlot75001.html')




fig7, ax7 = plt.subplots()
ax7.set_title('Multiple Samples with Different sizes')
ax7.boxplot(aFrame[clust=='"surface-"',:])
plt.show()





epochs=2
learning_rate = 1e-3
earlyStopping=keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001, patience=3, verbose=0, mode='min')
adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate / epochs)
z = encoder.predict([aFrame, neibF,  Sigma, weight_neibF])

#ae.compile(optimizer=adam, loss=ae_loss)
#autoencoder.compile(optimizer=adam, loss=custom_loss, metrics=[ mean_square_error_NN])
#ae.get_weights()

checkpoint = ModelCheckpoint('.', monitor='custom_loss', verbose=1, save_best_only=True, mode='max')
#logger = DBLogger(comment="An example run")
from sklearn.model_selection import train_test_split

sourceTr, source_test, neibF_Tr, neibF_test,  SigmaTr, Sigma_test, weight_neibFTr, weight_neibF_test= \
    train_test_split(aFrame, neibF,  Sigma, weight_neibF, test_size=0.15, shuffle= True)

epochs= 200
batch_size = 88
start = timeit.default_timer()

history=autoencoder.fit([sourceTr, neibF_Tr, SigmaTr, weight_neibFTr],
epochs = epochs,
shuffle=True,
callbacks= [earlyStopping],
batch_size = batch_size)#,
#validation_data =  ([source_test, neibF_test, Sigma_test, weight_neibF_test], None))
stop = timeit.default_timer()
#autoencoder.save('WEBERCELLS2D.h5')
#autoencoder.load('WEBERCELLS2D.h5')




# vae.set_weights(trained_weight)


# from keras.utils import plot_model
# plot_model(vae, to_file='/mnt/f/Brinkman group/current/Stepan/PyCharm/PhenographLevin13/model.png')

''' this model maps an input to its reconstruction'''



# vae.save('WEBERCELLS3D32lambdaPerpCorr0.01h5')
#vae.load('WEBERCELLS3D.h5')

# ae=load_model('Wang0_modell1.h5', custom_objects={'mean_square_error_weighted':mean_square_error_weighted, 'ae_loss':
#  ae_loss, 'mean_square_error_weightedNN' : mean_square_error_weightedNN})
# ae = load_model('Wang0_modell1.h5')


fig0 = plt.figure();
plt.plot(history.history['kl_loss'][10:]);

fig02 = plt.figure();
plt.plot(history.history['mean_square_error_NN']);
fig025 = plt.figure();
plt.plot(history.history['contractive']);
fig03 = plt.figure();
plt.plot(history.history['loss']);
# predict and extract latent variables

# gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = vae.predict([sourceTr, neibF_Tr, Sigma, weight_neibF])
len(x_test_vae)
# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_vaeMoeWeights.txt', x_test_vae)
# x_test_vae=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_ae001Pert.txt.txt')
encoder = Model([x, neib,  weight_neib], z_mean)
print(encoder.summary())
x_test_enc = encoder.predict([sourceTr, neibF_Tr, weight_neibF])

cl = 4;
bw = 0.02
fig1 = plt.figure();
ax = sns.violinplot(data=x_test_vae[lbls == cl, :], bw=bw);
plt.plot(cutoff)
plt.show();
# ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
plt.plot(cutoff)
bx = sns.violinplot(data=aFrame[lbls == cl, :], bw=bw);
plt.show();

cl = 1
fig4 = plt.figure();
b0 = sns.violinplot(data=x_test_vae[lbls == cl, :], bw=bw, color='skyblue');
b0 = sns.violinplot(data=aFrame[lbls == cl, :], bw=bw, color='black');
# b0.set_xticklabels(rs[cl-1, ]);
fig5 = plt.figure();
b0 = sns.violinplot(data=x_test_vae[lbls == cl, :], bw=bw, color='skyblue');
# b0.set_xticklabels(np.round(cutoff,2));
plt.show()

unique0, counts0 = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique0, counts0)).T)
num_clus = len(counts0)

from scipy import stats

conn = [sum((stats.itemfreq(lbls[Idx[x, :k]])[:, 1] / k) ** 2) for x in range(len(aFrame))]

plt.figure();
plt.hist(conn, 50);
plt.show()

nb = find_neighbors(x_test_vae, k3, metric='manhattan', cores=12)

connClean = [sum((stats.itemfreq(lbls[nb['idx'][x, :k]])[:, 6] / k) ** 2) for x in range(len(aFrame))]
plt.figure();
plt.hist(connClean, 50);
plt.show()

scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb2 = find_neighbors(x_test_enc, k, metric='manhattan', cores=12)
connClean2 = [sum((stats.itemfreq(lbls[nb2['idx'][x, :]])[:, 1] / k) ** 2) for x in range(len(aFrame))]
plt.figure();
plt.hist(connClean2, 50);
plt.show()
# print(np.mean(connNoNoise))

for cl in unique0:
    # print(np.mean(np.array(connNoNoise)[lbls==cl]))
    print(cl)
    print(np.mean(np.array(conn)[lbls == cl]))
    print(np.mean(np.array(connClean)[lbls == cl]))
    print(np.mean(np.array(connClean)[lbls == cl]) - np.mean(np.array(conn)[lbls == cl]))
    print(np.mean(np.array(connClean2)[lbls == cl]) - np.mean(np.array(conn)[lbls == cl]))

print(np.mean(np.array([np.mean(np.array(conn)[lbls == cl]) for cl in unique0])))
print(np.mean(np.array([np.mean(np.array(connClean)[lbls == cl]) for cl in unique0])))
print(np.mean(np.array([np.mean(np.array(connClean2)[lbls == cl]) for cl in unique0])))

# plotly in 3d

