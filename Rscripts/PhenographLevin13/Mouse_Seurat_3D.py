'''This script to visualise cytoff data using deep variational autoencoder with MMD  with neighbourhood denoising and
contracting knnCVAE, neighbourhood cleaning removed
Original publication of data set: https://pubmed.ncbi.nlm.nih.gov/26095251/
data : http://127.0.0.1:27955/library/HDCytoData/doc/Examples_and_use_cases.html,
# watcg cd 4 separated into
cd4  cd7+- cd15+
compare with tsne there
TODO ?? pretraining
Now with entropic weighting of neibourhoods
Innate‐like CD8+ T‐cells and NK cells: converging functions and phenotypes
Ayako Kurioka,corresponding author 1 , 2 Paul Klenerman, 1 , 2 and Christian B. Willbergcorresponding author 1 , 2
CD8+ T Cells and NK Cells: Parallel and Complementary Soldiers of Immunotherapy
Jillian Rosenberg1 and Jun Huang1,2
'''
#import keras
import tensorflow as tf
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Dropout, BatchNormalization
#from tensorflow.keras.utils import np_utils
#import np_utils
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import random
# from keras.utils import multi_gpu_model
import timeit
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.constraints import maxnorm
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
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
# import champ
# import ig
# import metric
# import dill
from tensorflow.keras.callbacks import TensorBoard
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from GetBest import GetBest

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)


def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}




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
k3 = k * 3
ID = 'Pr_sample_008_1_Unstim_3D'
'''
data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816
'''
source_dir = '/media/grines02/New Volume1/Box Sync/Box Sync/CyTOFdataPreprocess/pregnancy'
output_dir  = '/media/grines02/New Volume1/Box Sync/Box Sync/CyTOFdataPreprocess/pregnancy/output'
'''
#data0 = np.genfromtxt(source_dir + "/Gates_PTLG008_1_Unstim.fcs.csv" , names=None, dtype=float,  delimiter=',')
data0 = pd.read_csv(source_dir + "/Gates_PTLG008_1_Unstim.fcs.csv")
gating_channels = ["CD57", "CD19", "CD4", "CD8", "IgD", "CD11c", "CD16", "CD3", "CD38", "CD27", "CD14", "CXCR5", "CCR7", "CD45RA", "CD20", "CD127", "CD33", "CD28", "CD161", "TCRgd", "CD123", "CD56", "HLADR", "CD25", "CD235ab_CD61", "CD66", "CD45", "Tbet", "CD7", "FoxP3", "CD11b"]
MarkersMap = pd.read_csv(source_dir + "/MarkersInfo.csv")
data1 = data0.rename(columns=MarkersMap.set_index('Channels')['Markers'].to_dict())
markers = ['CD235ab_CD61',         'CD45',           
                          'CD66',                   'CD7',
               'CD19',       'CD45RA',        'CD11b',          'CD4',
               'CD8a',        'CD11c',        'CD123',         'CREB',
              'STAT5',          'p38',        'TCRgd',        'STAT1',
              'STAT3',           'S6',        'CXCR3',        'CD161',
               'CD33',     'MAPKAPK2',         'Tbet',            
              'FoxP3',                 'IkB',         'CD16',
               'NFkB',          'ERK',         'CCR9',         'CD25',
                'CD3',         'CCR7',         'CD15',         'CCR2',
              'HLADR',         'CD14',         'CD56']
data2 = data1[markers]
aFrame = data2.to_numpy() 
aFrame.shape
# set negative values to zero

aFrame[aFrame < 0] = 0
lbls= np.genfromtxt(source_dir + "/Gates_PTLG008_1_Unstim.fcs_LeafPopulations.csv" , names=None, skip_header=1, delimiter=',', dtype='U100')
#randomize order
IDX = np.random.choice(aFrame.shape[0], aFrame.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
aFrame= aFrame[IDX,:]
lbls = lbls[IDX]
len(lbls)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
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
original_dim=37
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
perp((Distances[:,0:k3]),       nrow,     original_dim,   neib_weight,          nn,          k3,   Sigma,    12)
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
#outfile = source_dir + '/Nowicka2017euclid.npz'
outfile = source_dir + '/Pr_008_1_Unstim_euclid_scaled.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma, markers=markers)
'''

outfile = source_dir + '/Pr_008_1_Unstim_euclid_scaled.npz'
k=30
#markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
# np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
npzfile = np.load(outfile)
weight_distALL = npzfile['Dist'];
# = weight_distALL[IDX,:]
aFrame = npzfile['aFrame'];
Dist = npzfile['Dist']
#cut_neibF = npzfile['cut_neibF'];
#cut_neibF = cut_neibF[IDX,:]
neibALL = npzfile['neibALL']
neibALL = neibALL[:, 0:30, :]

#neibALL  = neibALL [IDX,:]
#np.sum(cut_neibF != 0)
# plt.hist(cut_neibF[cut_neibF!=0],50)
Sigma = npzfile['Sigma']
lbls = npzfile['lbls'];
neib_weight = npzfile['neib_weight']
markers = npzfile['markers']
# [aFrame, neibF, cut_neibF, weight_neibF]
# training set
# targetTr = np.repeat(aFrame, r, axis=0)
targetTr = aFrame
neibF_Tr = neibALL
weight_neibF_Tr =weight_distALL
sourceTr = aFrame

# session set up

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()


# Model-------------------------------------------------------------------------
######################################################
# targetTr = np.repeat(aFrame, r, axis=0)
targetTr = aFrame
neibF_Tr = neibALL
weight_neibF_Tr =weight_distALL
sourceTr = aFrame

# session set up

#tf.config.threading.set_inter_op_parallelism_threads(0)
#tf.config.threading.set_intra_op_parallelism_threads(0)


nrow = aFrame.shape[0]
batch_size = 256
original_dim = 37
latent_dim = 3
intermediate_dim = 120
intermediate_dim2=120
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

SigmaTsq = Input(shape=(1,))
neib = Input(shape=(k, original_dim,))
# var_dims = Input(shape = (original_dim,))

x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu', name='intermediate')(x)
#h = Dense(intermediate_dim, activation='sigmoid', name='intermediate')(x)
#h = Dense(intermediate_dim, activation='softplus', name='intermediate')(x)
# h.set_weights(ae.layers[1].get_weights())
#z_mean =  Dense(latent_dim, name='z_mean', activation='sigmoid')(h)
z_mean =  Dense(latent_dim, activation=None, name='z_mean')(h)

encoder = Model([x, neib, SigmaTsq], z_mean, name='encoder')

# we instantiate these layers separately so as to reuse them later
#decoder_input = Input(shape=(latent_dim,))

decoder_h = Dense(intermediate_dim2, activation='relu')
decoder_mean = Dense(original_dim, activation='relu')
h_decoded = decoder_h(z_mean)
x_decoded_mean = decoder_mean(h_decoded)
#decoder = Model(decoder_input, x_decoded_mean, name='decoder')

#train_z = encoder([x, neib, SigmaTsq, weight_neib])
#train_xr = decoder(train_z)
autoencoder = Model(inputs=[x, neib, SigmaTsq], outputs=x_decoded_mean)

# Loss and optimizer ------------------------------------------------------
# rewrite this based on recommendations here
# https://www.tensorflow.org/guide/keras/train_and_evaluate

def compute_kernel(x,y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):   # [batch_size, z_dim] [batch_size, z_dim]
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

normSigma = nrow / sum(1 / Sigma)
weight_neibF = np.full((nrow, k), 1/k)

neib_weight3D = np.repeat(weight_neibF[:, :, np.newaxis], original_dim, axis=2)
w_mean_target = np.average(neibALL, axis=1, weights=neib_weight3D)

def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    msew = tf.keras.losses.mean_squared_error(y_true, y_pred)
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(msew, normSigma * 1/SigmaTsq ) # TODO Sigma -denomiator or nominator? try reverse, schek hpw sigma computed in UMAP
    #return tf.multiply(weightedN, 0.5)

lam=1e-4
def contractive(x, x_decoded_mean):
        W = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        m = encoder.get_layer('z_mean').output
        dm = m * (1 - m)  # N_batch x N_hidden
        return 1/normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
import sys


def deep_contractive(x, x_decoded_mean):
       W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
       Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
       W = K.transpose(W);  Z = K.transpose(Z); # N_hidden x N
       m = encoder.get_layer('intermediate').output
       dm = tf.linalg.diag(m * (1 - m))   # N_batch x N_hidden
       s = encoder.get_layer('z_mean').output
       ds = tf.linalg.diag(s * (1 - s))  # N_batch x N_hidden
       #tf.print(ds.shape)
       #return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
       S_1W = tf.einsum('akl,lj->akj', dm, W ) # N_batch x N_input ??
       #tf.print((S_1W).shape) #[None, 120]
       S_2Z = tf.einsum('akl,lj->akj', ds, Z ) # N_batch ?? TODO: use tf. and einsum and/or tile
       #tf.print((S_2Z).shape)
       diff_tens = tf.einsum('akl,alj->akj' , S_2Z, S_1W ) # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
       #tf.Print(K.sum(diff_tens ** 2))
       return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)


def deep_contractive(x, x_decoded_mean): # deep contractive with softplus in intermediate layer
       W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
       Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
       W = K.transpose(W);  Z = K.transpose(Z); # N_hidden x N
       m = encoder.get_layer('intermediate').output
       dm = tf.linalg.diag(1 - 1/(tf.math.exp(m)))   # N_batch x N_hidden
       s = encoder.get_layer('z_mean').output
       ds = tf.linalg.diag(s * (1 - s))  # N_batch x N_hidden
       #tf.print(ds.shape)
       #return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
       S_1W = tf.einsum('akl,lj->akj', dm, W ) # N_batch x N_input ??
       #tf.print((S_1W).shape) #[None, 120]
       S_2Z = tf.einsum('akl,lj->akj', ds, Z ) # N_batch ?? TODO: use tf. and einsum and/or tile
       #tf.print((S_2Z).shape)
       diff_tens = tf.einsum('akl,alj->akj' , S_2Z, S_1W ) # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
       #tf.Print(K.sum(diff_tens ** 2))
       return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)


def deep_contractive(x, x_decoded_mean):  # deep contractive with ReLu in intermediate layer
    W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
    Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N
    m = encoder.get_layer('intermediate').output
    dm = tf.linalg.diag((tf.math.sign(m)+1)/2)  # N_batch x N_hidden
    s = encoder.get_layer('z_mean').output
    ds = tf.linalg.diag(s * (1 - s))  # N_batch x N_hidden
    # tf.print(ds.shape)
    # return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
    S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
    # tf.print((S_1W).shape) #[None, 120]
    S_2Z = tf.einsum('akl,lj->akj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z,
                          S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)


def deep_contractive(x, x_decoded_mean):  # deep contractive with ReLu in all layersd
    W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
    Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N
    m = encoder.get_layer('intermediate').output
    dm = tf.linalg.diag((tf.math.sign(m)+1)/2)  # N_batch x N_hidden
    s = encoder.get_layer('z_mean').output
    ds = tf.linalg.diag((tf.math.sign(s)+1)/2)  # N_batch x N_hidden
    # tf.print(ds.shape)
    # return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
    S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
    # tf.print((S_1W).shape) #[None, 120]
    S_2Z = tf.einsum('akl,lj->akj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z,
                          S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)

def deep_contractive(x, x_decoded_mean):  # attempt to avoid vanishing derivative of sigmoid
    W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
    Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N
    m = encoder.get_layer('intermediate').output
    dm = tf.linalg.diag((tf.math.sign(m)+1)/2)  # N_batch x N_hidden
    s = encoder.get_layer('z_mean').output
    #ds = K.sqrt(tf.linalg.diag(s * (1 - s)))  # N_batch x N_hidden
    #ds = tf.linalg.diag(tf.math.sign(s*(1-s) ** 2 +0.1))
    #bs = np.shape(s)[0]
    #tf.print(bs)  # [None, 120]
    r = tf.linalg.einsum('aj->a', s**2)
    #tf.print(r.shape)
    #b_i = tf.eye(latent_dim)
    #tf.print(tf.shape(b_i))
    #ds = tf.einsum('alk,a ->alk', b_i,r)
    ds  = -2 * r + 1.5*r **2 + 1.5
    #tf.print(ds.shape)
    # return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axi0s=1)
    S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
    # tf.print((S_1W).shape) #[None, 120]
    S_2Z = tf.einsum('a,lj->alj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z,
                          S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam *(K.sum(diff_tens ** 2))






#contractive = deep_contractive


        #return lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)

def loss_mmd(x, x_decoded_mean):
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return compute_mmd(true_samples, z_mean)
nn=30
def custom_loss(x, x_decoded_mean):
    msew = mean_square_error_NN(x, x_decoded_mean)
    #
    return 1*msew + 1*loss_mmd(x, x_decoded_mean) + 1*deep_contractive(x, x_decoded_mean)
################################################################################################
#################################################################################################
#loss = custom_loss(x, x_decoded_mean)
#autoencoder.add_loss(loss)
autoencoder.compile(optimizer='adam', loss=custom_loss)
print(autoencoder.summary())
print(encoder.summary())
#print(decoder.summary())



#callbacks = [EarlyStoppingByLossVal( monitor='loss', value=0.01, verbose=0

#earlyStopping=EarlyStoppingByValLoss( monitor='val_loss', min_delta=0.0001, verbose=1, patience=10, restore_best_weights=True)
#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss')
epochs = 1000
batch_size=256
#history = autoencoder.fit([targetTr, neibF_Tr,  Sigma],
history = autoencoder.fit([targetTr, neibF_Tr[:,0:30,:],  Sigma], targetTr,
#history = autoencoder.fit([targetTr, neibF_Tr,  Sigma],w_mean_target,
                epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
                #validation_data=([targetTr[0:2000,:], neibF_Tr[0:2000,:],  Sigma[0:2000], weight_neibF[0:2000,:]], None),
                          # shuffle=True)
                #callbacks=[CustomMetrics()], verbose=2)#, validation_data=([targetTr, neibF_Tr,  Sigma, weight_neibF], None))
z = encoder.predict([aFrame, neibALL,  Sigma])
plt.plot(history.history['loss'][5:])
#plt.plot(history.history['val_loss'][5:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
#encoder.save_weights('encoder_weightsWEBERCELLS_2D_MMD_CONTRACTIVEk30_2000epochs_LAM_0.0001.h5')

#encoder.load_weights('encoder_weightsWEBERCELLS_2D_MMD_CONTRACTIVEk30_200epochs_LAM_0.0001.h5')

#z = encoder.predict([aFrame, neibF,  Sigma, weight_neibF])

#- visualisation -----------------------------------------------------------------------------------------------

import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter
import plotly.graph_objects as go
# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
# x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt')

nrow = np.shape(z)[0]
# subsIdx=np.random.choice(nrow,  500000)
num_lbls = (np.unique(lbls, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
zz = z[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
fig = go.Figure()
fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                mode='markers',
                marker=dict(
                    size=1,
                    color=num_lbls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=lbls,
                #hoverinfo='text')], filename='tmp.html')
                hoverinfo='text'))

fig.show()
plotly.offline.plot(fig, filename=ID + 'no_knn_denoising_HAT_potential_in CAE_MMD_1_DCAE_1_scaled.html')

# plot the same with clicable cluster colours
nrow = np.shape(z)[0]
# subsIdx=np.random.choice(nrow,  500000)
num_lbls = (np.unique(lbls, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
zz = z[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
fig = go.Figure()
lbls_list = np.unique(lbls)
nM=len(np.unique(lbls))
from matplotlib.colors import rgb2hex
import seaborn as sns
palette = sns.color_palette(None, nM)
colors = np.array([ rgb2hex(palette[i]) for i in range(len(palette)) ])

fig = go.Figure()
for m in range(nM):
    IDX = [x == lbls_list[m] for x in lbls]
    xs = x[IDX]; ys = y[IDX]; zs = zz[IDX];
    fig.add_trace(Scatter3d(x=xs, y=ys, z =zs,
                name = lbls_list[m],
                mode='markers',
                marker=dict(
                    size=1,
                    color=colors[m],  # set color to an array/list of desired values
                    opacity=0.5,
                ),
                text=lbls[IDX],
                #hoverinfo='text')], filename='tmp.html')
                hoverinfo='text'))
    fig.update_layout(yaxis=dict(range=[-3,3]))

vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

button_list = list([dict(label = np.unique(lbls)[m],
                    method = 'update',
                    args = [{'visible': vis_mat[m,:]},
                          {'title': np.unique(lbls)[m],
                           'showlegend':False}]) for m in range(len(np.unique(lbls)))])
fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=button_list
        )
    ])


fig.show()
plotly.offline.plot(fig, filename=ID + 'no_knn_denoising_knHAT_potential_in CAE_MMD_1_DCAE_5_scaledButtons.html')




# just the most important markers
#marker_sub=['CD45RA', 'CD19', 'CD22',
#            'CD11b', 'CD4','CD8', 'CD34', 'CD20',
#            'CXCR4', 'CD45','CD123', 'CD321',
#            'CD33', 'CD47','CD11c',
#            'CD7', 'CD15', 'CD16','CD44', 'CD38',
#            'CD3', 'CD61','CD117', 'CD49d',
#            'HLA-DR', 'CD64','CD41'
#            ] #
marker_sub=list(markers)
sub_idx =  np.random.choice(range(len(lbls)), 10000, replace  =False)
x = z[sub_idx, 0]
y = z[sub_idx, 1]
zz = z[sub_idx, 2]
lbls_s = lbls[sub_idx]
sFrame = aFrame[sub_idx,:]
result = [list(markers).index(i) for i in marker_sub]
sFrame = sFrame[:, result]

fig = go.Figure()
nM=len(marker_sub)
for m in range(nM):
    fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=sFrame[:,m],  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(title = marker_sub[m])
                        ),
                        text=lbls_s,
                        hoverinfo='text',
                        name = marker_sub[m]
                        ))


vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

button_list = list([dict(label = marker_sub[m],
                    method = 'update',
                    args = [{'visible': vis_mat[m,:]},
                          {'title': marker_sub[m],
                           'showlegend':False}]) for m in range(len(marker_sub))])
fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=button_list
        )
    ])
#TODO: add color by cluster

fig.show()
plotly.offline.plot(fig, filename=ID + 'no_knn_denoising_HAt_potential_DCAE_deep_MMD1_DCAE_5topMarkers.html')


# process a subset in limits
xlim = [-0.75, -0.25]
ylim = [-0.75, -0.35]
rows = np.logical_and( np.logical_and(xlim[0] <= z[:,0] , xlim[1] >= z[:,0]) , np.logical_and(ylim[0] <= z[:,1], ylim[1] >= z[:,1]) )
z_capture = z[rows, :]
z_capture.shape

# Model-------------------------------------------------------------------------
######################################################
# targetTr = np.repeat(aFrame, r, axis=0)
targetTr_cap = targetTr[rows,:]
neibF_Tr_cap=neibF_Tr[rows,:,:]
Sigma_cap = Sigma[rows]

# session set up

#tf.config.threading.set_inter_op_parallelism_threads(0)
#tf.config.threading.set_intra_op_parallelism_threads(0)


nrow = targetTr_cap.shape[0]
batch_size = 256
original_dim = 32
latent_dim = 2
intermediate_dim = 120
intermediate_dim2=120
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

SigmaTsq = Input(shape=(1,))
neib = Input(shape=(k, original_dim,))
x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu', name='intermediate')(x)
#h = Dense(intermediate_dim, activation='sigmoid', name='intermediate')(x)
#h = Dense(intermediate_dim, activation='softplus', name='intermediate')(x)
# h.set_weights(ae.layers[1].get_weights())
z_mean =  Dense(latent_dim, name='z_mean')(h)

encoder = Model([x, neib, SigmaTsq], z_mean, name='encoder')

# we instantiate these layers separately so as to reuse them later
#decoder_input = Input(shape=(latent_dim,))

decoder_h = Dense(intermediate_dim2, activation='relu')
decoder_mean = Dense(original_dim, activation='relu')
h_decoded = decoder_h(z_mean)
x_decoded_mean = decoder_mean(h_decoded)
#decoder = Model(decoder_input, x_decoded_mean, name='decoder')

#train_z = encoder([x, neib, SigmaTsq, weight_neib])
#train_xr = decoder(train_z)
autoencoder = Model(inputs=[x, neib, SigmaTsq], outputs=x_decoded_mean)

# Loss and optimizer ------------------------------------------------------
# rewrite this based on recommendations here
# https://www.tensorflow.org/guide/keras/train_and_evaluate

def compute_kernel(x,y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):   # [batch_size, z_dim] [batch_size, z_dim]
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

normSigma = nrow / sum(1 / Sigma_cap)
weight_neibF = np.full((nrow, k), 1/k)

neib_weight3D = np.repeat(weight_neibF[:, :, np.newaxis], original_dim, axis=2)
w_mean_target = np.average(neibF_Tr_cap, axis=1, weights=neib_weight3D)

def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    msew = tf.keras.losses.mean_squared_error(y_true, y_pred)
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(msew, normSigma * 1/SigmaTsq ) # TODO Sigma -denomiator or nominator? try reverse, schek hpw sigma computed in UMAP
    #return tf.multiply(weightedN, 0.5)

lam=1e-4
def contractive(x, x_decoded_mean):
        W = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        m = encoder.get_layer('z_mean').output
        dm = m * (1 - m)  # N_batch x N_hidden
        return 1/normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
import sys


def deep_contractive(x, x_decoded_mean):
       W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
       Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
       W = K.transpose(W);  Z = K.transpose(Z); # N_hidden x N
       m = encoder.get_layer('intermediate').output
       dm = tf.linalg.diag(m * (1 - m))   # N_batch x N_hidden
       s = encoder.get_layer('z_mean').output
       ds = tf.linalg.diag(s * (1 - s))  # N_batch x N_hidden
       #tf.print(ds.shape)
       #return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
       S_1W = tf.einsum('akl,lj->akj', dm, W ) # N_batch x N_input ??
       #tf.print((S_1W).shape) #[None, 120]
       S_2Z = tf.einsum('akl,lj->akj', ds, Z ) # N_batch ?? TODO: use tf. and einsum and/or tile
       #tf.print((S_2Z).shape)
       diff_tens = tf.einsum('akl,alj->akj' , S_2Z, S_1W ) # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
       #tf.Print(K.sum(diff_tens ** 2))
       return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)


def deep_contractive(x, x_decoded_mean): # deep contractive with softplus in intermediate layer
       W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
       Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
       W = K.transpose(W);  Z = K.transpose(Z); # N_hidden x N
       m = encoder.get_layer('intermediate').output
       dm = tf.linalg.diag(1 - 1/(tf.math.exp(m)))   # N_batch x N_hidden
       s = encoder.get_layer('z_mean').output
       ds = tf.linalg.diag(s * (1 - s))  # N_batch x N_hidden
       #tf.print(ds.shape)
       #return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
       S_1W = tf.einsum('akl,lj->akj', dm, W ) # N_batch x N_input ??
       #tf.print((S_1W).shape) #[None, 120]
       S_2Z = tf.einsum('akl,lj->akj', ds, Z ) # N_batch ?? TODO: use tf. and einsum and/or tile
       #tf.print((S_2Z).shape)
       diff_tens = tf.einsum('akl,alj->akj' , S_2Z, S_1W ) # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
       #tf.Print(K.sum(diff_tens ** 2))
       return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)


def deep_contractive(x, x_decoded_mean):  # deep contractive with ReLu in intermediate layer
    W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
    Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N
    m = encoder.get_layer('intermediate').output
    dm = tf.linalg.diag((tf.math.sign(m)+1)/2)  # N_batch x N_hidden
    s = encoder.get_layer('z_mean').output
    ds = tf.linalg.diag(s * (1 - s))  # N_batch x N_hidden
    # tf.print(ds.shape)
    # return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)
    S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
    # tf.print((S_1W).shape) #[None, 120]
    S_2Z = tf.einsum('akl,lj->akj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z,
                          S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)




#contractive = deep_contractive


        #return lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axis=1)

def loss_mmd(x, x_decoded_mean):
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return compute_mmd(true_samples, z_mean)
nn=30
def custom_loss(x, x_decoded_mean, z_mean):
    msew = mean_square_error_NN(x, x_decoded_mean)
    #
    return 1*msew + 1*loss_mmd(x, x_decoded_mean) + 0.1*deep_contractive(x, x_decoded_mean)
################################################################################################
#################################################################################################
loss = custom_loss(x, x_decoded_mean, z_mean)
autoencoder.add_loss(loss)
autoencoder.compile(optimizer='adam', metrics=[mean_square_error_NN, deep_contractive, custom_loss, loss_mmd])
print(autoencoder.summary())
print(encoder.summary())
#print(decoder.summary())



#callbacks = [EarlyStoppingByLossVal( monitor='loss', value=0.01, verbose=0

earlyStopping=EarlyStoppingByValLoss( monitor='val_loss', min_delta=0.0001, verbose=1, patience=10, restore_best_weights=True)
#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss')
epochs = 1300
batch_size=256
#history = autoencoder.fit([targetTr, neibF_Tr,  Sigma],
history = autoencoder.fit([targetTr_cap[:,:], neibF_Tr_cap[:,:],  Sigma_cap],
                epochs=epochs, batch_size=batch_size, verbose=1,
                #validation_data=([targetTr[0:2000,:], neibF_Tr[0:2000,:],  Sigma[0:2000], weight_neibF[0:2000,:]], None),
                           shuffle=True)
                #callbacks=[CustomMetrics()], verbose=2)#, validation_data=([targetTr, neibF_Tr,  Sigma, weight_neibF], None))
z = encoder.predict([targetTr_cap, neibF_Tr_cap,  Sigma_cap])
plt.plot(history.history['loss'][5:])
#plt.plot(history.history['val_loss'][5:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
#encoder.save_weights('encoder_weightsWEBERCELLS_2D_MMD_CONTRACTIVEk30_2000epochs_LAM_0.0001.h5')

#encoder.load_weights('encoder_weightsWEBERCELLS_2D_MMD_CONTRACTIVEk30_200epochs_LAM_0.0001.h5')

#z = encoder.predict([aFrame, neibF,  Sigma, weight_neibF])

#- visualisation -----------------------------------------------------------------------------------------------

import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter

# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
# x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt')

nrow = np.shape(z)[0]
# subsIdx=np.random.choice(nrow,  500000)
lbls_cap = lbls[rows]
num_lbls_cap = (np.unique(lbls_cap, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=1,
                    color=num_lbls_cap,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=lbls_cap,
                #hoverinfo='text')], filename='tmp.html')
                hoverinfo='text')], filename=ID + 'Subset_MMD_1_DCAE_0.1_scaled.html')



x = z[:, 0]
y = z[:, 1]
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
                        text=lbls,
                        hoverinfo='text'
                        )], filename='2dPerplexityk30weighted2000epochs_LAM_0.001_' + markers[m] + '.html')

# first attempt to plot with buttons
#def plot_data(df):

sub_idx =  np.random.choice(range(len(lbls)), 10000, replace  =False)
x = z[sub_idx, 0]
y = z[sub_idx, 1]
lbls_s = lbls[sub_idx]
sFrame = aFrame[sub_idx,:]

import plotly.graph_objects as go
fig = go.Figure()
nM=len(markers)
for m in range(nM):
    fig.add_trace(Scatter(x=x, y=y,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=sFrame[:,m],  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(title = markers[m])
                        ),
                        text=lbls_s,
                        hoverinfo='text',
                        name = markers[m]
                        ))


vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

button_list = list([dict(label = markers[m],
                    method = 'update',
                    args = [{'visible': vis_mat[m,:]},
                          {'title': markers[m],
                           'showlegend':False}]) for m in range(len(markers))])
fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=button_list
        )
    ])
#TODO: add color by cluster

fig.show()

# just the most important markers
marker_sub=['CD45RA', 'CD19', 'CD22',
            'CD11b', 'CD4','CD8', 'CD34', 'CD20',
            'CXCR4', 'CD45','CD123', 'CD321',
            'CD33', 'CD47','CD11c',
            'CD7', 'CD15', 'CD16','CD44', 'CD38',
            'CD3', 'CD61','CD117', 'CD49d',
            'HLA-DR', 'CD64','CD41'
            ] #
#marker_sub=markers
sub_idx =  np.random.choice(range(len(Sigma_cap)), 10000, replace  =False)
x = z[sub_idx, 0]
y = z[sub_idx, 1]
lbls_s = lbls_cap[sub_idx]
sFrame = targetTr_cap[sub_idx,:]
result = [markers.index(i) for i in marker_sub]
sFrame = sFrame[:, result]
import plotly.graph_objects as go
fig = go.Figure()
nM=len(marker_sub)
for m in range(nM):
    fig.add_trace(Scatter(x=x, y=y,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=sFrame[:,m],  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(title = marker_sub[m])
                        ),
                        text=lbls_s,
                        hoverinfo='text',
                        name = marker_sub[m]
                        ))


vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

button_list = list([dict(label = marker_sub[m],
                    method = 'update',
                    args = [{'visible': vis_mat[m,:]},
                          {'title': marker_sub[m],
                           'showlegend':False}]) for m in range(len(marker_sub))])
fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=button_list
        )
    ])
#TODO: add color by cluster

fig.show()
plotly.offline.plot(fig, filename=ID + 'Subset_MMD1_DCAE_0.1topMarkers.html')








