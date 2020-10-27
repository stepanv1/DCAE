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
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
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
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter
import plotly.graph_objects as go
import umap.umap_ as umap
import hdbscan
import phenograph

#/home/grines02/PycharmProjects/BIOIBFO25L/Rscripts/PhenographLevin13/utils_evaluation.py

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)


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

#ae_neib_compare  =   compare_neighbours(Idx, ae_neib, kmax=90)
#ae_neib_compare
#plt.plot(ae_neib_compare)

# computes cluster performance measures
lblsP = [str(x) for x in labelsHDBscanAttributes]
lblsT = lbls

print(compute_cluster_performance(lbls, labelsHDBscanAttributes))
print(compute_cluster_performance(lbls[lbls!='"unassigned"'], np.asarray(labelsHDBscanAttributes)[lbls!='"unassigned"'] ))

print(compute_cluster_performance(lbls, communities))
print(compute_cluster_performance(lbls[lbls!='"unassigned"'], communities[lbls!='"unassigned"']))

metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in labels]),:], np.array(labels)[np.array([i!= '-1.00' for i in labels])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in labels]),:], np.array(labels)[np.array([i!= '-1.00' for i in labels])])
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
# define metrics matrixs contrasts
#supervided
print(compute_cluster_performance(lbls, labelsUMAP))
print(compute_cluster_performance(lbls, labels))
# unsupervised
# gated:
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
# AE
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in labels]),:], np.array(labels)[np.array([i!= '-1.00' for i in labels])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in labels]),:], np.array(labels)[np.array([i!= '-1.00' for i in labels])])
# UMAP
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in labelsUMAP]),:], np.array(labelsUMAP)[np.array([i!= '-1.00' for i in labelsUMAP])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in labelsUMAP]),:], np.array(labelsUMAP)[np.array([i!= '-1.00' for i in labelsUMAP])])




########################## do umap with precomuted distances

mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(aFrame)
embedUMAP =  mapper.transform(aFrame)

clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=25, alpha=0.5, cluster_selection_method = 'leaf')

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(aFrame)
attrib_z=np.c_[z, principalComponents/10]

attrib_z=np.c_[z, np.array(aFrame)/10]
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=25, alpha=1.0, cluster_selection_method = 'leaf')
labelsHDBscanAttributes = clusterer.fit_predict(attrib_z)
#labelsHDBscanAttributes = clusterer.fit_predict(aFrame)
table(labelsHDBscanAttributes)
labelsHDBscanAttributes = [str(x) for  x in labelsHDBscanAttributes]
print(compute_cluster_performance(lbls, labelsHDBscanAttributes))
print(compute_cluster_performance(lbls[lbls!='"unassigned"'], np.array(labelsHDBscanAttributes)[lbls!='"unassigned"']))

# unsupervised
# gated:
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
# Attributed
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in labelsHDBscanAttributes]),:], np.array(labels)[np.array([i!= '-1.00' for i in labelsHDBscanAttributes])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in labelsHDBscanAttributes]),:], np.array(labelsHDBscanAttributes)[np.array([i!= '-1.00' for i in labelsHDBscanAttributes])])

num_lbls = (np.unique(labelsHDBscanAttributes, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
zz = z[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
labelsHDBscanAttributes=["%.2f" % x for x in labelsHDBscanAttributes]
lbls_list = np.unique(labelsHDBscanAttributes)
nM=len(np.unique(labelsHDBscanAttributes))
from matplotlib.colors import rgb2hex
import seaborn as sns
palette = sns.color_palette(None, nM)
colors = np.array([ rgb2hex(palette[i]) for i in range(len(palette)) ])

fig = go.Figure()
for m in range(nM):
    IDX = [x == lbls_list[m] for x in labelsHDBscanAttributes]
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
    fig.update_layout()

vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        updatemenus=[go.layout.Updatemenu(
        active=0,
        )
    ])
fig.show()

# do attributed clusteringn UMAP embedding
attrib_UMAP=np.c_[embedUMAP, np.array(aFrame)/8]
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=25, alpha=1.0, cluster_selection_method = 'leaf')
labelsHDBscanUMAPAttributes = clusterer.fit_predict(attrib_UMAP)
#labelsHDBscanAttributes = clusterer.fit_predict(aFrame)
table(labelsHDBscanUMAPAttributes)
table(communities)
print(compute_cluster_performance(lbls, labelsHDBscanUMAPAttributes))

# unsupervised
# gated:
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in lbls]),:], np.array(lbls)[np.array([i!= '-1.00' for i in lbls])])
# Attributed
metrics.calinski_harabasz_score(X[np.array([i!= '-1.00' for i in labelsHDBscanUMAPAttributes]),:], np.array(labelsHDBscanUMAPAttributes)[np.array([i!= '-1.00' for i in labelsHDBscanUMAPAttributes])])
metrics.calinski_harabasz_score(z[np.array([i!= '-1.00' for i in labelsHDBscanUMAPAttributes]),:], np.array(labelsHDBscanUMAPAttributes)[np.array([i!= '-1.00' for i in labelsHDBscanUMAPAttributes])])


# phenograph
communities, graph, Q = phenograph.cluster(aFrame)
print(compute_cluster_performance(lbls, communities))
print(compute_cluster_performance(lbls[lbls!='"unassigned"'], communities[lbls!='"unassigned"']))




num_lbls = (np.unique(lbls, return_inverse=True)[1])
from matplotlib.colors import rgb2hex
import seaborn as sns
nM=len(np.unique(lbls))
cluster_names=np.unique(lbls)
palette = sns.color_palette(None, nM)
colors = np.array([ rgb2hex(palette[i]) for i in range(len(palette)) ])

fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*embedUMAP.T,  c=num_lbls, cmap='Spectral', alpha=0.5, s=0.1)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(nM+1)-0.5)
cbar.set_ticks(np.arange(nM))
cbar.set_ticklabels(cluster_names)
plt.title('Levine32 Embedded via UMAP');

umap.plot.points(mapper, labels=lbls, color_key_cmap='Paired', background='black')

umap_neib = find_neighbors(embedUMAP, k_=90, metric='euclidean', cores=12)['idx']
umap_neib_compare  =   compare_neighbours(Idx, umap_neib, kmax=90)
ae_neib_compare
plt.plot(umap_neib_compare)

# load data
k = 30
k3 = k * 3
coeffCAE = 5
epochs = 400
ID = 'Levine32_MMD_01_3D_DCAE_'+ str(coeffCAE) + '_' + str(epochs) + '_kernelInit'
'''
data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816
'''
source_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess'
output_dir  = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/'

'''
data0 = np.genfromtxt(source_dir + "/Levine32_data.csv" , names=None, dtype=float, skip_header=1, delimiter=',')
aFrame = data0[:,:] 
aFrame.shape
# set negative values to zero
aFrame[aFrame < 0] = 0
lbls= np.genfromtxt(source_dir + "/Levine32_population.csv" , names=None, skip_header=0, delimiter=',', dtype='U100')
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
original_dim=32
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
outfile = source_dir + '/Levine32euclid_scaled.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)
'''

outfile = source_dir + '/Levine32euclid_scaled.npz'

markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
# np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
npzfile = np.load(outfile)
weight_distALL = npzfile['Dist'];
# = weight_distALL[IDX,:]
aFrame = npzfile['aFrame'];
Dist = npzfile['Dist']
#cut_neibF = npzfile['cut_neibF'];
#cut_neibF = cut_neibF[IDX,:]
neibALL = npzfile['neibALL']
#neibALL  = neibALL [IDX,:]
#np.sum(cut_neibF != 0)
# plt.hist(cut_neibF[cut_neibF!=0],50)
Sigma = npzfile['Sigma']
lbls = npzfile['lbls'];
neib_weight = npzfile['neib_weight']
# [aFrame, neibF, cut_neibF, weight_neibF]
# training set
# targetTr = np.repeat(aFrame, r, axis=0)
targetTr = aFrame
neibF_Tr = neibALL
weight_neibF_Tr =weight_distALL
sourceTr = aFrame
Idx = npzfile['Idx']

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


nrow = aFrame.shape[0]
batch_size = 256
original_dim = 32
latent_dim = 3
intermediate_dim = 120
intermediate_dim2=120
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

SigmaTsq = Input(shape=(1,))
neib = Input(shape=(k, original_dim,))
# var_dims = Input(shape = (original_dim,))
#
initializer = tf.keras.initializers.he_normal(12345)
x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer = initializer)(x)
z_mean =  Dense(latent_dim, activation=None, name='z_mean')(h)

encoder = Model([x, neib, SigmaTsq], z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='relu', name='intermediate2', kernel_initializer = initializer)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer = initializer)
h_decoded = decoder_h(z_mean)
x_decoded_mean = decoder_mean(h_decoded)
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
    return 1*msew + 1*loss_mmd(x, x_decoded_mean) + coeffCAE*deep_contractive(x, x_decoded_mean)
    #eturn 1 * msew +  coeffCAE * deep_contractive(x, x_decoded_mean)
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
epochs = epochs
batch_size=256
#history = autoencoder.fit([targetTr, neibF_Tr,  Sigma],
history = autoencoder.fit([targetTr, neibF_Tr,  Sigma],targetTr,
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
#encoder.save_weights(output_dir +'/'+ID + '_3D.h5')
#autoencoder.save_weights(output_dir +'/autoencoder_'+ID + '_3D.h5')
#np.savez(output_dir +'/'+ ID + '_latent_rep_3D.npz', z = z)

#ID='Levine32_MMD_1_3D_DCAE_5'
#encoder.load_weights('/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/Levine32_3D_DCAE_10_3D.h5')
#autoencoder.load_weights('/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/autoencoder_Levine32_MMD_1_3D_DCAE_freezing_experiment5_3D.h5')
encoder.load_weights(output_dir +'/'+ID + '_3D.h5')
autoencoder.load_weights(output_dir +'autoencoder_'+ID + '_3D.h5')

z = encoder.predict([aFrame, neibF_Tr,  Sigma, weight_neibF])

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


# plot the same with clicable cluster colours
nrow = np.shape(z)[0]
# subsIdx=np.random.choice(nrow,  500000)
num_lbls = (np.unique(lbls, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
zz = z[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21

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

from utils_evaluation import plot3D_cluster_colors
#reload(utils_evaluation)
fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=lbls)
fig.show()
html_str=plotly.io.to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "no_knn_denoising_knHAT_potential_inCAE_MMD_1_scaledButtons.html","w")
Html_file.write(html_str)
Html_file.close()




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
sub_idx =  np.random.choice(range(len(lbls)), 50000, replace  =False)
x = z[sub_idx, 0]
y = z[sub_idx, 1]
zz = z[sub_idx, 2]
lbls_s = lbls[sub_idx]
sFrame = aFrame[sub_idx,:]
result = [markers.index(i) for i in marker_sub]
sFrame = sFrame[:, result]

nM=len(marker_sub)
m=0
fig = go.Figure()
fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                        mode='markers',
                        marker=dict(
                            size=0.5,
                            color=sFrame[:,m],  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(xanchor = 'left', x=-0.05, len=0.5),
                            showscale = True
                        ),
                        text=lbls_s,
                        hoverinfo='text',
                        ))
for m in range(1,nM):
#for m in range(1,3):
    fig.add_trace(Scatter3d(x=x, y=y, z=zz,
                        mode='markers',
                        visible="legendonly",
                        marker=dict(
                            size=0.5,
                            color=sFrame[:,m],  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(xanchor = 'left', x=-0.05, len=0.5),
                            showscale = True
                        ),
                        text=lbls_s,
                        hoverinfo='text'
                        ))


vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

button_list = list([dict(label = marker_sub[m],
                    method = 'update',
                    args = [{'visible': vis_mat[m,:]},
                         # {'title': marker_sub[m],
                            {'showlegend':False}]) for m in range(len(marker_sub))])
fig.update_layout(
        showlegend=False,
        updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=button_list
        )
    ])
#TODO: add color by cluster
fig.show()
html_str=plotly.io.to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + 'no_knn_denoising_HAt_potential_DCAE_deep_MMD1_DCAE_10topMarkers.html',"w")
Html_file.write(html_str)
Html_file.close()

# hdbscan on z
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=25, alpha=0.5, cluster_selection_method = 'leaf')
labels = clusterer.fit_predict(z)
table(labels)
labels=["%.2f" % x for x in labels]
nrow = np.shape(z)[0]
# subsIdx=np.random.choice(nrow,  500000)
num_lbls = (np.unique(labels, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
zz = z[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
fig = go.Figure()
lbls_list = np.unique(labels)
nM=len(np.unique(labels))
from matplotlib.colors import rgb2hex
import seaborn as sns
palette = sns.color_palette(None, nM)
colors = np.array([ rgb2hex(palette[i]) for i in range(len(palette)) ])

fig = go.Figure()
for m in range(nM):
    IDX = [x == lbls_list[m] for x in labels]
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
    fig.update_layout()

vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        updatemenus=[go.layout.Updatemenu(
        active=0,
        )
    ])
fig.show()

# project z on 2d sphere
def projZ(x):
    r=np.sqrt(np.sum(x**2))
    return(x/r)

zR = np.apply_along_axis(projZ, 1, z)
clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf')
labels = clusterer.fit_predict(zR)
table(labels)
labels=["%.2f" % x for x in labels]
nrow = np.shape(zR)[0]
# subsIdx=np.random.choice(nrow,  500000)
num_lbls = (np.unique(labels, return_inverse=True)[1])
x = zR[:, 0]
y = zR[:, 1]
zz = zR[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
fig = go.Figure()
lbls_list = np.unique(labels)
nM=len(np.unique(labels))
from matplotlib.colors import rgb2hex
import seaborn as sns
palette = sns.color_palette(None, nM)
colors = np.array([ rgb2hex(palette[i]) for i in range(len(palette)) ])

fig = go.Figure()
for m in range(nM):
    IDX = [x == lbls_list[m] for x in labels]
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
    fig.update_layout()

vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        updatemenus=[go.layout.Updatemenu(
        active=0,
        )
    ])
fig.show()

###########################################################################################3
##### clustering using hidden layer

x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu', name='intermediate')(x)
encoder2 = Model([x], h, name='encoder2')
encoder2.layers[1].set_weights(encoder.layers[1].get_weights()) #= encoder.layers[1].get_weights()
activations0 = encoder2.predict([aFrame])
#drop dead neurons:
import prettyplotlib as ppl
df = pd.DataFrame(activations0)
fig, ax = plt.subplots()
df.boxplot(fontsize=3  )
plt.show()


from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
encoder2 = delete_channels(encoder2, encoder2.layers[1], np.arange(intermediate_dim)[np.max(activations0,axis=0) < 0.05])
activationsPruned = encoder2.predict([aFrame])
df = pd.DataFrame(activationsPruned)
fig, ax = plt.subplots()
df.boxplot()
plt.show()

encoderPruned = delete_channels(encoder, encoder.layers[1], np.arange(intermediate_dim)[np.max(activations0,axis=0) < 0.05])
zPr = encoderPruned.predict([aFrame, neibF_Tr,  Sigma, weight_neibF])

num_lbls = (np.unique(lbls, return_inverse=True)[1])
x = zPr[:, 0]
y = zPr[:, 1]
zz = zPr[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21

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
fig.show()
html_str=plotly.io.to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "no_knn_denoising_knHAT_potential_inCAE_MMD_1_scaledButtons.html","w")
Html_file.write(html_str)
Html_file.close()

activations = activations0
#drop dead neurons:
import prettyplotlib as ppl
df = pd.DataFrame(activations)
fig, ax = plt.subplots()
df.boxplot()
plt.show()

# prune autoencoder of last layer
autoencoderPruned = delete_layer(autoencoder, autoencoder.layers[6])
autoencoderPruned.summary()
act_h2 = autoencoderPruned.predict([aFrame, neibF_Tr,  Sigma, weight_neibF])
df = pd.DataFrame(act_h2)
fig, ax = plt.subplots()
df.boxplot()
plt.show()

df = pd.DataFrame(aFrame)
fig, ax = plt.subplots()
df.boxplot()
plt.show()



activations = sklearn.preprocessing.minmax_scale(activations, feature_range=(0, 1),  axis=0, copy=True)

bin_code = np.where(activations>0.6, 1, 0)
_, clusters = np.unique(bin_code, axis=0, return_inverse=True)
table(clusters)
clusters = [-1 if np.sum(a_==clusters) <= 1000 else a_ for a_ in clusters]
table(clusters)

num_lbls = (np.unique(clusters, return_inverse=True)[1])
x = zR[:, 0]
y = zR[:, 1]
zz = zR[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
fig = go.Figure()

nM=len(np.unique(clusters))
from matplotlib.colors import rgb2hex
import seaborn as sns
palette = sns.color_palette(None, nM)
colors = np.array([ rgb2hex(palette[i]) for i in range(len(palette)) ])
clusters = ["%.2f" % x for x in clusters]
lbls_list = np.unique(clusters)
fig = go.Figure()
for m in range(nM):
    IDX = [x == lbls_list[m] for x in clusters]
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
    fig.update_layout()

vis_mat=np.zeros((nM,nM), dtype=bool)
np.fill_diagonal(vis_mat, True)

fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        updatemenus=[go.layout.Updatemenu(
        active=0,
        )
    ])
fig.show()











