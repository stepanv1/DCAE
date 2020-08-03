'''This script to visualise cytoff data using deep variational autoencoder with MMD  with neighbourhood denoising and
contracting knnCVAE, neighbourhood cleaning removed
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
outfile = source_dir + '/Nowicka2017euclid.npz'
np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls,  Dist=Dist)
'''

#annealing schedule
#kl_weight = K.variable(value=0)
#kl_weight_lst = K.variable(np.array(frange_cycle_linear(0.0, 1.0, epochs, n_cycle=10, ratio=0.75)))


epsilon_std = 1.0


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
source_dir = '/media/FCS_local/Stepan/data/WeberLabels/'
outfile = source_dir + '/Nowicka2017euclid_scaled.npz'
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

# session set up

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
# Model-------------------------------------------------------------------------
nrow = aFrame.shape[0]
batch_size = 256
original_dim = 24
latent_dim = 2
intermediate_dim = 12
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]
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
    return  tf.multiply(msew, normSigma * (1/SigmaTsq) )

def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1))), axis=-1)
    weightedN = k * original_dim * K.dot(dst,
                                         K.transpose(weight_neib))  # not really a mean square error after we done this
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 * normSigma * (1/SigmaTsq) )
    #return tf.multiply(weightedN, 0.5)

lam=1e-4
def contractive(x, x_decoded_mean):
        W = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = encoder.get_layer('z_mean').output
        dh = h * (1 - h)  # N_batch x N_hidden
        return 1/normSigma * (SigmaTsq) * lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)
        #return lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

def loss_mmd(x, x_decoded_mean):
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return compute_mmd(true_samples, z_mean)

def custom_loss(x, x_decoded_mean, z_mean):
    msew = mean_square_error_NN(x, x_decoded_mean)
    #print('msew done', K.eval(msew))
    #mmd loss
    #loss_nll = K.mean(K.square(train_xr - x))
    #batch_size = batch_size #K.shape(train_z)[0]

    #print('batch_size')
    #latent_dim = latent_dim
    #print(K.shape(loss_mmd))
    #return msew +  1 * contractive()
    return msew + 1*loss_mmd(x, x_decoded_mean) + 1*contractive(x, x_decoded_mean)

loss = custom_loss(x, x_decoded_mean, z_mean)
autoencoder.add_loss(loss)
autoencoder.compile(optimizer='adam', metrics=[mean_square_error_NN, contractive, custom_loss, loss_mmd])
print(autoencoder.summary())
print(encoder.summary())
#print(decoder.summary())



#callbacks = [EarlyStoppingByLossVal( monitor='loss', value=0.01, verbose=0

earlyStopping=EarlyStoppingByValLoss( monitor='val_loss', min_delta=0.0001, verbose=1, patience=10, restore_best_weights=True)
#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss')
epochs = 200
history = autoencoder.fit([targetTr[:,], neibF_Tr[:,:],  Sigma[:], weight_neibF[:,:]],
                epochs=epochs, batch_size=batch_size, verbose=1,
                validation_data=([targetTr[0:2000,:], neibF_Tr[0:2000,:],  Sigma[0:2000], weight_neibF[0:2000,:]], None),
                           shuffle=True)
                #callbacks=[CustomMetrics()], verbose=2)#, validation_data=([targetTr, neibF_Tr,  Sigma, weight_neibF], None))
z = encoder.predict([aFrame, neibALL,  Sigma, weight_neibF])

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
num_lbls = (np.unique(lbls, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=1,
                    color=num_lbls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=num_lbls,
                #hoverinfo='text')], filename='tmp.html')
                hoverinfo='text')], filename='OLD_LOSS_200epochs_LAM_0.001_MMD_1_caled.html')


# same model as above but with optimized target loss

SigmaTsq = Input(shape=(1,))
neib = Input(shape=(k, original_dim,))
# var_dims = Input(shape = (original_dim,))

x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu')(x)
# h.set_weights(ae.layers[1].get_weights())
z_mean =  Dense(latent_dim, name='z_mean')(h)

encoder = Model([x, neib, SigmaTsq], z_mean, name='encoder')

# we instantiate these layers separately so as to reuse them later
#decoder_input = Input(shape=(latent_dim,))

decoder_h = Dense(intermediate_dim, activation='relu')
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
    return  tf.multiply(msew, normSigma * (1/SigmaTsq) )
    #return tf.multiply(weightedN, 0.5)

lam=1e-4
def contractive(x, x_decoded_mean):
        W = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = encoder.get_layer('z_mean').output
        dh = h * (1 - h)  # N_batch x N_hidden
        return 1/normSigma * (SigmaTsq) * lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)
        #return lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

def loss_mmd(x, x_decoded_mean):
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return compute_mmd(true_samples, z_mean)
nn=30
def custom_loss(x, x_decoded_mean, z_mean):
    msew = mean_square_error_NN(x, x_decoded_mean)
    #print('msew done', K.eval(msew))
    #mmd loss
    #loss_nll = K.mean(K.square(train_xr - x))
    #batch_size = batch_size #K.shape(train_z)[0]

    #print('batch_size')
    #latent_dim = latent_dim
    #print(K.shape(loss_mmd))
    #return msew +  1 * contractive()
    return nn*msew + 1*loss_mmd(x, x_decoded_mean) + 10*contractive(x, x_decoded_mean)

loss = custom_loss(x, x_decoded_mean, z_mean)
autoencoder.add_loss(loss)
autoencoder.compile(optimizer='adam', metrics=[mean_square_error_NN, contractive, custom_loss, loss_mmd])
print(autoencoder.summary())
print(encoder.summary())
#print(decoder.summary())



#callbacks = [EarlyStoppingByLossVal( monitor='loss', value=0.01, verbose=0

earlyStopping=EarlyStoppingByValLoss( monitor='val_loss', min_delta=0.0001, verbose=1, patience=10, restore_best_weights=True)
#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss')
epochs = 600
history = autoencoder.fit([targetTr[:,], neibF_Tr[:,:],  Sigma[:]],
                epochs=epochs, batch_size=batch_size, verbose=1,
                validation_data=([targetTr[0:2000,:], neibF_Tr[0:2000,:],  Sigma[0:2000], weight_neibF[0:2000,:]], None),
                           shuffle=True)
                #callbacks=[CustomMetrics()], verbose=2)#, validation_data=([targetTr, neibF_Tr,  Sigma, weight_neibF], None))
z = encoder.predict([aFrame, neibALL,  Sigma])
plt.plot(history.history['loss'][5:])
plt.plot(history.history['val_loss'][:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
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
num_lbls = (np.unique(lbls, return_inverse=True)[1])
x = z[:, 0]
y = z[:, 1]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=1,
                    color=num_lbls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=lbls,
                #hoverinfo='text')], filename='tmp.html')
                hoverinfo='text')], filename='OLD_LOSS_200epochs_LAM_0.001_MMD_1_caled.html')



x = aFrame[:, 0]
y = aFrame[:, 1]
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
x = aFrame[sub_idx, 0]
y = aFrame[sub_idx, 1]
lbls_s = lbls[sub_idx]

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(Scatter(x=x, y=y,
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=x,  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(title = markers[0])
                        ),
                        text=lbls_s,
                        hoverinfo='text',
                        name = markers[0]
                        ))

fig.add_trace(Scatter(x=x, y=y,
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=y,  # set color to an array/list of desired values
                            colorscale='Viridis',  # choose a colorscale
                            opacity=0.5,
                            colorbar=dict(title = markers[1])
                        ),
                        text=lbls_s,
                        hoverinfo='text',
                        name = markers[1]
                        ))

fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=list(
            [dict(label = markers[0],
                  method = 'update',
                  args = [{'visible': [True, False]},
                          {'title': markers[0],
                           'showlegend':False}]),
             dict(label = markers[1],
                  method = 'update',
                  args = [{'visible': [False, True]}, # the index of True aligns with the indices of plot traces
                          {'title': markers[1],
                           'showlegend':False}]),
             ])
        )
    ])

fig.show()







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

# check teh gradiaents
outputTensor = model.output #Or model.layers[index].output
  listOfVariableTensors = model.trainable_weights
gradients = k.gradients(outputTensor, listOfVariableTensors)
trainingExample = np.random.random((1,8))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})