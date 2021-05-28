import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
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
import umap.umap_ as umap
import pandas as pd
import timeit
from plotly.io import to_html
import plotly.io as pio
pio.renderers.default = "browser"

from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors

from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

def frange_anneal(n_epoch, ratio=0.25, shape='sin'):
    L = np.ones(n_epoch)
    if ratio ==0:
        return L
    for c in range(n_epoch):
        if c <= np.floor(n_epoch*ratio):
            if shape=='sqrt':
                norm = np.sqrt(np.floor(n_epoch*ratio))
                L[c] = np.sqrt(c)/norm
            if shape=='sin':
                Om = (np.pi/2/(n_epoch*ratio))
                L[c] =  np.sin(Om*c)
        else:
            L[c]=1
    return L



class AnnealingCallback(Callback):
    def __init__(self, weight, kl_weight_lst):
        self.weight = weight
        self.kl_weight_lst = kl_weight_lst

    def on_epoch_end(self, epoch, logs={}):
        new_weight = K.eval(self.kl_weight_lst[epoch])
        K.set_value(self.weight, new_weight)
        print("  Current DCAE Weight is " + str(K.get_value(self.weight)))

import ctypes
from numpy.ctypeslib import ndpointer
lib = ctypes.cdll.LoadLibrary("/home/grines02/PycharmProjects/BIOIBFO25L/Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #Sigma
                ctypes.c_size_t]



d= 5
# subspace clusters centers
original_dim = 30
d= 5 # main informative dimensions
sep=2
cl1_center = np.zeros(original_dim)
cl2_center = np.concatenate((np.ones(1), np.zeros(original_dim-1)), axis=0 )
cl3_center = np.concatenate((2*sep*np.ones(1), np.zeros(original_dim-1)), axis=0 )
cl4_center = np.concatenate((3*sep*np.ones(1), np.zeros(original_dim-1)), axis=0 )
cl5_center = np.concatenate((4*sep*np.ones(1), np.zeros(original_dim-1)), axis=0 )
cl6_center = np.concatenate((3*sep*np.ones(1), np.zeros(original_dim-1)), axis=0 )
cl6_center[1] = sep
cl7_center = np.concatenate((4*sep*np.ones(1), np.zeros(original_dim-1)), axis=0 )
cl7_center[2] = sep
# add big cluster of 'irrelevant' cells with very different expression set


# cluster populatiosn
ncl1 = ncl2 = ncl3 = ncl4  = ncl5 = ncl6 = ncl7 = 3000

# cluster labels
lbls = np.concatenate((np.zeros(ncl1), np.ones(ncl2), 2*np.ones(ncl3), 3*np.ones(ncl4), 4*np.ones(ncl5),
                       5*np.ones(ncl6), 6*np.ones(ncl7)), axis=0)
# expand in main informative  dimensions
'''
wd= 0.3
cl1 = cl1_center +  np.concatenate([np.random.uniform(low=-wd, high=wd, size=(ncl1,d)), np.zeros((ncl1,original_dim - d))], axis=1 )
cl2 = cl2_center +  np.concatenate([np.random.uniform(low=-wd, high=wd, size=(ncl2,d)), np.zeros((ncl1,original_dim - d))], axis=1 )
cl3 = cl3_center +  np.concatenate([np.random.uniform(low=-wd, high=wd, size=(ncl3,d)), np.zeros((ncl1,original_dim - d))], axis=1 )
cl4 = cl4_center +  np.concatenate([np.random.uniform(low=-wd, high=wd, size=(ncl4,d)), np.zeros((ncl1,original_dim - d))], axis=1 )
cl5 = cl5_center +  np.concatenate([np.random.uniform(low=-wd, high=wd, size=(ncl5,d)), np.zeros((ncl1,original_dim - d))], axis=1 )
cl6 = cl6_center +  np.concatenate([np.random.uniform(low=-wd, high=wd, size=(ncl6,d)), np.zeros((ncl1,original_dim - d))], axis=1 )
cl7 = cl7_center +  np.concatenate([np.random.uniform(low=-wd, high=wd, size=(ncl7,d)), np.zeros((ncl1,original_dim - d))], axis=1 )
'''
#sns.violinplot(data= cl1, bw = 0.1);sns.violinplot(data= cl2, bw = 0.1);
#noisy or not:
#noise_sig1 =  np.concatenate((np.zeros(20),  np.ones(10)), axis=0 )
#noise_sig2 = np.concatenate((np.ones(10), np.zeros(20)), axis=0 )
# add noise to orthogonal dimensions
'''
noise_scale =0.2
cl1_noisy = cl1 + np.concatenate([np.zeros((ncl1,d)), np.random.normal(loc=0, scale = noise_scale, size=(ncl1,original_dim - d))], axis=1 )
cl2_noisy = cl2 + np.concatenate([np.zeros((ncl2,d)), np.random.normal(loc=0, scale = noise_scale, size=(ncl2,original_dim - d))], axis=1 )
cl3_noisy = cl3 + np.concatenate([np.zeros((ncl3,d)), np.random.normal(loc=0, scale = noise_scale, size=(ncl3,original_dim - d))], axis=1 )
cl4_noisy = cl4 + np.concatenate([np.zeros((ncl4,d)), np.random.normal(loc=0, scale = noise_scale, size=(ncl4,original_dim - d))], axis=1 )
cl5_noisy = cl5 + np.concatenate([np.zeros((ncl5,d)), np.random.normal(loc=0, scale = noise_scale, size=(ncl5,original_dim - d))], axis=1 )

cl6_noisy = cl6 + np.concatenate([np.zeros((ncl6,d)), np.random.normal(loc=0, scale = noise_scale, size=(ncl6,original_dim - d))], axis=1 )
cl7_noisy = cl7 + np.concatenate([np.zeros((ncl7,d)), np.random.normal(loc=0, scale = noise_scale, size=(ncl7,original_dim - d))], axis=1 )
'''
#introduce correlation


# The desired covariance matrix.
#n <- 5
#p <- qr.Q(qr(matrix(rnorm(n^2), n)))
#Sigma <- crossprod(p, p*(5:1))
from sklearn import datasets
r = sklearn.datasets.make_spd_matrix(d,  random_state=12346)



# Generate the random samples.
y1 = np.random.multivariate_normal(cl1_center[:d], r, size=ncl1)
y2 = np.random.multivariate_normal(cl2_center[:d], r, size=ncl2)
y3 = np.random.multivariate_normal(cl3_center[:d], r, size= ncl3)
y4 = np.random.multivariate_normal(cl4_center[:d], r, size=ncl4)
y5 = np.random.multivariate_normal(cl5_center[:d], r, size=ncl5)
y6 = np.random.multivariate_normal(cl6_center[:d], r, size=ncl6)
y7 = np.random.multivariate_normal(cl7_center[:d], r, size=ncl7)


wd= 0.3
cl1 = cl1_center +  np.concatenate([y1, np.zeros((ncl1,original_dim - d))], axis=1 )
cl2 = cl2_center +  np.concatenate([y2, np.zeros((ncl1,original_dim - d))], axis=1 )
cl3 = cl3_center +  np.concatenate([y3, np.zeros((ncl1,original_dim - d))], axis=1 )
cl4 = cl4_center +  np.concatenate([y4, np.zeros((ncl1,original_dim - d))], axis=1 )
cl5 = cl5_center +  np.concatenate([y5, np.zeros((ncl1,original_dim - d))], axis=1 )
cl6 = cl6_center +  np.concatenate([y6, np.zeros((ncl1,original_dim - d))], axis=1 )
cl7 = cl7_center +  np.concatenate([y7, np.zeros((ncl1,original_dim - d))], axis=1 )

#sns.violinplot(data= cl1, bw = 0.1);sns.violinplot(data= cl2, bw = 0.1);
#noisy or not:
#noise_sig1 =  np.concatenate((np.zeros(20),  np.ones(10)), axis=0 )
#noise_sig2 = np.concatenate((np.ones(10), np.zeros(20)), axis=0 )
# add noise to orthogonal dimensions
noise_scale =0.2
cl1_noisy = cl1 + np.concatenate([np.zeros((ncl1,d)), np.abs(np.random.normal(loc=0, scale = noise_scale, size=(ncl1,original_dim - d)))], axis=1 )
cl2_noisy = cl2 + np.concatenate([np.zeros((ncl2,d)), np.abs(np.random.normal(loc=0, scale = noise_scale, size=(ncl2,original_dim - d)))], axis=1 )
cl3_noisy = cl3 + np.concatenate([np.zeros((ncl3,d)), np.abs(np.random.normal(loc=0, scale = noise_scale, size=(ncl3,original_dim - d)))], axis=1 )
cl4_noisy = cl4 + np.concatenate([np.zeros((ncl4,d)), np.abs(np.random.normal(loc=0, scale = noise_scale, size=(ncl4,original_dim - d)))], axis=1 )
cl5_noisy = cl5 + np.concatenate([np.zeros((ncl5,d)), np.abs(np.random.normal(loc=0, scale = noise_scale, size=(ncl5,original_dim - d)))], axis=1 )

cl6_noisy = cl6 + np.concatenate([np.zeros((ncl6,d)), np.abs(np.random.normal(loc=0, scale = noise_scale, size=(ncl6,original_dim - d)))], axis=1 )
cl7_noisy = cl7 + np.concatenate([np.zeros((ncl7,d)), np.abs(np.random.normal(loc=0, scale = noise_scale, size=(ncl7,original_dim - d)))], axis=1 )




#sns.violinplot(data= cl1_noisy, bw = 0.1);
#sns.violinplot(data= cl2_noisy, bw = 0.1);
#sns.violinplot(data= cl3_noisy, bw = 0.1);
#sns.violinplot(data= cl4_noisy, bw = 0.1);
#sns.violinplot(data= cl5_noisy, bw = 0.1);
#sns.violinplot(data= cl6_noisy, bw = 0.1);
#sns.violinplot(data= cl7_noisy, bw = 0.1);

noisy_clus = np.concatenate((cl1_noisy, cl2_noisy, cl3_noisy, cl4_noisy, cl5_noisy, cl6_noisy, cl7_noisy), axis=0)

scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
data =  scaler.fit_transform(noisy_clus)
mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=True).fit(data)
y =  mapper.transform(data)
cdict = {0: 'red', 1: 'blue', 2: 'green'}
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
dataset = pd.DataFrame()
dataset['x'] = y[:, 0]
dataset['y'] = y[:, 1]
dataset['color'] = [str(x) for x in lbls]

#pca
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(data)
yp = pca.transform(data)
#plt.scatter(y[:, 0], y[:, 1], c=color, s=1., cmap=plt.cm.Spectral)
dataset = pd.DataFrame()
dataset['x'] = yp[:, 0]
dataset['y'] = yp[:, 1]
dataset['color'] = [str(x) for x in lbls]


from matplotlib.colors import ListedColormap
classes = ['0', '1', '2', '3', '4','5', '6']
#values = [0, 0, 1, 2, 2, 2]
colours = ListedColormap(['k','b','y','g','r', 'm','c'], N=7)
fig01 = plt.figure();
scatter = plt.scatter(yp[:, 0], yp[:, 1], c=lbls.astype('float'), cmap=colours)
plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc='upper right')
plt.show()

from matplotlib.colors import ListedColormap
classes = ['0', '1', '2', '3', '4','5', '6']
#values = [0, 0, 1, 2, 2, 2]
colours = ListedColormap(['k','b','y','g','r', 'm','c'], N=7)
fig02 = plt.figure();
scatter = plt.scatter(y[:, 0], y[:, 1], c=lbls.astype('float'), cmap=colours)
plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc='upper right')
plt.show()

# preprocess before the deep
source_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/simulatedData'
output_dir  = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/simulatedData/output'


k = 30
k3 = k * 3
'''
aFrame = noisy_clus
aFrame.shape
# set negative values to zero
#aFrame[aFrame < 0] = 0
#randomize order
#IDX = np.random.choice(aFrame.shape[0], aFrame.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
#aFrame= aFrame[IDX,:]
#lbls = lbls[IDX]
len(lbls)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=12)
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
num_cores = 12
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=num_cores, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)

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
outfile = source_dir + '/7art_scaled_cor.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)
'''
source_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/simulatedData'
output_dir  = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/simulatedData/output'
outfile = source_dir + '/7art_scaled_cor.npz'
k=30
markers = np.arange(30).astype(str)
# np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
npzfile = np.load(outfile)


Sigma = npzfile['Sigma']
lbls = npzfile['lbls'];
lbls = lbls.astype(str)
aFrame = npzfile['aFrame'];
Dist = npzfile['Dist']
Idx = npzfile['Idx']
neibALL = npzfile['neibALL']
Sigma = npzfile['Sigma']

# session set up

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()


# Model-------------------------------------------------------------------------
######################################################
# targetTr = np.repeat(aFrame, r, axis=0)
k = 30
k3 = k * 3
coeffCAE = 15
epochs = 20000
ID = 'Art7'+ str(coeffCAE) + '_' + str(epochs) + '_210000samples_correlated'

DCAE_weight = K.variable(value=0)
DCAE_weight_lst = K.variable(np.array(frange_anneal(epochs, ratio=0)))

nrow = aFrame.shape[0]
batch_size = 256
latent_dim = 3
original_dim = aFrame.shape[1]
intermediate_dim =original_dim*3
intermediate_dim2=original_dim
# var_dims = Input(shape = (original_dim,))
#
initializer = tf.keras.initializers.he_normal(12345)
#initializer = None
SigmaTsq = Input(shape=(1,))
x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer = initializer)(x)
h1 = Dense(intermediate_dim2, activation='relu', name='intermediate2', kernel_initializer = initializer)(h)
z_mean =  Dense(latent_dim, activation=None, name='z_mean', kernel_initializer = initializer)(h1)


encoder = Model([x, SigmaTsq], z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='relu', name='intermediate3', kernel_initializer = initializer)
decoder_h1 = Dense(intermediate_dim, activation='relu', name='intermediate4', kernel_initializer = initializer)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer = initializer)
h_decoded = decoder_h(z_mean)
h_decoded2 = decoder_h1(h_decoded)
x_decoded_mean = decoder_mean(h_decoded2)
autoencoder = Model(inputs=[x, SigmaTsq], outputs=x_decoded_mean)

# Loss and optimizer ------------------------------------------------------
# rewrite this based on recommendations here
# https://www.tensorflow.org/guide/keras/train_and_evaluate

normSigma = nrow / sum(1 / Sigma)

lam=1e-4

def DCAE_loss(x, x_decoded_mean):  # attempt to avoid vanishing derivative of sigmoid
    U = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
    W = K.variable(value=encoder.get_layer('intermediate2').get_weights()[0])  # N x N_hidden
    Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
    U = K.transpose(U);
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = tf.linalg.diag((tf.math.sign(u) + 1) / 2)
    m = encoder.get_layer('intermediate2').output
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
    ds = DCAE_weight * (-2 * r + 1.5 * r ** 2) + 1.5 + 1.2 * (DCAE_weight - 1)
    #tf.print(ds.shape)
    # return 1 / normSigma * (SigmaTsq) * lam * K.sum(dm ** 2 * K.sum(W ** 2, axis=1), axi0s=1)
    # grow one level in front
    S_0W = tf.einsum('akl,lj->akj', du, U)
    S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
    # tf.print((S_1W).shape) #[None, 120]
    S_2Z = tf.einsum('a,lj->alj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z,  S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    diff_tens = tf.einsum('akl,alj->akj', diff_tens, S_0W)
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam *(K.sum(diff_tens ** 2))


#mmd staff TODO: try approximation for this
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
def loss_mmd(x, x_decoded_mean):
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    #true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval = -1., maxval = 1.0)
    return compute_mmd(true_samples, z_mean)


def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    msew = tf.keras.losses.mean_squared_error(y_true, y_pred)
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(msew, normSigma * 1/SigmaTsq ) # TODO Sigma -denomiator or nominator? try reverse, schek hpw sigma computed in UMAP

def ae_loss(weight, DCAE_weight_lst):
    def loss(x, x_decoded_mean):
        msew = mean_square_error_NN(x, x_decoded_mean)
        return msew + loss_mmd(x, x_decoded_mean) + coeffCAE * DCAE_loss(x, x_decoded_mean)
        # return K.mean(msew)
    return loss

autoencoder.summary()
import tensorflow_addons as tfa
#opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
opt = tf.keras.optimizers.SGD(learning_rate=0.01, decay=0, momentum=0.9, nesterov =False)
opt=tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
)
autoencoder.compile(optimizer=opt, loss=ae_loss(DCAE_weight, DCAE_weight_lst), metrics=[DCAE_loss, loss_mmd,  mean_square_error_NN])
#epochs=200
start = timeit.default_timer()
history = autoencoder.fit([aFrame, Sigma], aFrame,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[AnnealingCallback(DCAE_weight, DCAE_weight_lst)])
stop = timeit.default_timer()
z = encoder.predict([aFrame,  Sigma])

print(stop - start)

fig01 = plt.figure();
plt.plot(history.history['loss'][1000:]);
plt.title('loss')


fig0 = plt.figure();
plt.plot(history.history['DCAE_loss'][1000:]);
plt.title('DCAE_loss')


fig03 = plt.figure();
plt.plot(history.history['loss_mmd'][1000:]);
plt.title('loss_mmd')


fig03 = plt.figure();
plt.plot(history.history['mean_square_error_NN'][1000:]);
plt.title('mean_square_error')



encoder.save_weights(output_dir +'/'+ID + '_3D.h5')
autoencoder.save_weights(output_dir +'/autoencoder_'+ID + '_3D.h5')
np.savez(output_dir +'/'+ ID + '_latent_rep_3D.npz', z = z)

encoder.load_weights(output_dir +'/'+ID + '_3D.h5')
autoencoder.load_weights(output_dir +'/autoencoder_'+ID + '_3D.h5')
encoder.summary()
z = encoder.predict([aFrame, Sigma])


#- visualisation -----------------------------------------------------------------------------------------------

# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
# x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt')

fig = plot3D_cluster_colors(z, lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Buttons.html","w")
Html_file.write(html_str)
Html_file.close()

# cluster toplogy
# 0 1 2 3 4
#       5 6
# clustering UMAP representation
#mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(aFrame)
#embedUMAP =  mapper.transform(aFrame)
#np.savez('Pregnancy_' + 'embedUMAP.npz', embedUMAP=embedUMAP)
embedUMAP = np.load('Pregnancy_' + 'embedUMAP.npz')['embedUMAP']
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanUMAP = clusterer.fit_predict(embedUMAP)
table(labelsHDBscanUMAP)
print(compute_cluster_performance(lbls, labelsHDBscanUMAP))
#labelsHDBscanUMAP= [str(x) for  x in labelsHDBscanUMAP]#
fig = plot2D_cluster_colors(embedUMAP[0:10000,:], lbls=lbls[0:10000])
fig.show()

######################################3
# try SAUCIE
'''
import sys
sys.path.append("/home/grines02/PycharmProjects/BIOIBFO25L/SAUCIE/")
data = aFrame
from importlib import reload
import SAUCIE
#reload(SAUCIE)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
saucie = SAUCIE.SAUCIE(data.shape[1])
loadtrain = SAUCIE.Loader(data, shuffle=True)
saucie.train(loadtrain, steps=10000)

loadeval = SAUCIE.Loader(data, shuffle=False)
embedding = saucie.get_embedding(loadeval)
number_of_clusters, clusters = saucie.get_clusters(loadeval)
#np.savez('Art7_' + 'embedSAUCIE.npz', embedding=embedding,number_of_clusters=number_of_clusters, clusters=clusters)
'''
import os
os.chdir('/home/grines02/PycharmProjects/BIOIBFO25L')
os.getcwd()

embedding = np.load('Art7_' + 'embedSAUCIE.npz')['embedding']
clusters= np.load('Art7_' + 'embedSAUCIE.npz')['clusters']
print(compute_cluster_performance(lbls,  clusters))
print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], clusters[lbls!='"Unassgined"']))
#clusters= [str(x) for  x in clusters]
#fig = plot3D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(clusters)), lbls=np.asarray(clusters))
#fig.show()
fig = plot2D_cluster_colors(embedding[:,:], lbls=lbls, msize=5)
fig.show()
#1 TODO:  add msize option to all graphical function, defaulting to 1
#2 TODO: create more complex cluster structure, with 3 more clusters , one of which a side clusters and run teh demo




