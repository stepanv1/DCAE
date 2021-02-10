'''This script to visualise cytoff data using deep variational autoencoder with MMD   and
contracting AE,
'''
import tensorflow as tf
from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from GetBest import GetBest
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter
from plotly.io import to_html
import plotly.graph_objects as go
import umap.umap_ as umap
import hdbscan
import phenograph
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.io import to_html
import plotly.io as pio
pio.renderers.default = "browser"
import glob
import sklearn
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Dropout, BatchNormalization
#from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
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


#/home/grines02/PycharmProjects/BIOIBFO25L/Rscripts/PhenographLevin13/utils_evaluation.py

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)



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

# load data
k = 30
k3 = k * 3
coeffCAE = 5
epochs = 2000
ID = 'Pr_sample_008_1_MMD_01_3D_DCAE_h300_h200_hidden_7_layers_CAE_sgd'+ str(coeffCAE) + '_' + str(epochs) + '_kernelInit_tf2'
#ID = 'Pr_sample_008_1_MMD_01_3D_DCAE_h128_h63_h32_9_layers'+ str(coeffCAE) + '_' + str(epochs) + '_kernelInit_tf2'
#ID = 'Pr_sample_008_1_Unstim_3D'
'''
data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816
'''
source_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/pregnancy'
output_dir  = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/pregnancy/output'
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


#neibALL  = neibALL [IDX,:]
#np.sum(cut_neibF != 0)
# plt.hist(cut_neibF[cut_neibF!=0],50)
Sigma = npzfile['Sigma']
lbls = npzfile['lbls'];
#neib_weight = npzfile['neib_weight']
markers = list(npzfile['markers'])
# np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
npzfile = np.load(outfile)
#weight_distALL = npzfile['Dist'];
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
#neib_weight = npzfile['neib_weight']
# [aFrame, neibF, cut_neibF, weight_neibF]
# training set
# targetTr = np.repeat(aFrame, r, axis=0)
Idx = npzfile['Idx']

# session set up

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()


# Model-------------------------------------------------------------------------
######################################################
# targetTr = np.repeat(aFrame, r, axis=0)

DCAE_weight = K.variable(value=0)
DCAE_weight_lst = K.variable(np.array(frange_anneal(epochs, ratio=0)))

nrow = aFrame.shape[0]
batch_size = 256
original_dim = 37
latent_dim = 3
intermediate_dim =37*5
intermediate_dim2=37
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]
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
opt = tf.keras.optimizers.SGD(learning_rate=0.001, decay=0, momentum=0.01, nesterov =False)
autoencoder.compile(optimizer=opt, loss=ae_loss(DCAE_weight, DCAE_weight_lst), metrics=[DCAE_loss, loss_mmd, mean_square_error_NN])
#epochs=10
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
plt.plot(history.history['loss'][200:]);

fig0 = plt.figure();
plt.plot(history.history['DCAE_loss'][200:]);


fig03 = plt.figure();
plt.plot(history.history['loss_mmd'][200:]);

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


fig =plot3D_marker_colors(z, data=aFrame, markers=list(markers), sub_s = 20000, lbls=lbls)
#data2 =np.sqrt(1-(aFrame-1)**2)
#fig =plot3D_marker_colors(z, data=data2, markers=list(markers), sub_s = 20000, lbls=lbls)
fig =plot3D_marker_colors(z, data=aFrame, markers=list(markers), sub_s = 20000, lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Markers.html","w")
Html_file.write(html_str)
Html_file.close()

#smoothed marker distribution
data2 =np.sqrt(aFrame)
fig =plot3D_marker_colors(z, data=data2, markers=list(markers), sub_s = 20000, lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Smoothed_Markers.html","w")
Html_file.write(html_str)
Html_file.close()


import hdbscan
# clusteng hidden representation
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanZ = clusterer.fit_predict(z)
table(labelsHDBscanZ)
print(compute_cluster_performance(lbls, labelsHDBscanZ))
#labelsHDBscanZ= [str(x) for  x in labelsHDBscanZ]
#fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanZ))
#fig.show()

#compute innternal metrics
print(sklearn.metrics.calinski_harabasz_score(z, lbls))

attrib_z=np.c_[z, aFrame/5]
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanAttributes = clusterer.fit_predict(attrib_z)
#labelsHDBscanAttributes = clusterer.fit_predict(aFrame)
table(labelsHDBscanAttributes)
print(compute_cluster_performance(lbls, labelsHDBscanAttributes))
print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], labelsHDBscanAttributes[lbls!='"Unassgined"']))
labelsHDBscanAttributes= [str(x) for  x in labelsHDBscanAttributes]
fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanAttributes))
print(sklearn.metrics.calinski_harabasz_score(attrib_z, lbls))
fig.show()

# clustering projected z + aFrame
prZ = projZ(z)
attrib_z=np.c_[prZ, np.array(aFrame)/5]
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf')
labelsHDBscanAttributesPr = clusterer.fit_predict(attrib_z)
#labelsHDBscanAttributes = clusterer.fit_predict(aFrame)
table(labelsHDBscanAttributesPr)
print(compute_cluster_performance(lbls, labelsHDBscanAttributesPr))
print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], labelsHDBscanAttributesPr[lbls!='"Unassgined"']))
print(sklearn.metrics.calinski_harabasz_score(attrib_z, lbls))
labelsHDBscanAttributesPr = [str(x) for  x in labelsHDBscanAttributesPr]
x = prZ[:, 0]
y = prZ[:, 1]
zz = prZ[:, 2]
fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanAttributes))
fig.show()

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

attrib_z=np.c_[embedUMAP, np.array(aFrame)/5]
labelsHDBscanUMAPattr = clusterer.fit_predict(attrib_z)
table(labelsHDBscanUMAPattr)
print(compute_cluster_performance(lbls, labelsHDBscanUMAPattr))
print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], labelsHDBscanUMAPattr[lbls!='"Unassgined"']))
print(sklearn.metrics.calinski_harabasz_score(attrib_z, lbls))
#labelsHDBscanUMAPattr= [str(x) for  x in labelsHDBscanUMAPattr]
#fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanUMAPattr))
#fig.show()

# cluster with phenograph
#communities, graph, Q = phenograph.cluster(aFrame)
#np.savez('Pregnancy_Phenograph.npz', communities=communities, graph=graph, Q=Q)
communities =np.load('Pregnancy_Phenograph.npz')['communities']
print(compute_cluster_performance(lbls, communities))
print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], communities[lbls!='"Unassgined"']))

######################################3
# try SAUCIE
sys.path.append("/home/grines02/PycharmProjects/BIOIBFO25L/SAUCIE")
data = aFrame
from importlib import reload
import SAUCIE
#reload(SAUCIE)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
saucie = SAUCIE.SAUCIE(data.shape[1])
loadtrain = SAUCIE.Loader(data, shuffle=True)
saucie.train(loadtrain, steps=1000)

loadeval = SAUCIE.Loader(data, shuffle=False)
embedding = saucie.get_embedding(loadeval)
number_of_clusters, clusters = saucie.get_clusters(loadeval)
#np.savez('Pregnancy_' + 'embedSAUCIE.npz', embedding=embedding,number_of_clusters=number_of_clusters, clusters=clusters)
embedding = np.load('Pregnancy_' + 'embedSAUCIE.npz')['embedding']
clusters= np.load('Pregnancy_' + 'embedSAUCIE.npz')['clusters']
print(compute_cluster_performance(lbls,  clusters))
print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], clusters[lbls!='"Unassgined"']))
#clusters= [str(x) for  x in clusters]
#fig = plot3D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(clusters)), lbls=np.asarray(clusters))
#fig.show()
fig = plot2D_cluster_colors(embedding[0:100000,:], lbls=lbls[0:100000])
fig.show()

attrib_z=np.c_[embedding, np.array(aFrame)/5]
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanSAUCIEattr = clusterer.fit_predict(attrib_z)
table(labelsHDBscanSAUCIEattr)
print(compute_cluster_performance(lbls, labelsHDBscanSAUCIEattr))
print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], labelsHDBscanSAUCIEattr[lbls!='"Unassgined"']))
print(sklearn.metrics.calinski_harabasz_score(attrib_z, lbls))
labelsHDBscanSAUCIEattr= [str(x) for  x in labelsHDBscanSAUCIEattr]
fig = plot2D_cluster_colors(embedding, lbls=np.asarray(labelsHDBscanSAUCIEattr))
fig.show()

# create performance plots for paper
import os
os.chdir('/home/grines02/PycharmProjects/BIOIBFO25L')
os.getcwd()

embedding = np.load('Pregnancy_' + 'embedSAUCIE.npz')['embedding']
embedUMAP = np.load('Pregnancy_' + 'embedUMAP.npz')['embedUMAP']
PAPERPLOTS  = './PAPERPLOTS/'
#3 plots for paper
# how to export as png: https://plotly.com/python/static-image-export/ 2D
fig = plot3D_cluster_colors(z[lbls !='"Unassgined"', :  ], camera = dict(eye = dict(x=-1.5,y=1.5,z=0.3)),
                            lbls=lbls[lbls !='"Unassgined"'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ "Pregnancy.png")

fig = plot2D_cluster_colors(embedding[lbls !='"Unassgined"', :  ], lbls=lbls[lbls !='"Unassgined"'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ "Pregnancy_SAUCIE.png")

fig = plot2D_cluster_colors(embedUMAP[lbls !='"Unassgined"', :  ], lbls=lbls[lbls !='"Unassgined"'],legend=True)
fig.show()
fig.write_image(PAPERPLOTS+ "Pregnancy_UMAP.png")

#TODO:very importmant!!! scale all the output to be in unite square (or cube)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
embedding=  scaler.fit_transform(embedding)
embedUMAP= scaler.fit_transform(embedUMAP)
z= scaler.fit_transform(z)
z = z/np.sqrt(3.1415)
#prZ = projZ(z)
#prZ = scaler.fit_transform(prZ)
#prZ =prZ/np.sqrt(3.1415)
#DCAE
discontinuityDCAE, manytooneDCAE = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreDCAE = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreDCAE = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=12)

#discontinuityDCAE_prZ, manytooneDCAE_prZ = get_wsd_scores(aFrame, prZ, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
#onetomany_scoreDCAE_prZ = neighbour_onetomany_score(prZ, Idx, kmax=90, num_cores=12)[1]
#marker_similarity_scoreDCAE_prZ = neighbour_marker_similarity_score_per_cell(prZ, aFrame, kmax=90, num_cores=12)

#UMAP
discontinuityUMAP, manytooneUMAP = get_wsd_scores(aFrame, embedUMAP, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreUMAP= neighbour_onetomany_score(embedUMAP, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreUMAP = neighbour_marker_similarity_score_per_cell(embedUMAP, aFrame, kmax=90, num_cores=12)

#SAUCIE
discontinuitySAUCIE, manytooneSAUCIE = get_wsd_scores(aFrame, embedding, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreSAUCIE= neighbour_onetomany_score(embedding, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreSAUCIE = neighbour_marker_similarity_score_per_cell(embedding, aFrame, kmax=90, num_cores=12)

outfile2 = source_dir + '/' + ID+ '_PerformanceMeasures.npz'
np.savez(outfile2, discontinuityDCAE = discontinuityDCAE, manytooneDCAE= manytooneDCAE, onetomany_scoreDCAE= onetomany_scoreDCAE, marker_similarity_scoreDCAE= marker_similarity_scoreDCAE[1],
        discontinuityUMAP= discontinuityUMAP, manytooneUMAP= manytooneUMAP, onetomany_scoreUMAP= onetomany_scoreUMAP, marker_similarity_scoreUMAP= marker_similarity_scoreUMAP[1],
         discontinuitySAUCIE= discontinuitySAUCIE, manytooneSAUCIE= manytooneSAUCIE, onetomany_scoreSAUCIE= onetomany_scoreSAUCIE, marker_similarity_scoreSAUCIE= marker_similarity_scoreSAUCIE[1])

npzfile = np.load(outfile2)
discontinuityDCAE = npzfile['discontinuityDCAE']; manytooneDCAE= npzfile['manytooneDCAE']; onetomany_scoreDCAE= npzfile['onetomany_scoreDCAE']; marker_similarity_scoreDCAE= npzfile['marker_similarity_scoreDCAE'];
discontinuityUMAP= npzfile['discontinuityUMAP']; manytooneUMAP= npzfile['manytooneUMAP']; onetomany_scoreUMAP= npzfile['onetomany_scoreUMAP']; marker_similarity_scoreUMAP= npzfile['marker_similarity_scoreUMAP'];
discontinuitySAUCIE= npzfile['discontinuitySAUCIE']; manytooneSAUCIE= npzfile['manytooneSAUCIE']; onetomany_scoreSAUCIE= npzfile['onetomany_scoreSAUCIE']; marker_similarity_scoreSAUCIE=  npzfile['marker_similarity_scoreSAUCIE']
#Quick look into results
# TODO: data normalization by normalize_data_by_mean_pdist in y space
np.mean(discontinuityDCAE)
np.mean(manytooneDCAE)
np.mean(discontinuityUMAP)
np.mean(manytooneUMAP)
np.mean(discontinuitySAUCIE)
np.mean(manytooneSAUCIE)
np.mean(onetomany_scoreDCAE[29,:])
np.mean(marker_similarity_scoreDCAE[29])
np.mean(onetomany_scoreUMAP[29,:])
np.mean(marker_similarity_scoreUMAP[29])
np.mean(onetomany_scoreSAUCIE[29,:])
np.mean(marker_similarity_scoreSAUCIE[29])

np.median(discontinuityDCAE)
np.median(manytooneDCAE)
np.median(discontinuityUMAP)
np.median(manytooneUMAP)
np.median(discontinuitySAUCIE)
np.median(manytooneSAUCIE)
np.median(onetomany_scoreDCAE[29,:])
np.median(marker_similarity_scoreDCAE[29])
np.median(onetomany_scoreUMAP[29,:])
np.median(marker_similarity_scoreUMAP[29])
np.median(onetomany_scoreSAUCIE[29,:])
np.median(marker_similarity_scoreSAUCIE[29])

np.median(discontinuityDCAE_prZ)
np.median(manytooneDCAE_prZ)
np.median(onetomany_scoreDCAE_prZ[29,:])
np.median(marker_similarity_scoreDCAE_prZ[29])


plt.hist(onetomany_scoreSAUCIE[90,:],250)
plt.hist(onetomany_scoreDCAE[90,:],250)
plt.hist(onetomany_scoreUMAP[90,:],250)
plt.hist(discontinuityDCAE,250)
plt.hist(discontinuitySAUCIE,250)
plt.hist(discontinuityUMAP,250)

plt.hist(marker_similarity_scoreSAUCIE[29],250)
plt.hist(marker_similarity_scoreDCAE[29],250)
plt.hist(marker_similarity_scoreUMAP[29],250)
plt.hist(manytooneSAUCIE,250)
plt.hist(manytooneDCAE,250)
plt.hist(manytooneUMAP,250)



#plt.hist(z,250)
#plt.hist(embedding,250)
#build grpahs using above data
# now build plots and tables. 2 plots: 1 for onetomany_score, 1 marker_similarity_scoreDCAE on 2 methods
# table: Discontinuity and manytoone (2 columns) with 3 rows, each per method. Save as a table then stack with output on other  data , to create the final table
median_marker_similarity_scoreDCAE = np.median(marker_similarity_scoreDCAE, axis=1);median_marker_similarity_scoreSAUCIE = np.median(marker_similarity_scoreSAUCIE, axis=1);
median_marker_similarity_scoreUMAP = np.median(marker_similarity_scoreUMAP, axis=1);
df_sim = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_marker_similarity_scoreDCAE[1:], 'SAUCIE': median_marker_similarity_scoreSAUCIE[1:], 'UMAP': median_marker_similarity_scoreUMAP[1:]})
#fig1, fig2 = plt.subplots()
plt.plot('k', 'DCAE', data=df_sim, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_sim, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_sim, marker='x', color='olive', linewidth=2)
plt.legend()
plt.savefig(PAPERPLOTS  + 'LEVINE32_' + ID+ 'performance_marker_similarity_score.png')
plt.show()
plt.clf()
median_onetomany_scoreDCAE = np.median(onetomany_scoreDCAE, axis=1);median_onetomany_scoreSAUCIE = np.median(onetomany_scoreSAUCIE, axis=1);
median_onetomany_scoreUMAP = np.median(onetomany_scoreUMAP, axis=1);
df_otm = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_onetomany_scoreDCAE[1:], 'SAUCIE': median_onetomany_scoreSAUCIE[1:], 'UMAP': median_onetomany_scoreUMAP[1:]})
plt.plot('k', 'DCAE', data=df_otm, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_otm, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_otm, marker='x', color='olive', linewidth=2)
plt.savefig(PAPERPLOTS  + 'LEVINE32_' + ID+'_performance_onetomany_score.png')
plt.show()
#TODO:very importmant!!! scale all the output to be in unite square (or cube)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
embedding=  scaler.fit_transform(embedding)
embedUMAP= scaler.fit_transform(embedUMAP)
z= scaler.fit_transform(z)
z = z/np.sqrt(3.1415)
#prZ = projZ(z)
#prZ = scaler.fit_transform(prZ)
#prZ =prZ/np.sqrt(3.1415)
#DCAE
discontinuityDCAE, manytooneDCAE = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreDCAE = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreDCAE = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=12)

#discontinuityDCAE_prZ, manytooneDCAE_prZ = get_wsd_scores(aFrame, prZ, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
#onetomany_scoreDCAE_prZ = neighbour_onetomany_score(prZ, Idx, kmax=90, num_cores=12)[1]
#marker_similarity_scoreDCAE_prZ = neighbour_marker_similarity_score_per_cell(prZ, aFrame, kmax=90, num_cores=12)

#UMAP
discontinuityUMAP, manytooneUMAP = get_wsd_scores(aFrame, embedUMAP, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreUMAP= neighbour_onetomany_score(embedUMAP, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreUMAP = neighbour_marker_similarity_score_per_cell(embedUMAP, aFrame, kmax=90, num_cores=12)

#SAUCIE
discontinuitySAUCIE, manytooneSAUCIE = get_wsd_scores(aFrame, embedding, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreSAUCIE= neighbour_onetomany_score(embedding, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreSAUCIE = neighbour_marker_similarity_score_per_cell(embedding, aFrame, kmax=90, num_cores=12)

outfile2 = source_dir + '/' + ID+ '_PerformanceMeasures.npz'
#np.savez(outfile2, discontinuityDCAE = discontinuityDCAE, manytooneDCAE= manytooneDCAE, onetomany_scoreDCAE= onetomany_scoreDCAE, marker_similarity_scoreDCAE= marker_similarity_scoreDCAE[1],
#        discontinuityUMAP= discontinuityUMAP, manytooneUMAP= manytooneUMAP, onetomany_scoreUMAP= onetomany_scoreUMAP, marker_similarity_scoreUMAP= marker_similarity_scoreUMAP[1],
#         discontinuitySAUCIE= discontinuitySAUCIE, manytooneSAUCIE= manytooneSAUCIE, onetomany_scoreSAUCIE= onetomany_scoreSAUCIE, marker_similarity_scoreSAUCIE= marker_similarity_scoreSAUCIE[1])

npzfile = np.load(outfile2)
discontinuityDCAE = npzfile['discontinuityDCAE']; manytooneDCAE= npzfile['manytooneDCAE']; onetomany_scoreDCAE= npzfile['onetomany_scoreDCAE']; marker_similarity_scoreDCAE= npzfile['marker_similarity_scoreDCAE'];
discontinuityUMAP= npzfile['discontinuityUMAP']; manytooneUMAP= npzfile['manytooneUMAP']; onetomany_scoreUMAP= npzfile['onetomany_scoreUMAP']; marker_similarity_scoreUMAP= npzfile['marker_similarity_scoreUMAP'];
discontinuitySAUCIE= npzfile['discontinuitySAUCIE']; manytooneSAUCIE= npzfile['manytooneSAUCIE']; onetomany_scoreSAUCIE= npzfile['onetomany_scoreSAUCIE']; marker_similarity_scoreSAUCIE=  npzfile['marker_similarity_scoreSAUCIE']
#Quick look into results
# TODO: data normalization by normalize_data_by_mean_pdist in y space
np.mean(discontinuityDCAE)
np.mean(manytooneDCAE)
np.mean(discontinuityUMAP)
np.mean(manytooneUMAP)
np.mean(discontinuitySAUCIE)
np.mean(manytooneSAUCIE)
np.mean(onetomany_scoreDCAE[29,:])
np.mean(marker_similarity_scoreDCAE[29])
np.mean(onetomany_scoreUMAP[29,:])
np.mean(marker_similarity_scoreUMAP[29])
np.mean(onetomany_scoreSAUCIE[29,:])
np.mean(marker_similarity_scoreSAUCIE[29])

np.median(discontinuityDCAE)
np.median(manytooneDCAE)
np.median(discontinuityUMAP)
np.median(manytooneUMAP)
np.median(discontinuitySAUCIE)
np.median(manytooneSAUCIE)
np.median(onetomany_scoreDCAE[29,:])
np.median(marker_similarity_scoreDCAE[29])
np.median(onetomany_scoreUMAP[29,:])
np.median(marker_similarity_scoreUMAP[29])
np.median(onetomany_scoreSAUCIE[29,:])
np.median(marker_similarity_scoreSAUCIE[29])

plt.hist(onetomany_scoreSAUCIE[90,:],250)
plt.hist(onetomany_scoreDCAE[90,:],250)
plt.hist(onetomany_scoreUMAP[90,:],250)
plt.hist(discontinuityDCAE,250)
plt.hist(discontinuitySAUCIE,250)
plt.hist(discontinuityUMAP,250)

plt.hist(marker_similarity_scoreSAUCIE[29],250)
plt.hist(marker_similarity_scoreDCAE[29],250)
plt.hist(marker_similarity_scoreUMAP[29],250)
plt.hist(manytooneSAUCIE,250)
plt.hist(manytooneDCAE,250)
plt.hist(manytooneUMAP,250)



#plt.hist(z,250)
#plt.hist(embedding,250)
#build grpahs using above data
# now build plots and tables. 2 plots: 1 for onetomany_score, 1 marker_similarity_scoreDCAE on 2 methods
# table: Discontinuity and manytoone (2 columns) with 3 rows, each per method. Save as a table then stack with output on other  data , to create the final table
median_marker_similarity_scoreDCAE = np.median(marker_similarity_scoreDCAE, axis=1);median_marker_similarity_scoreSAUCIE = np.median(marker_similarity_scoreSAUCIE, axis=1);
median_marker_similarity_scoreUMAP = np.median(marker_similarity_scoreUMAP, axis=1);
df_sim = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_marker_similarity_scoreDCAE[1:], 'SAUCIE': median_marker_similarity_scoreSAUCIE[1:], 'UMAP': median_marker_similarity_scoreUMAP[1:]})
#fig1, fig2 = plt.subplots()
plt.plot('k', 'DCAE', data=df_sim, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_sim, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_sim, marker='x', color='olive', linewidth=2)
plt.legend()
plt.savefig(PAPERPLOTS  + 'LEVINE32_' + ID+ 'performance_marker_similarity_score.png')
plt.show()
plt.clf()
median_onetomany_scoreDCAE = np.median(onetomany_scoreDCAE, axis=1);median_onetomany_scoreSAUCIE = np.median(onetomany_scoreSAUCIE, axis=1);
median_onetomany_scoreUMAP = np.median(onetomany_scoreUMAP, axis=1);
df_otm = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_onetomany_scoreDCAE[1:], 'SAUCIE': median_onetomany_scoreSAUCIE[1:], 'UMAP': median_onetomany_scoreUMAP[1:]})
plt.plot('k', 'DCAE', data=df_otm, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_otm, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_otm, marker='x', color='olive', linewidth=2)
plt.savefig(PAPERPLOTS  + 'LEVINE32_' + ID+'_performance_onetomany_score.png')
plt.show()# tables
df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.1133, 0.1667, 0.1321], 'discontinuity': [0.1099, 0.0113, 0.0052]})
df_BORAI.to_csv(PAPERPLOTS  + 'Pregnancy_' + 'Borealis_measures.csv', index=False)
np.median(discontinuityDCAE)
#0.01565989388359918
np.median(manytooneDCAE)
#0.10611877287197075
np.median(discontinuitySAUCIE)
#0.009914790259467234
np.median(manytooneSAUCIE)

np.median(discontinuityUMAP)
#0.0013421323564317491
np.median(manytooneUMAP)
#0.11770417201150978
#plot perf
fig =plot3D_performance_colors(z[0:50000,:], perf=discontinuityDCAE, lbls=lbls[0:50000])
fig.show()
fig =plot3D_performance_colors(z[0:50000,:], perf=onetomany_scoreDCAE[30,:], lbls=lbls[0:50000])
fig.show()
fig =plot3D_performance_colors(z[0:50000,:], perf=manytooneDCAE, lbls=lbls[0:50000])
fig.show()
fig =plot3D_performance_colors(z[0:50000,:], perf=marker_similarity_scoreDCAE[0:50000], lbls=lbls[0:50000])
fig.show()

fig =plot2D_performance_colors(embedding[0:50000,:], perf=discontinuitySAUCIE, lbls=lbls[0:50000])
fig.show()
fig =plot2D_performance_colors(embedding[0:50000,:], perf=onetomany_scoreSAUCIE[30,:], lbls=lbls[0:50000])
fig.show()
fig =plot2D_performance_colors(embedding[0:50000,:], perf=manytooneSAUCIE, lbls=lbls[0:50000])
fig.show()
fig =plot2D_performance_colors(embedding[0:50000,:], perf=marker_similarity_scoreSAUCIE, lbls=lbls[0:50000])
fig.show()

fig =plot2D_performance_colors(embedUMAP[0:50000,:], perf=discontinuityUMAP, lbls=lbls[0:50000])
fig.show()
fig =plot2D_performance_colors(embedUMAP[0:50000,:], perf=onetomany_scoreSAUCIE[30,:], lbls=lbls[0:50000])
fig.show()
fig =plot2D_performance_colors(embedUMAP[0:50000,:], perf=manytooneSAUCIE, lbls=lbls[0:50000])
fig.show()
fig =plot2D_performance_colors(embedUMAP[0:50000,:], perf=marker_similarity_scoreSAUCIE, lbls=lbls[0:50000])
fig.show()


