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
coeffCAE = 1
epochs = 2000
ID = 'Shekhar_MMD_01_3D_DCAE_h300_h200_hidden_7_layers_CAE'+ str(coeffCAE) + '_' + str(epochs) + '_kernelInit_tf2'

source_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess'
output_dir  = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/'

'''
Get Shekhar 2016 following  https://colab.research.google.com/github/KrishnaswamyLab/SingleCellWorkshop/blob/master/
/exercises/Deep_Learning/notebooks/02_Answers_Exploratory_analysis_of_single_cell_data_with_SAUCIE.ipynb#scrollTo=KgaZZU8r4drd
import scprep
scprep.io.download.download_google_drive("1GYqmGgv-QY6mRTJhOCE1sHWszRGMFpnf", "data.pickle.gz")
scprep.io.download.download_google_drive("1q1N1s044FGWzYnQEoYMDJOjdWPm_Uone", "metadata.pickle.gz")
data_raw = pd.read_pickle("data.pickle.gz")
data_raw.head()
metadata = pd.read_pickle("metadata.pickle.gz")
# the batch ids are in the cell barcode names
metadata['batch'] = [int(index[7]) for index in metadata.index]
# for simplicity, we'll split the six batches into two groups -- 1-3 and 4-6
metadata['sample_id'] = np.where(metadata['batch'] < 4, 1, 2)
metadata['sample_name'] = np.where(metadata['batch'] < 4, 'Samples 1-3', 'Samples 4-6')
metadata.head()
pca_op = sklearn.decomposition.PCA(100)
pca = sklearn.decomposition.PCA(100)
pca = pca.fit(data_raw.to_numpy())
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

data = pca_op.fit_transform(data_raw.to_numpy())



n_features = data.shape[1]
n_features
scprep.plot.scatter2d(data, c=metadata['sample_name'], ticks=False, label_prefix="PC", legend_title="Batch")

# get labels
table(metadata['CELLTYPE'])
aFrame = data 
aFrame.shape
# set negative values to zero
#aFrame[aFrame < 0] = 0
lbls= metadata['CELLTYPE']
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
results = Parallel(n_jobs=6, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
original_dim=100
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
outfile = source_dir + '/Shenkareuclid_scaled.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)
'''
outfile = source_dir + '/Shenkareuclid_scaled.npz'
markers = np.arange(1,101)
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
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
# call load_data with allow_pickle implicitly set to true
lbls = npzfile['lbls'];
table(lbls)
1-len(lbls[lbls=='-1'])/len(lbls)
neib_weight = npzfile['neib_weight']
# restore np.load for future normal usage
np.load = np_load_old

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
original_dim = 100
latent_dim = 3
intermediate_dim =120
intermediate_dim2=16
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
autoencoder.compile(optimizer='adam', loss=ae_loss(DCAE_weight, DCAE_weight_lst), metrics=[DCAE_loss, loss_mmd])

start = timeit.default_timer()
history = autoencoder.fit([aFrame, Sigma], aFrame,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[AnnealingCallback(DCAE_weight, DCAE_weight_lst)])
stop = timeit.default_timer()
z = encoder.predict([aFrame,  Sigma])

print(stop - start)
fig0 = plt.figure();
plt.plot(history.history['DCAE_loss'][200:]);

fig01 = plt.figure();
plt.plot(history.history['loss'][200:]);

fig03 = plt.figure();
plt.plot(history.history['loss_mmd'][200:]);

encoder.save_weights(output_dir +'/'+ID + '_3D.h5')
autoencoder.save_weights(output_dir +'/autoencoder_'+ID + '_3D.h5')
np.savez(output_dir +'/'+ ID + '_latent_rep_3D.npz', z = z)

encoder.load_weights(output_dir +''+ID + '_3D.h5')
autoencoder.load_weights(output_dir +'autoencoder_'+ID + '_3D.h5')
encoder.summary()
z = encoder.predict([aFrame, Sigma])

#- visualisation and pefroramnce metric-----------------------------------------------------------------------------------------------


# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
# x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt')
x = z[:, 0]
y = z[:, 1]
zz = z[:, 2]

fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Buttons.html","w")
Html_file.write(html_str)
Html_file.close()


fig =plot3D_marker_colors(z, data=aFrame, markers=markers, sub_s = 20000, lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Markers.html","w")
Html_file.write(html_str)
Html_file.close()


# clusteng hidden representation
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=25, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanZ = clusterer.fit_predict(z)
table(labelsHDBscanZ)
print(compute_cluster_performance(lbls, labelsHDBscanZ))
labelsHDBscanZ= [str(x) for  x in labelsHDBscanZ]
fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanZ))
fig.show()

attrib_z=np.c_[z, aFrame[:,0:20]/20]
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanAttributes = clusterer.fit_predict(attrib_z)
#labelsHDBscanAttributes = clusterer.fit_predict(aFrame)
table(labelsHDBscanAttributes)
print(compute_cluster_performance(lbls, labelsHDBscanAttributes))
labelsHDBscanAttributes= [str(x) for  x in labelsHDBscanAttributes]
fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanAttributes))
fig.show()

# clustering projected z + aFrame
prZ = projZ(z)
attrib_z=np.c_[prZ, (aFrame)[:,0:20]/20]
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf')
labelsHDBscanAttributesPr = clusterer.fit_predict(attrib_z)
#labelsHDBscanAttributes = clusterer.fit_predict(aFrame)
table(labelsHDBscanAttributesPr)
labelsHDBscanAttributes = [str(x) for  x in labelsHDBscanAttributesPr]
print(compute_cluster_performance(lbls, labelsHDBscanAttributesPr))
print(compute_cluster_performance(lbls[lbls!='"unassigned"'], np.array(labelsHDBscanAttributesPr)[lbls!='"unassigned"']))
x = prZ[:, 0]
y = prZ[:, 1]
zz = prZ[:, 2]
fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanAttributes))
fig.show()

# clustering UMAP representation
mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(aFrame)
embedUMAP =  mapper.transform(aFrame)
#np.savez('Shekhar_' + 'embedUMAP.npz', embedUMAP=embedUMAP)
embedUMAP = np.load('Shekhar_' + 'embedUMAP.npz')['embedUMAP']
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanUMAP = clusterer.fit_predict(embedUMAP)
table(labelsHDBscanUMAP)
print(compute_cluster_performance(lbls, labelsHDBscanUMAP))
#labelsHDBscanUMAP= [str(x) for  x in labelsHDBscanUMAP]
fig = plot2D_cluster_colors(x=embedUMAP[:,0],y=embedUMAP[:,1], lbls=lbls)
fig.show()

attrib_z=np.c_[embedUMAP,(aFrame)[:,0:20]/20]
labelsHDBscanUMAPattr = clusterer.fit_predict(attrib_z)
table(labelsHDBscanUMAPattr)
print(compute_cluster_performance(lbls, labelsHDBscanUMAPattr))
#labelsHDBscanUMAPattr= [str(x) for  x in labelsHDBscanUMAPattr]
#fig = plot3D_cluster_colors(x=x,y=y,z=zz, lbls=np.asarray(labelsHDBscanUMAPattr))
#fig.show()

# cluster with phenograph
#communities, graph, Q = phenograph.cluster(aFrame)
#np.savez('Shekhar_Phenograph.npz', communities=communities, graph=graph, Q=Q)
communities =np.load('Shekhar_Phenograph.npz')['communities']
print(compute_cluster_performance(lbls, communities))
print(compute_cluster_performance(lbls[lbls!='"unassigned"'], communities[lbls!='"unassigned"']))


######################################3
# try SAUCIE
sys.path.append("/home/grines02/PycharmProjects/BIOIBFO25L/SAUCIE")
data = aFrame
from importlib import reload
import SAUCIE
reload(SAUCIE)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
saucie = SAUCIE.SAUCIE(data.shape[1])
loadtrain = SAUCIE.Loader(data, shuffle=True)
saucie.train(loadtrain, steps=10000)

loadeval = SAUCIE.Loader(data, shuffle=False)
embedding = saucie.get_embedding(loadeval)
#np.savez('Shekhar_' + 'embedSAUCIE.npz', embedding=embedding)
embedding = np.load('Shekhar_' + 'embedSAUCIE.npz')['embedding']
number_of_clusters, clusters = saucie.get_clusters(loadeval)
#np.savez('Shekhar_clusters' + 'embedSAUCIE.npz', number_of_clusters=number_of_clusters, clusters = clusters, embedding=embedding)
print(compute_cluster_performance(lbls,  clusters))
#clusters= [str(x) for  x in clusters]
#fig = plot3D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(clusters)), lbls=np.asarray(clusters))
#fig.show()
fig = plot2D_cluster_colors(embedding, lbls=lbls)
fig.show()

attrib_z=np.c_[embedding, (aFrame)[:,0:20]/20]
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=15, alpha=1.0, cluster_selection_method = 'leaf') #5,20
labelsHDBscanSAUCIEattr = clusterer.fit_predict(attrib_z)
table(labelsHDBscanSAUCIEattr)
print(compute_cluster_performance(lbls, labelsHDBscanSAUCIEattr))
labelsHDBscanSAUCIEattr= [str(x) for  x in labelsHDBscanSAUCIEattr]
fig = plot2D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(labelsHDBscanSAUCIEattr)), lbls=np.asarray(labelsHDBscanSAUCIEattr))
fig.show()

fig = plot3D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(clusters)), lbls=np.zeros(len(clusters)))
fig.show()

# create performance plots for paper
embedding = np.load('Shekhar_' + 'embedSAUCIE.npz')['embedding']
embedUMAP = np.load('Shekhar_' + 'embedUMAP.npz')['embedUMAP']

PAPERPLOTS  = './PAPERPLOTS/'
#3 plots for paper
# how to export as png: https://plotly.com/python/static-image-export/ 2D
fig = plot3D_cluster_colors(z[lbls !='-1', :  ], camera = dict(eye = dict(x=-1.5,y=1.5,z=0.3)),
                            lbls=lbls[lbls !='-1'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ "Shekhar.png")

fig = plot2D_cluster_colors(embedding[lbls !='-1', :  ], lbls=lbls[lbls !='-1'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ "Shekhar_SAUCIE.png")

fig = plot2D_cluster_colors(embedUMAP[lbls !='-1', :  ], lbls=lbls[lbls !='-1'],legend=True)
fig.show()
fig.write_image(PAPERPLOTS+ "Shekhar_UMAP.png")





#TODO:very importmant!!! scale all the output to be in unite square (or cube)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
embedding=  scaler.fit_transform(embedding)
embedUMAP= scaler.fit_transform(embedUMAP)
z= scaler.fit_transform(z)
z = z/np.sqrt(3.1415)
prZ = projZ(z)
prZ = scaler.fit_transform(prZ)
prZ =prZ/np.sqrt(3.1415)
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
#         discontinuityUMAP= discontinuityUMAP, manytooneUMAP= manytooneUMAP, onetomany_scoreUMAP= onetomany_scoreUMAP, marker_similarity_scoreUMAP= marker_similarity_scoreUMAP[1],
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
np.mean(onetomany_scoreDCAE[29])
np.mean(marker_similarity_scoreDCAE[29])
np.mean(onetomany_scoreUMAP[29,:])
np.mean(marker_similarity_scoreUMAP[29])
np.mean(onetomany_scoreSAUCIE[29,:])
np.mean(marker_similarity_scoreSAUCIE[29])

np.mean(discontinuityDCAE_prZ)
np.mean(manytooneDCAE_prZ)
np.mean(onetomany_scoreDCAE_prZ[29,:])
np.mean(marker_similarity_scoreDCAE_prZ[1][29])

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



plt.hist(z,250)
plt.hist(embedding,250)
PAPERPLOTS  = './PAPERPLOTS/'
#build grpahs using above data
# now build plots and tables. 2 plots: 1 for onetomany_score, 1 marker_similarity_scoreDCAE on 2 methods
# table: Discontinuity and manytoone (2 columns) with 3 rows, each per method. Save as a table then stack with output on other  data , to create the final table
median_marker_similarity_scoreDCAE = np.median(marker_similarity_scoreDCAE, axis=1);
median_marker_similarity_scoreSAUCIE = np.median(marker_similarity_scoreSAUCIE, axis=1);
median_marker_similarity_scoreUMAP = np.median(marker_similarity_scoreUMAP, axis=1);
df_sim = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_marker_similarity_scoreDCAE[1:], 'SAUCIE': median_marker_similarity_scoreSAUCIE[1:], 'UMAP': median_marker_similarity_scoreUMAP[1:]})
#fig1, fig2 = plt.subplots()
plt.plot('k', 'DCAE', data=df_sim, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_sim, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_sim, marker='x', color='olive', linewidth=2)
plt.legend()
plt.savefig(PAPERPLOTS  + 'Shekhar_' + 'performance_marker_similarity_score.png')
plt.show()
plt.clf()
median_onetomany_scoreDCAE = np.median(onetomany_scoreDCAE, axis=1);median_onetomany_scoreSAUCIE = np.median(onetomany_scoreSAUCIE, axis=1);
median_onetomany_scoreUMAP = np.median(onetomany_scoreUMAP, axis=1);
df_otm = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_onetomany_scoreDCAE[1:], 'SAUCIE': median_onetomany_scoreSAUCIE[1:], 'UMAP': median_onetomany_scoreUMAP[1:]})
plt.plot('k', 'DCAE', data=df_otm, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_otm, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_otm, marker='x', color='olive', linewidth=2)
plt.savefig(PAPERPLOTS  + 'Shekhar_' + 'performance_onetomany_score.png')
plt.show()
# tables
df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.5582, 0.5482, 0.5630], 'discontinuity': [0.2767, 0.0786, 0.0320]})
df_BORAI.to_csv(PAPERPLOTS  + 'Shekhar_' + 'Borealis_measures.csv', index=False)
np.median(discontinuityDCAE)
#0.01565989388359918
np.median(manytooneDCAE)
#0.10611877287197075
np.median(discontinuityUMAP)
#0.0013421323564317491
np.median(manytooneUMAP)
#0.11770417201150978
np.median(discontinuitySAUCIE)
#0.009914790259467234
np.median(manytooneSAUCIE)
#0.17852087116020135
