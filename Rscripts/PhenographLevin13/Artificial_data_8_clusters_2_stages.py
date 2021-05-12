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
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, show3d, plot3D_performance_colors, plot2D_performance_colors, \
    preprocess_artificial_clusters, generate_clusters, generate_clusters_pentagon

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
                L[c] =  np.sin(Om*c)**4
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
        print("  Current AP is " + str(K.get_value(self.weight)))

class EpochCounterCallback(Callback):
    def __init__(self, count, count_lst):
        self.count = count
        self.count_lst = count_lst

    def on_epoch_end(self, epoch, logs={}):
        new_count = K.eval(self.count_lst[epoch])
        K.set_value(self.count, new_count)
        #print("  Current AP is " + str(K.get_value(self.count)))



#d= 5
# subspace clusters centers
#original_dim = 30
# preprocess before the deep


k = 30
k3 = k * 3
source_dir = '/home/stepan/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/simulatedData'
output_dir  = '/home/stepan/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/simulatedData/output'
#outfile = source_dir + '/8art_scaled_cor_62000_15D_positive_signal.npz'
k=30
markers = np.arange(30).astype(str)
# np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
'''
npzfile = np.load(outfile)
import seaborn as sns
sns.violinplot(data=aFrame[lbls==5,:])
sns.violinplot(data=aFrame[:,:])


Sigma = npzfile['Sigma']
lbls = npzfile['lbls'];
lbls = lbls.astype(str)
aFrame = npzfile['aFrame'];
Dist = npzfile['Dist']
Idx = npzfile['Idx']
neibALL = npzfile['neibALL']
Sigma = npzfile['Sigma']
sns.violinplot(data=aFrame[lbls==-7,:])
'''
# session set up
if tf.__version__ == '2.3.0':
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.compat.v1.disable_eager_execution()


# Model-------------------------------------------------------------------------
######################################################
# targetTr = np.repeat(aFrame, r, axis=0)
k = 30
k3 = k * 3

list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
noisy_clus, lbls = generate_clusters_pentagon(num_noisy = 10, branches_loc = [0,0],  sep=3/2, pent_size=3/2)
aFrame=noisy_clus
aFrame= aFrame/np.max(aFrame)
#cm=np.corrcoef(aFrame[lbls==0,:], rowvar =False)



#scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
#scaler.fit_transform(aFrame)
#sns.violinplot(data=aFrame[lbls==6,:])

fig, axs = plt.subplots(nrows=8)
sns.violinplot(data=aFrame[lbls==0,:],  ax=axs[0]).set_title('0', rotation=-90, position=(1, 1), ha='left', va='bottom')
axs[0].set_ylim(0, 1)
sns.violinplot(data=aFrame[lbls==1,:],  ax=axs[1]).set_title('1', rotation=-90, position=(1, 2), ha='left', va='center')
axs[1].set_ylim(0, 1)
sns.violinplot(data=aFrame[lbls==2,:],  ax=axs[2]).set_title('2', rotation=-90, position=(1, 2), ha='left', va='center')
axs[2].set_ylim(0, 1)
sns.violinplot(data=aFrame[lbls==3,:],  ax=axs[3]).set_title('3', rotation=-90, position=(1, 2), ha='left', va='center')
axs[3].set_ylim(0, 1)
sns.violinplot(data=aFrame[lbls==4,:],  ax=axs[4]).set_title('4', rotation=-90, position=(1, 2), ha='left', va='center')
axs[4].set_ylim(0, 1)
sns.violinplot(data=aFrame[lbls==5,:],  ax=axs[5]).set_title('5', rotation=-90, position=(1, 2), ha='left', va='center')
axs[5].set_ylim(0, 1)
sns.violinplot(data=aFrame[lbls==6,:],  ax=axs[6]).set_title('6', rotation=-90, position=(1, 2), ha='left', va='center')
axs[6].set_ylim(0, 1)
sns.violinplot(data=aFrame[lbls==-7,:], ax=axs[7]).set_title('7', rotation=-90, position=(1, 2), ha='left', va='center')
axs[7].set_ylim(0, 1)


aFrame, Idx, Dist, Sigma, lbls, neibALL =  preprocess_artificial_clusters(aFrame, lbls, k=30, num_cores=12, outfile='test')


'''
mapper = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=True).fit(aFrame)
yUMAP =  mapper.transform(aFrame)

from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(aFrame)
yPCA = pca.transform(aFrame)

pca = decomposition.PCA(n_components=10)
pca.fit(aFrame)
yPCA3 = pca.transform(aFrame)



fig = plot2D_cluster_colors(yUMAP, lbls=lbls, msize=5)
fig.show()
fig = plot2D_cluster_colors(yPCA, lbls=lbls, msize=5)
fig.show()

'''

coeffCAE = 1
epochs = 500
ID = 'Art8_positive_signal_15D_constCAE_test2_branch' +  '[0,0]'   + str(coeffCAE) + '_' + str(epochs) + '_2stages_MMD'
# TODO try downweight mmd to the end of computation
#DCAE_weight = K.variable(value=0)
#DCAE_weight_lst = K.variable(np.array(frange_anneal(epochs, ratio=0)))
# TODO: Possible bug in here ast ther end of training after the switch
# check for possible discontinuities/singularities o last epochs, is shape of potenatial  to narroe at the end?

MMD_weight = K.variable(value=0)
MMD_weight_lst = K.variable( np.array(frange_anneal(int(epochs), ratio=0.80)) )


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


def DCAE3D_loss(x, x_decoded_mean):  # attempt to avoid vanishing derivative of sigmoid
    U = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
    W = K.variable(value=encoder.get_layer('intermediate2').get_weights()[0])  # N x N_hidden
    Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
    U = K.transpose(U);
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = tf.linalg.diag((tf.math.sign(u) + 1) / 2)
    m = encoder.get_layer('intermediate2').output
    dm = tf.linalg.diag((tf.math.sign(m) + 1) / 2)  # N_batch x N_hidden
    s = encoder.get_layer('z_mean').output
    #r = tf.linalg.einsum('aj->a', s ** 2)
    ds = tf.linalg.diag(tf.math.scalar_mul(0, s)+1)

    S_0W = tf.einsum('akl,lj->akj', du, U)
    S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
    # tf.print((S_1W).shape) #[None, 120]
    S_2Z = tf.einsum('akl,lj->akj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z, S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    diff_tens = tf.einsum('akl,alj->akj', diff_tens, S_0W)
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam * (K.sum(diff_tens ** 2))

def pot(alp, x):
    return np.select([(x < alp),(x >= alp)*(x<=1), x>1 ], [10, 0, 10])

alp=0.2
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

    r = tf.linalg.einsum('aj->a', s**2)

    ds= 500*tf.math.square(tf.math.abs(alp-r)) *  tf.dtypes.cast(tf.less(r, alp)  , tf.float32) + \
                (r**2-1)  * tf.dtypes.cast(tf.greater_equal(r, 1), tf.float32) + 0.1
    #0 * tf.dtypes.cast(tf.math.logical_and(tf.greater_equal(r, alp), tf.less(r, 1)), tf.float32) + \
    #ds = pot(0.1, r)
    S_0W = tf.einsum('akl,lj->akj', du, U)
    S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
    # tf.print((S_1W).shape) #[None, 120]
    S_2Z = tf.einsum('a,lj->alj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z,  S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    diff_tens = tf.einsum('akl,alj->akj', diff_tens, S_0W)
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam *(K.sum(diff_tens ** 2))


#1000.0*  np.less(r, alp).astype(int)  + \
#        0* (np.logical_and(np.greater_equal(r, alp), np.less(r, 1))).astype(int) + \
#        1000.0* np.greater_equal(r, 1).astype(int)


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
    #true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.)
    true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval = -1.5, maxval = 1.5)
    #true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=0.0, maxval=1.0)
    return compute_mmd(true_samples, z_mean)


def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    msew = tf.keras.losses.mean_squared_error(y_true, y_pred)
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(msew, normSigma * 1/SigmaTsq ) # TODO Sigma -denomiator or nominator? try reverse, schek hpw sigma computed in UMAP


def ae_loss(weight, MMD_weight_lst):
    def loss(x, x_decoded_mean):
        msew = mean_square_error_NN(x, x_decoded_mean)
        #return msew + 1*(1-MMD_weight) * loss_mmd(x, x_decoded_mean) + (MMD_weight + coeffCAE) * DCAE_loss(x, x_decoded_mean) #TODO: try 1-MMD insted 2-MMD
        return msew + 1 * (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + (coeffCAE) * DCAE_loss(x,
                                                                                                               x_decoded_mean)
        # return K.mean(msew)
    return loss
    #return K.switch(tf.equal(Epoch_count, 10),  loss1(x, x_decoded_mean), loss1(x, x_decoded_mean))


opt=tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
)

autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst), metrics=[DCAE_loss, loss_mmd,  mean_square_error_NN])

autoencoder.summary()
import tensorflow_addons as tfa
#opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
#opt=tf.keras.optimizers.Adam(
#    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
#)
#autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst), metrics=[DCAE_loss, loss_mmd,  mean_square_error_NN])


from tensorflow.keras.callbacks import Callback
save_period = 10
class plotCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % save_period  == 0 or epoch in range(200):
            z=encoder.predict([aFrame,  Sigma])
            fig = plot3D_cluster_colors(z, lbls=lbls)
            html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                               include_mathjax=False, post_script=None, full_html=True,
                               animation_opts=None, default_width='100%', default_height='100%', validate=True)
            html_dir = "/home/stepan/plots/"
            Html_file = open(html_dir + "/" + ID +'_epoch=' + str(epoch) + '_' + "_Buttons.html", "w")
            Html_file.write(html_str)
            Html_file.close()
callPlot = plotCallback()

start = timeit.default_timer()
history_multiple = autoencoder.fit([aFrame, Sigma], aFrame,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[AnnealingCallback(MMD_weight, MMD_weight_lst),
                             callPlot], verbose=2)
stop = timeit.default_timer()
z = encoder.predict([aFrame,  Sigma])
print(stop - start)
st=10;stp=epochs
fig01 = plt.figure();
plt.plot(history_multiple.history['loss'][st:stp]);
plt.title('loss')
fig02 = plt.figure();
plt.plot(history_multiple.history['DCAE_loss'][st:stp]);
plt.title('DCAE_loss')
fig03 = plt.figure();
plt.plot(history_multiple.history['loss_mmd'][st:stp]);
plt.title('loss_mmd')
fig04 = plt.figure();
plt.plot(history_multiple.history['mean_square_error_NN'][st:stp]);
plt.title('mean_square_error')
fig = plot3D_cluster_colors(z, lbls=lbls)
fig.show()

fig01 = plt.figure();
plt.plot(history_multiple.history['loss'][st:stp], label= 'loss', c = 'red');
plt.plot(history_multiple.history['DCAE_loss'][st:stp], label= 'DCAE_loss', c = 'green');
plt.plot(history_multiple.history['loss_mmd'][st:stp], label= 'loss_mmd', c = 'blue');
plt.plot(history_multiple.history['mean_square_error_NN'][st:stp], label= 'mean_square_error_NN', c = 'black');
plt.legend(loc="upper right")


encoder.save_weights(output_dir +'/'+ID + '_3D.h5')
autoencoder.save_weights(output_dir +'/autoencoder_'+ID + '_3D.h5')
np.savez(output_dir +'/'+ ID + '_latent_rep_3D.npz', z = z)

encoder.load_weights(output_dir +'/'+ID + '_3D.h5')
autoencoder.load_weights(output_dir +'/autoencoder_'+ID + '_3D.h5')
encoder.summary()
z = encoder.predict([aFrame, Sigma])

import sys
os.chdir('/home/stepan/PycharmProjects/BIOIBFO25L/')
sys.path.append("/home/stepan/PycharmProjects/BIOIBFO25L/SAUCIE")
data = aFrame

fig = plot3D_cluster_colors(z, lbls=lbls)
fig.show()
fig =plot3D_marker_colors(z, data=aFrame, markers=list(markers[0:15]), sub_s = 62000, lbls=lbls)
fig.show()

# cluster toplogy
# 0 1 2 3 4
#       5 6
# clustering UMAP representation
mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(aFrame)
embedUMAP =  mapper.transform(aFrame)
#np.savez('Art8_' + 'embedUMAP.npz', embedUMAP=embedUMAP)
embedUMAP = np.load('Art8_' + 'embedUMAP.npz')['embedUMAP']
#labelsHDBscanUMAP= [str(x) for  x in labelsHDBscanUMAP]#
fig = plot2D_cluster_colors(embedUMAP, lbls=lbls)
fig.show()

#generate list of barnches:





######################################3
# try SAUCIE

import sys
#sys.path.append("/home/stepan/SAUCIE/")
data = aFrame
from importlib import reload
import SAUCIE
reload(SAUCIE)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
saucie = SAUCIE.SAUCIE(data.shape[1])
loadtrain = SAUCIE.Loader(data, shuffle=True)
saucie.train(loadtrain, steps=10000)

loadeval = SAUCIE.Loader(data, shuffle=False)
embedding = saucie.get_embedding(loadeval)
number_of_clusters, clusters = saucie.get_clusters(loadeval)
np.savez('Art8_' + 'embedSAUCIE.npz', embedding=embedding,number_of_clusters=number_of_clusters, clusters=clusters)

import os
os.chdir('/home/stepan/PycharmProjects/BIOIBFO25L')
os.getcwd()

embedding = np.load('Art8_' + 'embedSAUCIE.npz')['embedding']
clusters= np.load('Art8_' + 'embedSAUCIE.npz')['clusters']
#print(compute_cluster_performance(lbls,  clusters))
#print(compute_cluster_performance(lbls[lbls!='"Unassgined"'], clusters[lbls!='"Unassgined"']))
#clusters= [str(x) for  x in clusters]
#fig = plot3D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(clusters)), lbls=np.asarray(clusters))
#fig.show()
fig = plot2D_cluster_colors(embedding[:,:], lbls=lbls, msize=5)
fig.show()
#1 TODO:  add all the performance metrics and run on Levine32. If ok, create package for installation



