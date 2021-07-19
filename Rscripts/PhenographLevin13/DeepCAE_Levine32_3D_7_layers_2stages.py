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



import ctypes
from numpy.ctypeslib import ndpointer
lib = ctypes.cdll.LoadLibrary("/home/grinek/PycharmProjects/BIOIBFO25L/Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #Sigma
                ctypes.c_size_t]

#d= 5
# subspace clusters centers
#original_dim = 30
# preprocess before the deep


k = 30
k3 = k * 3
coeffCAE = 1
epochs = 100
ID = 'Levine32_MMD_01_3D_DCAE_h96_h32_hidden_7_layers'+ str(coeffCAE) + '_' + str(epochs) + '_2stages_not+scaled'
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'CyTOFdataPreprocess/'
output_dir  = DATA_ROOT + 'Real_sets/DCAE_output/'

'''
#data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
#https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816

data0 = np.genfromtxt(source_dir + "/Levine32_data.csv" , names=None, dtype=float, skip_header=1, delimiter=',')
aFrame = data0[:,:] 
aFrame.shape
aFrame.min(axis=0)
aFrame.max(axis=0)
sns.violinplot(data= aFrame, bw = 0.1);plt.show()
aFrame.min()
# set negative values to zero and shift minima to zero
aFrame = aFrame  - aFrame.min(axis=0)

aFrame.min(axis=0)
aFrame.max(axis=0)
sns.violinplot(data= aFrame, bw = 0.1);plt.show()
lbls= np.genfromtxt(source_dir + "Levine32_population.csv" , names=None, skip_header=0, delimiter=',', dtype='U100')
#randomize order
IDX = np.random.choice(aFrame.shape[0], aFrame.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
aFrame= aFrame[IDX,:]
lbls = lbls[IDX]
len(lbls)
#scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
#scaler.fit_transform(aFrame)

aFrame= aFrame/np.max(aFrame)
#cm=np.corrcoef(aFrame[lbls==0,:], rowvar =False)
sns.violinplot(data= aFrame, bw = 0.1);plt.show()

ly=0.1
lx=-0.1
fig, axs = plt.subplots(nrows=8)
sns.violinplot(data=aFrame[lbls=='"CD8_T_cells"',:],  ax=axs[0]).set_title('0', rotation=-90, position=(1, 1), ha='left', va='bottom')
axs[0].set_ylim(lx, ly)
sns.violinplot(data=aFrame[lbls=='"CD16+_NK_cells"',:],  ax=axs[1]).set_title('1', rotation=-90, position=(1, 2), ha='left', va='center')
axs[1].set_ylim(lx, ly)
sns.violinplot(data=aFrame[lbls=='"CD16-_NK_cells"',:],  ax=axs[2]).set_title('2', rotation=-90, position=(1, 2), ha='left', va='center')
axs[2].set_ylim(lx, ly)
sns.violinplot(data=aFrame[lbls=='"CD34+CD38+CD123+_HSPCs"',:],  ax=axs[3]).set_title('3', rotation=-90, position=(1, 2), ha='left', va='center')
axs[3].set_ylim(lx, ly)
sns.violinplot(data=aFrame[lbls=='"CD34+CD38+CD123-_HSPCs"',:],  ax=axs[4]).set_title('4', rotation=-90, position=(1, 2), ha='left', va='center')
axs[4].set_ylim(lx, ly)
sns.violinplot(data=aFrame[lbls=='"CD34+CD38lo_HSCs"',:],  ax=axs[5]).set_title('5', rotation=-90, position=(1, 2), ha='left', va='center')
axs[5].set_ylim(lx, ly)
sns.violinplot(data=aFrame[lbls=='"CD34+CD38lo_HSCs"',:],  ax=axs[6]).set_title('6', rotation=-90, position=(1, 2), ha='left', va='center')
axs[6].set_ylim(lx, ly)
sns.violinplot(data=aFrame[lbls=='"CD8_T_cells"',:], ax=axs[7]).set_title('7', rotation=-90, position=(1, 2), ha='left', va='center')
axs[7].set_ylim(lx, ly)




cm=np.corrcoef(aFrame[lbls=='"CD16+_NK_cells"',:], rowvar =False)
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
results = Parallel(n_jobs=16, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
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
outfile = source_dir + '/Levine32euclid_not_scaled.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)
'''
outfile = source_dir + '/Levine32euclid_not_scaled.npz'
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





# "CD16+_NK_cells"
# "CD16-_NK_cells"
# "Mature_B_cells"

# session set up

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()


# Model-------------------------------------------------------------------------
######################################################
# targetTr = np.repeat(aFrame, r, axis=0)

# Model-------------------------------------------------------------------------
######################################################
# targetTr = np.repeat(aFrame, r, axis=0)
k = 30
k3 = k * 3




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
        return msew + 1*(1-MMD_weight) * loss_mmd(x, x_decoded_mean) + (MMD_weight + coeffCAE) * DCAE_loss(x, x_decoded_mean) #TODO: try 1-MMD insted 2-MMD
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
            html_dir = "/media/grinek/Seagate/Real_sets/DCAE_output"
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
st=0; stp=100
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

#ID='Levine32_MMD_1_3D_DCAE_5'
#encoder.load_weights('/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/Levine32_3D_DCAE_10_3D.h5')
#autoencoder.load_weights('/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/autoencoder_Levine32_MMD_1_3D_DCAE_freezing_experiment5_3D.h5')
#ID = 'Levine32_MMD_01_3D_DCAE_5_3500_kernelInit'
encoder.load_weights(output_dir +''+ID + '_3D.h5')
autoencoder.load_weights(output_dir +'autoencoder_'+ID + '_3D.h5')
encoder.summary()
z = encoder.predict([aFrame, neibF_Tr,  Sigma])

#- visualisation and pefroramnce metric-----------------------------------------------------------------------------------------------
# np.savetxt('/mnt/f/Brinkman group/current/grinek/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
# x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/grinek/WangData/WangDataPatient/x_test_encBcells3d.txt')
x = z[:, 0]
y = z[:, 1]
zz = z[:, 2]

fig = plot3D_cluster_colors(z, camera = dict(eye = dict(x=-0.2,y=0.2,z=1.5)), lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "media/grinek/Seagate/Real_sets/DCAE_output"
Html_file= open(html_dir + "/"+ID + "_Buttons.html","w")
Html_file.write(html_str)
Html_file.close()


fig =plot3D_marker_colors(z, data=aFrame, markers=markers, sub_s = 50000, lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/grinekv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Markers.html","w")
Html_file.write(html_str)
Html_file.close()

#stretch low signal
aFramesqrt =np.sqrt(aFrame)
fig =plot3D_marker_colors(z, data=aFramesqrt, markers=markers, sub_s = 50000, lbls=lbls)
fig.show()

# clusteng hidden representation

# clustering UMAP representation
#mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(aFrame)
#embedUMAP =  mapper.transform(aFrame)
#np.savez('LEVINE32_' + 'embedUMAP.npz', embedUMAP=embedUMAP)
#embedUMAP = np.load('LEVINE32_' + 'embedUMAP.npz')['embedUMAP']
######################################3
# try SAUCIE
os.chdir('/home/grines02/PycharmProjects/BIOIBFO25L/')
sys.path.append("/home/grines02/PycharmProjects/BIOIBFO25L/SAUCIE")
data = aFrame
from importlib import reload
#import SAUCIE
#reload(SAUCIE)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#saucie = SAUCIE.SAUCIE(data.shape[1])
#loadtrain = SAUCIE.Loader(data, shuffle=True)
#saucie.train(loadtrain, steps=100000)

#loadeval = SAUCIE.Loader(data, shuffle=False)
#embedding = saucie.get_embedding(loadeval)
#np.savez('LEVINE32_' + 'embedSAUCIE_100000.npz', embedding=embedding)
embedding = np.load('LEVINE32_' + 'embedSAUCIE_100000.npz')['embedding']
#number_of_clusters, clusters = saucie.get_clusters(loadeval)
#print(compute_cluster_performance(lbls,  clusters))
#clusters= [str(x) for  x in clusters]
#fig = plot3D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(clusters)), lbls=np.asarray(clusters))
#fig.show()
fig = plot2D_cluster_colors(embedding, lbls=lbls)
fig.show()



z_mr =  neighbour_marker_similarity_score(z, aFrame, kmax=90)
embedding_mr =  neighbour_marker_similarity_score(embedding, aFrame, kmax=90)
embedUMAP_mr = neighbour_marker_similarity_score(embedUMAP, aFrame, kmax=90)
np.savez(ID + '_marker_similarity.npz', z_mr = z_mr,  embedding_mr=embedding_mr, embedUMAP_mr=embedUMAP_mr)
npobj =  np.load(ID + '_marker_similarity.npz')
z_mr,embedding_mr,embedUMAP_mr  = npobj ['z_mr'] , npobj['embedding_mr'],  npobj['embedUMAP_mr'],
z_mr[89]
embedding_mr[89]
embedUMAP_mr[89]
# plot
df = pd.DataFrame({'k':range(0,90)[2:],  'DCAE': z_mr[3:], 'SAUCIE': embedding_mr[3:], 'UMAP': embedUMAP_mr[3:]})

# multiple line plot
plt.plot('k', 'DCAE', data=df, marker='o', markerfacecolor='blue', markersize=2, color='skyblue', linewidth=4)
plt.plot('k', 'SAUCIE', data=df, marker='', color='olive', linewidth=2)
plt.plot('k', 'UMAP', data=df, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.legend()

# create performance plots for paper
embedding = np.load('LEVINE32_' + 'embedSAUCIE.npz')['embedding']
embedUMAP = np.load('LEVINE32_' + 'embedUMAP.npz')['embedUMAP']
PAPERPLOTS  = './PAPERPLOTS/'
#3 plots for paper
# how to export as png: https://plotly.com/python/static-image-export/ 2D
fig = plot3D_cluster_colors(z[lbls !='"unassigned"', :  ], camera = dict(eye = dict(x=-0.2,y=0.2,z=1.5)),
                            lbls=lbls[lbls !='"unassigned"'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ ID+"LEVINE32.png")

fig = plot2D_cluster_colors(embedding[lbls !='"unassigned"', :  ], lbls=lbls[lbls !='"unassigned"'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ ID+ "LEVINE32_SAUCIE.png")

fig = plot2D_cluster_colors(embedUMAP[lbls !='"unassigned"', :  ], lbls=lbls[lbls !='"unassigned"'],legend=True)
fig.show()
fig.write_image(PAPERPLOTS+ ID+ "LEVINE32_UMAP.png")


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
onetomany_scoreSAUCIE= neighbour_onetomany_score(embedding, Idx, kmax =90, num_cores=12)[1]
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
# tables
df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.1011, 0.1800, 0.1182], 'discontinuity': [0.0047, 0.1800, 0.0009]})
df_BORAI.to_csv(PAPERPLOTS  + 'LEVINE32_' + ID+ 'Borealis_measures.csv', index=False)
np.median(discontinuityDCAE)
#0.004765862888760037
np.median(manytooneDCAE)
#0.1011053429110994
np.median(discontinuityUMAP)
#0.0008730
np.median(manytooneUMAP)
#0.11820161593197807
np.median(discontinuitySAUCIE)
#0.01005548636118571
np.median(manytooneSAUCIE)
#0.1800277093476539

#plot perf
fig =plot3D_performance_colors(z, perf=discontinuityDCAE, lbls=lbls)
fig.show()
fig =plot3D_performance_colors(z, perf=onetomany_scoreDCAE[30,:], lbls=lbls)
fig.show()
fig =plot3D_performance_colors(z, perf=manytooneDCAE, lbls=lbls)
fig.show()
fig =plot3D_performance_colors(z, perf=marker_similarity_scoreDCAE, lbls=lbls)
fig.show()

fig =plot2D_performance_colors(embedding, perf=discontinuitySAUCIE, lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedding, perf=onetomany_scoreSAUCIE[30,:], lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedding, perf=manytooneSAUCIE, lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedding, perf=marker_similarity_scoreSAUCIE, lbls=lbls)
fig.show()

fig =plot2D_performance_colors(embedUMAP, perf=discontinuityUMAP, lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedUMAP, perf=onetomany_scoreSAUCIE[30,:], lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedUMAP, perf=manytooneSAUCIE, lbls=lbls)
fig.show()





# add global metrics
from scipy.spatial import ConvexHull, convex_hull_plot_2d
points = np.random.rand(300000, 2)   # 30 random points in 2-D
hull = ConvexHull(points)
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#We could also have directly used the vertices of the hull, which for 2-D are guaranteed to be in counterclockwise order:
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()


hull = ConvexHull(aFrame[0:100,:])

for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#We could also have directly used the vertices of the hull, which for 2-D are guaranteed to be in counterclockwise order:
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()


def convex_hull(A):
    vertices = A.T.tolist()
    vertices = [ i + [ 1 ] for i in vertices ]
    poly = PyNormaliz.Cone(vertices = vertices)
    hull_vertices = np.array([entry[:-1] for entry in poly.VerticesOfPolyhedron()]).T
    hull_indices = find_column_indices(A, hull_vertices)
    return hull_indices
from scipy import spatial
zzz0= spatial.distance_matrix(aFrame[1000:2001,:], aFrame[1000:2001,:], p=2, threshold=1000000)
#plt.hist(zzz,100)
zzz1 =spatial.distance_matrix(z[1000:2001,:], z[1000:2001,:], p=2, threshold=1000000)
zzz2 =spatial.distance_matrix(embedding[1000:2001,:], embedding[1000:2001,:], p=2, threshold=1000000)
zzz3 =spatial.distance_matrix(embedUMAP[1000:2001,:], embedUMAP[1000:2001,:], p=2, threshold=1000000)
#plt.hist(zzz,100)
from scipy.spatial.distance import pdist
zzz0 =pdist(aFrame[1000:2001,:])
zzz1 =pdist(z[1000:2001,:])
zzz2 =pdist(embedding[1000:2001,:])
zzz3 =pdist(embedUMAP[1000:2001,:])
#TODO switch distance here into geodisec on a spheeeeere d_g = 2*arcsin(d/(2r))*r for zzz1

# Tennenbaum [43]
#http://www.sci.utah.edu/~beiwang/publications/TopoDR_BeiWang_2018.pdf
from scipy import stats
1.0-stats.spearmanr(zzz0, zzz1).correlation**2
1.0-stats.pearsonr(zzz0, zzz2).correlation**2
1.0-stats.pearsonr(zzz0, zzz3).correlation**2
from numpy import random
idx=random.choice(500500,100000)

plt.scatter(zzz0.flatten()[idx], 2*zzz1.flatten()[idx], s=0.01)
plt.scatter(zzz0.flatten()[idx], zzz2.flatten()[idx], s=0.01)
plt.scatter(zzz0.flatten()[idx], zzz3.flatten()[idx], s=0.01)

plt.scatter(zzz0.flatten()[:1000], zzz1.flatten()[:1000])
plt.scatter(zzz0.flatten()[:1000], zzz2.flatten()[:1000])
plt.scatter(zzz0.flatten()[:1000], zzz3.flatten()[:1000])


1.0-stats.spearmanr(zzz0[zzz0>0.5].flatten(), zzz1[zzz0>0.5].flatten()).correlation**2
1.0-stats.spearmanr(zzz0[zzz0>0.5].flatten(), zzz2[zzz0>0.5].flatten()).correlation**2
1.0-stats.spearmanr(zzz0[zzz0>0.5].flatten(), zzz3[zzz0>0.5].flatten()).correlation**2

# more measures:
#file:///home/grinek/Downloads/Agreement_Analysis_of_Quality_Measures_for_Dimensi.pdf
###########################################################################
#viisualization using metric as coulurs, add to supplements later
# try kit and may be modify for subset clustering (use utils evelauation to improve this)
from utils_evaluation import get_wsd_scores
embedUMAP = np.load('LEVINE32_' + 'embedUMAP.npz')['embedUMAP']
embedding = np.load('LEVINE32_' + 'embedSAUCIE.npz')['embedding']
z= np.load(output_dir +'/'+ ID + '_latent_rep_3D.npz')['z']
n_neighbors = 30
discontinuitySAUCIE, manytooneSAUCIE = get_wsd_scores(aFrame, embedding, n_neighbors, num_meandist=10000)
vmaxSAUCIE = np.percentile(np.hstack(
    (discontinuitySAUCIE,
     manytooneSAUCIE)), 95)
fig = plt.figure()
plt.hist(discontinuitySAUCIE,500)
plt.title('discontinuitySAUCIE' + ' median ' + str(np.median(discontinuitySAUCIE)))
plt.show()
fig2 = plt.figure()
plt.hist(manytooneSAUCIE,500)
plt.title('manytooneSAUCIE' + ' median ' + str(np.median(manytooneSAUCIE)))
plt.show()

discontinuityZ, manytooneZ = get_wsd_scores(aFrame, z, n_neighbors, num_meandist=10000)
vmaxZ = np.percentile(np.hstack(
    (discontinuityZ,
     manytooneZ)), 95)
fig3 = plt.figure()
plt.hist(discontinuityZ,500)
plt.title('discontinuityZ' + ' median ' + str(np.median(discontinuityZ)))
plt.show()
fig4 = plt.figure()
plt.hist(manytooneZ,500)
plt.title('manytooneZ' + ' median ' + str(np.median(manytooneZ)))
plt.show()

discontinuityUMAP, manytooneUMAP = get_wsd_scores(aFrame, embedUMAP, n_neighbors, num_meandist=10000)
vmaxUMAP = np.percentile(np.hstack(
    (discontinuityUMAP,
     manytooneUMAP)), 95)
fig5 = plt.figure()
plt.hist(discontinuityUMAP,500)
plt.title('discontinuityUMAP' + ' median ' + str(np.median(discontinuityUMAP)))
plt.show()
fig6 = plt.figure()
plt.hist(manytooneUMAP,500)
plt.title('manytooneUMAP' + ' median ' + str(np.median(manytooneUMAP)))
plt.show()





# compare SAUCIE results and ours using cross-decomposition analysis
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
cca = CCA(n_components=2)
cca.fit(aFrame, prZ)
cca.score(aFrame, prZ)

cca.fit(aFrame, z)
cca.score(aFrame, z)

cca.fit(aFrame, embedding )
cca.score(aFrame, embedding)

cca.fit(aFrame, embedUMAP )
cca.score(aFrame, embedUMAP)

#get spherical
# https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
# https://pypi.org/project/ai.cs/
# basemap
#https://stackoverflow.com/questions/7889826/python-basemap-stereographic-map



from statsmodels.multivariate.cancorr import CanCorr
cc1 = CanCorr(z, aFrame)
cc1._fit()
res1 = cc1.corr_test()
res1.summary()

cc2 = CanCorr(embedding, aFrame)
cc2._fit()
res2 = cc2.corr_test()
res2.summary()

from sklearn.manifold import Isomap
embedISO = Isomap(n_components=2)
sub_idx = np.random.choice(range(np.shape(prZ)[0]), 50000, replace=False)
fit = embedISO.fit(prZ[sub_idx,:])
sub_idx = np.random.choice(range(np.shape(prZ)[0]), 100000, replace=False)
ISOmap = fit.transform(prZ[sub_idx,:])
fig = plot2D_cluster_colors(x=ISOmap[:, 0],y=ISOmap[:, 1], lbls=lbls[sub_idx])
fig.show()

from sklearn.cross_decomposition import  CCA
cca = CCA(n_components=2)
cca.fit(aFrame[sub_idx,:], ISOmap)
U_c, V_c = cca.fit_transform(aFrame[sub_idx,:], ISOmap)
np.corrcoef(U_c.T, V_c.T)[0,1]
cca.score(aFrame[sub_idx,:], ISOmap)
cca.fit(aFrame[sub_idx,:], embedding[sub_idx,:] )
cca.score(aFrame[sub_idx,:], embedding[sub_idx,:])

cca.fit(ISOmap, aFrame[sub_idx,:])
U_c, V_c = cca.fit_transform(ISOmap, aFrame[sub_idx,:])
np.corrcoef(U_c.T, V_c.T)[0,1]
cca.score(ISOmap, aFrame[sub_idx,:])
cca.fit(embedding[sub_idx], aFrame[sub_idx,:])
cca.score(embedding[sub_idx], aFrame[sub_idx,:])



#cca.fit(aFrame, z)
#cca.score(aFrame, z)






from sklearn.decomposition import PCA, KernelPCA
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=False, gamma=10)
X_kpca = kpca.fit_transform(z)

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
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/grinekv1.github.io/_includes"
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











