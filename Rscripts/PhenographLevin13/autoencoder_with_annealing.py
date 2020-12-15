'''This script to visualise cytoff data using deep variational autoencoder with neighbourhood denoising and
contracting
TODO ?? pretraining
Now with entropic weighting of neibourhoods
Innate‐like CD8+ T‐cells and NK cells: converging functions and phenotypes
Ayako Kurioka,corresponding author 1 , 2 Paul Klenerman, 1 , 2 and Christian B. Willbergcorresponding author 1 , 2
CD8+ T Cells and NK Cells: Parallel and Complementary Soldiers of Immunotherapy
Jillian Rosenberg1 and Jun Huang1,2
'''

import keras
import tensorflow as tf
import numpy as np
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


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


class AnnealingCallback(Callback):
    def __init__(self, weight, kl_weight_lst):
        self.weight = weight
        self.kl_weight_lst = kl_weight_lst

    def on_epoch_end(self, epoch, logs={}):
        new_weight = K.eval(self.kl_weight_lst[epoch])
        K.set_value(self.weight, new_weight)
        print("Current KL Weight is " + str(K.get_value(self.weight)))


# import vpsearch as vp
# def find_neighbors(data, k_, metric='manhattan', cores=12):
#   res = vp.find_nearest_neighbors(data, k_, cores, metric)
#   return {'dist':np.array(res[1]), 'idx': np.int32(np.array(res[0]))}


# from libKMCUDA import kmeans_cuda, knn_cuda
# def find_neighbors(data, k_, metric='euclidean', cores=12):
#    ca = kmeans_cuda(np.float32(data), 25, metric="euclidean", verbosity=1, seed=3, device=0)
#    neighbors = knn_cuda(k_, np.float32(data), *ca, metric=metric, verbosity=1, device=0)
#    return {'dist':0, 'idx': np.int32(neighbors)}

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
source_dir = "/home/grines02/PycharmProjects/BIOIBFO25L/data/data/"
'''
#file_list = glob.glob(source_dir + '/*.txt')
data0 = np.genfromtxt(source_dir + "d_matrix.txt"
, names=None, dtype=float, skip_header=1)
aFrame = data0[:,1:]
# set negative values to zero
aFrame[aFrame < 0] = 0
patient_table = np.genfromtxt(source_dir + "label_patient.txt", names=None, dtype='str', skip_header=1, delimiter=" ", usecols = (1, 2, 3))
lbls=patient_table[:,0]
len(lbls)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=12)
Idx = nb['idx']; Dist = nb['dist']
'''
data0 = np.genfromtxt(source_dir + "d_matrix.txt"
                      , names=None, dtype=float, skip_header=1)
clust = np.genfromtxt(source_dir + "label_patient.txt", names=None, dtype='str', skip_header=1, delimiter=" ",
                      usecols=(1, 2, 3))[:, 0]
outfile = outfile = '/home/grines02/PycharmProjects/BIOIBFO25L/data/WeberLabels/Nowicka2017euclid_scaled.npz'
# np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls,  Dist=Dist)
npzfile = np.load(outfile)
lbls = npzfile['lbls'];
Idx = npzfile['Idx'];
aFrame = npzfile['aFrame'];
Dist = npzfile['Dist']
# transform labels into natural numbers
patient_table = lbls
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(lbls)
lbls = le.transform(lbls)

# lbls2=npzfile['lbls'];Idx2=npzfile['Idx'];aFrame2=npzfile['aFrame'];
# cutoff2=npzfile['cutoff']; Dist2 =npzfile['Dist']
cutoff = np.repeat(0.1, 24)
batch_size = 50
original_dim = 24
latent_dim = 3
intermediate_dim = 12
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

epochs = 120
# annealing schedule
kl_weight = K.variable(value=0)
kl_weight_lst = K.variable(np.array(frange_cycle_linear(0.0, 1.0, epochs, n_cycle=4, ratio=0.95)))

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
#outfile = source_dir + '/Nowicka2017euclidFeatures.npz'
# np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
#npzfile = np.load(outfile)
#weight_distALL = npzfile['weight_distALL'];
#cut_neibF = npzfile['cut_neibF'];
#neibALL = npzfile['neibALL']
#np.sum(cut_neibF != 0)
# plt.hist(cut_neibF[cut_neibF!=0],50)


targetTr = aFrame

sourceTr = aFrame
tf.compat.v1.disable_eager_execution()
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
# h.set_weights(ae.layers[1].get_weights())
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='relu')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

   # return tf.multiply(weightedN, 0.5)


# def mean_square_error_NN(y_true, y_pred):
#    #dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
#    dst = tf.multiply(K.transpose(1/SigmaTsq), K.mean(K.square(neib - K.expand_dims(y_pred, 1)), axis=-1))
#    #weightedN = K.dot(dst, K.transpose(weight_neib))
#    weightedN = K.dot(dst, K.transpose(weight_neib))
#    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
#    return  tf.multiply(weightedN, 0.5 )


def kl_loss(x, x_decoded_mean):
    return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


# annealing variables

# KL weight (to be used by total loss and by annealing scheduler)
mse = tf.keras.losses.MeanSquaredError()
def vae_loss(weight, kl_weight_lst):
    def loss(x, x_decoded_mean):
        msew = mse(x, x_decoded_mean)
        # pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
        return K.mean(msew + kl_weight * kl_loss(x, x_decoded_mean))
        # return K.mean(msew)
    return loss


# y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model([x], x_decoded_mean)
vae.summary()

# vae.set_weights(trained_weight)


# from keras.utils import plot_model
# plot_model(vae, to_file='/mnt/f/Brinkman group/current/Stepan/PyCharm/PhenographLevin13/model.png')

''' this model maps an input to its reconstruction'''

learning_rate = 1e-3


# adam = Adam(lr=learning_rate, epsilon=0.001)

# ae.compile(optimizer=adam, loss=ae_loss)
vae.compile(optimizer='adam', loss=vae_loss(kl_weight, kl_weight_lst), metrics=[kl_loss])
# ae.get_weights()


# logger = DBLogger(comment="An example run")

start = timeit.default_timer()

# here to set weights ti uniform , by default they are perplexity weighted
weight_neibF = np.full((nrow, k), 1 / k)
history = vae.fit([sourceTr], targetTr,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[AnnealingCallback(kl_weight, kl_weight_lst)])
stop = timeit.default_timer()
# vae.save('WEBERCELLS3D32lambdaPerpCorr0.01h5')
# vae.load('WEBERCELLS3D.h5')

# ae=load_model('Wang0_modell1.h5', custom_objects={'mean_square_error_weighted':mean_square_error_weighted, 'ae_loss':
#  ae_loss, 'mean_square_error_weightedNN' : mean_square_error_weightedNN})
# ae = load_model('Wang0_modell1.h5')

print(stop - start)
fig0 = plt.figure();
plt.plot(history.history['kl_loss'][:]);

fig02 = plt.figure();
plt.plot(history.history['loss']);

# encoder = Model([x, neib, cut_neib], encoded2)
encoder = Model([x], z_mean)
print(encoder.summary())

# predict and extract latent variables

# gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = vae.predict([sourceTr])
len(x_test_vae)
# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_vaeMoeWeights.txt', x_test_vae)
# x_test_vae=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_ae001Pert.txt.txt')
x_test_enc = encoder.predict([sourceTr])

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

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter

# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
# x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt')

nrow = np.shape(x_test_enc)[0]
# subsIdx=np.random.choice(nrow,  500000)

x = x_test_enc[:, 0]
y = x_test_enc[:, 1]
z = x_test_enc[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
plot([Scatter3d(x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=1,
                    color=lbls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=clust,
                hoverinfo='text')])

x = x_test_enc[:, 0]
y = x_test_enc[:, 1]
z = x_test_enc[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
plot([Scatter3d(x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=1,
                    color=aFrame[:, 0],  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=clust,
                hoverinfo='text')])

# umap graph to compare

# np.savetxt('/home/grines02/PycharmProjects/BIOIBFO25L/data/data/umap_embedding.txt', standard_embedding)
standard_embedding = np.loadtxt('/home/grines02/PycharmProjects/BIOIBFO25L/data/data/umap_embedding.txt')
'''
import umap
standard_embedding = umap.UMAP(n_neighbors=30, n_components=3).fit_transform(aFrame)
'''
x = standard_embedding[:, 0]
y = standard_embedding[:, 1]
z = standard_embedding[:, 2]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
plot([Scatter3d(x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=1,
                    color=lbls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=clust,
                hoverinfo='text')])

plot([Scatter3d(x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=1,
                    color=aFrame[:, 0],  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=clust,
                hoverinfo='text')])

cl = 2
plot([Scatter3d(x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=[2 if x == cl else 0.5 for x in lbls],
                    color=['red' if x == cl else 'green' for x in lbls],  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=patient_table[:, 0],
                hoverinfo='text')])

x = x_test_enc[:, 0]
y = x_test_enc[:, 1]
plot([Scatter(x=x, y=y,
              mode='markers',
              marker=dict(
                  size=1,
                  color=lbls / 5,  # set color to an array/list of desired values
                  colorscale='Viridis',  # choose a colorscale
                  opacity=1,
              ),
              text=patient_table[:, 0],
              hoverinfo='text')])
