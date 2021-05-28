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
outfile = '/home/grines02/PycharmProjects/BIOIBFO25L/data/WeberLabels/Nowicka2017euclid_scaled.npz'
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


batch_size = 256
original_dim = 24
latent_dim = 3
intermediate_dim = 12


epochs = 10

epsilon_std = 1.0

features = aFrame
IdxF = Idx[:, ]
nrow = np.shape(features)[0]
rk = range(k3)

# plt.hist(cut_neibF[cut_neibF!=0],50)

# [aFrame, neibF, cut_neibF, weight_neibF]
# training set
# targetTr = np.repeat(aFrame, r, axis=0)
targetTr = aFrame

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



def kl_loss(x, x_decoded_mean):
    return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

#annealing variables

# KL weight (to be used by total loss and by annealing scheduler)




# y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [x_decoded_mean])
vae.summary()

# vae.set_weights(trained_weight)


# from keras.utils import plot_model
# plot_model(vae, to_file='/mnt/f/Brinkman group/current/Stepan/PyCharm/PhenographLevin13/model.png')

''' this model maps an input to its reconstruction'''


#adam = Adam(lr=learning_rate, epsilon=0.001)
# ae.compile(optimizer=adam, loss=ae_loss)
mse = tf.keras.losses.MeanSquaredError()
m = tf.keras.metrics.MeanSquaredError()
vae.compile(optimizer='adam',  loss= mse, metrics=[m,kl_loss ])
# ae.get_weights()

#checkpoint = ModelCheckpoint('.', monitor='loss', verbose=1, save_best_only=True, mode='max')
# logger = DBLogger(comment="An example run")
#def newloss(x, x_decoded_mean):
#    return mse(x, x_decoded_mean) + kl_loss(x, x_decoded_mean)

class LossCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch,  logs=None):
    if (epoch < 10):
      vae.add_loss(mse(x, x_decoded_mean))
    else:
      print(epoch)
      def custom_loss(x, x_decoded_mean):
            def _custom_loss():
                loss= mse(x, x_decoded_mean)+ epoch * kl_loss(x, x_decoded_mean)
                return  loss
            return _custom_loss
      vae.add_loss(lambda: custom_loss(x, x_decoded_mean))
'''
class NewCallback(Callback):
    def __init__(self, current_epoch):
        self.current_epoch = current_epoch

    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.current_epoch, epoch)

def loss_wrapper(t_change, current_epoch):
    def custom_loss(y_true, y_pred):
        # compute loss_1 and loss_2
        bool_case_1=K.less(current_epoch,t_change)
        num_case_1=K.cast(bool_case_1,"float32")
        loss = (num_case_1)*loss_1 + (1-num_case_1)*loss_2
        return loss
    return custom_loss
'''
#https://colab.research.google.com/github/SachsLab/IntracranialNeurophysDL/blob/master/notebooks/05_04_betaVAE_TFP.ipynb#scrollTo=8DiCpDslah5W
K.clear_session()
K.set_floatx('float32')
tf.random.set_seed(42)

kl_beta = K.variable(1.0, name="kl_beta")
if self.kl_warmup:
    kl_warmup_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs: K.set_value(
            kl_beta, K.min([epoch / self.kl_warmup, 1])
        )
    )





z_mean, z_log_sigma = KLDivergenceLayer(beta=kl_beta)([z_mean, z_log_sigma])

vae.compile(optimizer='adam',  loss= Kloss, metrics=[m,kl_loss ])

N_EPOCHS = 20
history = vae.fit(x=aFrame, y=aFrame,
                        epochs=N_EPOCHS,
                        callbacks=[kl_beta_cb],
                        verbose=1)




# here to set weights ti uniform , by default they are perplexity weighted
history = vae.fit(x=aFrame, y=aFrame,
                  batch_size=batch_size,
                  epochs=15,
                  shuffle=True,
                  callbacks=[LossCallback()], verbose=2)
stop = timeit.default_timer()
# vae.save('WEBERCELLS3D32lambdaPerpCorr0.01h5')
#vae.load('WEBERCELLS3D.h5')

# ae=load_model('Wang0_modell1.h5', custom_objects={'mean_square_error_weighted':mean_square_error_weighted, 'ae_loss':
#  ae_loss, 'mean_square_error_weightedNN' : mean_square_error_weightedNN})
# ae = load_model('Wang0_modell1.h5')

fig03 = plt.figure();
plt.plot(history.history['loss']);
plt.plot(history.history['kl_loss']);

# encoder = Model([x, neib, cut_neib], encoded2)
encoder = Model(x, z_mean)
print(encoder.summary())

# predict and extract latent variables

# gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = vae.predict([sourceTr, neibF_Tr, cut_neibF_Tr, Sigma, weight_neibF])
len(x_test_vae)
# np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_vaeMoeWeights.txt', x_test_vae)
# x_test_vae=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_ae001Pert.txt.txt')
x_test_enc = encoder.predict(aFrame)

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

from plotly import __version__
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
standard_embedding=np.loadtxt('/home/grines02/PycharmProjects/BIOIBFO25L/data/data/umap_embedding.txt')
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
