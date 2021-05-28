'''This script to clean cytoff data using deep stacked autoencoder with neighbourhood denoising and
contracting
'''
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
from keras.utils import multi_gpu_model
import timeit
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.constraints import maxnorm
import readline
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
stsne = importr('stsne')
from sklearn.preprocessing import MinMaxScaler
from kerasvis import DBLogger
#Start the keras visualization server with
#export FLASK_APP=kerasvis.runserver
#flask run
import seaborn as sns
import warnings
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import regularizers
import champ
import igraph

def naive_power(m, n):
    m = np.asarray(m)
    res = m.copy()
    for i in range(1,n):
        res *= m
    return res

class CustomMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('mean_square_error_weightedNN'):
                print
                logs[k]
            if k.endswith('mean_square_error_weighted'):
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
        current_train = logs.get('vae_loss')
        #current_train = logs.get('loss')
        if current_train is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(current_train,self.criterion):
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))




def find_neighbors(data, k_, metric='L2', cores=12):
    res = stsne.matrix_search(data, k_, cores)
    return {'dist':np.array(res[0]), 'idx': np.int32(np.array(res[1])-1)}


import fcsparser

ROOT = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL'
DATA_DIR = "/benchmark_data_sets/"
DATA = 'Samusik_all.fcs'

meta, data0 = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=False, reformat_meta=True, output_format='ndarray')
meta['_channels_']
#extract markers
l=data0[:, 53]
data0 = data0[:, 8:47]
#from keras.datasets import mnist
#load data
k=30
data0[data0<0]=0


#file_list = glob.glob(source_dir + '/*.txt')

lbls = l
aFrame = data0
cutoff = np.ones(39)*0.1
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k, metric='L2', cores=12)
Idx=nb['idx']

outfile = 'MatSamusik001.npz'
np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls, cutoff=cutoff)
npzfile = np.load(outfile)
lbls=npzfile['lbls'];Idx=npzfile['Idx'];aFrame=npzfile['aFrame'];




#model parameters
batch_size = 12
original_dim = 39
latent_dim = 117
intermediate_dim = 78
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]
epochs = 10
epsilon_std = 1.0
U=10#energy barier

k=30#nearest neighbours

#generate neighbours data
features=aFrame
IdxF = Idx[:, :k]
nrow = np.shape(features)[0]
b=0
neibF = np.zeros((nrow, k, original_dim))
cnz = np.zeros((original_dim))
cut_neibF = np.zeros((nrow, original_dim))
weight_distF = np.zeros((nrow, k))
weight_neibF = np.zeros((nrow, k))
rk=range(k)

sigmaBer = np.sqrt(cutoff*(1-cutoff))
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing
def singleInput(i):
     #nei = np.zeros(( k, original_dim))
     #cut_nei = np.zeros((nrow, original_dim))
     #weight_di = np.zeros((k))
     #weight_nei = np.zeros((k))
     nei =  features[IdxF[i,:],:]
     cnz=[np.sum(np.where(nei[:, j] == 0, 0,1))/ k for j in range(original_dim)]
     cut_nei= np.array([0 if (cnz[j] >= cutoff[j] or cutoff[j]>0.5) else (U/naive_power(cutoff[j], 2)) * naive_power((cutoff[j] - cnz[j])/sigmaBer[j] , 2) for j in range(original_dim)])
     weight_di = [sum(((features[i] - nei[k_i,]) / (1 + cut_nei))**2) for k_i in rk]
     d_lock_max=np.max(weight_di)
     d_lock_min = np.min(weight_di)
     weight_nei = np.exp((-k * (weight_di- d_lock_min)/(d_lock_max-d_lock_min)))
     return [nei, cut_nei, weight_nei, i]

inputs = range(nrow)
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(singleInput)(i) for i in inputs)
for i in range(nrow):
 neibF[i,] = results[i][0]
for i in range(nrow):
    cut_neibF[i,] = results[i][1]
for i in range(nrow):
    weight_neibF[i,] = results[i][2]


#regulariztion, not feed forward
neib = Input(shape = (k, original_dim, ))
cut_neib = Input(shape = (original_dim,))
#var_dims = Input(shape = (original_dim,))
weight_neib = Input(shape = (k,))

#pretrain input layers

trained_weight = []
X_train_tmp = aFrame
for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
    print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
# Create AE and training
    pretrain_input = Input(shape=(n_in,))
    encoder = Dense(n_out, activation='tanh')(pretrain_input)
    decoder = Dense(n_in, activation='linear')(encoder)
    ae = Model(input=pretrain_input,output=decoder)
    encoder_temp = Model(input=pretrain_input, output=encoder)
    ae.compile(loss='mean_squared_error', optimizer='RMSprop')
    ae.fit(X_train_tmp, X_train_tmp, batch_size=256, epochs=10)
# Store trainined weight
    trained_weight = trained_weight + encoder_temp.get_weights()
# Update training data
    X_train_tmp = encoder_temp.predict(X_train_tmp)
ae.summary()
print('Fine-tuning:')

'''this is our input placeholder'''
x = Input(shape=(original_dim,))
''' "encoded" is the encoded representation of the input'''
encoded1 = Dense(intermediate_dim, activation='selu')(x)

encoded2 = Dense(latent_dim, activation='selu')(encoded1)

x_decoded3 = Dense(intermediate_dim, activation='selu')(encoded2)

x_decoded4 = Dense(original_dim, activation='relu')(x_decoded3)
ae = Model([x, neib, cut_neib, weight_neib], x_decoded4)
#ae = Model([x, neib, cut_neib], x_decoded4)
ae.summary()

ae.set_weights(trained_weight)
#ae.get_weights()

''' this model maps an input to its reconstruction'''

def mean_square_error_weighted(y_true, y_pred):
    return K.mean(K.square(y_pred  - y_true/ (cut_neib + 1)), axis=-1)
    #return K.mean(K.square((y_pred - y_true)/(cut_neib+1)) , axis=-1)

def mean_square_error_weightedNN(y_true, y_pred):
    dst = K.mean(K.square((neib / (tf.expand_dims(cut_neib, 1) + 1) - K.expand_dims(y_pred, 1)) ), axis=-1)
    #dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1) ), axis=-1)
    ww = weight_neib
    return K.dot(dst, K.transpose(ww))

#def pen_zero(y_pred, cut_neib):
#    return(K.sum(K.abs((y_pred*cut_neib)), axis=-1))

#def  kl_l(z_mean, z_log_var):
#    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


def ae_loss(x, x_decoded_mean):
    #msew = original_dim * losses.mean_squared_error(x, x_decoded_mean)
    msew = original_dim * mean_square_error_weighted(x, x_decoded_mean)
    msewNN = 1/k*original_dim * mean_square_error_weightedNN(x, x_decoded_mean)

    #penalty_zero = pen_zero(x_decoded_mean, cut_neib)
    #return K.mean(msew)
    #return K.mean(0.1*msewNN+msew)# +  0.001*penalty_zero)
    return K.mean(msewNN)  # +  0.001*penalty_zero)


learning_rate = 1e-3
#earlyStopping=CustomEarlyStopping(criterion=0.0001, patience=3, verbose=1)
adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate / epochs)

#ae.compile(optimizer=adam, loss=ae_loss)
ae.compile(optimizer=adam, loss=ae_loss, metrics=[mean_square_error_weightedNN,mean_square_error_weighted])
#ae.get_weights()

checkpoint = ModelCheckpoint('.', monitor='ae_loss', verbose=1, save_best_only=True, mode='max')
logger = DBLogger(comment="An example run")
start = timeit.default_timer()
b_sizes = range(10,110,10); i=9
#for i in range(10) :


history=ae.fit([aFrame, neibF, cut_neibF, weight_neibF], aFrame,
batch_size=batch_size,
epochs = epochs,
shuffle=True,
callbacks=[CustomMetrics(), logger])
stop = timeit.default_timer()
ae.save('Samusik0_model001.h5')
#ae = load_model('Samusik0_model.h5')

print(stop - start)
fig0 = plt.figure();
plt.plot(history.history['loss']);
fig01 = plt.figure();
plt.plot(history.history['mean_square_error_weightedNN']);
fig02 = plt.figure();
plt.plot(history.history['mean_square_error_weighted']);
#fig03 = plt.figure();
#plt.plot(history.history['pen_zero']);
print(ae.summary())

# build a model to project inputs on the latent space
#encoder = Model([x, neib, cut_neib], encoded2)
encoder = Model([x, neib, cut_neib, weight_neib], encoded2)
print(encoder.summary())

# predict and extract latent variables

#gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = ae.predict([aFrame, neibF, cut_neibF, weight_neibF])
len(x_test_vae)
#np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/Samusik_x_test_vae001.txt', x_test_vae)
x_test_vae=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/Samusik_x_test_vae001.txt')
x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])
#len(x_test_enc)
#3,8,13
cl=12;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
#ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);

fig4= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
#b0.set_xticklabels(rs[cl-1, ]);
fig5= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0.set_xticklabels(np.round(cutoff,2));

unique0, counts0 = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique0, counts0)).T)
num_clus=len(counts0)


#start second iteration with un-corrupted neighbourhoods
##########################################################

from scipy import stats
conn = [sum((stats.itemfreq(lbls[IdxF[x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(conn,50);
np.mean(conn)

nb=find_neighbors(x_test_vae, k, metric='L2', cores=12)
connClean = [sum((stats.itemfreq(lbls[nb['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean,50);
np.mean(connClean)
#start second iteration with un-corrupted neighbourhoods
IdxF=nb['idx']
k=30
nrow = np.shape(features)[0]
b=0
neibF = np.zeros((nrow, k, original_dim))
cnz = np.zeros((original_dim))
cut_neibF = np.zeros((nrow, original_dim))
weight_distF = np.zeros((nrow, k))
weight_neibF = np.zeros((nrow, k))
rk=range(k)

sigmaBer = np.sqrt(cutoff*(1-cutoff))
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing
def singleInput(i):
     #nei = np.zeros(( k, original_dim))
     #cut_nei = np.zeros((nrow, original_dim))
     #weight_di = np.zeros((k))
     #weight_nei = np.zeros((k))
     nei =  features[IdxF[i,:],:]
     cnz=[np.sum(np.where(nei[:, j] == 0, 0,1))/ k for j in range(original_dim)]
     cut_nei= np.array([0 if (cnz[j] >= cutoff[j] or cutoff[j]>0.5) else (U/naive_power(cutoff[j], 2)) * naive_power((cutoff[j] - cnz[j])/sigmaBer[j] , 2) for j in range(original_dim)])
     weight_di = [sum(((features[i] - nei[k_i,]) / (1 + cut_nei))**2) for k_i in rk]
     d_lock_max=np.max(weight_di)
     d_lock_min = np.min(weight_di)
     weight_nei = np.exp((-k * (weight_di- d_lock_min)/(d_lock_max-d_lock_min)))
     return [nei, cut_nei, weight_nei, i]

inputs = range(nrow)
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(singleInput)(i) for i in inputs)
for i in range(nrow):
 neibF[i,] = results[i][0]
for i in range(nrow):
    cut_neibF[i,] = results[i][1]
for i in range(nrow):
    weight_neibF[i,] = results[i][2]

print('Fine-tuning:')

'''this is our input placeholder'''
x = Input(shape=(original_dim,))
''' "encoded" is the encoded representation of the input'''
encoded1 = Dense(intermediate_dim, activation='selu')(x)

encoded2 = Dense(latent_dim, activation='selu')(encoded1)

x_decoded3 = Dense(intermediate_dim, activation='selu')(encoded2)

x_decoded4 = Dense(original_dim, activation='relu')(x_decoded3)
ae = Model([x, neib, cut_neib, weight_neib], x_decoded4)
#ae = Model([x, neib, cut_neib], x_decoded4)
ae.summary()

ae.set_weights(trained_weight)
#ae.get_weights()

''' this model maps an input to its reconstruction'''

def mean_square_error_weighted(y_true, y_pred):
    return K.mean(K.square(y_pred  - y_true/ (cut_neib + 1)), axis=-1)
    #return K.mean(K.square((y_pred - y_true)/(cut_neib+1)) , axis=-1)

def mean_square_error_weightedNN(y_true, y_pred):
    dst = K.mean(K.square((neib / (tf.expand_dims(cut_neib, 1) + 1) - K.expand_dims(y_pred, 1)) ), axis=-1)
    #dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1) ), axis=-1)
    ww = weight_neib
    return K.dot(dst, K.transpose(ww))

#def pen_zero(y_pred, cut_neib):
#    return(K.sum(K.abs((y_pred*cut_neib)), axis=-1))

#def  kl_l(z_mean, z_log_var):
#    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


def ae_loss(x, x_decoded_mean):
    #msew = original_dim * losses.mean_squared_error(x, x_decoded_mean)
    msew = original_dim * mean_square_error_weighted(x, x_decoded_mean)
    msewNN = 1/k*original_dim * mean_square_error_weightedNN(x, x_decoded_mean)

    #penalty_zero = pen_zero(x_decoded_mean, cut_neib)
    #return K.mean(msew)
    #return K.mean(0.1*msewNN+msew)# +  0.001*penalty_zero)
    return K.mean(msewNN)  # +  0.001*penalty_zero)


learning_rate = 1e-4
#earlyStopping=CustomEarlyStopping(criterion=0.0001, patience=3, verbose=1)
adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate / epochs)

#ae.compile(optimizer=adam, loss=ae_loss)
ae.compile(optimizer=adam, loss=ae_loss, metrics=[mean_square_error_weightedNN,mean_square_error_weighted])
#ae.get_weights()

checkpoint = ModelCheckpoint('.', monitor='ae_loss', verbose=1, save_best_only=True, mode='max')
logger = DBLogger(comment="An example run")
start = timeit.default_timer()
b_sizes = range(10,110,10); i=9
#for i in range(10) :


history=ae.fit([aFrame, neibF, cut_neibF, weight_neibF], aFrame,
batch_size=batch_size,
epochs = epochs,
shuffle=True,
callbacks=[CustomMetrics(), logger])
stop = timeit.default_timer()
ae.save('Samusik0_model001_2nd_run.h5')
#ae = load_model('Samusik0_model.h5')

print(stop - start)
fig0 = plt.figure();
plt.plot(history.history['loss']);
fig01 = plt.figure();
plt.plot(history.history['mean_square_error_weightedNN']);
fig02 = plt.figure();
plt.plot(history.history['mean_square_error_weighted']);
#fig03 = plt.figure();
#plt.plot(history.history['pen_zero']);
print(ae.summary())

# build a model to project inputs on the latent space
#encoder = Model([x, neib, cut_neib], encoded2)
encoder = Model([x, neib, cut_neib, weight_neib], encoded2)
print(encoder.summary())

# predict and extract latent variables

#gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = ae.predict([aFrame, neibF, cut_neibF, weight_neibF])
len(x_test_vae)
np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/Samusik_x_test_vae001_2nd_run.txt', x_test_vae)
x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])
#len(x_test_enc)
#3,8,13
cl=12;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
#ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);

fig4= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
#b0.set_xticklabels(rs[cl-1, ]);
fig5= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0.set_xticklabels(np.round(cutoff,2));

unique0, counts0 = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique0, counts0)).T)
num_clus=len(counts0)

from scipy import stats
k=30
conn = [sum((stats.itemfreq(lbls[Idx[x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(conn,50);


nb=find_neighbors(x_test_vae, k, metric='L2', cores=12)
connClean = [sum((stats.itemfreq(lbls[nb['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean,50);






import dill
#filename = 'globalsave.pkl'
#dill.dump_session(filename)
#dill.load_session(filename)
#ae = load_model('Wang0_model.h5')


from sklearn.cluster import Birch
import sklearn.metrics
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('manhattan')

idx=np.random.choice(np.shape(aFrame)[0], 50000, replace=False)

brc0=Birch(threshold=0.5, branching_factor=50, n_clusters=len(unique0), compute_labels=True, copy=True)
brc0.fit(aFrame[idx,])
lbls0=brc0.predict(aFrame[idx,  ])

brc1= Birch(threshold=0.5, branching_factor=50, n_clusters=len(unique0), compute_labels=True, copy=True)
brc1.fit(x_test_vae[lbls!=14 and lbls!=15,])
lbls1=brc1.predict(x_test_vae[lbls!=14 and lbls!=15,])

print(sklearn.metrics.adjusted_mutual_info_score(lbls, lbls0))
print(sklearn.metrics.adjusted_mutual_info_score(lbls, lbls1))

unique, counts = np.unique(lbls0, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, )[-24:])


unique, counts = np.unique(lbls1, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, )[-24:])

unique, counts = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, )[-24:])

#number of zeroes before and after
sum(sum(features==0)) #7964609
sum(sum(x_test_vae==0))#9881212
sum(sum(x_test_vae<=0.01))#11012548

#true number of zeroes
#zr= 1-(1*rs)
#pycharmsum([counts[xx+1] * sum(zr[xx+1,:]) for xx in range(cl)])#13704320


import numpy as np
>>> import DBSCAN_multiplex as DB

>>> data = np.random.randn(15000, 7)
>>> N_iterations = 50
>>> N_sub = 9 * data.shape[0] / 10
>>> subsamples_matrix = np.zeros((N_iterations, N_sub), dtype = int)
>>> for i in xrange(N_iterations):
        subsamples_matrix[i] = np.random.choice(data.shape[0], N_sub, replace = False)
>>> eps, labels_matrix = DB.DBSCAN(data, minPts = 3, subsamples_matrix = subsamples_matrix, verbose = True)


from sklearn.cluster import MiniBatchKMeans
>>> X = np.random.randn(50000, 7)
>>> %timeit MiniBatchKMeans(30).fit(X)
1 loops, best of 3: 114 ms per loop

hdbscan_ = hdbscan.HDBSCAN()
hdbscan_data = benchmark_algorithm(dataset_sizes, hdbscan_.fit, (), {})


import hdbscan
from sklearn.datasets import make_blobs

data = make_blobs(1000)

clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
cluster_labels = clusterer.fit_predict(data)
hierarchy = clusterer.cluster_hierarchy_
alt_labels = hierarchy.get_clusters(0.100, 5)
hierarchy.plot()









