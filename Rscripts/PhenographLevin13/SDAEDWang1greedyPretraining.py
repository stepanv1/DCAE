'''This script to clean cytoff data using deep stacked autoencoder with neighbourhood denoising and
contracting
Now with greedy pretraining
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
#from keras.utils import multi_gpu_model
import timeit
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.constraints import maxnorm
import readline
import os
os.environ['R_HOME'] = '/home/grines02/R/x86_64-pc-linux-gnu-library/3.6'

import rpy2
from rpy2.robjects.packages import importr
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#stsne = importr('stsne')
#subspace = importr('subspace')
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
#import champ
#import ig

def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique':unique, 'counts':counts}

desc=["Y89Di", "Cd112Di", "In115Di", "La139Di", "Pr141Di", "Nd142Di", "Nd143Di", "Nd144Di", "Nd145Di" , "Nd146Di",  "Sm147Di" ,   "Nd148Di"   ,   "Sm149Di"    ,  "Nd150Di"   ,   "Eu151Di" ,  "Sm152Di"   ,   "Eu153Di"   ,   "Sm154Di"   ,   "Gd155Di"   ,   "Gd156Di"    ,  "Gd157Di"  ,   "Gd158Di" , "Tb159Di"   ,   "Gd160Di"  ,"Dy161Di"   ,   "Dy162Di"   ,   "Dy163Di"  ,    "Dy164Di"   ,  "Ho165Di" , "Er166Di" , "Er167Di",   "Er168Di"  , "Tm169Di"  ,   "Er170Di"   ,   "Yb171Di"   ,  "Yb172Di",   "Yb173Di"    ,  "Yb174Di"  ,    "Lu175Di"  ,   "Yb176Di"]

markers = ["89Y-CD45" ,  "112Cd-CD45RA"   ,   "115In-CD8"   ,  "139La-CD19"  ,  "141Pr-CD137"  ,   "142Nd-CD57"  , "143Nd-HLA_DR"   ,  "144Nd-CCR5"  , "145Nd-CD45RO" ,
           "146Nd-FOXP3" ,   "147Sm-CD62L"  ,  "148Nd-PD_L1"   ,  "149Sm-CD56"   ,  "150Nd-LAG3"  ,   "151Eu-ICOS" ,    "152Sm-CCR6"  ,  "153Eu-TIGIT"  ,   "154Sm-TIM3",
           "155Gd-PD1"  ,  "156Nd-CXCR3" ,    "157Gd-GITR" ,   "158Gd-CD134"  ,   "159Tb-CCR7"   ,  "160Gd-Tbet"   , "161Dy-CTLA4"  ,   "162Dy-CD27"   ,  "163Dy-BTLA",
           "164Dy-CCR4"   , "165Ho-CD101"  ,  "166Er-EOMES"  ,  "167Er-GATA3"  ,  "168Er-CD40L" ,    "169Tm-CD25"   ,   "170Er-CD3" ,   "171Yb-CXCR5" ,    "172Yb-CD38",
           "173Yb-GrnzB"   ,   "174Yb-CD4", "175Lu-Perforin"  ,  "176Yb-CD127"]


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



import vpsearch as vp
def find_neighbors(data, k_, metric='L2', cores=12):
   res = vp.find_nearest_neighbors(data, k_, cores)
   return {'dist':np.array(res[1]), 'idx': np.int32(np.array(res[0]))}

#from libKMCUDA import kmeans_cuda, knn_cuda
#def find_neighbors(data, k_, metric='euclidean', cores=12):
#    ca = kmeans_cuda(np.float32(data), 25, metric="euclidean", verbosity=1, seed=3, device=0)
#    neighbors = knn_cuda(k_, np.float32(data), *ca, metric=metric, verbosity=1, device=0)
#    return {'dist':0, 'idx': np.int32(neighbors)}






#load data
k=30
'''
source_dir = "/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient"
#file_list = glob.glob(source_dir + '/*.txt')
data0 = np.genfromtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/AllPatients_data0.txt'
, names=None, dtype=float, skip_header=1)
lbls = np.int32(data0[:,40])

aFrame = data0[:,:40]
cutoff = np.genfromtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/cutoff.txt'
, delimiter=' ', skip_header=1)
patient =  np.int32(data0[:, 41])
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k, metric='L2', cores=12)
Idx = nb['idx']; Dist = nb['dist']
'''
outfile = 'Mat0.npz'
#np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls, cutoff=cutoff, patient=patient, Dist=Dist)
npzfile = np.load(outfile)
lbls=npzfile['lbls'];Idx=npzfile['Idx'];aFrame=npzfile['aFrame'];patient=npzfile['patient'];
cutoff=npzfile['cutoff']; Dist =npzfile['Dist']




#model parameters
batch_size = 10
original_dim = 40
latent_dim = 120
intermediate_dim = 80
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


#compute weights
import ctypes
from numpy.ctypeslib import ndpointer
#del lib
#del perp
#import _ctypes
#_ctypes.dlclose(lib._handle )
#del perp

lib = ctypes.cdll.LoadLibrary("/mnt/f/Brinkman group/current/Stepan/Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,ctypes.c_size_t]

#razobratsya s nulem (index of 0-neares neightbr )!!!!
weights = np.empty((nrow, 10*3))
perp(Dist[:,0:30],  nrow,  40,   weights,        10,            30,      12)
#( double* dist, int N, int D,  double* P,  double perplexity, int K, int num_threads)
np.shape(weights)
weights[0,:]
plt.figure();plt.plot(x=weight_neibF[0,:],y=weights[0,:])


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
#ae.save('Wang0_model001.h5')
#ae = load_model('Wang0_model.h5')

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
np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_ae001.txt', x_test_vae)
x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])
#len(x_test_enc)
#3,8,13
cl=15;bw=0.02
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

scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb2=find_neighbors(x_test_enc, k, metric='L2', cores=12)
connClean2 = [sum((stats.itemfreq(lbls[nb2['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean2,50)
#print(np.mean(connNoNoise))
print(np.mean(conn))
print(np.mean(connClean))
print(np.mean(connClean2))

for cl in unique0 :
#print(np.mean(np.array(connNoNoise)[lbls==cl]))
    print(cl)
    print(np.mean(np.array(conn)[lbls==cl]))
    print(np.mean(np.array(connClean)[lbls==cl]))
    print(np.mean(np.array(connClean)[lbls==cl])-np.mean(np.array(conn)[lbls==cl]))
    print(np.mean(np.array(connClean2)[lbls == cl]) - np.mean(np.array(conn)[lbls == cl ]))

print(np.mean( np.array([np.mean(np.array(conn)[lbls==cl]) for cl in unique0] ) ))
print(np.mean( np.array([np.mean(np.array(connClean)[lbls==cl]) for cl in unique0] ) ))
print(np.mean( np.array([np.mean(np.array(connClean2)[lbls==cl]) for cl in unique0] ) ))



import dill
#filename = 'globalsave.pkl'
#dill.dump_session(filename)
#dill.load_session(filename)
#ae = load_model('Wang0_model.h5')


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
ae.save('Wang0_model001_2nd_run.h5')
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
#np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/Wang0_x_test_vae001_2nd_run.txt', x_test_vae)
x_test_vae=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/Wang0_x_test_vae001_2nd_run.txt')
x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])
#len(x_test_enc)
#3,8,13
#139la-CD19
#157Gd-GITR
cl=9;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
#ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);

plt.rcParams['xtick.labelsize'] = 8
cl=13;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
ax.set_xticklabels(markers);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
bx.set_xticklabels(markers);




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


for cl in unique0 :
#print(np.mean(np.array(connNoNoise)[lbls==cl]))
    print(cl)
    print(np.mean(np.array(conn)[lbls==cl]))
    print(np.mean(np.array(connClean)[lbls==cl]))
    print(np.mean(np.array(connClean)[lbls==cl])-np.mean(np.array(conn)[lbls==cl]))
    print(np.mean(np.array(connClean2)[lbls == cl]) - np.mean(np.array(conn)[lbls == cl ]))

print(np.mean( np.array([np.mean(np.array(conn)[lbls==cl]) for cl in unique0] ) ))
print(np.mean( np.array([np.mean(np.array(connClean)[lbls==cl]) for cl in unique0] ) ))
print(np.mean( np.array([np.mean(np.array(connClean2)[lbls==cl]) for cl in unique0] ) ))






from sklearn.cluster import Birch
import sklearn.metrics
from sklearn.neighbors import DistanceMetric


pt=2
brc0=Birch(threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=True, copy=True)
brc0.fit(aFrame[patient==pt,:])
lbls0=brc0.predict(aFrame[patient==pt,:])

brc1= Birch(threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=True, copy=True)
brc1.fit(x_test_vae[patient==pt,:])
lbls1=brc1.predict(x_test_vae[patient==pt,:])

print(sklearn.metrics.adjusted_mutual_info_score(lbls[patient==pt], lbls0))
print(sklearn.metrics.adjusted_mutual_info_score(lbls[patient==pt], lbls1))

unique, counts = np.unique(lbls0, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, )[-14:])


unique, counts = np.unique(lbls1, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, )[-14:])

unique, counts = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, ))


conm0=sklearn.metrics.confusion_matrix(lbls[patient==pt], lbls0)
conm1=sklearn.metrics.confusion_matrix(lbls[patient==pt], lbls1)


#all data
clusterer0all = hdbscan.HDBSCAN(algorithm ='boruvka_balltree', min_cluster_size=25, min_samples=1,metric='euclidean', core_dist_n_jobs=10)
clusterer0all.fit(aFrame)
clusterer1all = hdbscan.HDBSCAN(algorithm ='boruvka_balltree', min_cluster_size=25, min_samples=1,metric='euclidean', core_dist_n_jobs=10)
clusterer1all.fit(x_test_vae)

np.savez("DBSCAN.npz", clusterer0all=clusterer0all, clusterer1all=clusterer1all)

DBSCANres = np.load("DBSCAN.npz")
clusterer0all=DBSCANres['clusterer0all'];clusterer1all=DBSCANres['clusterer1all']

table(clusterer0all.labels_)
table(clusterer1all.labels_)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]


clb0All = clusterer0all.labels_
clb0All[clusterer0all.labels_==-1]=2
#color =  plt.cm.Vega20c( (4./3*np.arange(4*3/4)).astype(int) )
color=[sorted_names[i] for i in (np.arange(0,100,10)) ]
fig, ax = plt.subplots()
for g in np.unique(clb0All):
    i = np.where(clb0All == g)
    ax.scatter(Y1all[i,0], Y1all[i,1], label=g, marker='.', s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()

clb1All = clusterer1all.labels_
clb1All[clusterer1all.labels_==-1]=9
#color =  plt.cm.Vega20c((4./3*np.arange(10)).astype(int))
color=[sorted_names[i] for i in (np.arange(0,160,16)) ]
fig, ax = plt.subplots()
for g in np.unique(clb1All[(lbls!=11)*(lbls!=15)]):
    i = np.where(clb1All[(lbls!=11)*(lbls!=15)] == g)
    ax.scatter(Y1all[i,0], Y1all[i,1], label=g, marker='.', s=0.1, c=color[g], alpha=0.5)
ax.legend(markerscale=10)
plt.show()






pt=2
#faster dbscan attempt
import hdbscan
clusterer0 = hdbscan.HDBSCAN(algorithm ='boruvka_balltree', min_cluster_size=25, min_samples=10, metric='euclidean', core_dist_n_jobs=10)
clusterer0.fit(aFrame[patient==pt,:])


clusterer1 = hdbscan.HDBSCAN(algorithm ='boruvka_balltree', min_cluster_size=25, min_samples=10, metric='euclidean', core_dist_n_jobs=10)
clusterer1.fit(x_test_vae[patient==pt,:])

unique, counts = np.unique(clusterer0.labels_, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, )[-14:])
plt.figure();plt.hist(clusterer0.probabilities_,50);

unique, counts = np.unique(clusterer1.labels_, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
print(np.sort(counts, )[-14:])
plt.figure();plt.hist(clusterer1.probabilities_,50);

uniqueT, countsT = np.unique(lbls[patient==pt], return_counts=True)
print('%d %d', np.asarray((uniqueT, countsT)).T)
print(np.sort(countsT, ))

sklearn.metrics.confusion_matrix(lbls[patient==pt], clusterer0.labels_)
sklearn.metrics.confusion_matrix(lbls[patient==pt], clusterer1.labels_)


from MulticoreTSNE import MulticoreTSNE as TSNE

tsne = TSNE(n_jobs=12,  n_iter=10000)
Y0 = tsne.fit_transform(aFrame[(patient==pt)*(lbls!=11),:])
tsne = TSNE(n_jobs=12,  n_iter=10000)
Y1 = tsne.fit_transform(x_test_vae[(patient==pt)*(lbls!=11),:])

from sklearn.manifold import TSNE as oldTSNE

otsne = oldTSNE(init="pca", n_iter=5000, metric='manhattan')
Y0o = otsne.fit_transform(aFrame[(patient==pt)*(lbls!=11),:])
otsne = oldTSNE(init="pca", n_iter=5000, metric='manhattan')
Y1o = otsne.fit_transform(x_test_vae[(patient==pt)*(lbls!=11),:])

np.savez('otsne.npz', Y0o=Y0o, Y1o=Y1o)
ots=np.load("otsne.npz")

from matplotlib.pyplot import cm
#color=cm.tab20c(np.linspace(0,1,16))
color =  plt.cm.Vega20c( (4./3*np.arange(24*3/4)).astype(int) )
fig, ax = plt.subplots()
for g in np.unique(lbls[(patient==pt)*(lbls!=11)]):
    i = np.where(lbls[(patient==pt)*(lbls!=11)] == g)
    ax.scatter(Y0o[i,0], Y0o[i,1], label=g, marker='.', s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()

fig, ax = plt.subplots()
for g in np.unique(lbls[(patient==pt)*(lbls!=11)]):
    i = np.where(lbls[(patient==pt)*(lbls!=11)] == g)
    ax.scatter(Y1o[i,0], Y1o[i,1], label=g, marker='.',s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()

clb1=clusterer1.labels_
clb1[clusterer1.labels_==-1]=16
color =  plt.cm.Vega20c((4./3*np.arange(24*3/4)).astype(int) )
fig, ax = plt.subplots()
for g in np.unique(clb1[lbls[patient==pt]!=11]):
    i = np.where(clb1[lbls[patient==pt]!=11] == g)
    ax.scatter(Y1o[i,0], Y1o[i,1], label=g, marker='.',s=1, c= color[g])
ax.legend(markerscale=10)
plt.show()

clb0=clusterer0.labels_
clb0[clusterer0.labels_==-1]=16
fig, ax = plt.subplots()
for g in np.unique(clb0[lbls[patient==pt]!=11]):
    i = np.where(clb0[lbls[patient==pt]!=11] == g)
    ax.scatter(Y1o[i,0], Y1o[i,1], label=g, marker='.',s=1, c= color[g])
ax.legend(markerscale=10)
plt.show()






#plt.colorbar(ticks=range(len(countsT
#)-2))
#plt.clim(1, len(countsT
#))

tsne = oldTSNE(   n_iter= 5000, n_components=3)
Y03D = tsne.fit_transform(aFrame[(patient==pt)*(lbls!=11),:])
tsne = oldTSNE(  n_iter= 5000, n_components=3)
Y13D = tsne.fit_transform(x_test_vae[(patient==pt)*(lbls!=11),:])


plt.figure()
color =  plt.cm.Vega20c( (4./3*np.arange(24*3/4)).astype(int) )
ax = plt.subplot(projection='3d')
for g in np.unique(lbls[(patient==pt)*(lbls!=11)]):
    i = np.where(lbls[(patient==pt)*(lbls!=11)] == g)
    ax.scatter(Y03D[i,0], Y03D[i,1], Y03D[i,2], label=g, marker='.', s=10, c=color[g])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.legend(markerscale=10)
plt.show()

fig=plt.figure(facecolor='white')
fig.patch.set_facecolor('white')
ax = plt.subplot(projection='3d', facecolor='white')
for g in np.unique(lbls[(patient==pt)*(lbls!=11)]):
    i = np.where(lbls[(patient==pt)*(lbls!=11)] == g)
    ax.scatter(Y13D[i,0], Y13D[i,1], Y13D[i,2], label=g, marker='.', s=10, c=color[g])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.legend(markerscale=10)
plt.show()


ax = plt.subplot(projection='3d')
ax.scatter(Y03D[:,0], Y03D[:,1], Y03D[:,2])


tsne = TSNE(n_jobs=12,  n_iter=5000)
Y0all = tsne.fit_transform(aFrame)
tsne = TSNE(n_jobs=12,  n_iter=5000)
Y1all = tsne.fit_transform(x_test_vae)


from tsne import bh_sne

Y0all =  bh_sne(aFrame, max_iter=10000, d=3)

Y1all = bh_sne(x_test_vae, max_iter=10000, d=3)

np.savez("3Dtsne10000.npz", Y0all=Y0all, Y1all=Y1all)

fl = np.load("tsne5000.npz")
Y0all=fl['Y0all'];Y1all=fl['Y1all']


color =  plt.cm.Vega20c( (4./3*np.arange(24*3/4)).astype(int) )
fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y0all[i,0], Y0all[i,1], label=g, marker='.', s=0.2, c=color[g], alpha=0.9)
ax.legend(markerscale=10)
plt.show()

fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y1all[i,0], Y1all[i,1], label=g, marker='.', s=0.2, c=color[g], alpha=0.9)
ax.legend(markerscale=10)
plt.show()

color=['yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow']
color[15] ='red'
fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y0all[i,0], Y0all[i,1], label=g, marker='.', s=0.2, c=color[g], alpha=0.9)
ax.legend(markerscale=10)
plt.show()

fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y1all[i,0], Y1all[i,1], label=g, marker='.', s=0.2, c=color[g], alpha=0.9)
ax.legend(markerscale=10)
plt.show()





fig, ax = plt.subplots()
for g in np.unique(patient):
    i = np.where(patient == g)
    ax.scatter(Y0all[i,0], Y0all[i,1], label=g, marker='.', s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()

fig, ax = plt.subplots()
for g in np.unique(patient):
    i = np.where(patient == g)
    ax.scatter(Y1all[i,0], Y1all[i,1], label=g, marker='.', s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()









import sklearn
print(sklearn.metrics.adjusted_mutual_info_score(clusterer0.labels_, lbls[patient==1]))
print(sklearn.metrics.adjusted_mutual_info_score(clusterer1.labels_, lbls[patient==1]))




clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
cluster_labels = clusterer.fit_predict(data)
hierarchy = clusterer.cluster_hierarchy_
alt_labels = hierarchy.get_clusters(0.100, 5)
hierarchy.plot()









