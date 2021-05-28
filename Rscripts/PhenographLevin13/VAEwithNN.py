'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob
from keras.layers import Input, Dense, Lambda, Layer, Dropout, BatchNormalization
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
import seaborn as sns
import warnings
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import regularizers


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



#from keras.datasets import mnist
#load data

source_dir = "/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/DA"
file_list = glob.glob(source_dir + '/*.txt')

[aFrame, lbls, Dists, Idx, rs] = [np.genfromtxt(x, delimiter=' ', skip_header=0, skip_footer=0) for x in file_list ]
Idx = np.int32(Idx)
lbls = np.int32(lbls)
rs = np.int32(rs)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
k=30
nb=find_neighbors(aFrame, k, metric='L2', cores=12)
Idx = nb['idx']
Dists = nb['dist']


#model parameters
batch_size = 100
original_dim = 39
latent_dim = 8
intermediate_dim = 20

epochs = 50
epsilon_std = 1.0
U=1#energy barier
d=(0.1)**2#neighbourhood radius
k=30#nearest neighbours
#generator

cutoff = np.count_nonzero(aFrame)/(np.shape(aFrame)[0]*np.shape(aFrame)[1])

def generatorNN(features, features2, Idx, Dists, batch_size,  k_=k, shuffle=True):
 # Create empty arrays to contain batch of features and labels#
    #x = np.zeros((batch_size, 39))
    #neib = np.zeros((batch_size, k_, 39))
    #cut_neib = np.zeros((batch_size, 39))
    Dists = Dists[:,:k]
    Idx = Idx[:, :k]
    l=len(features)
    dim2=np.shape(features)[1]
    b=0
    ac= U/np.power(1-cutoff, 15)
    if shuffle == True:
        random.shuffle(features)
    while True:
        index = range(b, min(b + batch_size, l))
        lind = len(index)

        if b+lind>=l:
            if shuffle == True:
                random.shuffle(features)
            b=0

        b = b + lind
        neib = np.zeros((lind, k_, original_dim))
        cut_neib = np.zeros((lind, original_dim))
        #var_dims = np.zeros((lind, original_dim))
        x = features[index]
        weight_neib =  np.exp(-Dists[index,]/d)

        for i in range(lind):
             neib[i,] = [ features[z, ] for z in Idx[index[i]] ]
             for j in range(dim2):
                 cnz = np.count_nonzero(neib[i, :, j]) / k_
                 #print(U if cnz <= cutoff else ac * (1 - cnz) ** 2)
                 #cut_neib[i, j] = U if cnz <= cutoff else ac * np.power((1 - cnz) ,15)
                 cut_neib[i, j] =  ac * np.power((1 - cnz) ,15)
                 #var_dims[i, j] = np.var(neib[i, :, j])+0.0000001

        yield ([x, neib, cut_neib, weight_neib],x)

gen = generatorNN(aFrame, aFrame, Idx, Dists, batch_size,  k_=k )
#gen_pred = generator(aFrame, aFrame, batch_size, nn=Idx, k_=30, shuffle = False)

neib = Input(shape = (k, original_dim, ))
cut_neib = Input(shape = (original_dim,))
#var_dims = Input(shape = (original_dim,))

weight_neib = Input(shape = (k,))

x = Input(shape=(original_dim,))
xB = BatchNormalization()(x)

h = Dense(intermediate_dim,  activation='relu')(xB)
#hD = Dropout(0.2)(h)
hD = BatchNormalization()(h)
z_mean = Dense(latent_dim,
           W_regularizer=regularizers.l1(0.001))(hD)
z_log_var = Dense(latent_dim)(hD)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#zD = Dropout(0.2)(z)
zD = BatchNormalization()(z)
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(zD)
#h_decodedD = Dropout(0.2)(h_decoded)
h_decodedD = BatchNormalization()(h_decoded)
x_decoded_mean = decoder_mean(h_decodedD)





def mean_square_error_weighted(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)/(cut_neib+1)) , axis=-1)

def mean_square_error_weightedNN(y_true, y_pred):
    dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    ww = weight_neib
    return K.dot(dst, K.transpose(ww))

def pen_zero(y_pred, cut_neib):
    return(K.sum(K.square((y_pred*cut_neib)), axis=-1))

def  kl_l(z_mean, z_log_var):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

def vae_loss(x, x_decoded_mean):
    msew = original_dim * mean_square_error_weighted(x, x_decoded_mean)
    msewNN = original_dim * mean_square_error_weightedNN(x, x_decoded_mean)
    kl_loss = kl_l(z_mean, z_log_var)
    penalty_zero = pen_zero(x_decoded_mean, cut_neib)
    return K.mean((msewNN+msew) + 0.01*kl_loss + 0.1*penalty_zero)

vae = Model([x, neib, cut_neib, weight_neib], x_decoded_mean)
#vae.compile(optimizer='rmsprop', loss=vae_loss)
#sgd = SGD(lr=0.01, momentum=0.1, decay=0.00001, nesterov=True)
learning_rate = 1e-4
epochs=3
#earlyStopping=CustomEarlyStopping(criterion=0.0001, patience=3, verbose=1)
adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate / epochs)
vae.compile(optimizer=adam, loss=vae_loss, metrics=[mean_square_error_weightedNN,mean_square_error_weighted, pen_zero, kl_l])

checkpoint = ModelCheckpoint('.', monitor='vaekl_l_loss', verbose=1, save_best_only=True, mode='max')
logger = DBLogger(comment="An example run")
start = timeit.default_timer()
epochs=10
b_sizes = range(10,110,10); i=9
#for i in range(10) :
    gen = generatorNN(aFrame, aFrame, Idx, Dists, batch_size, k_= k , shuffle = True )

    history=vae.fit_generator(gen,
    steps_per_epoch=np.ceil(len(aFrame)/batch_size),
    epochs = epochs,
    #shuffle=True,
    use_multiprocessing=True, callbacks=[CustomMetrics(), logger], workers=12)
stop = timeit.default_timer()
print(stop - start)
fig0 = plt.figure()
plt.plot(history.history['loss'])
fig01 = plt.figure()
plt.plot(history.history['mean_square_error_weightedNN'])
fig02 = plt.figure()
plt.plot(history.history['mean_square_error_weighted'])
print(vae.summary())

# build a model to project inputs on the latent space
encoder = Model([x, neib, cut_neib, weight_neib], z_mean)

# predict and extract latent variables
gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, 1000,  k_=k, shuffle = False)
x_test_vae = vae.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/1000), use_multiprocessing=True)
len(x_test_vae)
gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, 1000, k_=k, shuffle = False)
x_test_enc = encoder.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/1000), use_multiprocessing=True)

cl=1;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
bx.set_xticklabels(rs[cl-1, ])
fig3 = plt.figure();
bz = sns.violinplot(data= x_test_enc[lbls==cl , :], bw =bw);
fig4= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
b0.set_xticklabels(rs[cl-1, ])

import matplotlib.cm as cm
colors = cm.con(np.linspace(0, 1, 24))

fig = plt.figure()
for cl in range(1,24,5):
    sns.violinplot(data=x_test_enc[lbls == cl+1, :], bw=bw, color=colors[cl]);
    bx.set_xticklabels(rs[cl,])

unique, counts = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)

from scipy import stats
conn = [sum((stats.itemfreq(lbls[Idx[x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(conn,20)

nb=find_neighbors(x_test_vae, k, metric='L2', cores=12)
connClean = [sum((stats.itemfreq(lbls[nb['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean,20)

nb2=find_neighbors(x_test_enc, k, metric='L2', cores=12)
connClean2 = [sum((stats.itemfreq(lbls[nb2['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean2,20)



#calculate new nearest neighbours
#and create new generator

vae = Model([x, neib, cut_neib, weight_neib], x_decoded_mean)
#vae.compile(optimizer='rmsprop', loss=vae_loss)
#sgd = SGD(lr=0.01, momentum=0.1, decay=0.00001, nesterov=True)
learning_rate = 1e-3
epochs=10
adam = Adam(lr=learning_rate, epsilon=0.1, decay = learning_rate / epochs)
vae.compile(optimizer=adam, loss=vae_loss, metrics=[mean_square_error_weightedNN,mean_square_error_weighted, pen_zero])

start = timeit.default_timer()
b_sizes = range(10,110,10); i=9
#for i in range(10) :
    gen = generatorNN(aFrame, aFrame, nb['idx'], nb['dist'], batch_size, k_= k , shuffle = True )
    history=vae.fit_generator(gen,
    steps_per_epoch=np.ceil(len(aFrame)/batch_size),
    epochs = epochs,
    shuffle=True,
    use_multiprocessing=True, callbacks=[CustomMetrics()], workers=12)
stop = timeit.default_timer()
print(stop - start)
fig0 = plt.figure()
plt.plot(history.history['loss'])
fig01 = plt.figure()
plt.plot(history.history['mean_square_error_weightedNN'])
fig02 = plt.figure()
plt.plot(history.history['mean_square_error_weighted'])
print(vae.summary())

# build a model to project inputs on the latent space
encoder = Model([x, neib, cut_neib, weight_neib], z_mean)

# display a 2D plot of the digit classes in  install nmslibthe latent space
gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, 10000,  k_=k, shuffle = False)
x_test_vae = vae.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/10000), use_multiprocessing=True)
len(x_test_vae)
gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, 10000, k_=k, shuffle = False)
x_test_enc = encoder.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/10000), use_multiprocessing=True)

cl=11;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
bx.set_xticklabels(rs[cl-1, ])
fig3 = plt.figure();
bz = sns.violinplot(data= x_test_enc[lbls==cl , :], bw =bw);
fig4= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
b0.set_xticklabels(rs[cl-1, ])




