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
import seaborn as sns
import warnings
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import regularizers
import champ
import igraph


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

#[aFrame, lbls, Dists, Idx, rs] = [np.genfromtxt(x, delimiter=' ', skip_header=0, skip_footer=0) for x in file_list ]
#Idx = np.int32(Idx)
#lbls = np.int32(lbls)
#rs = np.int32(rs)
#scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
#scaler.fit_transform(aFrame)
#nb=find_neighbors(aFrame, k, metric='L2', cores=12)
#Idx = nb['idx']
#Dists = nb['dist']
outfile = 'Mat.npz'
#np.savez(outfile, Idx=Idx, Dists=Dists, aFrame=aFrame, lbls=lbls, rs=rs)
npzfile = np.load(outfile)
lbls=npzfile['lbls'];Idx=npzfile['Idx'];aFrame=npzfile['aFrame'];rs=npzfile['rs'];Dists=npzfile['Dists'];

#model parameters
batch_size = 100
original_dim = 39
latent_dim = 39*24
intermediate_dim = 90


nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

epochs = 10
epsilon_std = 1.0
U=1000#energy barier
#d=(np.mean(Dists[:,:]))#neighbourhood radius
k=30#nearest neighbours
#generator

#cutoff = np.count_nonzero(aFrame)/(np.shape(aFrame)[0]*np.shape(aFrame)[1])
np.shape(rs)
#cutoff = np.count_nonzero(aFrame[:,0])/(np.shape(aFrame)[0])
cutoff = 2/k


import threading
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator

def generatorNN(features, features2, Idx, Dists, batch_size,  k_, shuffle=True):
 # Create empty arrays to contain batch of features and labels#
    #x = np.zeros((batch_size, 39))
    #neib = np.zeros((batch_size, k_, 39))
    #cut_neib = np.zeros((batch_size, 39))
    Dists = Dists[:,:k_]
    Idx = Idx[:, :k_]
    l=len(features)
    dim2=np.shape(features)[1]
    b=0
    ac= U/np.power(cutoff, 4)
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
        #weight_neib =  np.exp(-Dists[index,]/d)
        weight_dist = np.zeros((lind, k_))
        weight_neib = np.zeros((lind, k_))
        rk=range(k_)
        for i in range(lind):
             neib[i,] = [ features[z, ] for z in Idx[index[i],:]  ]
             for j in range(dim2):
                 cnz = np.count_nonzero(neib[i, :, j]) / k_
                 #print(U if cnz <= cutoff else ac * (1 - cnz) ** 2)
                 cut_neib[i, j] = 0 if cnz >= cutoff else ac * np.power((cutoff - cnz) , 4)
                 #print(cut_neib[i, j])
             weight_dist[i, ] = [sum(((x[i]-neib[i, k_i, ])/(1+cut_neib[i,]))**2) for k_i in rk]
             #weight_dist[i,] =[sum(((x[i]-neib[i, k_i, ]))**2) for k_i in rk]
             d_lock_max=np.max(weight_dist[i,])
             d_lock_min = np.min(weight_dist[i, ])
             #weight_neib[i, ] = 1/((-k_ * (weight_dist[i, ]- d_lock_min)/(d_lock_max-d_lock_min))**2 +0.5)
             weight_neib[i,] = np.exp((-k_ * (weight_dist[i, ]- d_lock_min)/(d_lock_max-d_lock_min)))
             #weight_neib[i,] = 1
        yield ([x, neib, cut_neib, weight_neib],x)

gen = generatorNN(aFrame, aFrame, Idx, Dists, batch_size,  k_=k )
#gen_pred = generator(aFrame, aFrame, batch_size, nn=Idx, k_=30, shuffle = False)

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

print('Fine-tuning:')

'''this is our input placeholder'''
x = Input(shape=(original_dim,))
''' "encoded" is the encoded representation of the input'''
encoded1 = Dense(intermediate_dim, activation='selu',
activity_regularizer=regularizers.l2(0.0000))(x)

encoded2 = Dense(latent_dim, activation='selu',
activity_regularizer=regularizers.l2(0.0000))(encoded1)

x_decoded3 = Dense(intermediate_dim, activation='selu',
activity_regularizer=regularizers.l2(0.0000))(encoded2)

x_decoded4 = Dense(original_dim, activation='relu',
activity_regularizer=regularizers.l2(0.0000))(x_decoded3)
ae = Model([x, neib, cut_neib, weight_neib], x_decoded4)
ae.set_weights(trained_weight)
#ae.get_weights()

dropout1 = AlphaDropout(0.1)
dropout2 = AlphaDropout(0.2)
dropout3 = AlphaDropout(0.1)

y = dropout1(encoded1)
x = fc2(y)
x = dropout2(x)
predictors = predictions(x)



''' this model maps an input to its reconstruction'''

def mean_square_error_weighted(y_true, y_pred):
    return K.mean(K.square((y_pred / (cut_neib + 1) - y_true) ), axis=-1)
    #return K.mean(K.square((y_pred - y_true)/(cut_neib+1)) , axis=-1)

def mean_square_error_weightedNN(y_true, y_pred):
    dst = K.mean(K.square((neib / (tf.expand_dims(cut_neib, 1) + 1) - K.expand_dims(y_pred, 1)) ), axis=-1)
    #dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1) ), axis=-1)
    ww = weight_neib
    return K.dot(dst, K.transpose(ww))

def pen_zero(y_pred, cut_neib):
    return(K.sum(K.abs((y_pred*cut_neib)), axis=-1))

def  kl_l(z_mean, z_log_var):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


def ae_loss(x, x_decoded_mean):
    #msew = original_dim * losses.mean_squared_error(x, x_decoded_mean)
    msew = original_dim * mean_square_error_weighted(x, x_decoded_mean)
    msewNN = 1/k*original_dim * mean_square_error_weightedNN(x, x_decoded_mean)

    penalty_zero = pen_zero(x_decoded_mean, cut_neib)
    #return K.mean(msew+0*msewNN+0*penalty_zero)
    return K.mean((msewNN+msew) +  0.00*penalty_zero)

#vae.compile(optimizer='rmsprop', loss=vae_loss)
#sgd = SGD(lr=0.01, momentum=0.1, decay=0.00001, nesterov=True)
learning_rate = 1e-3
#earlyStopping=CustomEarlyStopping(criterion=0.0001, patience=3, verbose=1)
adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate / epochs)

ae.compile(optimizer=adam, loss=ae_loss, metrics=[mean_square_error_weightedNN,mean_square_error_weighted, pen_zero])
#ae.get_weights()

checkpoint = ModelCheckpoint('.', monitor='ae_loss', verbose=1, save_best_only=True, mode='max')
logger = DBLogger(comment="An example run")
start = timeit.default_timer()
b_sizes = range(10,110,10); i=9
#for i in range(10) :
    gen = generatorNN(aFrame, aFrame, Idx, Dists, batch_size, k_= k , shuffle = True )

    history=ae.fit_generator(gen,
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
fig03 = plt.figure()
plt.plot(history.history['pen_zero'])
print(ae.summary())

# build a model to project inputs on the latent space
encoder = Model([x, neib, cut_neib, weight_neib], encoded2)
print(encoder.summary())

# predict and extract latent variables
batch_size = 100
gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = ae.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/batch_size), workers=10, max_queue_size=10, use_multiprocessing=False)
len(x_test_vae)
gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size, k_=k, shuffle = False)
x_test_enc = encoder.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/batch_size),  workers=10, max_queue_size=10)
len(x_test_enc)

cl=19;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
bx.set_xticklabels(rs[cl-1, ])
#fig3 = plt.figure();
#bz = sns.violinplot(data= x_test_enc[lbls==cl , :], bw =bw);
fig4= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
b0.set_xticklabels(rs[cl-1, ])

#import matplotlib.cm as cm
#colors = cm.con(np.linspace(0, 1, 24))

#fig = plt.figure()
#for cl in range(1,24,5):
#    sns.violinplot(data=x_test_enc[lbls == cl+1, :], bw=bw, color=colors[cl]);
#    bx.set_xticklabels(rs[cl,])

unique, counts = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)

from scipy import stats
#aFrameNoNoise = [aFrame[x, rs[lbls[x]-1, :]] for x in range(len(aFrame)) ]
#nbNoNoise=find_neighbors(np.array(aFrameNoNoise), k, metric='L2', cores=12)
#connNoNoise=[sum((stats.itemfreq(lbls[nbNoNoise['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
#plt.figure();plt.hist(connNoNoise,10)
#print(np.mean(connNoNoise))
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

cl=13
#print(np.mean(np.array(connNoNoise)[lbls==cl]))
print(np.mean(np.array(conn)[lbls==cl]))
print(np.mean(np.array(connClean)[lbls==cl]))
print(np.mean(np.array(connClean2)[lbls==cl]))

champ.louvain_ext.parallel_louvain(graph0, start=1, fin=1, numruns=12,
                                   maxpt=None, numprocesses=12, attribute=None, weight=None, node_subset=None, progress=True)

champ.louvain_ext.parallel_louvain(graph0, start=1, fin=1, numruns=12,
                                   maxpt=None, numprocesses=12, attribute=None, weight=None, node_subset=None, progress=True)


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
x_test_vae = vae.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/10000), use_multiprocessing=True, workers=12)
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




