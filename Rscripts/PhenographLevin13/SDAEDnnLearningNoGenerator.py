'''This script to clean cytoff data using deep stacked autoencoder with neighbourhood denoiding and
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
batch_size = 10
original_dim = 39
latent_dim = 117
intermediate_dim = 78


nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

epochs = 10
epsilon_std = 1.0
U=10#energy barier
#d=(np.mean(Dists[:,:]))#neighbourhood radius
k=30#nearest neighbours
#generator

#cutoff = np.count_nonzero(aFrame)/(np.shape(aFrame)[0]*np.shape(aFrame)[1])
np.shape(rs)
#cutoff = np.count_nonzero(aFrame[:,0])/(np.shape(aFrame)[0])
#calculate cutoff per channel

'''
cutoff =np.zeros(( np.shape(aFrame)[1]))
zeros_per_dim = np.zeros((len(set(lbls)), np.shape(aFrame)[1]))
for i in range(np.shape(aFrame)[1]):
    for j in set(lbls):
        if not rs[j-1, i]:
            print((i,j))
            zeros_per_dim[j-1, i] = sum(aFrame[lbls==(j),  i ]==0)
unique, counts = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)
for i in range(np.shape(aFrame)[1]):
 print(sum(zeros_per_dim[:,i]), sum(counts* (1-1*rs[:,i])))
 cutoff[i]=1-sum(zeros_per_dim[:,i])/sum(counts* (1-1*rs[:,i]))
cutoff
cutoff[1]=0
'''
cutoff = np.array([0.16290101, 0.  , 0.7206715 , 0.3107677 , 0.28672348,
       0.56945339, 0.34172146, 0.48290333, 0.35362354, 0.58688839,
       0.6111398 , 0.27726631, 0.29951124, 0.32618416, 0.31768792,
       0.49285161, 0.42619707, 0.44307881, 0.75080272, 0.54131919,
       0.44071592, 0.41755393, 0.67803699, 0.57918511, 0.27221648,
       0.54869376, 0.5784086 , 0.33442838, 0.31299828, 0.42947868,
       0.34699622, 0.37014286, 0.45298288, 0.70190171, 0.53360225,
       0.36184329, 0.61460673, 0.6325026 , 0.70764004])


#plt.figure(); plt.hist(cutoff);





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
#ae.summary()

#dropout1 = AlphaDropout(0.0)
#dropout2 = AlphaDropout(0.0)
#dropout3 = AlphaDropout(0.0)

#encoded1_l = ae.layers[-4]
#encoded2_l = ae.layers[-3]
#x_decoded3_l = ae.layers[-2]
#x_decoded4_l = ae.layers[-1]

#y = dropout1(encoded1_l.output)
#y = encoded2_l(y)
#y = dropout2(y)
#y = x_decoded3_l(y)
#y = dropout3(y)
#predictors = x_decoded4_l(y)

#ae = Model([x, neib, cut_neib, weight_neib], predictors)
#ae = Model([x, neib, cut_neib], predictors)
#ae.summary()

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

#vae.compile(optimizer='rmsprop', loss=vae_loss)
#sgd = SGD(lr=0.01, momentum=0.1, decay=0.00001, nesterov=True)
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
batch_size = 100
#gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = ae.predict([aFrame, neibF, cut_neibF, weight_neibF])
len(x_test_vae)
x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])
#len(x_test_enc)

cl=9;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
bx.set_xticklabels(rs[cl-1, ]);
#fig3 = plt.figure();
#bz = sns.violinplot(data= x_test_enc[lbls==cl , :], bw =bw);
fig4= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
b0.set_xticklabels(rs[cl-1, ]);
fig5= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0.set_xticklabels(np.round(cutoff,2));




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

for cl in range(24) :
#print(np.mean(np.array(connNoNoise)[lbls==cl]))
    print(cl+1)
    print(np.mean(np.array(conn)[lbls==cl+1]))
    print(np.mean(np.array(connClean)[lbls==cl+1]))
    print(np.mean(np.array(connClean)[lbls==cl+1])-np.mean(np.array(conn)[lbls==cl+1]))
    print(np.mean(np.array(connClean2)[lbls == cl + 1]) - np.mean(np.array(conn)[lbls == cl + 1]))

np.mean( np.array([np.mean(np.array(conn)[lbls==cl+1]) for cl in range(24)] ) )
np.mean( np.array([np.mean(np.array(connClean)[lbls==cl+1]) for cl in range(24)] ) )
np.mean( np.array([np.mean(np.array(connClean2)[lbls==cl+1]) for cl in range(24)] ) )

print(np.mean(np.array(connClean2)[lbls==cl]))


from sklearn.cluster import Birch
import sklearn.metrics
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('manhattan')

brc0=Birch(threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=True, copy=True)
brc0.fit(aFrame)
lbls0=brc0.predict(aFrame)

brc1= Birch(threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=True, copy=True)
brc1.fit(x_test_vae)
lbls1=brc1.predict(x_test_vae)

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
zr= 1-(1*rs)
sum([counts[xx+1] * sum(zr[xx+1,:]) for xx in range(cl)])#13704320



#true number of zeroes

#epochs=4
#IdxF =nb['idx']

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
b0.set_xticklabels(rs[cl-1, ]);




