'''This script to clean cytoff data using deep stacked autoencoder with neighbourhood denoising and
contracting
Now with greedy pretraining
'''
import keras
import tensorflow as tf
import numpy as np
import matplotlib as plt
#from scipy.stats import norm
#import glob
from keras.layers import Input, Dense, Lambda, Layer, Dropout, BatchNormalization
#from keras.layers.noise import AlphaDropout
#from keras.utils import np_utils
from keras.models import Model
from keras import backend as K
#from keras import metrics
#import random
#from keras.utils import multi_gpu_model
import timeit
#from keras.optimizers import SGD
#from keras.optimizers import Adagrad
#from keras.optimizers import Adadelta
from keras.optimizers import Adam
#from keras.constraints import maxnorm
#import readline
#mport rpy2
#from rpy2.robjects.packages import importr
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#stsne = importr('stsne')
#subspace = importr('subspace')
import sklearn
from sklearn.preprocessing import MinMaxScaler
#from kerasvis import DBLogger
#Start the keras visualization server with
#export FLASK_APP=kerasvis.runserver
#flask run
import seaborn as sns
import warnings
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
#from keras.models import load_model
#from keras import regularizers
#import champ
#import ig
#import metric
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce

def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique':unique, 'counts':counts}

desc=["Y89Di", "Cd112Di", "In115Di",  "Pr141Di", "Nd142Di", "Nd143Di", "Nd144Di", "Nd145Di" , "Nd146Di",  "Sm147Di" ,   "Nd148Di"   ,   "Sm149Di"    ,  "Nd150Di"   ,   "Eu151Di" ,  "Sm152Di"   ,   "Eu153Di"   ,   "Sm154Di"   ,   "Gd155Di"   ,   "Gd156Di"    ,    "Gd158Di" , "Tb159Di"   ,   "Gd160Di"  ,"Dy161Di"   ,   "Dy162Di"   ,   "Dy163Di"  ,    "Dy164Di"   ,  "Ho165Di" , "Er166Di" , "Er167Di",   "Er168Di"  , "Tm169Di"  ,   "Er170Di"   ,   "Yb171Di"   ,  "Yb172Di",   "Yb173Di"    ,  "Yb174Di"  ,    "Lu175Di"  ,   "Yb176Di"]

markers = ["89Y-CD45" ,  "112Cd-CD45RA"   ,   "115In-CD8"   ,    "141Pr-CD137"  ,   "142Nd-CD57"  , "143Nd-HLA_DR"   ,  "144Nd-CCR5"  , "145Nd-CD45RO" ,
           "146Nd-FOXP3" ,   "147Sm-CD62L"  ,  "148Nd-PD_L1"   ,  "149Sm-CD56"   ,  "150Nd-LAG3"  ,   "151Eu-ICOS" ,    "152Sm-CCR6"  ,  "153Eu-TIGIT"  ,   "154Sm-TIM3",
           "155Gd-PD1"  ,  "156Nd-CXCR3" ,      "158Gd-CD134"  ,   "159Tb-CCR7"   ,  "160Gd-Tbet"   , "161Dy-CTLA4"  ,   "162Dy-CD27"   ,  "163Dy-BTLA",
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
perp=k
k3=k*3

source_dir = "/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient"
#file_list = glob.glob(source_dir + '/*.txt')
data0 = np.genfromtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/AllPatients.txt'
, names=None, dtype=float, skip_header=1)
patient =  np.int32(data0[:, 39])
IDX=np.logical_or.reduce((patient==1, patient==2, patient==3))
lbls = np.int32(data0[IDX,38])
len(lbls)
aFrame = data0[IDX,:38]
cutoff = np.genfromtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/cutoff.txt'
, delimiter=' ', skip_header=1)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k3, metric='L2', cores=12)
Idx = nb['idx']; Dist = nb['dist']
outfile = 'Mat3.npz'
#np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls, cutoff=cutoff, patient=patient, Dist=Dist)
#npzfile = np.load(outfile)
lbls=npzfile['lbls'];Idx=npzfile['Idx'];aFrame=npzfile['aFrame'];patient=npzfile['patient'];
cutoff=npzfile['cutoff']; Dist =npzfile['Dist']



k=30
perp=k
k3=k*3

#model parameters
batch_size = 10
original_dim = 38
latent_dim = 120

epochs = 10
epsilon_std = 1.0
U=10#energy barier


D=np.shape(aFrame)[1]
#model parameters
M=3 # number of classes
N=D*3 # number of categorical distributions
batch_size = 100
original_dim = D
latent_dim = M*D

nb_hidden_layers = [original_dim, latent_dim, original_dim]

#generate neighbours data
features=aFrame
IdxF = Idx[:, ]
nrow = np.shape(features)[0]
b=0
neibALL = np.zeros((nrow, k3, original_dim))
cnz = np.zeros((original_dim))
cut_neibF = np.zeros((nrow, original_dim))
weight_distALL = np.zeros((nrow, k3))
weight_neibALL = np.zeros((nrow, k3))
rk=range(k3)

sigmaBer = np.sqrt(cutoff*(1-cutoff))
from joblib import Parallel, delayed
import multiprocessing
def singleInput(i):
     nei =  features[IdxF[i,:],:]
     cnz=[np.sum(np.where(nei[:k, j] == 0, 0,1))/ k for j in range(original_dim)]
     cut_nei= np.array([0 if (cnz[j] >= cutoff[j] or cutoff[j]>0.5) else
                        (U/naive_power(cutoff[j], 2)) * naive_power((cutoff[j] - cnz[j])/sigmaBer[j] , 2) for j in range(original_dim)])
     weight_di = [sum(((features[i] - nei[k_i,]) / (1 + cut_nei))**2) for k_i in rk]
     return [nei, cut_nei, weight_di, i]


inputs = range(nrow)
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(singleInput)(i) for i in inputs)
for i in range(nrow):
 neibALL[i,] = results[i][0]
for i in range(nrow):
    cut_neibF[i,] = results[i][1]
for i in range(nrow):
    weight_distALL[i,] = results[i][2]
del results

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

#razobratsya s nulem (index of 0-nearest neightbr )!!!!

perp(np.sqrt(weight_distALL[:,0:k3]),   nrow,     38,   weight_neibALL,          k,          k*3,        12)
#(          double* dist,           int N,    int D,  double* P,     double perplexity, int K, int num_threads)
np.shape(weight_neibALL)
plt.plot(weight_neibALL[5,])

#get neighbors with top k weights and normalize
#[aFrame, neibF, cut_neibF, weight_neibF]

topk = np.argsort(weight_neibALL, axis=1)[:,-k:]
topk= np.apply_along_axis(np.flip, 1, topk,0)

weight_neibF=np.array([ weight_neibALL[i, topk[i]] for i in range(len(topk))])
neibF=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])
weight_neibF=sklearn.preprocessing.normalize(weight_neibF, axis=1, norm='l1')


plt.plot(weight_neibF[2503,])

#regulariztion, not feed forward
neib = Input(shape = (k, original_dim, ))
cut_neib = Input(shape = (original_dim,))
#var_dims = Input(shape = (original_dim,))
weight_neib = Input(shape = (k,))

#pretrain input layers
#aFrame=x_test_vae

trained_weight = []
X_train_tmp = aFrame
for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
    print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
# Create AE and training
    pretrain_input = Input(shape=(n_in,))
    encoder = Dense(n_out, activation='linear')(pretrain_input)
    decoder = Dense(n_in, activation='selu')(encoder)
    ae = Model(input=pretrain_input,output=decoder)
    encoder_temp = Model(input=pretrain_input, output=encoder)
    ae.compile(loss='mean_squared_error', optimizer='RMSprop')
    ae.fit(X_train_tmp, X_train_tmp, batch_size=256, epochs=10)
# Store trainined weight
    trained_weight = trained_weight + encoder_temp.get_weights()
# Update training data
    X_train_tmp = encoder_temp.predict(X_train_tmp)

ae.summary()
from keras.utils.vis_utils import plot_model
plot_model(ae, to_file='model.png')

print('Fine-tuning:')

'''this is our input placeholder'''
x = Input(shape=(original_dim,))
''' "encoded" is the encoded representation of the input'''
encoded1 = Dense(latent_dim, activation='linear')(x)
x_decoded2 = Dense(original_dim, activation='selu')(encoded1)


ae = Model([x, neib, cut_neib, weight_neib], x_decoded2)
#ae = Model([x, neib, cut_neib], x_decoded4)
ae.summary()

ae.set_weights(trained_weight)
#ae.get_weights()


epsilon_std = 0.01
max_temperature = 0.66
min_temperature = 0.66
anneal_rate = np.log(max_temperature/min_temperature)/(epochs-1)

tau = K.variable(max_temperature, name="temperature")

logits_y = Dense(M*N)(encoded1)

def sampling(z):
    U = K.random_uniform(K.shape(z), 0, 1)
    ly = z - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
    ly = softmax(K.reshape(ly, (-1, N, M)) / tau)
    ly = K.reshape(ly, (-1, N*M))
    return ly

y = Lambda(sampling, output_shape=(M*N,))(logits_y)




#ae.summary()


ae = Model([x, neib, cut_neib, weight_neib], x_decoded2)


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

def gumbel_loss(x, x_hat):
    q_y = K.reshape(logits_y, (-1, N, M))
    q_y = softmax(q_y)
    log_q_y = K.log(q_y + 1e-20)
    kl_tmp = q_y * (log_q_y - K.log(1.0/M))
    KL = K.sum(kl_tmp, axis=(1, 2))
    elbo = original_dim * bce(x, x_hat) - KL
    #elbo = - KL
    return elbo

def  kl_l(z_mean, z_log_var):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


def ae_loss(x, x_decoded_mean):
    #msew = original_dim * losses.mean_squared_error(x, x_decoded_mean)
    msew = original_dim * mean_square_error_weighted(x, x_decoded_mean)
    #msewNN = 1/k*original_dim * mean_square_error_weightedNN(x, x_decoded_mean)
    g_l= gumbel_loss(x/(cut_neib + 1), x_decoded_mean)
    #penalty_zero = pen_zero(x_decoded_mean, cut_neib)
    #return K.mean(msew+0*msewNN+0*penalty_zero)
    #return K.mean(0.01*msewNN) +  g_l
    return K.mean(msew) + 0.1*g_l

#vae.compile(optimizer='rmsprop', loss=vae_loss)
#sgd = SGD(lr=0.01, momentum=0.1, decay=0.00001, nesterov=True)
learning_rate = 1e-4
#earlyStopping=CustomEarlyStopping(criterion=0.0001, patience=3, verbose=1)
adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate )

ae.compile(optimizer=adam, loss=ae_loss, metrics=[mean_square_error_weightedNN,mean_square_error_weighted, gumbel_loss, bce], )
#ae.get_weights()

checkpoint = ModelCheckpoint('.', monitor='ae_loss', verbose=1, save_best_only=True, mode='max')
logger = DBLogger(comment="An example run")
start = timeit.default_timer()
#b_sizes = range(10,110,10); i=9

history = ae.fit([aFrame, neibF, cut_neibF, weight_neibF], aFrame,
                     batch_size=batch_size,
                     epochs=epochs,
                     shuffle=True,
                     callbacks=[CustomMetrics(), logger])

#K.set_value(tau, np.max([max_temperature * np.exp(- anneal_rate * e), min_temperature]))
print(K.get_value(tau))
stop = timeit.default_timer()



stop = timeit.default_timer()

print(stop - start)
fig0 = plt.figure()
plt.plot(history.history['loss'])
fig01 = plt.figure()
plt.plot(history.history['mean_square_error_weightedNN'])
fig02 = plt.figure()
plt.plot(history.history['mean_square_error_weighted'])
fig03 = plt.figure()
#plt.plot(history.history['pen_zero'])
#print(ae.summary())

# build a model to project inputs on the latent space
encoder = Model([x, neib, cut_neib, weight_neib], encoded2)
print(encoder.summary())

#sparsify results
#argmax_y = K.max(K.reshape(logits_y, (-1, N, M)), axis=-1, keepdims=True)
#argmax_y = K.equal(K.reshape(logits_y, (-1, N, M)), argmax_y)
#encoder2 = K.function([x], [argmax_y])



# predict and extract latent variables
x_test_vae = ae.predict([aFrame, neibF, cut_neibF, weight_neibF])
len(x_test_vae)
x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])

#code, x_hat_test = encoder2([aFrame])

argmax_y = np.max(np.reshape(x_test_enc,(-1, N, M)), axis=-1, keepdims=True)
argmax_y = np.equal(np.reshape(x_test_enc, (-1, N, M)), argmax_y)
code=argmax_y.astype(int)
codeFlat = np.reshape(code, (len(aFrame), M*N))

cl=1;bw=0.02
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
scaler.fit_transform(x_test_enc)
nb2=find_neighbors(x_test_enc, k, metric='L2', cores=12)
connClean2 = [sum((stats.itemfreq(lbls[nb2['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean2,50)
#print(np.mean(connNoNoise))
print(np.mean(conn))
print(np.mean(connClean))
print(np.mean(connClean2))

nb3=find_neighbors(codeFlat, k, metric='L2', cores=12)
connClean3 = [sum((stats.itemfreq(lbls[nb3['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean3,50)



cl=24
#print(np.mean(np.array(connNoNoise)[lbls==cl]))
print(np.mean(np.array(conn)[lbls==cl]))
print(np.mean(np.array(connClean)[lbls==cl]))
print(np.mean(np.array(connClean2)[lbls==cl]))

from sklearn.preprocessing import normalize
x_test_norm = normalize(x_test_enc2)
correlations = np.dot(x_test_norm[lbls==11, ], np.transpose(x_test_norm[lbls==16, ]))
np.mean(correlations)


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




