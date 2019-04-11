'''This script to clean cytoff data using deep stacked autoencoder with neighbourhood denoising and
contracting
Now with greedy pretraining
Now with entropic weighting of neibourhoods
No perturbation
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
#from keras.utils import multi_gpu_model
import timeit
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.constraints import maxnorm
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
from keras.regularizers import l2, l1
from keras.models import load_model
from keras import regularizers
#import champ
#import ig
#import metric
#import dill
from keras.callbacks import TensorBoard
#from GetBest import GetBest

tensorboard = TensorBoard(log_dir='./logs',  histogram_freq=0,
                          write_graph=True, write_images=True)



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



#using approximate hnswlib
import hnswlib
from scipy.ndimage.interpolation import shift
def find_neighbors(data, k_, metric='euclidean', cores=12):
    dim = np.shape(data)[1]
    num_elements = np.shape(data)[0]
    data_labels = np.arange(num_elements)
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=5000, M=16)
    # Element insertion (can be called several times):
    p.add_items(data, data_labels)
    # Controlling the recall by setting ef:
    p.set_ef(5000)  # ef should always be > k
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=k_+1)
    #correct results
    IDX = distances[:,0]!=0
    distances = np.array([shift(distances[i,:], 1, cval=0) if IDX[i] else distances[i,:] for i in data_labels ])
    labels = np.array([shift(labels[i, :], 1, cval=0) if IDX[i] else labels[i, :] for i in data_labels])
    return {'dist':distances[:,1:], 'idx':labels[:,1:]}





#import vpsearch as vp
#def find_neighbors(data, k_, metric='manhattan', cores=12):
#   res = vp.find_nearest_neighbors(data, k_, cores, metric)
#   return {'dist':np.array(res[1]), 'idx': np.int32(np.array(res[0]))}

#from libKMCUDA import kmeans_cuda, knn_cuda
#def find_neighbors(data, k_, metric='euclidean', cores=12):
#    ca = kmeans_cuda(np.float32(data), 25, metric="euclidean", verbosity=1, seed=3, device=0)
#    neighbors = knn_cuda(k_, np.float32(data), *ca, metric=metric, verbosity=1, device=0)
#    return {'dist':0, 'idx': np.int32(neighbors)}






#load data
k=30
perp=k
k3=k*3
'''
source_dir = "/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient"
#file_list = glob.glob(source_dir + '/*.txt')
data0 = np.genfromtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/rln4Asin5Transform.txt'
, names=None, dtype=float, skip_header=1)
aFrame = data0[:,:38] 

lbls=np.genfromtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/rln4NUmlbls.txt'
, names=None, dtype=int, skip_header=1)

popnames = [line.strip() for line in open('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/rln4names.txt', 'r')]
popnames=popnames[1:]

len(lbls)
cutoff = np.genfromtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/cutoff.txt'
, delimiter=' ', skip_header=1)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=12)
Idx = nb['idx']; Dist = nb['dist']
'''
outfile = 'Mat1pat4Asinh5TransformManhattan.npz'
#np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls, cutoff=cutoff,  Dist=Dist)
npzfile = np.load(outfile)
lbls=npzfile['lbls'];Idx=npzfile['Idx'];aFrame=npzfile['aFrame'];
cutoff=npzfile['cutoff']; Dist =npzfile['Dist']
#lbls2=npzfile['lbls'];Idx2=npzfile['Idx'];aFrame2=npzfile['aFrame'];
#cutoff2=npzfile['cutoff']; Dist2 =npzfile['Dist']
plt.plot(Dist[5,])
popnames = [line.strip() for line in open('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/rln4names.txt', 'r')]
popnames=popnames[1:]

#model parameters
batch_size = 500
original_dim = 38
latent_dim = 2
intermediate_dim = 19
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]
epochs = 1000
epsilon_std = 1.0
U=10#energy barier
r=0#number of replicas in training data set
v=0#number of replicas in validation data set


#nearest neighbours

#generate neighbours data
features=aFrame
IdxF = Idx[:, ]
nrow = np.shape(features)[0]
b=0
neibALL = np.zeros((nrow, k3, original_dim))
cnz = np.zeros((original_dim))
cut_neibF = np.zeros((nrow, original_dim))
weight_distALL = np.zeros((nrow, k3))

rk=range(k3)

from scipy.stats import binom
cutoff=np.repeat(0.1,38)
sigmaBer = np.sqrt(cutoff*(1-cutoff)/k)
#precompute pmf for all cutoffs at given k
probs=1-cutoff
pmf= np.zeros((original_dim, k+1))
for j in range(original_dim):
    rb=binom(k, probs[j])
    pmf[j, :] = (1-rb.cdf(range(k+1)))**10
def singleInput(i):
     nei =  features[IdxF[i,:],:]
     cnz=[np.sum(np.where(nei[:k, j] == 0, 0,1)) for j in range(original_dim)]
     #cut_nei= np.array([0 if (cnz[j] >= cutoff[j] or cutoff[j]>0.5) else
     #                   (U/(cutoff[j]**2)) * ( (cutoff[j] - cnz[j]) / sigmaBer[j] )**2 for j in range(original_dim)])
     cut_nei = np.array([U *  pmf[j,:][cnz[j]] for j in range(original_dim)])
#weighted distances computed in L1 metric
     weight_di = [sum((np.square(features[i] - nei[k_i,] / (1 + cut_nei)))) for k_i in rk]
     return [nei, cut_nei, weight_di, i]


inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
for i in range(nrow):
 neibALL[i,] = results[i][0]
for i in range(nrow):
    cut_neibF[i,] = results[i][1]
for i in range(nrow):
    weight_distALL[i,] = results[i][2]
del results

np.sum(cut_neibF!=0)
#plt.hist(cut_neibF[cut_neibF!=0],50)


#compute weights
import ctypes
from numpy.ctypeslib import ndpointer
#del lib
#del perp
#import _ctypes
#_ctypes.dlclose(lib._handle )
#del perp
sigmaTsne= np.zeros(nrow)
lib = ctypes.cdll.LoadLibrary("/mnt/f/Brinkman group/current/Stepan/Clibs/perp8.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double, ctypes.c_size_t, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]
weight_neibALL = np.zeros((nrow, k3))

#razobratsya s nulem (index of 0-nearest neightbr )!!!!
#here si the fifference with equlid -no sqrt
perp(np.sqrt((weight_distALL[:,0:k3])),   nrow,     38,   weight_neibALL,          k,         k*3,          sigmaTsne,         12)
#(          double* dist,       int N,    int D,  double* P,     double perplexity, int K,    double* Sigma,   int num_threads)
np.shape(weight_neibALL)
plt.plot(weight_neibALL[200,])
plt.plot(Dist[200,]/10)

np.min(sigmaTsne)

np.max(sigmaTsne)

# the histogram of the data
plt.hist((sigmaTsne), bins='auto')
plt.grid(True)
plt.show()


#get neighbors with top k weights and normalize and rearrange
#[aFrame, neibF, cut_neibF, weight_neibF]

topk = np.argsort(weight_neibALL, axis=1)[:,-k:]
topk= np.apply_along_axis(np.flip, 1, topk,0)

weight_neibF=np.array([ weight_neibALL[i, topk[i]] for i in range(len(topk))])
neibF=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])
weight_neibF=sklearn.preprocessing.normalize(weight_neibF, axis=1, norm='l1')
plt.plot(weight_neibF[5,]);plt.show()


#pert = np.zeros((nrow*(r+v), original_dim))
#def Perturbation(i):
#    sample=np.stack([np.random.choice(neibF[i,:,j],  r+v, p=weight_neibF[i,:]) for j in range(original_dim)],axis=-1)
#    return sample
#resSample = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(Perturbation, check_pickle=False)(i) for i in inputs)
#data = np.vstack(resSample)
#del resSample

#IDXtr= np.tile(np.concatenate((np.repeat(True,r),np.repeat(False,v))),  nrow)
#IDXval= np.tile(np.concatenate((np.repeat(False,r),np.repeat(True,v))),  nrow)

#[aFrame, neibF, cut_neibF, weight_neibF]
#training set
#targetTr = np.repeat(aFrame, r, axis=0)
targetTr = aFrame
neibF_Tr = neibF
cut_neibF_Tr = cut_neibF
weight_neibF_Tr = weight_neibF
sourceTr = aFrame
#validation set
#targetVal = np.repeat(aFrame, v, axis=0)
#neibF_Val = np.repeat(neibF, v, axis =0)
#cut_neibF_Val = np.repeat(cut_neibF, v, axis =0)
#weight_neibF_Val = np.repeat(weight_neibF, v, axis =0)
#sourceVal = data[IDXval, ]
#[sourceTr, neibF_Tr, cut_neibF_Tr, weight_neibF_Tr], targetTr
#[sourceVal, neibF_Val, cut_neibF_Val, weight_neibF_Val], targetVal

#regulariztion, not feed forward


#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(ae).create(prog='dot', format='svg'))




SigmaTsq = Input(shape = (1,))
neib = Input(shape = (k, original_dim, ))
cut_neib = Input(shape = (original_dim,))
#var_dims = Input(shape = (original_dim,))
weight_neib = Input(shape = (k,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
#h.set_weights(ae.layers[1].get_weights())
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


def mean_square_error_NN(y_true, y_pred):
    #dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, K.transpose(weight_neib))
    return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    #return  tf.multiply(weightedN, 0.5 )

def kl_loss(x, x_decoded_mean):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

def vae_loss(x, x_decoded_mean):
    msew = k*original_dim * mean_square_error_NN(x, x_decoded_mean)
    #pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(msew + kl_loss(x, x_decoded_mean))

#y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model([x, neib, cut_neib, SigmaTsq, weight_neib], x_decoded_mean)
vae.summary()

#vae.set_weights(trained_weight)


from keras.utils import plot_model
plot_model(vae, to_file='/mnt/f/Brinkman group/current/Stepan/PyCharm/PhenographLevin13/model.png')

''' this model maps an input to its reconstruction'''


learning_rate = 1e-3
earlyStopping=keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001, patience=3, verbose=0, mode='min')
adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate / epochs)

#ae.compile(optimizer=adam, loss=ae_loss)
vae.compile(optimizer=adam, loss=vae_loss, metrics=[kl_loss, mean_square_error_NN])
#ae.get_weights()

checkpoint = ModelCheckpoint('.', monitor='loss', verbose=1, save_best_only=True, mode='max')
#logger = DBLogger(comment="An example run")

start = timeit.default_timer()
history=vae.fit([sourceTr, neibF_Tr, cut_neibF_Tr, sigmaTsne, weight_neibF], targetTr,
batch_size=batch_size,
epochs = epochs,
shuffle=True,
callbacks=[CustomMetrics(),  tensorboard, earlyStopping])
stop = timeit.default_timer()
vae.save('Wang0_modelnol1.h5')


from keras.models import load_model
#ae=load_model('Wang0_modell1.h5', custom_objects={'mean_square_error_weighted':mean_square_error_weighted, 'ae_loss':
#  ae_loss, 'mean_square_error_weightedNN' : mean_square_error_weightedNN})
#ae = load_model('Wang0_modell1.h5')

print(stop - start)
fig0 = plt.figure();
plt.plot(history.history['loss']);

print(stop - start)
fig0 = plt.figure();
plt.plot(history.history['kl_loss'][10:]);

fig02 = plt.figure();
plt.plot(history.history['mean_square_error_NN']);

#encoder = Model([x, neib, cut_neib], encoded2)
encoder = Model([x, neib, cut_neib, SigmaTsq, weight_neib], z_mean)
print(encoder.summary())

# predict and extract latent variables

#gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
x_test_vae = vae.predict([sourceTr, neibF_Tr, cut_neibF_Tr, sigmaTsne,  weight_neibF])
len(x_test_vae)
#np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_vaeMoeWeights.txt', x_test_vae)
#x_test_vae=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_ae001Pert.txt.txt')
x_test_enc = encoder.predict([sourceTr, neibF_Tr, cut_neibF_Tr, sigmaTsne, weight_neibF])

cl=4;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
plt.plot(cutoff)
plt.show();
#ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
plt.plot(cutoff)
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
plt.show();

cl=1
fig4= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
#b0.set_xticklabels(rs[cl-1, ]);
fig5= plt.figure();
b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
b0.set_xticklabels(np.round(cutoff,2));
plt.show()

unique0, counts0 = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique0, counts0)).T)
num_clus=len(counts0)

from scipy import stats

conn = [sum((stats.itemfreq(lbls[Idx[x,:k]])[:,1] / k)**2) for x in range(len(aFrame)) ]

plt.figure();plt.hist(conn,50);plt.show()


nb=find_neighbors(x_test_vae, k3, metric='manhattan', cores=12)


connClean = [sum((stats.itemfreq(lbls[nb['idx'][x,:k]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean,50);plt.show()

scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb2=find_neighbors(x_test_enc, k, metric='manhattan', cores=12)
connClean2 = [sum((stats.itemfreq(lbls[nb2['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean2,50);plt.show()
#print(np.mean(connNoNoise))

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


#plotly in 3d

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter

#np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt', x_test_enc)
#x_test_enc=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_encBcells3d.txt')

nrow=np.shape(x_test_enc)[0]
#subsIdx=np.random.choice(nrow,  500000)

x=x_test_enc[:,0]
y=x_test_enc[:,1]
z=x_test_enc[:,2]

plot([Scatter3d(x=x,y=y,z=z,
                mode='markers',
        marker=dict(
        size=1,
        color = lbls/5,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.5,
            ),
                text=popnames,
                hoverinfo = 'text')])



cl=9
plot([Scatter3d(x=x,y=y,z=z,
                mode='markers',
        marker=dict(
        size=[2 if x==cl else 0.5 for x in lbls],
        color = ['red' if x==cl else 'green' for x in lbls],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.5,
            ),
                text=popnames,
                hoverinfo = 'text')])


x=x_test_enc[:,0]
y=x_test_enc[:,1]
plot([Scatter(x=x,y=y,
                mode='markers',
        marker=dict(
        size=1,
        color = lbls/5,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=1,
            ),
                text=popnames,
                hoverinfo = 'text')])





#second run
for iii in range(10):
    print('run number: ', iii)
    IdxF_old=IdxF
    #del nb2
    IdxF = nb['idx']
    del nb
    diffNeib= IdxF - IdxF_old
    print('number of first neighbors didn\'t change:', sum(diffNeib[:,0]==0) )

    features=aFrame

    nrow = np.shape(features)[0]
    b=0
    neibALL = np.zeros((nrow, k3, original_dim))
    cnz = np.zeros((original_dim))
    cut_neibF = np.zeros((nrow, original_dim))
    weight_distALL = np.zeros((nrow, k3))
    weight_neibALL = np.zeros((nrow, k3))
    rk=range(k3)

    sigmaBer = np.sqrt(cutoff*(1-cutoff))

    inputs = range(nrow)
    from joblib import Parallel, delayed
    from pathos import multiprocessing
    num_cores = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(num_cores, maxtasksperchild=1000)
    results = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
    for i in range(nrow):
     neibALL[i,] = results[i][0]
    for i in range(nrow):
        cut_neibF[i,] = results[i][1]
    for i in range(nrow):
        weight_distALL[i,] = results[i][2]
    del results

    #compute weights

    perp((weight_distALL[:,0:k3]),   nrow,     38,   weight_neibALL,          k,          k*3,        12)
    #(          double* dist,           int N,    int D,  double* P,     double perplexity, int K, int num_threads)
    np.shape(weight_neibALL)
    #plt.plot(weight_neibALL[5,])

    #get neighbors with top k weights and normalize
    #[aFrame, neibF, cut_neibF, weight_neibF]

    topk = np.argsort(weight_neibALL, axis=1)[:,-k:]
    topk= np.apply_along_axis(np.flip, 1, topk,0)

    weight_neibF=np.array([ weight_neibALL[i, topk[i]] for i in range(len(topk))])
    neibF=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])
    weight_neibF=sklearn.preprocessing.normalize(weight_neibF, axis=1, norm='l1')
    #plt.plot(weight_neibF[200,]);plt.show()


    pert = np.zeros((nrow*(r+v), original_dim))
    resSample = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(Perturbation, check_pickle=False)(i) for i in inputs)
    data = np.vstack(resSample)
    del resSample

    IDXtr= np.tile(np.concatenate((np.repeat(True,r),np.repeat(False,v))),  nrow)
    IDXval= np.tile(np.concatenate((np.repeat(False,r),np.repeat(True,v))),  nrow)

    #[aFrame, neibF, cut_neibF, weight_neibF]
    #training set
    targetTr = np.repeat(aFrame, r, axis=0)
    targetTr = np.vstack((aFrame, targetTr))
    neibF_Tr = np.repeat(neibF, r, axis =0)
    neibF_Tr = np.vstack((neibF, neibF_Tr))
    cut_neibF_Tr = np.repeat(cut_neibF, r, axis =0)
    cut_neibF_Tr = np.vstack((cut_neibF, cut_neibF_Tr))
    weight_neibF_Tr = np.repeat(weight_neibF, r, axis =0)
    weight_neibF_Tr = np.vstack((weight_neibF, weight_neibF_Tr))
    sourceTr = np.vstack((aFrame, data[IDXtr, ]))
    #validation set
    targetVal = np.repeat(aFrame, v, axis=0)
    neibF_Val = np.repeat(neibF, v, axis =0)
    cut_neibF_Val = np.repeat(cut_neibF, v, axis =0)
    weight_neibF_Val = np.repeat(weight_neibF, v, axis =0)
    sourceVal = data[IDXval, ]
    #[sourceTr, neibF_Tr, cut_neibF_Tr, weight_neibF_Tr], targetTr
    #[sourceVal, neibF_Val, cut_neibF_Val, weight_neibF_Val], targetVal

    #regulariztion, not feed forward

    '''this is our input placeholder'''
    x = Input(shape=(original_dim,))
    ''' "encoded" is the encoded representation of the input'''
    encoded1 = Dense(intermediate_dim, activation='selu')(x)

    encoded2 = Dense(latent_dim, activation='selu')(encoded1)

    x_decoded3 = Dense(intermediate_dim, activation='selu')(encoded2)

    x_decoded4 = Dense(original_dim, activation='relu')(x_decoded3)
    ae = Model([x, neib, cut_neib, weight_neib], x_decoded4)
    # ae = Model([x, neib, cut_neib], x_decoded4)
    ae.summary()

    ae.set_weights(trained_weight)


    #ae.get_weights()

    ''' this model maps an input to its reconstruction'''


    learning_rate = 1e-3
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=3, verbose=0, mode='min')
    adam = Adam(lr=learning_rate, epsilon=0.001, decay = learning_rate / epochs)

    #ae.compile(optimizer=adam, loss=ae_loss)
    ae.compile(optimizer=adam, loss=ae_loss, metrics=[mean_square_error_weighted, mse])
    #ae.get_weights()

    #checkpoint = ModelCheckpoint('.', monitor='ae_loss', verbose=1, save_best_only=True, mode='max')
    logger = DBLogger(comment="An example run")
    start = timeit.default_timer()
    #b_sizes = range(10,110,10); i=9
    #for i in range(10) :

    start = timeit.default_timer()
    history = ae.fit([sourceTr, neibF_Tr, cut_neibF_Tr, weight_neibF_Tr], targetTr,
                     validation_data=([sourceVal, neibF_Val, cut_neibF_Val, weight_neibF_Val], targetVal),
                     batch_size=batch_size,
                     epochs=epochs,
                     shuffle=True,
                     callbacks=[CustomMetrics(), logger, tensorboard,
                                GetBest(monitor='val_loss', verbose=1, mode='min'), earlyStopping])

    stop = timeit.default_timer()
    ae.save('Wang0_modelPert2ndrun.h5')
    #tr_weights = ae.get_weights()
    from keras.models import load_model
    #ae=load_model('Wang0_modelPert2ndrun.h5', custom_objects={'mean_square_error_weighted':mean_square_error_weighted, 'ae_loss':
    #    ae_loss, 'mean_square_error_weightedNN' : mean_square_error_weightedNN})
    #ae = load_model('Wang0_modelPert2ndrun.h5')
    print(stop - start)
    '''
    fig0 = plt.figure();
    plt.plot(history.history['loss']);
    fig01 = plt.figure();
    plt.plot(history.history['mean_square_error_weightedNN']);
    fig02 = plt.figure();
    plt.plot(history.history['mean_square_error_weighted']);
    fig02_1 = plt.figure();
    plt.plot(history.history['val_loss']);
    plt.show()
    #fig03 = plt.figure();
    #plt.plot(history.history['pen_zero']);
    print(ae.summary())
    '''
    # build a model to project inputs on the latent space
    #encoder = Model([x, neib, cut_neib], encoded2)

    #print(encoder.summary())

    # predict and extract latent variables

    #gen_pred = generatorNN(aFrame, aFrame, Idx, Dists, batch_size=batch_size,  k_=k, shuffle = False)
    x_test_vae = ae.predict([aFrame, neibF, cut_neibF, weight_neibF])
    len(x_test_vae)
    np.savetxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_ae001Pert2ndrun.txt', x_test_vae)
    #x_test_vae=np.loadtxt('/mnt/f/Brinkman group/current/Stepan/WangData/WangDataPatient/x_test_ae001Pert2ndrun.txt')
    '''
    x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])
    #len(x_test_enc)
    #3,8,13
    cl=13;bw=0.02
    fig1 = plt.figure();
    ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
    plt.show();
    #ax.set_xticklabels(rs[cl-1, ]);
    fig2 = plt.figure();
    bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
    plt.show();
    
    cl=1
    fig4= plt.figure();
    b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
    b0=sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');
    #b0.set_xticklabels(rs[cl-1, ]);
    fig5= plt.figure();
    b0=sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='skyblue');
    b0.set_xticklabels(np.round(cutoff,2));
    plt.show()
    
    unique0, counts0 = np.unique(lbls, return_counts=True)
    print('%d %d', np.asarray((unique0, counts0)).T)
    num_clus=len(counts0)
    
    from scipy import stats
    
    #conn = [sum((stats.itemfreq(lbls[Idx[x,:k]])[:,1] / k)**2) for x in range(len(aFrame)) ]
    plt.figure();plt.hist(conn,50);plt.show()
    '''

    nb=find_neighbors(x_test_vae, k3, metric='manhattan', cores=12)

connClean = [sum((stats.itemfreq(lbls[nb['idx'][x,:k]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean,50);plt.show()

encoder = Model([x, neib, cut_neib, weight_neib], encoded2)
x_test_enc = encoder.predict([aFrame, neibF, cut_neibF, weight_neibF])
encoder2 = Model([x, neib, cut_neib, weight_neib], x_decoded3)
x_test_enc2 = encoder2.predict([aFrame, neibF, cut_neibF, weight_neibF])
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb2=find_neighbors(x_test_enc, k, metric='manhattan', cores=12)
connClean2 = [sum((stats.itemfreq(lbls[nb2['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean2,50);plt.show()
nb3=find_neighbors(x_test_enc2, k, metric='manhattan', cores=12)
connClean3 = [sum((stats.itemfreq(lbls[nb3['idx'][x,:]])[:,1] / k)**2) for x in range(len(aFrame)) ]
plt.figure();plt.hist(connClean3,50);plt.show()
#print(np.mean(connNoNoise))

for cl in unique0 :
#print(np.mean(np.array(connNoNoise)[lbls==cl]))
    print(cl)
    print(np.mean(np.array(conn)[lbls==cl]))
    print(np.mean(np.array(connClean)[lbls==cl]))
    print(np.mean(np.array(connClean)[lbls==cl])-np.mean(np.array(conn)[lbls==cl]))
    print(np.mean(np.array(connClean2)[lbls == cl]) - np.mean(np.array(conn)[lbls == cl ]))
    print(np.mean(np.array(connClean3)[lbls == cl]) - np.mean(np.array(conn)[lbls == cl ]))

print(np.mean( np.array([np.mean(np.array(conn)[lbls==cl]) for cl in unique0] ) ))
print(np.mean( np.array([np.mean(np.array(connClean)[lbls==cl]) for cl in unique0] ) ))
print(np.mean( np.array([np.mean(np.array(connClean2)[lbls==cl]) for cl in unique0] ) ))
print(np.mean( np.array([np.mean(np.array(connClean3)[lbls==cl]) for cl in unique0] ) ))








from MulticoreTSNE import MulticoreTSNE as TSNE

tsne = TSNE(n_jobs=12,  n_iter=5000, metric='manhattan')
Y0 = tsne.fit_transform(aFrame)
tsne = TSNE(n_jobs=12,  n_iter=5000)
Y1 = tsne.fit_transform(x_test_vae, metric='manhattan')
tsne = TSNE(n_jobs=12,  n_iter=5000)
Y2 = tsne.fit_transform(x_test_enc, metric='manhattan')
np.savez('otsnePertnewlbls.npzMulti', Y0=Y0, Y1=Y1)


fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls== g)
    ax.scatter(Y0[i,0], Y0[i,1], label=g, marker='.', s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()

fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls==g)
    ax.scatter(Y1[i,0], Y1[i,1], label=g, marker='.',s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()


from sklearn.manifold import TSNE as oldTSNE
colors = ['red', 'brown', 'yellow', 'green', 'blue']
#from plt.colors import LinearSegmentedColormap
#cmap = LinearSegmentedColormap.from_list('name', colors)

otsne = oldTSNE(init="pca", n_iter=5000, metric='manhattan')
Y0o = otsne.fit_transform(aFrame)
otsne = oldTSNE(init="pca", n_iter=5000, metric='manhattan')
Y1o = otsne.fit_transform(x_test_vae)
otsne = oldTSNE(init="pca", n_iter=5000, metric='manhattan')
Y1o = otsne.fit_transform(x_test_enc)

np.savez('otsnePertMAnhattanDenoise.npz', Y0o=Y0o, Y1o=Y1o)
#np.savez('otsnePertMAnhattanDenoise.npz', Y0o=Y0o, Y1o=Y1o)
ots=np.load("otsne.npz")
Y0o=ots['Y0o'];Y1o=ots['Y1o']; #Y0=ots['Y0'];Y1=ots['Y1'];


from matplotlib.pyplot import cm
#color=cm.tab20c(np.linspace(0,1,16))
color =  plt.cm.Vega20c( (4./3*np.arange(24*3/4)).astype(int) )
color=np.random.permutation(color)


fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y0o[i,0], Y0o[i,1], label=g, marker='.', s=1, c=color[g])
ax.legend(markerscale=10)
plt.show()


fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y1o[i,0], Y1o[i,1], label=g, marker='.', s=1, c=color[g])
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
importlib.reload(phenograph2)


from tsne import bh_sne

Y0all3 =  bh_sne(aFrame, max_iter=1000, d=3)

Y1all3 = bh_sne(x_test_vae, max_iter=1000, d=3)

np.savez("3Dtsne10000.npz", Y0all3=Y0all3, Y1all3=Y1all3)

fl = np.load("/home/sgrinek/anaconda3/tsne5000.npz")
Y0all=fl['Y0all'];Y1all=fl['Y1all']


#import plt.cm
color =  ['black', 'red', 'orange', 'yellow', 'green', 'black', 'brown']
fig, ax = plt.subplots()
for g in np.unique(patient):
    i = np.where(patient == g)
    ax.scatter(Y0all[i,0], Y0all[i,1], label=g, marker='.', s=0.2, c=color[g], alpha=0.9)
ax.legend(markerscale=10)
plt.show()


# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
#cmaplist[0] = (.5,.5,.5,1.0)
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
nl=len(table(lbls)['unique'])
bounds = np.linspace(0,nl,nl+1)
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1,1, figsize=(6,6))
scat = ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=lbls, alpha=0.9, cmap=cmap, norm=norm)
ax2 = fig.add_axes([0.90, 0.1, 0.03, 0.8])
cb = ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
ax.set_title('Manual labels')
ax2.set_ylabel('clusters', size=8)

plt.show()

fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y0o[i,0], Y0o[i,1], label=g, marker='.', s=0.2, c=color[g], alpha=0.9)
ax.legend(markerscale=10)
plt.show()

color=['yellow','yellow', 'red','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow']
#color[6] ='red'
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

color=['yellow','yellow', 'red','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow','yellow', 'yellow']
#color[6] ='red'
fig, ax = plt.subplots()
for g in np.unique(lbls):
    i = np.where(lbls == g)
    ax.scatter(Y0all[i,0], Y0all[i,1], label=g, marker='.', s=0.2, c=aFrame[:,35], alpha=0.9)
ax.legend(markerscale=10)
plt.show()


markers = ["89Y-CD45" ,  "112Cd-CD45RA"   ,   "115In-CD8"   ,    "141Pr-CD137"  ,   "142Nd-CD57"  , "143Nd-HLA_DR"   ,
           "144Nd-CCR5"  , "145Nd-CD45RO" ,   "146Nd-FOXP3" ,   "147Sm-CD62L"  ,  "148Nd-PD_L1"   ,  "149Sm-CD56"   ,
           "150Nd-LAG3"  ,  "151Eu-ICOS" ,    "152Sm-CCR6"  ,  "153Eu-TIGIT"  ,   "154Sm-TIM3",      "155Gd-PD1"  ,
           "156Nd-CXCR3" ,  "158Gd-CD134" ,    "159Tb-CCR7" ,  "160Gd-Tbet"   ,   "161Dy-CTLA4"  ,   "162Dy-CD27"   ,
           "163Dy-BTLA",    "164Dy-CCR4"   , "165Ho-CD101"  ,  "166Er-EOMES"  ,  "167Er-GATA3"  ,  "168Er-CD40L" ,
           "169Tm-CD25"   ,   "170Er-CD3" ,   "171Yb-CXCR5" ,    "172Yb-CD38",   "173Yb-GrnzB"   ,   "174Yb-CD4",
           "175Lu-Perforin"  ,"176Yb-CD127"]



m=35
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("outputBeforeDenoise.pdf")
for m in range(38):
    colors = ['blue', 'green', 'yellow', 'brown', 'red']
    #import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('name', colors)
    fig, ax = plt.subplots()
    im=ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=aFrame[:,m], cmap=cmap, alpha=0.9, )
    fig.colorbar(im, ax=ax, orientation='horizontal')
    plt.title(markers[m])
    pdf.savefig(fig)
pdf.close()

pdf = matplotlib.backends.backend_pdf.PdfPages("outputAfterDenoise.pdf")
for m in range(38):
    colors = ['blue', 'green', 'yellow', 'brown', 'red']
    #import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('name', colors)
    fig, ax = plt.subplots()
    im=ax.scatter(Y1o[:,0], Y1o[:,1],  marker='.', s=0.2, c=aFrame[:,m], cmap=cmap, alpha=0.9, )
    fig.colorbar(im, ax=ax, orientation='horizontal')
    plt.title(markers[m])
    pdf.savefig(fig)
pdf.close()






colors = ['blue', 'green', 'yellow', 'brown', 'red']
#import matplotlib
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('name', colors)
fig, ax = plt.subplots()
im=ax.scatter(Y1o[:,0], Y1o[:,1],  marker='.', s=0.2, c=lbls, cmap=cmap, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal')
plt.title(markers[m])
plt.show()




fig, ax = plt.subplots()
im=ax.scatter(Y0all[:,0], Y0all[:,1],  marker='.', s=0.2, c=data0[:,38], cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('populations')
plt.show()

fig, ax = plt.subplots()
im=ax.scatter(Y0all[:,0], Y0all[:,1],  marker='.', s=0.2, c=data0[:,39], cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('patients')
plt.show()

IDX =  np.array((patient==5) + (patient==4) + (patient==6))
IDX2=(np.logical_and.reduce((lbls!=11,  lbls!=9, lbls!=4, lbls!=15, lbls==1)))
fig, ax = plt.subplots()
im=ax.scatter(Y0all[list(IDX),0][IDX2], Y0all[IDX,1][IDX2],  marker='.', s=0.2, c=communities[IDX2], cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('communities')
plt.show()

IDX =  np.array((patient==5) + (patient==4) + (patient==6))
IDX2=(np.logical_and.reduce((lbls!=11,  lbls!=9, lbls!=4, lbls!=15, lbls==1)))
fig, ax = plt.subplots()
im=ax.scatter(Y0all[list(IDX),0][IDX2], Y0all[IDX,1][IDX2],  marker='.', s=0.2, c=communitiesC[IDX2], cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('communitiesC')
plt.show()


IDX =  np.array((patient==5) + (patient==4) + (patient==6))
IDX2=(np.logical_and.reduce((lbls!=11, lbls!=15, lbls!=9, lbls!=4)))
fig, ax = plt.subplots()
im=ax.scatter(Y0all[list(IDX),0][IDX2], Y0all[IDX,1][IDX2],  marker='.', s=0.2, c=lbls[IDX2], cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('lbls')
plt.show()





fig, ax = plt.subplots()
im=ax.scatter(Y0all[list(IDX),0], Y0all[IDX,1],  marker='.', s=0.2, c=communitiesC, cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('communitiesC')
plt.show()

fig, ax = plt.subplots()
im=ax.scatter(Y0all[list(IDX),0], Y0all[IDX,1],  marker='.', s=0.2, c=lbls, cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('communitiesC')
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


from phenograph2 import cluster
#import importlib
#importlib.reload(phenograph2)

#pt3=patient[np.logical_or.reduce((patient==4, patient==5, patient==6))]
communities, graph, Q = cluster(aFrame, k=60, nn_method='vptree', n_jobs=12)
communitiesC, graphC, QC = cluster(x_test_vae, k=60, nn_method='vptree', n_jobs=12)
communitiesE, graphE, QE = cluster(x_test_enc, k=60, nn_method='vptree', n_jobs=12)
communitiesE2, graphE2, QE2 = cluster(x_test_enc2, k=60, nn_method='vptree', n_jobs=12)
table(communities);
table(communitiesC);
table(communitiesE);
table(communitiesE2);

print(sklearn.metrics.adjusted_mutual_info_score(communities[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))

print(sklearn.metrics.adjusted_mutual_info_score(communitiesC[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE2[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))

print(sklearn.metrics.adjusted_mutual_info_score(communities,  lbls))

print(sklearn.metrics.adjusted_mutual_info_score(communitiesC,  lbls))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE,  lbls))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE2,  lbls))


pt=4
idxk=lbls!=11, lbls!=1, lbls!=6, lbls!=14, lbls!=15
import phenograph
#pt3=patient[np.logical_or.reduce((patient==4, patient==5, patient==6))]
communities, graph, Q = phenograph.cluster(aFrame, k=60, nn_method='kdtree', n_jobs=12, primary_metric='manhattan')
communitiesC, graphC, QC = phenograph.cluster(x_test_vae, k=60, nn_method='kdtree', n_jobs=12,  primary_metric='manhattan')
communitiesE, graphE, QE = phenograph.cluster(x_test_enc, k=60, nn_method='kdtree', n_jobs=12,  primary_metric='manhattan')
communitiesE2, graphE2, QE2 = phenograph.cluster(x_test_enc2, k=60, nn_method='kdtree', n_jobs=12,  primary_metric='manhattan')
table(communities);
table(communitiesC);
table(communitiesE);
table(communitiesE2);

print(sklearn.metrics.adjusted_mutual_info_score(communities[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))

print(sklearn.metrics.adjusted_mutual_info_score(communitiesC[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE2[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))

print(sklearn.metrics.adjusted_mutual_info_score(communities,  lbls))

print(sklearn.metrics.adjusted_mutual_info_score(communitiesC,  lbls))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE,  lbls))
print(sklearn.metrics.adjusted_mutual_info_score(communitiesE2,  lbls))

#clustering using graph weights created by perplexity
import scipy
from scipy.sparse import coo_matrix
kk=latent_dim*3
weight_dist_enc = np.zeros((nrow, kk))
weight_neib_enc = np.zeros((nrow, kk))
k=60
neibMat=find_neighbors(x_test_enc, kk, metric='euclidean', cores=12)
weight_dist_enc=neibMat['dist']
DimD=latent_dim#intermediate_dim
perp(np.sqrt(weight_dist_enc[:,0:kk]),   nrow,             DimD  ,      weight_neib_enc,          k,          kk,        12)
#(               double* dist,           int N,            int D,          double* P,     double perplexity, int K, int num_threads)
np.shape(weight_neib_enc)

weight_neib_enc_1 = np.append(np.zeros( (nrow, 1), dtype='int64'), weight_neib_enc, axis=1)
weight_neib_enc_1 = weight_neib_enc_1[:, 0:(k+1)]
data_flat=np.ndarray.flatten(weight_neib_enc_1)
zind = np.arange(nrow)
zind.shape = (nrow,1)
mat_enc_1 = np.append(zind, neibMat['idx'][:, 0:k], axis=1)
col_enc_1 = np.ndarray.flatten(mat_enc_1)
row_enc_1=np.repeat(zind, k+1)
adj_mat = coo_matrix((data_flat, (row_enc_1, col_enc_1)), shape=(nrow,nrow))
##take smallest of two weights to symmetrize the matrix
adj_mat = scipy.sparse.csr_matrix(adj_mat)
#adj_sym=adj_mat.mean(adj_mat.transpose(copy=True))
adj_sym =1/2*(adj_mat+adj_mat.transpose(copy=True))
import igraph
rows, cols = adj_sym.nonzero()
vals = adj_sym.data
edges = zip(rows, cols)
G = igraph.Graph(edges=list(edges), directed=False)
G.es['weight'] = vals
G.vs['label'] = zind
Louv_res =  igraph.Graph.community_multilevel(G, weights='weight', return_levels=True)

pcl1=np.int32(Louv_res[1].membership)
pcl2=np.int32(Louv_res[2].membership)

idxk=lbls!=11, lbls!=1, lbls!=6, lbls!=14, lbls!=15
print(sklearn.metrics.adjusted_mutual_info_score(pcl1[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))
print(sklearn.metrics.adjusted_mutual_info_score(pcl2[np.logical_and.reduce((idxk))],
                                                             lbls[np.logical_and.reduce((idxk))]))
print(sklearn.metrics.adjusted_mutual_info_score(pcl1, lbls))
print(sklearn.metrics.adjusted_mutual_info_score(pcl2, lbls))



ots=np.load("otsne.npz")
Y0o=ots['Y0o'];Y1o=ots['Y1o']; #Y0=ots['Y0'];Y1=ots['Y1'];

# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
#cmaplist[0] = (.5,.5,.5,1.0)
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
nl=len(table(lbls)['unique'])
bounds = np.linspace(0,nl,nl+1)
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1,1, figsize=(6,6))
scat = ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=lbls, alpha=0.9, cmap=cmap, norm=norm)
ax2 = fig.add_axes([0.90, 0.1, 0.03, 0.8])
cb = ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
ax.set_title('Manual labels')
ax2.set_ylabel('clusters', size=8)

nl=len(table(communities)['unique'])
bounds = np.linspace(0,nl,nl+1)
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1,1, figsize=(6,6))
scat = ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=communities, alpha=0.9, cmap=cmap, norm=norm)
ax2 = fig.add_axes([0.90, 0.1, 0.03, 0.8])
cb = ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
ax.set_title('NO process labels')
ax2.set_ylabel('clusters', size=8)


nl=len(table(communitiesC)['unique'])
bounds = np.linspace(0,nl,nl+1)
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1,1, figsize=(6,6))
scat = ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=communitiesC, alpha=0.9, cmap=cmap, norm=norm)
ax2 = fig.add_axes([0.90, 0.1, 0.03, 0.8])
cb = ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
ax.set_title('AE labels')
ax2.set_ylabel('clusters', size=8)


nl=len(table(communitiesE)['unique'])
bounds = np.linspace(0,nl,nl+1)
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1,1, figsize=(6,6))
scat = ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=communitiesE, alpha=0.9, cmap=cmap, norm=norm)
ax2 = fig.add_axes([0.90, 0.1, 0.03, 0.8])
cb = ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
ax.set_title('Encoder labels')
ax2.set_ylabel('clusters', size=8)























fig, ax = plt.subplots()
im=ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=communitiesE, cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('populations')
plt.show()

fig, ax = plt.subplots()
im=ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=communitiesC, cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('populations')
plt.show()

fig, ax = plt.subplots()
im=ax.scatter(Y0o[:,0], Y0o[:,1],  marker='.', s=0.2, c=communities, cmap=plt.cm.jet, alpha=0.9, )
fig.colorbar(im, ax=ax, orientation='horizontal', cmap=plt.cm.jet)
plt.title('populations')
plt.show()





print(sklearn.metrics.adjusted_mutual_info_score(communities[np.logical_and.reduce((lbls2!=11, lbls2!=15, lbls2!=9, lbls2!=4))],
                                                             lbls2[np.logical_and.reduce((lbls2!=11, lbls2!=15, lbls2!=9, lbls2!=4))]))

print(sklearn.metrics.adjusted_mutual_info_score(communitiesC[np.logical_and.reduce((lbls2!=11, lbls2!=15, lbls2!=9, lbls2!=4))],
                                                             lbls2[np.logical_and.reduce((lbls2!=11, lbls2!=15, lbls2!=9, lbls2!=4))]))
table(communities[np.logical_and.reduce((lbls!=11, lbls!=15, lbls!=9, lbls!=4))])
table(communitiesC[np.logical_and.reduce((lbls!=11, lbls!=15, lbls!=9, lbls!=4))])

map_pairs, cont = metric.get_map_pairs(communities[lbls!=11], lbls[lbls!=11])
map_pairsC, contC =metric.get_map_pairs(communitiesC[lbls!=11], lbls[lbls!=11])



conm0=sklearn.metrics.confusion_matrix(lbls2[lbls2!=11], lbls2[lbls2!=11])
conm1=sklearn.metrics.confusion_matrix(lbls2[lbls2!=11], communitiesC[lbls2!=11])

plt.matshow(np.log(cont+0.000001))
plt.matshow(np.log(contC+0.000001))
plt.matshow((cont))
plt.matshow((contC))
 iter, iter_iter

import dill                            #pip install dill --user
filename = 'globalsave.pkl'
dill.dump_session(filename)

# and to load the session again:

import sys
sys.path.insert(0, '/mnt/f/Brinkman group/current/Stepan/bin/stsne_pack/')
import stsne
mat=aFrame[np.random.randint(aFrame.shape[0], size=6000), :]
mat2=aFrame[np.random.randint(aFrame.shape[0], size=6000), :]

Y=stsne.run_control(mat, 6000, 10, 1000, no_dims=2, theta=0.5, )


Y2 = stsne.run_existing(mat2, mat, 6000, Y, 10, 1000, 250, no_dims=2, theta=0.5, )
plt.figure(); plt.scatter(Y[:,0], Y[:,1],  marker='.', s=1, color='blue');
plt.scatter(Y2[0][:,0], Y2[0][:,1],  marker='.', s=1, color='red');  plt.show()
plt.scatter(Y2[1][:,0], Y2[1][:,1],  marker='.', s=1, color='red');  plt.show()

nn=6
plt.figure()
plt.plot(np.log(weight_neib_enc[lbls==3,:][nn,:]), c='red')
plt.plot(np.log(weight_neib_enc[lbls==13,:][nn,:]), c='blue')
plt.show()

