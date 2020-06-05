# script generate 2 sibdimensional clusters
# of 5 and 10 dims in 30 dimensional space
# within each cluster distribution is uniform
# we then add Gaussian noise in complemetary dimensions
# clusters is used to comapre preformance with
# a) Vanilla DAE
# b) Theoretical formula from paper with two noise sources (second data set with second noise is created)
# c) straitforward analysis with knn neighbours instead second noise
import tensorflow as tf

print(tf.__version__)
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras.datasets import mnist
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

from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)
def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

lib = ctypes.cdll.LoadLibrary("/home/grines02/PycharmProjects/BIOIBFO25L/Rscripts/bin/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,ctypes.c_size_t]




# two subspace clusters centers
original_dim = 30
cl1_center = np.zeros(original_dim)
cl2_center = np.concatenate((np.ones(20),  np.zeros(10)), axis=0 )
ncl1 =ncl2=100000
'''
cl1_center = np.zeros(original_dim)
cl2_center = np.concatenate((np.ones(20),  np.zeros(10)), axis=0 )
ncl1 =ncl2=100000
lbls = np.concatenate((np.zeros(ncl1), np.ones(ncl2)), axis=0)

cl1 = cl1_center +  np.concatenate([np.zeros((ncl1,20)),  np.random.uniform(low=-1, high=1, size=(ncl1,10))], axis=1 )
cl2 = cl2_center +  np.concatenate([np.random.uniform(low=-1, high=1, size=(ncl2,10)),  np.zeros((ncl2,20))], axis=1 )
sns.violinplot(data= cl1, bw = 0.1);sns.violinplot(data= cl2, bw = 0.1);
#noisy or not:
noise_sig1 =  np.concatenate((np.zeros(20),  np.ones(10)), axis=0 )
noise_sig2 = np.concatenate((np.ones(10), np.zeros(20)), axis=0 )
noise_scale =1
# add noise to orthogonal dimensions
cl1_noisy = cl1 + np.concatenate([np.random.normal(loc=0, scale = noise_scale, size=(ncl1,20)), np.zeros((ncl1,10))], axis=1 )
cl2_noisy = cl2 + np.concatenate([ np.zeros((ncl2, 10)), np.random.normal(loc=0, scale = noise_scale, size=(ncl2,20))], axis=1 )
sns.violinplot(data= cl1_noisy, bw = 0.1);sns.violinplot(data= cl2_noisy, bw = 0.1);

# create noisy neighbours, 30 per each initial point,  neighbours live in parallel dims
# and add orthogonal noise
def Perturbation(i, cl, noise_sig, nn):
    ncol = np.shape(cl)[1]
    nrow = np.shape(cl)[0]
    sample= cl[i,:] + (noise_sig-1)* np.random.normal(loc=0, scale = noise_scale, size=(nn,30)) +\
            noise_sig * np.random.normal(loc=0, scale = noise_scale, size=(nn,30))
    return sample
nn=30
resSample1 = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(Perturbation,
                check_pickle=False)(i, cl1, noise_sig1, nn) for i in range(np.shape(cl1_noisy)[0]))
#cl1_noisy_nn = np.vstack(resSample1)
cl1_noisy_nn = np.zeros((ncl1, nn, 30))
for i in range(ncl1):
    cl1_noisy_nn[i,:,:] = resSample1[i]

resSample2 = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(Perturbation,
                check_pickle=False)(i, cl2, noise_sig2, nn) for i in range(np.shape(cl2_noisy)[0]))
cl2_noisy_nn = np.zeros((ncl2, nn, 30))
for i in range(ncl2):
    cl2_noisy_nn[i,:,:] = resSample2[i]
del resSample1, resSample2

# find neighbours an define clusters consiting solely from perturbed data
def find_neighbors(data, k_, metric='euclidean', cores=12):
    tree = NearestNeighbors(n_neighbors=k_, algorithm="ball_tree", leaf_size=30, metric=metric,  metric_params=None, n_jobs=cores)
    tree.fit(data)
    dist, ind = tree.kneighbors(return_distance=True)
    return {'dist': np.array(dist), 'idx': np.array(ind)}

noisy_clus = np.concatenate(np.vstack((cl1_noisy_nn[:, np.random.choice(cl1_noisy_nn.shape[1], 1, replace=False), :],
                            cl2_noisy_nn[:, np.random.choice(cl2_noisy_nn.shape[1], 1, replace=False), :] )), axis=0)
# visualise with umap

import umap
standard_embedding = umap.UMAP(n_neighbors=30, n_components=2).fit_transform(noisy_clus)
x = standard_embedding[:, 0]
y = standard_embedding[:, 1]
# analog of tsne plot fig15 from Nowizka 2015, also see fig21
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter3d, Figure, Layout, Scatter
plot([Scatter(x=x, y=y,
                mode='markers',
                marker=dict(
                    size=1,
                    color=lbls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.5,
                ),
                text=lbls,
                hoverinfo='text')])

# find orthogonal distances
# zero's in noise_sig are noisy dims
def ort_dist(i, noise_sig, cl, cl_center):
    dist= np.sum(np.square((cl[i,:] - cl_center) * (noise_sig-1)))
    return dist
resSample1 = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(ort_dist,
                check_pickle=False)(i, noise_sig1, noisy_clus[lbls==0,:], cl1_center)
                                                                 for i in range(np.shape(noisy_clus[lbls==0,:])[0]))
cl1_ort_dist = np.array(resSample1)
resSample2 = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(ort_dist,
                check_pickle=False)(i, noise_sig2, noisy_clus[lbls==1,:], cl2_center)
                                                                 for i in range(np.shape(noisy_clus[lbls==1,:])[0]))
cl2_ort_dist = np.array(resSample2)


nrow=noisy_clus.shape[0]
# find nearest neighbours
nb=find_neighbors(noisy_clus, nn, metric='manhattan', cores=12)
Idx = nb['idx']; Dist = nb['dist']


def singleInput(i):
    nei = noisy_clus[Idx[i, :], :]
    return [nei, i]


nrow=noisy_clus.shape[0]
# find nearest neighbours
nn=30

rk=range(nn)
def singleInput(i):
     nei =  noisy_clus[Idx[i,:],:]
     di = [np.sqrt(sum(np.square(noisy_clus[i] - nei[k_i,]))) for k_i in rk]
     return [nei, di, i]

inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
neibALL = np.zeros((nrow, nn, original_dim))
Distances = np.zeros((nrow, nn))
neib_weight = np.zeros((nrow, nn))
for i in range(nrow):
    neibALL[i,] = results[i][0]
for i in range(nrow):
    Distances[i,] = results[i][1]
#Compute perpelexities
perp((Distances[:,0:nn]),       nrow,     original_dim,   neib_weight,          nn,          nn,        12)
      #(     double* dist,      int N,    int D,       double* P,     double perplexity,    int K, int num_threads)
np.shape(neib_weight)
plt.plot(neib_weight[10,])
#sort and normalise weights
topk = np.argsort(neib_weight, axis=1)[:,-nn:]
topk= np.apply_along_axis(np.flip, 1, topk,0)
neib_weight=np.array([ neib_weight[i, topk[i]] for i in range(len(topk))])
neib_weight=sklearn.preprocessing.normalize(neib_weight, axis=1, norm='l1')
neibALL=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])

plt.plot(neib_weight[5,]);plt.show()

outfile = './data/ArtBulbz.npz'
np.savez(outfile, Idx=Idx, cl1=cl1, cl2=cl2, noisy_clus=noisy_clus,
         lbls=lbls,  Dist=Dist, cl1_noisy_nn =cl1_noisy_nn,
         cl2_noisy_nn=cl2_noisy_nn, cl1_noisy =cl1_noisy,
         cl2_noisy=cl2_noisy,cl1_ort_dist=cl2_ort_dist, cl2_ort_dist=cl2_ort_dist,
         neibALL=neibALL, neib_weight= neib_weight)

import dill                            
filepath = 'session_ArtdataGeneration.pkl'
dill.dump_session(filepath) # Save the session
dill.load_session(filepath)

'''

outfile = './data/ArtBulbz.npz'
npzfile = np.load(outfile)
lbls = npzfile['lbls'];
Idx=npzfile['Idx']; cl1=npzfile['cl1']; cl2=npzfile['cl2']; noisy_clus=npzfile['noisy_clus'];
lbls=npzfile['lbls'];  Dist=npzfile['Dist']; cl1_noisy_nn =npzfile['cl1_noisy_nn'];
cl2_noisy_nn=npzfile['cl2_noisy_nn'];
cl1_noisy =npzfile['cl1_noisy'];
cl2_noisy=npzfile['cl2_noisy'];
cl1_ort_dist=npzfile['cl2_ort_dist'];
cl2_ort_dist=npzfile['cl2_ort_dist'];
neibALL=npzfile['neibALL']
neib_weight= npzfile['neib_weight']

'''
# scale data
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))

nrow = cl1_noisy.shape[0] + cl2_noisy.shape[0]
for i in range(nrow):
    neibALL[i,] =(neibALL[i,]+5)/5
for i in range(cl1_noisy.shape[0]):
    cl2_noisy_nn[i,] = (cl2_noisy_nn[i,] + 5)/5
    cl1_noisy_nn[i,] = (cl1_noisy_nn[i,] + 5)/5
cl1 = (cl1+5)/5; cl2 = (cl2+5)/5;
cl1_noisy = (cl1_noisy+5)/5; cl2_noisy=(cl2_noisy+5)/5;
noisy_clus = (noisy_clus+5)/5;
'''
#sns.violinplot(data= cl2_noisy, bw = 0.1);
# some globals
act= 'selu'
####################################################################
############################### run vanilla DAE ####################

nn=30
# (a) Deep denoising Autoencoder
model_name = 'DAE'
# this is the size of our encoded representations
encoding_dim = 2

# this is our input placeholder
input = Input(shape=(30, ))
# "encoded" is the encoded representation of the inputs
encoded = Dense(10, activation=act)(input)
encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(10, activation=act)(encoded)
decoded = Dense(30, activation='linear')(decoded)
# this model maps an input to its reconstruction
autoencoderDAE = Model(input, decoded)
# Separate Encoder model
# this model maps an input to its encoded representation
encoderDAE = Model(input, encoded)
# Separate Decoder model
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim, ))
# retrieve the layers of the autoencoder model
decoder_layer1 = autoencoderDAE.layers[-2]
decoder_layer2 = autoencoderDAE.layers[-1]
# create the decoder model
decoderDAE = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))
# Train to reconstruct MNIST digits
# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoderDAE.compile(optimizer='adadelta', loss='mean_squared_error')
# prepare input data from noisy output and clean input
cl2_noisy= cl2_noisy; cl1_noisy= cl1_noisy;cl1= cl1;cl2= cl2
(x_train, y_train) = (np.vstack((cl1_noisy, cl2_noisy)), np.vstack((cl1, cl2)) )
X_train, X_test,  Y_train, Y_test, lbls_train, lbls_test = train_test_split(x_train, y_train, lbls, test_size=0.33, random_state=42)
print(x_train.shape)
print(y_train.shape)
# Train autoencoder
my_epochs = 500
history_DAE= autoencoderDAE.fit(X_train, Y_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(X_test, Y_test),
                verbose=2, workers =  multiprocessing.cpu_count(), use_multiprocessing  = True )
fig0 = plt.figure();
plt.plot(history_DAE.history['loss'][100:])
plt.plot(history_DAE.history['val_loss'][100:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Visualize the reconstructed encoded representations

# encode and decode some digits
# note that we take them from the *test* set
encoded_bulbs = encoderDAE.predict(X_test)
decoded_bulbs = decoderDAE.predict(encoded_bulbs)
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.1)
plt.title(model_name)
plt.show();
figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.1)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')

# now plot decoded bulbs co-dimensions in input and output data:
#input:
cl=0;bw=0.02
fig = plt.figure()
sns.violinplot(data= Y_test[lbls_test==cl , :], bw = bw);
plt.savefig(os.getcwd() + '/plots/' + 'cluster1_nonoise.png')
fig = plt.figure()
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.savefig(os.getcwd() + '/plots/' + 'cluster1_noise.png')
cl=1;bw=0.02
fig = plt.figure()
sns.violinplot(data= Y_test[lbls_test==cl , :], bw = bw);
plt.savefig(os.getcwd() + '/plots/' + 'cluster2_nonoise.png')
fig = plt.figure()
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.savefig(os.getcwd() + '/plots/' + 'cluster2_noise.png')


#output:
fig2 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==cl , :], bw = bw);
plt.show();

##############################################################################################################
#########################################################################################################
# b) Theoretical formula from paper with two noise sources (second data set with second noise is created)
model_name  = 'True NN'

# add noise to orthogonal dimensions
#form neighbour data for y_train of autoencoder

k=nn;intermediate_dim=15; latent_dim = 2
k_neib = Input(shape = (k, original_dim, ))
#var_dims = Input(shape = (original_dim,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation=act)(x)
#h.set_weights(ae.layers[1].get_weights())
z = Dense(latent_dim, activation='sigmoid')(h)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation=act)
decoder_mean = Dense(original_dim, activation='linear')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, K.ones((nn,original_dim)))
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def ae_loss(x, x_decoded_mean):
    msew = k*original_dim * mean_square_error_NN(x, x_decoded_mean)
    #pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(msew)

aeNNtrue = Model([x, k_neib], x_decoded_mean)
aeNNtrue.summary()

aeNNtrue.compile(optimizer='adadelta', loss=ae_loss, metrics=[mean_square_error_NN])
#split data into train and test
(x_train, neib) = (np.vstack((cl1_noisy, cl2_noisy)), np.vstack((cl1_noisy_nn, cl2_noisy_nn)) )
X_train, X_test,  neib_train, neib_test, lbls_train, lbls_test = train_test_split(x_train, neib, lbls, test_size=0.33, random_state=42)

history_DAEnnTrue= aeNNtrue.fit([X_train, neib_train], X_train, epochs=800, batch_size=256,
                      shuffle=True,
                      validation_data=([X_test, neib_test], X_test),
                verbose=2, workers =  multiprocessing.cpu_count(), use_multiprocessing  = True )
plt.plot(history_DAEnnTrue.history['loss'][600:])
plt.plot(history_DAEnnTrue.history['val_loss'][600:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

encoderNNtrue = Model([x, k_neib], z)
print(encoderNNtrue.summary())
decoderNNtrue = aeNNtrue
print(decoderNNtrue.summary())

encoded_bulbs = encoderNNtrue.predict([X_test, neib_test])
decoded_bulbs = decoderNNtrue.predict([X_test, neib_test])
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = lbls_test, s = 0.5)
plt.title(model_name)
plt.show()
figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.1)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')

# now plot decoded bulbs co-dimensions in input and output data:
#input:
cl=1;bw=0.2
fig1 = plt.figure();
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.show();
#output:
fig2 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==cl , :], bw = bw);
plt.show();
#############################################################################################
#############################################################################################
# c) analysis with knn neighbours instead second noise
model_name = 'Estimated NN'

print('compute training sources and targets...')

from scipy.stats import binom
#split data into train and test
(x_train, neib) = (noisy_clus, neibALL)
#nn_lbls = np.array([item for item in lbls for i in range(nn)]).astype(int)
X_train, X_test,  neib_train, neib_test, lbls_train, lbls_test, ort_train, ort_test \
    = train_test_split(x_train, neib, lbls, np.concatenate((cl1_ort_dist,cl2_ort_dist)), test_size=0.33, random_state=42)

# create model and train
k=nn;original_dim=30;intermediate_dim=15; intermediate_dim2 =6; latent_dim = 2
k_neib = Input(shape = (k, original_dim, ))
#var_dims = Input(shape = (original_dim,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation=act)(x)
h2 =  Dense(intermediate_dim2, activation=act)(h)
#h.set_weights(ae.layers[1].get_weights())
z = Dense(latent_dim, activation='sigmoid')(h2)

# we instantiate these layers separately so as to reuse them later
decoder_h2 = Dense(intermediate_dim2, activation=act)
decoder_h = Dense(intermediate_dim, activation=act)
decoder_mean = Dense(original_dim, activation='linear')
h_decoded2 = decoder_h2(z)
h_decoded = decoder_h(h_decoded2)
x_decoded_mean = decoder_mean(h_decoded)


def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, K.ones((nn,original_dim)))
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def ae_loss(x, x_decoded_mean):
    msew = k*original_dim * mean_square_error_NN(x, x_decoded_mean)
    #pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(msew)

ae_nn = Model([x, k_neib], x_decoded_mean)
ae_nn.summary()

ae_nn.compile(optimizer='adadelta', loss=ae_loss, metrics=[mean_square_error_NN])

history_DAEnn= ae_nn.fit([X_train, neib_train], X_train, epochs=1000, batch_size=256,
                      shuffle=True,
                      validation_data=([X_test, neib_test], X_test),
                verbose=2, workers =  multiprocessing.cpu_count(), use_multiprocessing  = True )
plt.plot(history_DAEnn.history['loss'][00:])
plt.plot(history_DAEnn.history['val_loss'][00:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

encoderDAEnn = Model([x, k_neib], z)
print(encoderDAEnn.summary())
decoderDAEnn = ae_nn
print(decoderDAEnn.summary())

encoded_bulbs = encoderDAEnn.predict([X_test, neib_test])
decoded_bulbs = decoderDAEnn.predict([X_test, neib_test])
#encoded_bulbs = encoderDAEnn.predict([X_train, neib_train])
#decoded_bulbs = decoderDAEnn.predict([X_train, neib_train])
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = lbls_test, s=0.1)
plt.title(model_name)
plt.show()
figDist = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = ort_test, s=0.1,
            cmap='viridis')
plt.colorbar()
plt.title(model_name)
plt.show()
plt.savefig(os.getcwd() + '/plots/' + model_name  +  'ort_dist_bulbs.png')

figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.1)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')
# now plot decoded bulbs co-dimensions in input and output data:
#input:
cl=1;bw=0.2
fig1 = plt.figure();
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.show();
#output:
fig2 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==cl , :], bw = bw);
plt.show();
#############################################################################################
#########################################################################
# d) Compare with vanilla autoencoder without nn on noisy data
model_name = 'Autoencoder'
encoding_dim = 2

# this is our input placeholder
input = Input(shape=(30, ))
# "encoded" is the encoded representation of the inputs
encoded = Dense(10, activation=act)(input)
encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(10, activation=act)(encoded)
decoded = Dense(30, activation='linear')(decoded)
# this model maps an input to its reconstruction
autoencoder = Model(input, decoded)
# Separate Encoder model
# this model maps an input to its encoded representation
encoder = Model(input, encoded)
# Separate Decoder model
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim, ))
# retrieve the layers of the autoencoder model
decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))
# Train to reconstruct MNIST digits
# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Train autoencoder
my_epochs = 1000
history_AE= autoencoder.fit(X_train, X_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(X_test, X_test),
                verbose=2, workers =  multiprocessing.cpu_count(), use_multiprocessing  = True )
fig0 = plt.figure();
plt.plot(history_AE.history['loss'][100:])
plt.plot(history_AE.history['val_loss'][100:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Visualize the reconstructed encoded representations

# encode and decode some digits
# note that we take them from the *test* set
encoded_bulbs = encoder.predict(X_test)
decoded_bulbs = decoder.predict(encoded_bulbs)
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = lbls_test, s = 0.1)
plt.title(model_name)
plt.show()
figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.1)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')
# now plot decoded bulbs co-dimensions in input and output data:
#input:
figDist = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = ort_test, s=0.5,
            cmap='viridis')
plt.colorbar()
plt.title(model_name)
plt.show()
plt.savefig(os.getcwd() + '/plots/' + model_name  +  'ort_dist_bulbs.png')


fig1 = plt.figure();
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.show();
#output:
fig2 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==cl , :], bw = bw);
plt.show();

#############################################################################################
#############################################################################################
model_name = 'Sigma-weighted NN'
# e) analysis with knn neighbours
# and sigma-weighted from sklearn.neighbors import NearestNeighbors


noisy_clus = np.concatenate(np.vstack((cl1_noisy_nn[:, np.random.choice(cl1_noisy_nn.shape[1], 1, replace=False), :],
                            cl2_noisy_nn[:, np.random.choice(cl2_noisy_nn.shape[1], 1, replace=False), :] )), axis=0)
#noisy_clus = np.concatenate(np.vstack((cl1_noisy_nn,
#                           cl2_noisy_nn )), axis=0)

#split data into train and test
(x_train, neib) = (noisy_clus, neibALL)
#nn_lbls = np.array([item for item in lbls for i in range(nn)]).astype(int)
X_train, X_test,  neib_train, neib_test, lbls_train, lbls_test, weight_train, weight_test =\
    train_test_split(x_train, neib, lbls, neib_weight, test_size=0.33, random_state=42)

# create model and train
k=nn;original_dim=30;intermediate_dim=15; latent_dim = 2
weight_neib = Input(shape = (k,))
k_neib = Input(shape = (k, original_dim, ))
#var_dims = Input(shape = (original_dim,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation=act)(x)
#h.set_weights(ae.layers[1].get_weights())
z = Dense(latent_dim, activation='sigmoid')(h)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation=act)
decoder_mean = Dense(original_dim, activation='linear')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, K.transpose(weight_neib))
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def ae_loss(x, x_decoded_mean):
    msew = k*original_dim * mean_square_error_NN(x, x_decoded_mean)
    #pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(msew)

ae_nnsigma = Model([x, k_neib,  weight_neib], x_decoded_mean)
ae_nnsigma.summary()

ae_nnsigma.compile(optimizer='adadelta', loss=ae_loss, metrics=[mean_square_error_NN])
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./models/model_knn_ae{epoch:08d}.h5', period=100)
history_nnsigma= ae_nnsigma.fit([X_train, neib_train, weight_train], X_train, epochs=50, batch_size=256,
                      shuffle=True,
                      validation_data=([X_test, neib_test, weight_test], X_test),
                verbose=2, workers =  multiprocessing.cpu_count(), use_multiprocessing  = True,
                         callbacks=[checkpoint])
plt.plot(history_nnsigma.history['loss'][00:])
plt.plot(history_nnsigma.history['val_loss'][00:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

encoder_nnsigma = Model([x, k_neib, weight_neib], z)
print(encoder_nnsigma.summary())
decoder_nnsigma = ae_nnsigma
print(decoder_nnsigma.summary())

encoded_bulbs = encoder_nnsigma.predict([X_test, neib_test, weight_test])
decoded_bulbs = decoder_nnsigma.predict([X_test, neib_test, weight_test])
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = lbls_test, s = 0.1)
plt.title(model_name)
plt.show()

figDist = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = ort_test, s=0.1,
            cmap='viridis')
plt.colorbar()
plt.title(model_name)
plt.show()
plt.savefig(os.getcwd() + '/plots/' + model_name  +  'ort_dist_bulbs.png')


figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.1)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')
# now plot decoded bulbs co-dimensions in input and output data:
#input:
cl=1;bw=0.2
fig1 = plt.figure();
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.show();
#output:
fig2 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==cl , :], bw = bw);
plt.show();
###########################
# visualize training process
# load models exrtract encoder and decoder, visualise..
#model_number  = range(100, 3100, 100)
model_number  = [100, 200, 300, 500, 1000, 3000]
from keras import models
for i in model_number:
    model_i = './models/model_knn_ae' + str(i).zfill(8) + '.h5'
    ae_i= models.load_model(model_i,  custom_objects={'ae_loss':ae_loss,
                                                      'mean_square_error_NN':mean_square_error_NN})
    ae_i.summary()
    enc = ae_i.layers[1](x)
    enc = ae_i.layers[2](enc)
    # create the decoder model
    enc_i = Model(x, enc)
    enc_i.summary()

    encoded_bulbs = enc_i.predict(X_test)
    decoded_bulbs = ae_i.predict(X_test)
    fig0 = plt.figure();
    plt.scatter(encoded_bulbs[:, 0], encoded_bulbs[:, 1], c=lbls_test, s = 0.5)
    plt.show()

    cl = 1;
    bw = 0.2
    fig1 = plt.figure();
    ax = sns.violinplot(data=X_test[lbls_test == cl, :], bw=bw);
    plt.show();
    # output:
    fig2 = plt.figure();
    bx = sns.violinplot(data=decoded_bulbs[lbls_test == cl, :], bw=bw);
    plt.show();


##################################################################
# int itself in the target function
##
# f) analysis with knn neighbours instead second noise
model_name = 'Estimated NN+'

nrow=noisy_clus.shape[0]
############# add x itself
#############################

neibALLPlus = np.zeros((nrow, nn+1, original_dim))
for i in range(nrow):
 neibALLPlus[i,] = np.row_stack((neibALL[i,], x_train[i,]))
#split data into train and test
(x_train, neibP) = (noisy_clus, neibALLPlus)
#nn_lbls = np.array([item for item in lbls for i in range(nn)]).astype(int)
X_train, X_test,  neib_trainP, neib_testP, lbls_train, lbls_test = train_test_split(x_train, neibP, lbls, test_size=0.33, random_state=42)

# create model and train
k=nn+1
original_dim=30;intermediate_dim=15; latent_dim = 2
k_neibP = Input(shape = (k, original_dim, ))
#var_dims = Input(shape = (original_dim,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation=act)(x)
#h.set_weights(ae.layers[1].get_weights())
z = Dense(latent_dim, activation='sigmoid')(h)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation=act)
decoder_mean = Dense(original_dim, activation='linear')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neibP - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, K.ones((k,original_dim)))
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def ae_loss(x, x_decoded_mean):
    msew = k*original_dim * mean_square_error_NN(x, x_decoded_mean)
    #pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(msew)

ae_nnP = Model([x, k_neibP], x_decoded_mean)
ae_nnP.summary()

ae_nnP.compile(optimizer='adadelta', loss=ae_loss, metrics=[mean_square_error_NN])

history_DAEnnP= ae_nnP.fit([X_train, neib_trainP], X_train, epochs=500, batch_size=256,
                      shuffle=True,
                      validation_data=([X_test, neib_testP], X_test),
                verbose=2, workers =  multiprocessing.cpu_count(), use_multiprocessing  = True )
plt.plot(history_DAEnnP.history['loss'][00:])
plt.plot(history_DAEnnP.history['val_loss'][00:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

encoderDAEnnP = Model([x, k_neibP], z)
print(encoderDAEnnP.summary())
decoderDAEnnP = ae_nnP
print(decoderDAEnnP.summary())

encoded_bulbs = encoderDAEnnP.predict([X_test, neib_testP])
decoded_bulbs = decoderDAEnnP.predict([X_test, neib_testP])
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = lbls_test, s=0.5)
plt.title(model_name)
plt.show()
figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.5)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')
figDist = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = ort_test, s=0.5,
            cmap='viridis')
plt.colorbar()
plt.title(model_name)
plt.show()
plt.savefig(os.getcwd() + '/plots/' + model_name  +  'ort_dist_bulbs.png')
# now plot decoded bulbs co-dimensions in input and output data:
#input:
cl=1;bw=0.2
fig1 = plt.figure();
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.show();
#output:
fig2 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==cl , :], bw = bw);
plt.show();

###########################################################
# g) add regularization by VAE
model_name = 'Estimated NN_MMD'
#split data into train and test
(x_train, neib) = (noisy_clus, neibALL)
#nn_lbls = np.array([item for item in lbls for i in range(nn)]).astype(int)
X_train, X_test,  neib_train, neib_test, lbls_train, lbls_test, ort_train, ort_test \
    = train_test_split(x_train, neib, lbls, np.concatenate((cl1_ort_dist,cl2_ort_dist)), test_size=0.33, random_state=42)

# create model and train
k=nn;original_dim=30;intermediate_dim=15; latent_dim = 2
k_neib = Input(shape = (k, original_dim, ))
#var_dims = Input(shape = (original_dim,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation=act)(x)
#h.set_weights(ae.layers[1].get_weights())

z_mean =  Dense(latent_dim, activation='sigmoid', name='z_mean')(h)
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation=act)
decoder_mean = Dense(original_dim, activation='linear')
h_decoded = decoder_h(z_mean)
x_decoded_mean = decoder_mean(h_decoded)

# adding MMD

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


def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, K.ones((nn,original_dim)))
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def custom_loss(x, x_decoded_mean, z_mean):
    msew = mean_square_error_NN(x, x_decoded_mean)
    #print('msew done', K.eval(msew))
    #mmd loss
    #loss_nll = K.mean(K.square(train_xr - x))
    #batch_size = batch_size #K.shape(train_z)[0]
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    #print('batch_size')
    #latent_dim = latent_dim
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    loss_mmd = compute_mmd(true_samples, z_mean)
    #print(K.shape(loss_mmd))
    #return msew +  1 * contractive()
    return msew + loss_mmd

ae_nnMMD = Model(inputs=[x, k_neib], outputs=x_decoded_mean)
ae_nnMMD.summary()
loss = custom_loss(x, x_decoded_mean, z_mean)
ae_nnMMD.compile(optimizer='adam', loss=loss, metrics=[mean_square_error_NN,  custom_loss])


history_DAEnnMMD= ae_nnMMD.fit([X_train, neib_train], X_train, epochs=5,
                      shuffle=True,
                      validation_data=([X_test, neib_test], X_test),
                verbose=2, workers =  multiprocessing.cpu_count(), use_multiprocessing  = True )
plt.plot(history_DAEnnMMD.history['loss'][00:])
plt.plot(history_DAEnnMMD.history['val_loss'][00:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

encoderDAEnnMMD = Model([x, k_neib], z)
print(encoderDAEnnMMD.summary())
decoderDAEnnMMD = ae_nnMMD
print(decoderDAEnnMMD.summary())

encoded_bulbs = encoderDAEnnMMD.predict([X_test, neib_test])
decoded_bulbs = decoderDAEnnMMD.predict([X_test, neib_test])
#encoded_bulbs = encoderDAEnn.predict([X_train, neib_train])
#decoded_bulbs = decoderDAEnn.predict([X_train, neib_train])
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = lbls_test, s=0.1)
plt.title(model_name)
plt.show()
figDist = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = ort_test, s=0.1,
            cmap='viridis')
plt.colorbar()
plt.title(model_name)
plt.show()
plt.savefig(os.getcwd() + '/plots/' + model_name  +  'ort_dist_bulbs.png')

figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls_test, s = 0.5)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')
# now plot decoded bulbs co-dimensions in input and output data:
#input:
cl=1;bw=0.2
fig1 = plt.figure();
ax = sns.violinplot(data= X_test[lbls_test==cl , :], bw = bw);
plt.show();
#output:
fig2 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==cl , :], bw = bw);
plt.show();
