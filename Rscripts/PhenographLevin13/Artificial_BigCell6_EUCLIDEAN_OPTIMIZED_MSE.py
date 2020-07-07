# script generates 2 sibdimensional clusters
# of 5 and 10 dims in 30 dimensional space
# within each cluster distribution is uniform
# we then add Gaussian noise in complemetary dimensions
# clusters is used to comapre preformance with
# a) Vanilla DAE
# b) Theoretical formula from paper with two noise sources (second data set with second noise is created)
# c) straitforward analysis with knn neighbours instead second noise


import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
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
import h5py

from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)
def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

lib = ctypes.cdll.LoadLibrary("./Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #Sigma
                ctypes.c_size_t]




# two subspace clusters centers
original_dim = 30
'''
# find neighbours an define clusters consiting solely from perturbed data
def find_neighbors(data, k_, metric='euclidean', cores=12):
    tree = NearestNeighbors(n_neighbors=k_, algorithm="ball_tree", leaf_size=30, metric=metric,  metric_params=None, n_jobs=cores)
    tree.fit(data)
    dist, ind = tree.kneighbors(return_distance=True)
    return {'dist': np.array(dist), 'idx': np.array(ind)}

source_dir = "/media/FCS_local/Stepan/data/"
#file_list = glob.glob(source_dir + '/*.txt')
data0 = np.genfromtxt(source_dir + "ArtCells6_Big.csv", names=None, dtype=float, skip_header=1, delimiter=',')
aFrame = data0[:,1:]

lbls=np.genfromtxt(source_dir + "ArtCells6_Big_labels.csv", names=None, dtype=int, skip_header=1, delimiter=',')[:,1]

len(lbls)
k3 = 90
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=48)
Idx = nb['idx']; Dist = nb['dist']


nrow=Idx.shape[0]
# find nearest neighbours


def singleInput(i):
    nei = noisy_clus[Idx[i, :], :]
    return [nei, i]

# find nearest neighbours
nn=30

rk=range(k3)
def singleInput(i):
     nei =  aFrame[Idx[i,:],:]
     di = [np.sqrt(sum(np.square(aFrame[i] - nei[k_i,]))) for k_i in rk]
     return [nei, di, i]

nrow = len(lbls)
inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = 48
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=48, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
neibALL = np.zeros((nrow, k3, original_dim))
Distances = np.zeros((nrow, k3))
neib_weight = np.zeros((nrow, k3))
Sigma = np.zeros(nrow, dtype=float)
for i in range(nrow):
    neibALL[i,] = results[i][0]
for i in range(nrow):
    Distances[i,] = results[i][1]
#Compute perpelexities
nn=30
perp((Distances[:,0:k3]),       nrow,     original_dim,   neib_weight,          nn,          k3,   Sigma,    48)
      #(     double* dist,      int N,    int D,       double* P,     double perplexity,    int K, int num_threads)
np.shape(neib_weight)
plt.plot(neib_weight[1,])
#sort and normalise weights
topk = np.argsort(neib_weight, axis=1)[:,-nn:]
topk= np.apply_along_axis(np.flip, 1, topk,0)
neib_weight=np.array([ neib_weight[i, topk[i]] for i in range(len(topk))])
neib_weight=sklearn.preprocessing.normalize(neib_weight, axis=1, norm='l1')
neibALL=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])

plt.plot(neib_weight[1,:]);plt.show()

outfile = './data/ArtCell6_Big_Euclid.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)
'''

###########################################
#load preprocessed data
nn=30
outfile = './data/ArtCell6_Big_Euclid.npz'
npzfile = np.load(outfile)
lbls = npzfile['lbls'];
Idx=npzfile['Idx'][:,0:nn];
lbls=npzfile['lbls'];
Dist=npzfile['Dist'][:,0:nn];
neibALL=npzfile['neibALL'][:,0:nn,:];
neib_weight= npzfile['neib_weight']
Sigma = npzfile['Sigma']
aFrame = npzfile['aFrame']


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
config =tf.ConfigProto(device_count = {'GPU': 0, 'CPU':28},
      intra_op_parallelism_threads=50,
      inter_op_parallelism_threads=50,
                       allow_soft_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)
####################################################################
############################### run vanilla DAE ####################
#############################################################################################
#############################################################################################
# c) analysis with knn neighbours instead second noise

model_name = 'Estimated NN_6_Big_Euclid'

print('compute training sources and targets...')

neibALLmean = np.mean(neibALL, axis=1)
from scipy.stats import binom
#split data into train and test
(x_train, mean_neib) = (aFrame, neibALLmean)
#nn_lbls = np.array([item for item in lbls for i in range(nn)]).astype(int)
X_train, X_test,  mean_neib_train, mean_neib_test, lbls_train, lbls_test  \
    = train_test_split(x_train, mean_neib, lbls, test_size=0.1, random_state=42)

# create model and train
k=nn;original_dim=30;intermediate_dim=15; intermediate_dim2 =6; latent_dim = 2
#k_neib = Input(shape = (k, original_dim, ))
#var_dims = Input(shape = (original_dim,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation=act)(x)
#h2 =  Dense(intermediate_dim2, activation=act)(h)
#h.set_weights(ae.layers[1].get_weights())
z = Dense(latent_dim, activation='sigmoid')(h)

# we instantiate these layers separately so as to reuse them later
#decoder_h2 = Dense(intermediate_dim2, activation=act)
decoder_h = Dense(intermediate_dim, activation=act)
decoder_mean = Dense(original_dim, activation='linear')
#h_decoded2 = decoder_h2(z)
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

#take manhatten metric for loss function
Kones = K.ones((nn,original_dim))
def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.abs((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, Kones)
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def ae_loss(x, x_decoded_mean):
    msew = k*original_dim * mean_square_error_NN(x, x_decoded_mean)
    #pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(msew)

ae_nn = Model(x, x_decoded_mean)
ae_nn.summary()

ae_nn.compile(optimizer='adadelta', loss=tf.keras.losses.MeanSquaredError())

history_DAEnn= ae_nn.fit(X_train, mean_neib_train, epochs=2000, batch_size=1024,
                      shuffle=True,
                      validation_data=(X_test, mean_neib_test),
                verbose=2)
plt.plot(history_DAEnn.history['loss'][500:])
plt.plot(history_DAEnn.history['val_loss'][500:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

encoderDAEnn = Model(x, z)
print(encoderDAEnn.summary())
decoderDAEnn = ae_nn
print(decoderDAEnn.summary())

encoded_bulbs = encoderDAEnn.predict(x_train)
decoded_bulbs = decoderDAEnn.predict(x_train)
#encoded_bulbs = encoderDAEnn.predict([X_train, neib_train])
#decoded_bulbs = decoderDAEnn.predict([X_train, neib_train])
fig0 = plt.figure();
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = lbls, s=0.1)
plt.title(model_name)
plt.show()
#figDist = plt.figure();
#plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = ort_test, s=0.1,
#            cmap='viridis')
#plt.colorbar()
#plt.title(model_name)
#plt.show()
#plt.savefig(os.getcwd() + '/plots/' + model_name  +  'ort_dist_bulbs.png')

figSave0 = plt.figure()
plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c=lbls, s = 0.1)
plt.title(model_name)
plt.savefig(os.getcwd() + '/plots/' + model_name  +  '_bulbs.png')
# now plot decoded bulbs co-dimensions in input and output data:
#input:
cols = 6
rows = 2
plt.figure()
gs = plt.GridSpec(rows, cols)
for cl in range(1,7):
    bw=0.2
    i = cl-1
    j = 0
    plt.subplot(gs[j,i])
    plt.title(cl)
    sns.violinplot(data= x_train[lbls==cl , :], bw = bw);
    plt.subplot(gs[j+1, i])
    sns.violinplot(data= decoded_bulbs[lbls==cl , :], bw = bw);
plt.tight_layout(pad=0.1)
plt.show();
import umap
import numba
# see how denoising looks
numba.set_num_threads(48)

reducer = umap.UMAP(n_neighbors=nn)
embeddingTr = reducer.fit_transform(x_train)
fig0 = plt.figure();
plt.scatter(embeddingTr[:,0], embeddingTr[:,1], c = lbls, s=0.1)
plt.title(model_name+'train Bulbs UMAP')
plt.show()

embeddingDec = reducer.fit_transform(decoded_bulbs)
fig0 = plt.figure();
plt.scatter(embeddingDec[:,0], embeddingDec[:,1], c = lbls, s=0.1)
plt.title(model_name+'decoded Bulbs UMAP')
plt.show()

reducer = umap.UMAP(n_neighbors=nn)
embeddingEnc = reducer.fit_transform(encoded_bulbs)
fig0 = plt.figure();
plt.scatter(embeddingEnc[:,0], embeddingEnc[:,1], c = lbls, s=0.1)
plt.title(model_name+'encoded Bulbs UMAP')
plt.show()







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
                verbose=2)
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
figcl0 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==0 , :], bw = bw);
plt.show();
figcl1 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==1 , :], bw = bw);
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


Kones = K.ones((nn,original_dim))
def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, Kones)
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def ae_loss(x, x_decoded_mean):
    msew = k*original_dim * mean_square_error_NN(x, x_decoded_mean)
    #pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(msew)

ae_nnsigma = Model([x, k_neib,  weight_neib], x_decoded_mean)
ae_nnsigma.summary()

ae_nnsigma.compile(optimizer='adadelta', loss=ae_loss, metrics=[mean_square_error_NN])
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./models/model_knn_ae{epoch:08d}.h5', period=2)
history_nnsigma= ae_nnsigma.fit([X_train, neib_train, weight_train], X_train, epochs=3000, batch_size=256,
                      shuffle=True,
                      validation_data=([X_test, neib_test, weight_test], X_test),
                verbose=2)#, callbacks=[checkpoint])
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

#figDist = plt.figure();
#plt.scatter(encoded_bulbs[:,0], encoded_bulbs[:,1], c = ort_test, s=0.1,
#            cmap='viridis')#
#plt.colorbar()
#plt.title(model_name)
#plt.show()
#plt.savefig(os.getcwd() + '/plots/' + model_name  +  'ort_dist_bulbs.png')


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
figcl0 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==0 , :], bw = bw);
plt.show();
figcl1 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==1 , :], bw = bw);
plt.show();

###########################
# visualize training process
# load models exrtract encoder and decoder, visualise..
#model_number  = range(100, 3100, 100)
model_number  = [100, 200, 300, 500, 1000, 3000]
from tensorflow.keras import models
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


Kones = K.ones((original_dim, nn+1))
def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, Kones)
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
                verbose=2)
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
figcl0 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==0 , :], bw = bw);
plt.show();
figcl1 = plt.figure();
bx = sns.violinplot(data= decoded_bulbs[lbls_test==1 , :], bw = bw);
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
    return  weightedN * 0.5


def compute_kernel(x, y):
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

def custom_loss(train_z, train_xr, train_x):
    """
    Critical loss builder
		:param train_z: latent code
		:param train_xr: reconstructed data
		:param train_x: training data, the input data
		:return: new loss
		"""
    'So, we first get the mmd loss'
    'First, sample from random noise'
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    'calculate mmd loss'
    loss_mmd = compute_mmd(true_samples, z_mean)

    'Then, also get the reconstructed loss'
    #loss_nll = K.mean(K.square(train_xr - train_x))

    'Add them together, then you can get the final loss'
    loss =loss_mmd
    return loss


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
                verbose=2)
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
