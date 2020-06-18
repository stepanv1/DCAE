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
cl1_center = np.zeros(original_dim)
cl2_center = np.concatenate((np.ones(20),  np.zeros(10)), axis=0 )
ncl1 =ncl2=100000

outfile = './data/ArtBulbz.npz'
npzfile = np.load(outfile)
lbls = npzfile['lbls'];
Idx=npzfile['Idx']; cl1=npzfile['cl1']; cl2=npzfile['cl2']; noisy_clus=npzfile['noisy_clus'];
lbls=npzfile['lbls'];  Dist=npzfile['Dist'];
cl1_noisy_nn =npzfile['cl1_noisy_nn'];
cl2_noisy_nn=npzfile['cl2_noisy_nn'];
cl1_noisy =npzfile['cl1_noisy'];
cl2_noisy=npzfile['cl2_noisy'];
cl1_ort_dist=npzfile['cl2_ort_dist'];
cl2_ort_dist=npzfile['cl2_ort_dist'];
neibALL=npzfile['neibALL']
neib_weight= npzfile['neib_weight']

#sns.violinplot(data= cl2_noisy, bw = 0.1);
# some globals
act= 'selu'
config =tf.ConfigProto(device_count = {'GPU': 0, 'CPU':28},
      intra_op_parallelism_threads=50,
      inter_op_parallelism_threads=50,
                       allow_soft_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)

nn=30
###########################################################
# g) add regularization by VAE
model_name = 'Estimated NN_MMD'
#split data into train and test
(x_train, neib) = (noisy_clus, neibALL)
#nn_lbls = np.array([item for item in lbls for i in range(nn)]).astype(int)
X_train, X_test,  neib_train, neib_test, lbls_train, lbls_test, ort_train, ort_test,  weight_neib_train, weight_neib_test \
    = train_test_split(x_train, neib, lbls, np.concatenate((cl1_ort_dist,cl2_ort_dist)), neib_weight, test_size=0.33, random_state=42)

# create model and train
k=nn;original_dim=30;intermediate_dim=15; latent_dim = 2
k_neib = Input(shape = (k, original_dim, ))
#var_dims = Input(shape = (original_dim,))
#weight_neib = Input(shape=(k,))
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
#analytical estimate of MMD from https://arxiv.org/pdf/1901.03227.pdf
# When CodeNorm is not used , set adaptive = True
def smmd(z_mean, scale =1./8., adaptive = False):
    nf = tf.cast( tf.shape(z_mean)[0], "float32")
    latent_dim = tf.cast( tf.shape(z_mean)[1] , "float32")

    norms2 = tf.reduce_sum(tf.square( z_mean ), axis =1, keepdims = True )
    dotprods = tf.matmul(z_mean , z_mean , transpose_b = True )
    dists2 = norms2 + tf.transpose( norms2 ) - 2. * dotprods
    if adaptive :
        mean_norms2 = tf.reduce_mean( norms2 )
        gamma2 = tf.stop_gradient( scale * mean_norms2 )
    else :
        gamma2 = scale * latent_dim

    variance = ( gamma2 /(2.+ gamma2 ) )**latent_dim + \
                ( gamma2 /(4.+ gamma2 ) )**( latent_dim /2.) - \
                2.*( gamma2**2./((1.+ gamma2 )*(3.+ gamma2 ) ) )**( latent_dim /2.)
    variance = 2. * variance /( nf *( nf -1.) )
    variance_normalization = ( variance )**( -1./2.)

    Ekzz = ( tf.reduce_sum( tf.exp( - dists2 /(2.* gamma2 ) ) ) - nf ) /(( nf * nf - nf ) )
    Ekzn = ( gamma2 /(1.+ gamma2 ) ) **( latent_dim /2.) *\
        tf.reduce_mean ( tf.exp( - norms2 /(2.*(1.+ gamma2 ) ) ) )
    Eknn = ( gamma2 /(2.+ gamma2 ) ) **( latent_dim /2.)

    return variance_normalization *( Ekzz - 2.* Ekzn + Eknn )
'''
def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1))), axis=-1)
    weightedN = k * original_dim * K.dot(dst,
                                         K.transpose(weight_neib))  # not really a mean square error after we done this
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    #return  tf.multiply(weightedN, 0.5 * normSigma * (1/SigmaTsq) )
    return tf.multiply(weightedN, 0.5)
'''
Kones = K.ones((nn,original_dim))
def mean_square_error_NN(y_true, y_pred):
    dst = K.mean(K.square((k_neib - K.expand_dims(y_pred, 1))) , axis=-1)
    weightedN = K.dot(dst, Kones)
    #return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(weightedN, 0.5 )

def custom_loss(x, x_decoded_mean):
    msew = mean_square_error_NN(x, x_decoded_mean)
    loss_mmd = smmd(z_mean, scale = 1./8. , adaptive = False )
    #print(K.shape(loss_mmd))
    #return msew +  1 * contractive()
    return 0.001*loss_mmd  + msew


ae_nnMMD = Model(inputs=[x, k_neib], outputs=x_decoded_mean)
ae_nnMMD.summary()
#loss = custom_loss(x, x_decoded_mean, z_mean)
#ae_nnMMD.compile(optimizer='adam', loss=custom_loss, metrics=[mean_square_error_NN,  custom_loss, smmd])
ae_nnMMD.compile(optimizer='adam', loss=custom_loss, metrics=[mean_square_error_NN, custom_loss])


history_DAEnnMMD= ae_nnMMD.fit([X_train, neib_train], X_train, epochs=5000,
                      shuffle=True,
                      validation_data=([X_test, neib_test], X_test),
                verbose=1, batch_size=256)
plt.plot(history_DAEnnMMD.history['loss'][50:])
plt.plot(history_DAEnnMMD.history['val_loss'][50:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

encoderDAEnnMMD = Model([x, k_neib], z_mean, name='encoder')
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
