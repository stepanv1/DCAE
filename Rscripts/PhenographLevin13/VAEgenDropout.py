'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob
from keras.layers import Input, Dense, Lambda, Layer, Dropout
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

import seaborn as sns

#from keras.datasets import mnist
#load data

source_dir = "/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL/DA"
file_list = glob.glob(source_dir + '/*.txt')

[aFrame, lbls, Dists, Idx, rs] = [np.genfromtxt(x, delimiter=' ', skip_header=0, skip_footer=0) for x in file_list ]
Idx = np.int32(Idx)
lbls = np.int32(lbls)
rs = np.int32(rs)

#model parameters
batch_size = 10
original_dim = 39
latent_dim = 39*24
intermediate_dim = 30

epochs = 10
epsilon_std = 1.0
U=1
#generator

cutoff = np.count_nonzero(aFrame)/(np.shape(aFrame)[0]*np.shape(aFrame)[1])

def generator(features, features2,  batch_size, nn, k_=30, shuffle=True):
 # Create empty arrays to contain batch of features and labels#
    x = np.zeros((batch_size, 39))
    neib = np.zeros((batch_size, k_, 39))
    cut_neib = np.zeros((batch_size, 39))
    l=len(features)
    dim2=np.shape(features)[1]
    b=0
    ac= U/np.power(1-cutoff, 15)
    if shuffle == True:
        random.shuffle(features)
    while True:
        index = range(b, min(b + batch_size, l))
        lind = len(index)
        b = b + lind
        if b>=l:
            if shuffle == True:
                random.shuffle(features)
            b=0
        neib = np.zeros((lind, k_, 39))
        cut_neib = np.zeros((lind, 39))
        x = features[index]
        for i in range(lind):
             neib[i,] = [ features[z, ] for z in Idx[index[i]] ]
             for j in range(dim2):
                 cnz = np.count_nonzero(neib[i, :, j]) / k_
                 #print(U if cnz <= cutoff else ac * (1 - cnz) ** 2)
                 cut_neib[i, j] = U if cnz <= cutoff else ac * np.power((1 - cnz) ,15)

        yield ([x, cut_neib],x)

gen = generator(aFrame, aFrame, batch_size, nn=Idx, k_=30 )
#gen_pred = generator(aFrame, aFrame, batch_size, nn=Idx, k_=30, shuffle = False)

cut_neib = Input(shape = (original_dim,))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim,  activation='relu')(x)
hD = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(hD)
z_log_var = Dense(latent_dim)(hD)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

zD = Dropout(0.5)(z)
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='relu' )
h_decoded = decoder_h(zD)
h_decodedD = Dropout(0.2)(h_decoded)
x_decoded_mean = decoder_mean(h_decodedD)

def mean_square_error_weighted(y_true, y_pred):
    return K.mean((K.square(y_pred - y_true)/(cut_neib+1))/K.var(y_true) , axis=-1)

def vae_loss(x, x_decoded_mean):
    msew = original_dim * mean_square_error_weighted(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    pen_zero = K.sum(K.square(x*cut_neib), axis=-1)
    return K.mean(100*msew + 0.1*kl_loss + pen_zero)


#y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model([x, cut_neib], x_decoded_mean)

#vae.compile(optimizer='rmsprop', loss=vae_loss)
#sgd = SGD(lr=0.01, momentum=0.1, decay=0.00001, nesterov=True)
vae.compile(optimizer='Adam', loss=vae_loss)

#x_train = aFrame[:400000,:]
#x_test = aFrame[1:4000,:]

#vae.fit(x_train, x_train,
#        shuffle=True,
#        epochs=epochs,
#        batch_size=batch_size,
#        validation_data=(x_test, x_test)
#        )

start = timeit.default_timer()
b_sizes = range(10,110,10); i=9
#for i in range(10) :
    gen = generator(aFrame, aFrame, 30, nn=Idx, k_=30 )
    history=vae.fit_generator(gen,
    steps_per_epoch=np.ceil(len(aFrame)/b_sizes[i]),
    epochs = 9,
    shuffle=True,
    use_multiprocessing=True, workers=12)
stop = timeit.default_timer()
print(stop - start)
fig0 = plt.figure()
plt.plot(history.history['loss'])
print(vae.summary())

# build a model to project inputs on the latent space
encoder = Model([x, cut_neib], z_mean)

# display a 2D plot of the digit classes in  install nmslibthe latent space
gen_pred = generator(aFrame, aFrame, 10000, nn=Idx, k_=30, shuffle = False)
x_test_vae = vae.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/10000), use_multiprocessing=True)
len(x_test_vae)
gen_pred = generator(aFrame, aFrame, 10000, nn=Idx, k_=30, shuffle = False)
x_test_enc = encoder.predict_generator(gen_pred, steps=np.ceil(len(aFrame)/10000), workers=12)

cl=5;bw=0.02
fig1 = plt.figure();
ax = sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw);
ax.set_xticklabels(rs[cl-1, ]);
fig2 = plt.figure();
bx = sns.violinplot(data= aFrame[lbls==cl , :], bw =bw);
bx.set_xticklabels(rs[cl-1, ])
fig3 = plt.figure();
bz = sns.violinplot(data= x_test_enc[lbls==cl , :], bw =bw);
fig4= plt.figure();
sns.violinplot(data= x_test_vae[lbls==cl , :], bw =bw, color='red');
sns.violinplot(data= aFrame[lbls==cl , :], bw =bw, color='black');

import matplotlib.cm as cm
colors = cm.con(np.linspace(0, 1, 24))

fig = plt.figure()
for cl in range(1,24,5):
    sns.violinplot(data=x_test_enc[lbls == cl+1, :], bw=bw, color=colors[cl]);
    bx.set_xticklabels(rs[cl,])



unique, counts = np.unique(lbls, return_counts=True)
print('%d %d', np.asarray((unique, counts)).T)

