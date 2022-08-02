'''
Demonstrates splitting effect of DR on uniformly distrubuted data in 10D to 2D
line 1045 - paper plots
'''
import timeit
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import tensorflow as tf
from plotly.io import to_html
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import pickle
pio.renderers.default = "browser"

from utils_evaluation import plot3D_cluster_colors, table
from utils_model import frange_anneal, relu_derivative, elu_derivative, tanh_derivative, linear_derivative,\
    sigmoid_derivative


class saveEncoder(Callback):
    def __init__(self,  encoder, ID, epochs, output_dir, save_period):
        super(Callback, self).__init__()
        self.encoder = encoder
        self.ID = ID
        self.epochs = epochs
        self.output_dir = output_dir
        self.save_period = save_period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_period == 0 and epoch != 0:
            self.encoder.save_weights(
                self.output_dir + '/' + self.ID + "_encoder_" + 'epochs' + str(self.epochs) + '_epoch=' + str(
                    epoch) + '_3D.h5')


from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()


DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Split_demo/'
output_dir  = DATA_ROOT + 'Artificial_sets/Split_demo/'
PLOTS = DATA_ROOT + 'Artificial_sets/Split_demo/PLOTS/'

#####################################################################################
ID1 = 'Split_demo_tanh'

epochs=30000

############################################################################################################
# 10 dimensional example
# generate data
nrow = 30000
s=1

inp_d =10
#TODO: uncomment before submission
#aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))
#np.savez(output_dir + '/' + ID1 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' + ID1 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.01
latent_dim = 2
original_dim = aFrame.shape[1]
intermediate_dim = original_dim*2
intermediate_dim2 = original_dim * 1

initializer = tf.keras.initializers.he_normal(12345)

X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='elu', name='intermediate', kernel_initializer=initializer)(X)
h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h2)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='elu', name='intermediate3', kernel_initializer=initializer)(z_mean)
decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='elu', name='output', kernel_initializer=initializer)(decoder_h2)
autoencoder = Model(inputs=X, outputs=decoder_mean)

def DCAE_loss(y_true, y_pred):  # (x, x_decoded_mean):  # attempt to avoid vanishing derivative of sigmoid
    U = encoder.get_layer('intermediate').trainable_weights[0]
    W = encoder.get_layer('intermediate2').trainable_weights[0]
    Z = encoder.get_layer('z_mean').trainable_weights[0]  # N x N_hidden
    U = K.transpose(U);
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = elu_derivative(u)
    m = encoder.get_layer('intermediate2').output
    dm = elu_derivative(m)
    s = encoder.get_layer('z_mean').output
    ds = tanh_derivative(s)

    diff_tens = tf.einsum('al,lj->alj', ds, Z)
    diff_tens = tf.einsum('al,ajl->ajl', dm, diff_tens)
    diff_tens = tf.einsum('ajl,lk->ajk', diff_tens, W)
    u_U = tf.einsum('al,lj->alj', du, U)
    diff_tens = tf.einsum('ajl,alk->ajk', diff_tens, u_U)
    return lam * K.sqrt(K.sum(diff_tens ** 2, axis=[1, 2]))

def loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
)

autoencoder.compile(optimizer=opt, loss=loss,
                    metrics=[loss, DCAE_loss])

autoencoder.summary()

save_period = 100
batch_size = 32

# start = timeit.default_timer()
# history_multiple = autoencoder.fit(aFrame, aFrame,
#                                    batch_size=batch_size,
#                                    epochs=epochs,
#                                    shuffle=True,
#                                    callbacks=[saveEncoder(encoder=encoder, ID=ID1, epochs=epochs,
#                                                                   output_dir=output_dir, save_period=save_period)],
#                                    verbose=1)
# stop = timeit.default_timer()
# z = encoder.predict([aFrame])
# print(stop - start)

# Comment this after the first run
#encoder.save_weights(output_dir + '/' + ID1 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
#autoencoder.save_weights(output_dir + '/autoencoder_' + ID1 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
#np.savez(output_dir + '/' + ID1 + "_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
#with open(output_dir + '/' + ID1 + 'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#    pickle.dump(history_multiple.history, file_pi)



#
encoder.load_weights(output_dir + '/' + ID1 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' + ID1 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
history = pickle.load(open(output_dir + '/' + ID1 + 'epochs' + str(epochs) + '_history',  "rb"))
st = 10;
stp = 10000 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')
fig02 = plt.figure();
plt.plot(history['DCAE_loss'][st:stp]);
plt.title('DCAE_loss')


for i in range(original_dim):
    fig01 = plt.figure();
    col=i
    plt.scatter(z[:,0], z[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

###############################################################################################################
# Include DCAE
epochs=30000
lam=0.01
ID2 = 'Split_demo_with_DCAE_lam_' + str(lam)
def DCAE_loss(y_true, y_pred):
    U = encoder.get_layer('intermediate').trainable_weights[0]
    W = encoder.get_layer('intermediate2').trainable_weights[0]
    Z = encoder.get_layer('z_mean').trainable_weights[0]  # N x N_hidden
    U = K.transpose(U);
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = elu_derivative(u)
    m = encoder.get_layer('intermediate2').output
    dm = elu_derivative(m)
    s = encoder.get_layer('z_mean').output
    ds = tanh_derivative(s)

    #r = tf.linalg.einsum('aj->a', s ** 2)  # R^2 in reality. to think this trough

    # pot = 500 * tf.math.square(alp - r) * tf.dtypes.cast(tf.less(r, alp), tf.float32) + \
    #      500 * (r - 1) * tf.dtypes.cast(tf.greater_equal(r, 1), tf.float32) + 1
    # pot=1
    #pot = tf.math.square(r - 1) + 1

    #ds = tf.einsum('ak,a->ak', ds, pot)
    diff_tens = tf.einsum('al,lj->alj', ds, Z)
    diff_tens = tf.einsum('al,ajl->ajl', dm, diff_tens)
    diff_tens = tf.einsum('ajl,lk->ajk', diff_tens, W)
    u_U = tf.einsum('al,lj->alj', du, U)
    diff_tens = tf.einsum('ajl,alk->ajk', diff_tens, u_U)
    return lam * K.sqrt(K.sum(diff_tens ** 2, axis=[1, 2]))


X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='elu', name='intermediate', kernel_initializer=initializer)(X)
h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h2)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='elu', name='intermediate3', kernel_initializer=initializer)(z_mean)
decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='elu', name='output', kernel_initializer=initializer)(decoder_h2)
autoencoder = Model(inputs=X, outputs=decoder_mean)

# optimize matrix multiplication


def MSE(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def loss(y_true, y_pred):
    return  MSE(y_true, y_pred) + DCAE_loss(y_true, y_pred)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
)

autoencoder.compile(optimizer=opt, loss=loss,
                    metrics=[loss, DCAE_loss, MSE])

autoencoder.summary()

save_period = 100
batch_size = 32

# start = timeit.default_timer()
# history_multiple = autoencoder.fit(aFrame, aFrame,
#                                    batch_size=batch_size,
#                                    epochs=epochs,
#                                    shuffle=True,
#                                    callbacks=[saveEncoder(encoder=encoder, ID=ID2, epochs=epochs,
#                                                                   output_dir=output_dir, save_period=save_period)],
#                                    verbose=1)
# stop = timeit.default_timer()
# z = encoder.predict([aFrame])
# print(stop - start)

#encoder.save_weights(output_dir + '/' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
#autoencoder.save_weights(output_dir + '/autoencoder_' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
#np.savez(output_dir + '/' + ID2 + "_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
#with open(output_dir + '/' + ID2 + "_tanh"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#    pickle.dump(history_multiple.history, file_pi)
encoder.load_weights(output_dir + '/' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)

history = pickle.load(open(output_dir + '/' + ID2 + "_tanh"+'epochs' + str(epochs) + '_history',  "rb"))
st = 10;
stp = 10000 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')

st = 10;
stp = 10000 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')
fig02 = plt.figure();
plt.plot(history['DCAE_loss'][st:stp]);
plt.title('DCAE_loss')


for i in range(original_dim):
    fig01 = plt.figure();
    col=i
    plt.scatter(z[:,0], z[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

####################################################################################################################
npzfile = np.load(output_dir + '/' + ID1 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

npzfile1 = np.load(output_dir + '/' + ID1 + "_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz')
z1 = npzfile1['z']
history1 = pickle.load(open(output_dir + '/'  + ID1 +  'epochs'+ str(epochs)+ '_history',  "rb"))

npzfile2 = np.load(output_dir + '/' + ID2 + "_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz')
z2 = npzfile2['z']
history2 = pickle.load(open(output_dir + '/' + ID2 + "_tanh"+'epochs' + str(epochs) + '_history',  "rb"))

from matplotlib import rcParams
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(1,3,width_ratios=[6,6,0.5])
dpi= 350
rcParams['savefig.dpi'] = dpi
fig = plt.figure(dpi = dpi, figsize=(12,5))
#plt.xticks(fontsize=14)
# First subplot
ax1 = fig.add_subplot(gs[0])
ax1.grid(False)
ax1.set_title('MSE',fontsize=30)
col=3

ax2 = fig.add_subplot(gs[1])
ax2.set_title('MSE + DCAE', fontsize=30,ha='center')
ax2.scatter(z2[:,0], z2[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
ax2.grid(False)

amap = ax1.scatter(z1[:,0], z1[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
fig.colorbar(amap, cax=fig.add_subplot(gs[2]))
#ax1.axis('off')
#ax2.axis('off')
fig.tight_layout(pad = 1.0)
plt.savefig(PLOTS + 'splitting_demo.eps', dpi=dpi, format='eps')
plt.show()

########################################################################################################
# investigate embedding surface in HD coordinates
#DCAE + MSE
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
encoder.load_weights(output_dir + '/' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest_DCAE = autoencoder.predict(aFrame)
z_DCAE = encoder.predict(aFrame)
dcae_ae = autoencoder

encoder.load_weights(output_dir + '/' + ID1 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' + ID1 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest_MSE = autoencoder.predict(aFrame)
z_MSE = encoder.predict(aFrame)
mse_ae = autoencoder

fig01 = plt.figure();
boxplot=sns.boxplot(data=pd.DataFrame(A_rest_DCAE))
boxplot.axes.set_title("DCAE", fontsize=16)
plt.show()

fig01 = plt.figure();
boxplot=sns.boxplot(data=pd.DataFrame(A_rest_MSE))
boxplot.axes.set_title("MSE", fontsize=16)
plt.show()

# #create a 3d mesh
# nm=25
# nx=nm; ny=nm; nz = nm
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# zs = np.linspace(0, 1, nz)
# xv, yv, zv = np.meshgrid(x, y, zs, sparse=False)
# nz = nx*ny*nm
# a = np.reshape(xv, nz)
# b = np.reshape(yv, nz)
# c = np.reshape(zv, nz)
# mesh_array = np.c_[a,b,c ]
# mesh_map =  autoencoder.predict(mesh_array)
# mesh_z = encoder.predict(mesh_array)

A_rest_DCAE_3D  =A_rest_DCAE[:, [0,4,9]]
dpi =300
fig = plt.figure(dpi=dpi, figsize=(10, 10))
ax = Axes3D(fig)
sns.reset_orig()
ax.scatter(xs=A_rest_DCAE_3D[:, 0], ys=A_rest_DCAE_3D[:,1], zs=A_rest_DCAE_3D[:,2], s=0.1)
plt.show()


aFrame_3D  =aFrame[:, [3,5,8]]
dpi =300
fig = plt.figure(dpi=dpi, figsize=(10, 10))
ax = Axes3D(fig)
sns.reset_orig()
ax.scatter(xs=aFrame_3D [:, 0], ys=aFrame_3D [:,1], zs=aFrame_3D [:,2], s=0.1)
plt.show()






########################################################################################################
# PLOTS for paper
# 1. Histogram in z,
# 2. plot of splitting overlaying mappings of border and D-set
ID13 = 'Split_demo_2D_to_1D_ELU_8_16_tanh'
epochs=500000

nrow = 10000
s=1
inp_d = 2
#TODO: uncomment later
aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))

#np.savez(output_dir + '/' + ID13 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
#npzfile = np.load(output_dir + '/' +ID13 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
#aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = inp_d
intermediate_dim = original_dim * 4
intermediate_dim2= original_dim * 8
# remove one unit to see of splitting stops -1 -still splits
#                                           -2

initializer = tf.keras.initializers.he_normal(12345)
X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='elu', name='intermediate', kernel_initializer=initializer)(X)
#h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='elu', name='intermediate3', kernel_initializer=initializer)(z_mean)
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='elu', name='output', kernel_initializer=initializer)(decoder_h)
autoencoder = Model(inputs=X, outputs=decoder_mean)


# loss for 2 layer encoder
def DCAE_2l(y_true, y_pred):
    U = encoder.get_layer('intermediate').trainable_weights[0]
    Z = encoder.get_layer('z_mean').trainable_weights[0]  # N x N_hidden
    U = K.transpose(U);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = elu_derivative(u)
    s = encoder.get_layer('z_mean').output
    ds = tanh_derivative(s)
    diff_tens = tf.einsum('al,lj->alj', ds, Z)
    u_U = tf.einsum('al,lj->alj', du, U)
    diff_tens = tf.einsum('ajl,alk->ajk', diff_tens, u_U)
    return lam * K.sqrt(K.sum(diff_tens ** 2, axis=[1, 2]))



def loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-07, amsgrad=True
)

autoencoder.compile(optimizer=opt, loss=loss,
                    metrics=[loss, DCAE_2l])

autoencoder.summary()

save_period = 100
batch_size = 250
#
# start = timeit.default_timer()
# history_multiple = autoencoder.fit(aFrame, aFrame,
#                                    batch_size=batch_size,
#                                    epochs=epochs,
#                                    shuffle=True,
#                                    callbacks=[saveEncoder(encoder=encoder, ID=ID13, epochs=epochs,
#                                                                   output_dir=output_dir, save_period=save_period)],
#                                    verbose=1)
# stop = timeit.default_timer()
# z = encoder.predict([aFrame])
# print(stop - start)
#
# encoder.save_weights(output_dir + '/' +ID13 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
# autoencoder.save_weights(output_dir + '/autoencoder_' +ID13 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
#
# np.savez(output_dir + '/' +ID13+ "_linear_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
# with open(output_dir + '/' +ID13 + "_linear"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#    pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' +ID13 +  "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' +ID13  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
print(output_dir + '/autoencoder_' +ID13  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')

# extract decoder
decoder_input = Input(shape=(latent_dim,))
x= Dense(intermediate_dim2, activation='relu', name='intermediate3', kernel_initializer=initializer)(decoder_input )
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoded= Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(x)
decoder = Model(inputs=decoder_input, outputs=decoded)
decoder.summary()
weights_list = autoencoder.get_weights()[4:8]
decoder.set_weights(weights_list)
dec_map = decoder.predict(np.linspace(-1, 1, num=1000))

#histogram of 1D representation, paper
fig01 = plt.figure();
plt.hist(z,500)

#for w in autoencoder.trainable_weights:
#    print(K.eval(w))

#np.arctanh(2* (z- np.min(z) / (np.max(z) - np.min(z)) ) -1 )
for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z , c=aFrame[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()



fig01 = plt.figure();
col=1
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=z,  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

history = pickle.load(open(output_dir + '/' +ID13 + "_linear"+'epochs' + str(epochs) + '_history',  "rb"))
st = 4000;
stp = 500000 #len(history['loss'])
fig01 = plt.figure();
plt.plot((history['loss'][st:stp]));
plt.title('loss')
fig = plt.figure();
plt.plot((history['DCAE_2l'][st:stp]));
plt.title('DCAE_2l')


z_gap = np.linspace(-0.97, 0.97, num=1000)
z_gap = z_gap[(np.logical_or(z_gap<0.14, z_gap<0.14))]
gap_map = decoder.predict(z_gap)
# fig01 = plt.figure();
# col=1
# ax = fig01.add_subplot(projection='3d')
# #ax.set_zlim3d(0.1,0.2)
# #zs =np.arctanh(2* (z- np.min(z) / (np.max(z) - np.min(z)) ) -1 )
# p = ax.scatter3D(aFrame[:,0], aFrame[:,1], z, c = z, s=0.1)
# fig01.colorbar(p)
# #ax.scatter3D(A_rest[:,0], A_rest[:,1], -1, c=aFrame[:,col],  cmap='winter', s=1)
# ax.scatter3D(gap_map[:,0], gap_map[:,1], -1, c='red',s=1)

# plot a mapping portrait of autoencoder
# create a mesh in a square in [-1,1] cube
nx=200; ny=100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y, sparse=False)
nz = nx*ny
a = np.reshape(xv, nz)
b = np.reshape(yv, nz)
mesh_array = np.c_[a,b ]
mesh_map =  autoencoder.predict(mesh_array)
mesh_z = encoder.predict(mesh_array)

cmap = plt.get_cmap('jet_r')
# fig01 = plt.figure();
# colors = cmap(mesh_z[:, 0])
# p = plt.scatter(x=mesh_map[:,0], y=mesh_map[:,1], c=colors, s=1)
# fig01.colorbar(p)
# for i in range(nz):
#     color = cmap(mesh_z[i, 0])
#     plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , c=colors[i,:], linestyle="--", markersize=0.1 ,linewidth=0.1)
# #p = plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=z, s=0.1)
# #plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
# plt.title("Encoder mapping  vs input mesh autoencoder mapping (blue)")
# plt.show()

fig01 = plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c='red',  cmap='winter', s=3)
#plt.ylim((0.1,0.2));plt.xlim((0.20,0.24));
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("autoencoder random input mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

# plot a 'corner' are of mapping
# nx=100; ny=100
# x = np.linspace(0.6, 1, nx)
# y = np.linspace(0.0, 0.6, ny)
# xv, yv = np.meshgrid(x, y, sparse=False)
# nz = nx*ny
# a = np.reshape(xv, nz)
# b = np.reshape(yv, nz)
# mesh_array = np.c_[a,b ]
# mesh_map =  autoencoder.predict(mesh_array)
#
# fig01 = plt.figure();
# plt.xlim([0.6, 1])
# plt.ylim([0.0, 0.6])
# plt.scatter(x=dec_map[:,0], y=dec_map[:,1], c='red',  cmap='winter', s=3)
# for i in range(nz):
#     plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
# #plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
# plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
# plt.title("Decoder mapping (red) vs input mesh autoencoder mapping (blue)")
# plt.show()

fig01 = plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c='red',  cmap='winter', s=3)
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("autoencoder random input mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

# plot mapping of the lower bottom of the square
# plot a mapping portrait of autoencoder
# create a mesh in a square in [-1,1] cube
# TO INLCLUDE in papeer
nx=100; ny=1
x = np.linspace(0, 1, nx)
y = np.linspace(0, 0.01, ny)
xv, yv = np.meshgrid(x, y, sparse=False)
nz = nx*ny
a = np.reshape(xv, nz)
b = np.reshape(yv, nz)
mesh_array = np.c_[a,b ]
mesh_map =  autoencoder.predict(mesh_array)
mesh_z = encoder.predict(mesh_array)

nx=100; ny=1
x = np.linspace(0.5320,0.5400, nx)
y = np.linspace(0, 0.01, ny)
xv, yv = np.meshgrid(x, y, sparse=False)
nz = nx*ny
a = np.reshape(xv, nz)
b = np.reshape(yv, nz)
mesh_array2 = np.c_[a,b ]
mesh_map2 =  autoencoder.predict(mesh_array2)
mesh_z2 = encoder.predict(mesh_array2)

cmap = plt.get_cmap('jet_r')
from matplotlib import colors, cm
cnorm = colors.Normalize(vmin=-1, vmax=1)
smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
#color = cmap(np.linspace(-1, 1, num=1000))
enc_map = encoder.predict(aFrame)
#plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(nrows=4, figsize=(8, 19), gridspec_kw={'height_ratios': [3, 3, 1, 1]})
p= ax[0].scatter(x=A_rest[:,0], y=A_rest[:,1], cmap='jet_r', c=enc_map[:,0], vmin=-1, vmax=1 ,   s=3)
for i in range(nx):
    color = smap.to_rgba(mesh_z[i,0] )
    im=ax[0].plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , c=color ,   linestyle="--", markersize=0.1 ,linewidth=0.5)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
ax[0].scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
ax[0].title.set_text('(a) Decoder mapping of latent variable overlayed with autoencoder mapping')

# add subplot for the mapping on gap
p= ax[1].scatter(x=A_rest[:,0], y=A_rest[:,1], cmap='jet_r', c=enc_map[:,0], vmin=-1, vmax=1 ,   s=3)
for i in range(nz):
    color = smap.to_rgba(mesh_z2[i,0] )
    im=ax[1].plot( [mesh_array2[i,0], mesh_map2[i,0]],  [mesh_array2[i,1], mesh_map2[i,1]] , c=color ,   linestyle="--", markersize=0.1 ,linewidth=0.5)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
ax[1].scatter( mesh_array2[:,0],   mesh_array2[:,1], c='green', s=1)
ax[1].title.set_text('(b) Decoder mapping of points near D-set')

z_bottom = z[aFrame[:,1] < 0.1]
Y,X = np.histogram(z_bottom, 200, normed=1)
x_span = X.max()-X.min()
cm = plt.cm.get_cmap('jet_r')
C = [cm(((x-X.min())/x_span)) for x in X]
ax[2].bar(X[:-1],Y,color=C,width=X[1]-X[0])
ax[2].title.set_text('(c) Histogram of latent variable for y<0.1')

Y,X = np.histogram(z, 200, normed=1)
x_span = X.max()-X.min()
cm = plt.cm.get_cmap('jet_r')
C = [cm(((x-X.min())/x_span)) for x in X]
ax[3].bar(X[:-1],Y,color=C,width=X[1]-X[0])
ax[3].title.set_text('(d) Histogram of latent variable')

cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(p, cax=cbar_ax)
plt.show()
fig.savefig(PLOTS + '/'+"ELU_split_2D" + ".eps", format='eps', dpi=350)
fig.savefig(PLOTS + '/'+"ELU_split_2D" + ".tif", format='tif', dpi=350)
plt.close()
##############################################################################################
#plot gradient loss
from  tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
x_tensor = tf.convert_to_tensor(aFrame, dtype=tf.float32)
with tf.GradientTape(persistent=True) as t:
    t.watch(x_tensor)
    mapx = autoencoder(x_tensor)
    #result =inputx
    #gradients = t.gradient(tf.keras.losses.mean_squared_error(x_tensor, mapx), x_tensor)#.numpy()
    gradients = t.gradient( mapx.data, x_tensor)
    print('Gradients: ', gradients)


# an input layer
x_true = Input(shape=2)
# compute loss based on model's output and input
ce = K.mean(tf.keras.losses.mean_squared_error(x_true, autoencoder.output))
# compute gradient of loss with respect to inputs
grad_ce = K.gradients(ce, autoencoder.inputs)
# create a function to be able to run this computation graph
func = K.function(autoencoder.inputs + [x_true], grad_ce)
output = func([aFrame, aFrame])[0]

norm_out  = np.sqrt(output[:, 0]**2 + output[:, 1]**2 )

fig, ax = plt.subplots(nrows=1, figsize=(8, 6))
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c='g',   s=3)
p = plt.scatter(x=aFrame[:,0], y=aFrame[:,1], cmap='jet_r', c=norm_out, vmin=norm_out.min(), vmax=norm_out.max(), s=1000*(norm_out) )
ax.set_rasterized(True)
fig.colorbar(p)
fig.savefig(PLOTS + '/'+"ELU_D_set_by_gradient_loss_2D" + ".eps", format='eps', dpi=350)
fig.savefig(PLOTS + '/'+"ELU_D_set_by_gradient_loss_2D" + ".tif", format='tif', dpi=350)
plt.show()


nz=nx=100
x1 = 0.9696; y1 = 0.988
x0 = 0.43
k = (y1 - 0.0)/(x1 - x0)
b = y1 - k * x1
d_set_inp_x = np.linspace(x0,0.47, nx)
d_set_inp_y = k * d_set_inp_x + b
d_set_inp = np.c_[d_set_inp_x, d_set_inp_y]
d_set_map = autoencoder.predict(d_set_inp)
d_set_mesh  = encoder.predict(d_set_inp)

fig = plt.figure()
p= plt.scatter(x=A_rest[:,0], y=A_rest[:,1], cmap='jet_r', c=enc_map[:,0], vmin=-1, vmax=1 ,   s=3)
for i in range(nz):
    color = smap.to_rgba(d_set_mesh[i,0] )
    im=plt.plot( [d_set_inp[i,0], d_set_map[i,0]],  [d_set_inp[i,1], d_set_map[i,1]] , c=color ,   linestyle="--", markersize=0.1 ,linewidth=0.5)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( d_set_inp[:,0],   d_set_inp[:,1], c='black', s=3)
#plt.title.set_text('(b) Decoder mapping of points near D-set')
plt.show()




fig01 = plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c='red',  cmap='winter', s=3)
cmap = plt.get_cmap('jet_r')
for i in range(nz):
    color = cmap(mesh_z[i, 0])
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , c= color,linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("autoencoder random input mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

# plot the gap related to the set above
z_b = encoder.predict(mesh_array)
for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,100), y=z_b , c=mesh_array[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

fig01 =plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=z,  cmap='winter', s=3)
fig01.colorbar(p)

# is it a retraction? Looks like shift
retract =  autoencoder.predict(A_rest)
deltaMap =  retract - A_rest
from matplotlib import colors, cm
cnorm = colors.Normalize(vmin=0, vmax=np.sqrt(deltaMap[:, 0]**2 + deltaMap[:, 1]**2).max())
smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)

n=100
fig01 = plt.figure();
plt.scatter(x=A_rest[:n,0], y=A_rest[:n,1], c='black',  cmap='winter', s=3)
plt.scatter(x= retract[:n,0], y= retract[:n,1], c='green',  cmap='winter', s=5, marker = "P")
cmap = plt.get_cmap('jet_r')
for i in range(n):
    color =  smap.to_rgba(np.sqrt(deltaMap[i, 0]**2 + deltaMap[i, 1]**2))
    plt.plot( [A_rest[i,0], retract[i,0]],  [A_rest[i,1], retract[i,1]] , c= color,linestyle="--", markersize=0.1 ,linewidth=0.5, )
    #plt.xlim([0.22, 0.3]); plt.ylim([0.2, 0.32]);
fig01.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap))
plt.show()
