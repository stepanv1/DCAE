'''
Demonstrates splitting effect of DR on uniformly distrubuted data in 10D to 2D
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

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID1, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)

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

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID2, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)

#encoder.save_weights(output_dir + '/' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
#autoencoder.save_weights(output_dir + '/autoencoder_' + ID2 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
#np.savez(output_dir + '/' + ID2 + "_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
#with open(output_dir + '/' + ID2 + "_tanh"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#    pickle.dump(history_multiple.history, file_pi)
history = pickle.load(open(output_dir + '/' + ID2 + "_tanh"+'epochs' + str(epochs) + '_history',  "rb"))
st = 10;
stp = 10000 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')



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
#low dimensional demonstration

ID3 = 'Split_demo_LD'

epochs=10000

nrow = 10000
s=1
inp_d =2
#TODO: uncomment later
aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))
np.savez(output_dir + '/' + ID3 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' + ID3 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = aFrame.shape[1]
intermediate_dim = 2
intermediate_dim2= 2

initializer = tf.keras.initializers.he_normal(12345)

X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(X)
h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h2)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim, activation='relu', name='intermediate3', kernel_initializer=initializer)(z_mean)
decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(decoder_h2)
autoencoder = Model(inputs=X, outputs=decoder_mean)

def loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-07, amsgrad=True
)

autoencoder.compile(optimizer=opt, loss=loss,
                    metrics=[loss])

autoencoder.summary()

save_period = 100
batch_size = 250

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID3, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)

encoder.save_weights(output_dir + '/' + ID3 + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.save_weights(output_dir + '/autoencoder_' + ID3 + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
np.savez(output_dir + '/' + ID3+ "_tanh_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
with open(output_dir + '/' + ID3 + "_tanh"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
    pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' + ID3 +  "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' + ID3  + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
print(output_dir + '/autoencoder_' + ID3  + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')

fig01 = plt.figure();
plt.hist(z,200)
from scipy.stats import spearmanr
spearmanr(z, aFrame[:,0])
spearmanr(z[z<0.378],aFrame[np.where(z<0.378)[0],1] )


for w in autoencoder.trainable_weights:
    print(K.eval(w))

fig01 = plt.figure();
col = 1
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for w in encoder.trainable_weights:
    print(K.eval(w))


fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


history = pickle.load(open(output_dir + '/' + ID3 + "_tanh"+'epochs' + str(epochs) + '_history',  "rb"))
st = 10;
stp = 10000 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')

fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

####################################################################################################
#start with assigned weights
ID4 = 'Split_demo_assigned weights'
epochs=10000

nrow = 10000
s=1
inp_d = 2
#TODO: uncomment later
aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))
np.savez(output_dir + '/' + ID4 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' + ID4 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = aFrame.shape[1]
intermediate_dim = 2


initializer = tf.keras.initializers.he_normal(12345)

X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(X)
#h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim, activation='relu', name='intermediate3', kernel_initializer=initializer)(z_mean)
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(decoder_h)
autoencoder = Model(inputs=X, outputs=decoder_mean)

# assign encoder  weights to create split from the beginning
w1 = np.zeros((2, 2)) # two input neurons for two neurons at the hidden layer
b1 = np.zeros((2,))   # one bias neuron for two neurons in the hidden layer
w2 = np.zeros((2, 1)) # two input neurons for one output neuron of encoder
b2 = np.zeros((1,))   # one bias for one output neuron of encoder

w1[0, 0] =  0.1;  w1[1, 0] =  100; # the weights for the first hidden neuron
b1[0]    = 0.1  # bias for the first neuron
w1[0, 1] =  0.1; w1[1, 1] =   -100 # the weights for the second hidden
b1[1]    = 0.1 # bias for the second neuron

w2[0, 0] =  0.01 # weight for the first input of the output neuron
w2[1, 0] =  -0.01 # weight for the second input of the output neuron
b2[0]    =  0 # bias for the output neuron

encoder.set_weights([w1, b1, w2, b2])

encoder.predict(np.array([[0.1,0.01]]))
encoder.predict(np.array([[0.1,-0.01]]))

for w in encoder.trainable_weights:
    print(K.eval(w))
z = encoder.predict([aFrame])
fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:, col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

plt.hist(z,200)


def loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True
)

autoencoder.compile(optimizer=opt, loss=loss,
                    metrics=[loss])

autoencoder.summary()

save_period = 100
batch_size = 250

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID4, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)

encoder.save_weights(output_dir + '/' + ID4 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.save_weights(output_dir + '/autoencoder_' + ID4 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
np.savez(output_dir + '/' + ID4+ "_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
with open(output_dir + '/' + ID4 + 'epochs' + str(epochs) + '_history', 'wb') as file_pi:
    pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' + ID4 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' + ID4 + "_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)


fig01 = plt.figure();
plt.hist(z,200)
from scipy.stats import spearmanr
spearmanr(z, aFrame[:,1])

fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:, col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, 0], y=z, c=aFrame[:, 0], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for w in encoder.trainable_weights:
    print(K.eval(w))


for w in autoencoder.trainable_weights:
    print(K.eval(w))



for i in range(original_dim):
    fig01 = plt.figure();
    col=i
    plt.scatter(aFrame[:,i], z, c=aFrame[:,col],  cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()


history = pickle.load(open(output_dir + '/' + ID4 + 'epochs' + str(epochs) + '_history',  "rb"))
st = 10;
stp = 10000 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')


fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


########################################################################################################
#low dimensional demonstration

ID5 = 'Split_demo_HD_to_1D'

epochs=10000

nrow = 10000
s=1
inp_d = 10
#TODO: uncomment later
#aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))
#np.savez(output_dir + '/' + ID5 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' +ID5 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = aFrame.shape[1]
intermediate_dim = original_dim *3
intermediate_dim2= original_dim *2

initializer = tf.keras.initializers.he_normal(12345)

X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(X)
#h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim, activation='relu', name='intermediate3', kernel_initializer=initializer)(z_mean)
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(decoder_h)
autoencoder = Model(inputs=X, outputs=decoder_mean)

def loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-07, amsgrad=True
)

autoencoder.compile(optimizer=opt, loss=loss,
                    metrics=[loss])

autoencoder.summary()

save_period = 100
batch_size = 250

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID3, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)

#encoder.save_weights(output_dir + '/' +ID5 + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
#autoencoder.save_weights(output_dir + '/autoencoder_' +ID5 + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
#np.savez(output_dir + '/' +ID5+ "_tanh_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
#with open(output_dir + '/' +ID5 + "_tanh"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#    pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' +ID5 +  "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' +ID5  + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
print(output_dir + '/autoencoder_' +ID5  + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')

fig01 = plt.figure();
plt.hist(z,200)
from scipy.stats import spearmanr
spearmanr(z, aFrame[:,1])
spearmanr(z[z<0.378],aFrame[np.where(z<0.378)[0],1] )


for w in autoencoder.trainable_weights:
    print(K.eval(w))

fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

fig01 = plt.figure();
col = 2
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for w in encoder.trainable_weights:
    print(K.eval(w))


fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


history = pickle.load(open(output_dir + '/' +ID5 + "_tanh"+'epochs' + str(epochs) + '_history',  "rb"))
st = 1000;
stp = 10000 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')

fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

########################################################################################################
#low dimensional demonstration with the main component 0 10 yomes more extended

ID6 = 'Split_demo_HD_to_1D_main component'

epochs=100000

nrow = 10000
s=1
inp_d = 10
#TODO: uncomment later
aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))
aFrame[:, 1:inp_d] = aFrame[:, 1:inp_d] /2
np.savez(output_dir + '/' + ID6 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' +ID6 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = aFrame.shape[1]
intermediate_dim = original_dim*3
intermediate_dim2= original_dim *2

initializer = tf.keras.initializers.he_normal(12345)

X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(X)
#h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim, activation='relu', name='intermediate3', kernel_initializer=initializer)(z_mean)
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(decoder_h)
autoencoder = Model(inputs=X, outputs=decoder_mean)

# loss for 2 layer encoder
def DCAE_2l(y_true, y_pred):
    U = encoder.get_layer('intermediate').trainable_weights[0]
    Z = encoder.get_layer('z_mean').trainable_weights[0]  # N x N_hidden
    U = K.transpose(U);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = relu_derivative(u)
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

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID6, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)

# encoder.save_weights(output_dir + '/' +ID6 + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
# autoencoder.save_weights(output_dir + '/autoencoder_' +ID6 + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
# np.savez(output_dir + '/' +ID6+ "_tanh_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
# with open(output_dir + '/' +ID6 + "_tanh"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#      pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' +ID6 +  "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' +ID6  + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
print(output_dir + '/autoencoder_' +ID6  + "_tanh_"  + 'epochs' + str(epochs) + '_3D.h5')

fig01 = plt.figure();
plt.hist(z,200)
from scipy.stats import spearmanr
spearmanr(z, aFrame[:,1])
spearmanr(z[z<0.378],aFrame[np.where(z<0.378)[0],1] )


for w in autoencoder.trainable_weights:
    print(K.eval(w))

fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

fig01 = plt.figure();
col = 2
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for w in encoder.trainable_weights:
    print(K.eval(w))


fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


history = pickle.load(open(output_dir + '/' +ID6 + "_tanh"+'epochs' + str(epochs) + '_history',  "rb"))
st = 44000;
stp = 100000 #len(history['loss'])
fig01 = plt.figure();
plt.plot((history['loss'][st:stp]));
plt.title('loss')
fig = plt.figure();
plt.plot((history['DCAE_2l'][st:stp]));
plt.title('DCAE_2l')

fig01 = plt.figure();
col=1
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


########################################################################################################
#low dimensional demonstration with the main component 0 10 yomes more extended

ID8 = 'Split_demo_2D_to_1D_by_latent_linear'
epochs=100000

nrow = 10000
s=1
inp_d = 2
#TODO: uncomment later
aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))
aFrame[:, 1:inp_d] = aFrame[:, 1:inp_d]
np.savez(output_dir + '/' + ID8 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' +ID8 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = aFrame.shape[1]
intermediate_dim = original_dim*2
intermediate_dim2= original_dim * 4

initializer = tf.keras.initializers.he_normal(12345)

X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(X)
#h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='relu', name='intermediate3', kernel_initializer=initializer)(z_mean)
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(decoder_h)
autoencoder = Model(inputs=X, outputs=decoder_mean)

# loss for 2 layer encoder
def DCAE_2l(y_true, y_pred):
    U = encoder.get_layer('intermediate').trainable_weights[0]
    Z = encoder.get_layer('z_mean').trainable_weights[0]  # N x N_hidden
    U = K.transpose(U);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = relu_derivative(u)
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

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID8, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)
#
#encoder.save_weights(output_dir + '/' +ID8 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
#autoencoder.save_weights(output_dir + '/autoencoder_' +ID8 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
#np.savez(output_dir + '/' +ID8+ "_linear_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
#with open(output_dir + '/' +ID8 + "_linear"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#    pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' +ID8 +  "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' +ID8  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
print(output_dir + '/autoencoder_' +ID8  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')

fig01 = plt.figure();
plt.hist(z,200)
from scipy.stats import spearmanr
spearmanr(z, aFrame[:,1])
spearmanr(z[z<0.378],aFrame[np.where(z<0.378)[0],1] )


for w in autoencoder.trainable_weights:
    print(K.eval(w))

fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=np.tanh(z), c=aFrame[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

fig01 = plt.figure();
col = 2
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, 1], y=z, c=aFrame[:, 1], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for w in encoder.trainable_weights:
    print(K.eval(w))


fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


history = pickle.load(open(output_dir + '/' +ID8 + "_linear"+'epochs' + str(epochs) + '_history',  "rb"))
st = 44000;
stp = 100000 #len(history['loss'])
fig01 = plt.figure();
plt.plot((history['loss'][st:stp]));
plt.title('loss')
fig = plt.figure();
plt.plot((history['DCAE_2l'][st:stp]));
plt.title('DCAE_2l')

fig01 = plt.figure();
col=1
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

########################################################################################################
#
ID9 = 'Split_demo_2D_to_1D_by_latent_linear_small_decoder_nodes_4_16_tanh'
epochs=100000

nrow = 10000
s=1
inp_d = 2
#TODO: uncomment later
aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))

np.savez(output_dir + '/' + ID9 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' +ID9 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = inp_d
intermediate_dim = original_dim * 2
intermediate_dim2= original_dim * 8
# remove one unit to see of splitting stops -1 -still splits
#                                           -2

initializer = tf.keras.initializers.he_normal(12345)
X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(X)
#h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='relu', name='intermediate3', kernel_initializer=initializer)(z_mean)
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(decoder_h)
autoencoder = Model(inputs=X, outputs=decoder_mean)


# loss for 2 layer encoder
def DCAE_2l(y_true, y_pred):
    U = encoder.get_layer('intermediate').trainable_weights[0]
    Z = encoder.get_layer('z_mean').trainable_weights[0]  # N x N_hidden
    U = K.transpose(U);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = relu_derivative(u)
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

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID9, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)
#
#encoder.save_weights(output_dir + '/' +ID9 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
#autoencoder.save_weights(output_dir + '/autoencoder_' +ID9 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')

#np.savez(output_dir + '/' +ID9+ "_linear_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
#with open(output_dir + '/' +ID9 + "_linear"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
#    pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' +ID9 +  "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' +ID9  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
print(output_dir + '/autoencoder_' +ID9  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')

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


fig01 = plt.figure();
plt.hist(z,500)

fig01 = plt.figure();
plt.hist(z[np.logical_and(z>0.14, z<0.19)],500)


from scipy.stats import spearmanr
spearmanr(z, aFrame[:,1])
spearmanr(z[z<0.-0.085],aFrame[np.where(z<0.-0.085)[0],0] )


for w in autoencoder.trainable_weights:
    print(K.eval(w))

fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()
#np.arctanh(2* (z- np.min(z) / (np.max(z) - np.min(z)) ) -1 )
for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z , c=aFrame[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

for dim in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(aFrame[:,dim], y=z, s=0.1)
    plt.title('dimension' + str(dim))
    plt.colorbar()




fig01 = plt.figure();
col = 0
plt.scatter(aFrame[:, col], y=z,  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, col], y=z,  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for w in encoder.trainable_weights:
    print(K.eval(w))


fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col=1
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


history = pickle.load(open(output_dir + '/' +ID9 + "_linear"+'epochs' + str(epochs) + '_history',  "rb"))
st = 4000;
stp = 100000 #len(history['loss'])
fig01 = plt.figure();
plt.plot((history['loss'][st:stp]));
plt.title('loss')
fig = plt.figure();
plt.plot((history['DCAE_2l'][st:stp]));
plt.title('DCAE_2l')

fig01 = plt.figure();
col=1
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

#from mpl_toolkits import mplot3d
fig01 = plt.figure();
col=1
ax = fig01.add_subplot(projection='3d')
#ax.set_zlim3d(0.1,0.2)
#zs =np.arctanh(2* (z- np.min(z) / (np.max(z) - np.min(z)) ) -1 )
p = ax.scatter3D(aFrame[:,0], aFrame[:,1], z, c = z, s=0.1)
fig01.colorbar(p)
ax.scatter3D(A_rest[:,0], A_rest[:,1], -1, c=aFrame[:,col],  cmap='winter', s=1)
#plt.title('color ' + str(col))
#plt.colorbar()
fig01 = plt.figure();
col=1
ax = fig01.add_subplot(projection='3d')
#ax.set_zlim3d(0.1,0.2)
#zs =np.arctanh(2* (z- np.min(z) / (np.max(z) - np.min(z)) ) -1 )
p = ax.scatter3D(aFrame[:,0], aFrame[:,1], z, c = z, s=0.1)
fig01.colorbar(p)
#ax.scatter3D(A_rest[:,0], A_rest[:,1], -1, c=aFrame[:,col],  cmap='winter', s=1)
ax.scatter3D(dec_map[:,0], dec_map[:,1], -1, c=np.linspace(-1, 1, num=1000), s=1)

z_gap = np.linspace(-0.97, 0.97, num=1000)
z_gap = z_gap[(np.logical_or(z_gap<0.14, z_gap<0.14))]
gap_map = decoder.predict(z_gap)
fig01 = plt.figure();
col=1
ax = fig01.add_subplot(projection='3d')
#ax.set_zlim3d(0.1,0.2)
#zs =np.arctanh(2* (z- np.min(z) / (np.max(z) - np.min(z)) ) -1 )
p = ax.scatter3D(aFrame[:,0], aFrame[:,1], z, c = z, s=0.1)
fig01.colorbar(p)
#ax.scatter3D(A_rest[:,0], A_rest[:,1], -1, c=aFrame[:,col],  cmap='winter', s=1)
ax.scatter3D(gap_map[:,0], gap_map[:,1], -1, c='red',s=1)

# plot a mapping portrait of autoencoder
# create a mesh in a square in [-1,1] cube
nx=100; ny=100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y, sparse=False)
nz = nx*ny
a = np.reshape(xv, nz)
b = np.reshape(yv, nz)
mesh_array = np.c_[a,b ]
mesh_map =  autoencoder.predict(mesh_array)

fig01 = plt.figure();
plt.scatter(x=dec_map[:,0], y=dec_map[:,1], c='red',  cmap='winter', s=3)
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("Decoder mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

fig01 = plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c='red',  cmap='winter', s=3)
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("autoencoder random input mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

# plot a 'corner' are of mapping
nx=100; ny=100
x = np.linspace(0.6, 1, nx)
y = np.linspace(0.0, 0.6, ny)
xv, yv = np.meshgrid(x, y, sparse=False)
nz = nx*ny
a = np.reshape(xv, nz)
b = np.reshape(yv, nz)
mesh_array = np.c_[a,b ]
mesh_map =  autoencoder.predict(mesh_array)

fig01 = plt.figure();
plt.xlim([0.6, 1])
plt.ylim([0.0, 0.6])
plt.scatter(x=dec_map[:,0], y=dec_map[:,1], c='red',  cmap='winter', s=3)
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("Decoder mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

fig01 = plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c='red',  cmap='winter', s=3)
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("autoencoder random input mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

# plot mapping of the lower bottom of the aquare
# plot a mapping portrait of autoencoder
# create a mesh in a square in [-1,1] cube
nx=1000; ny=1
x = np.linspace(0, 1, nx)
y = np.linspace(0, 0.01, ny)
xv, yv = np.meshgrid(x, y, sparse=False)
nz = nx*ny
a = np.reshape(xv, nz)
b = np.reshape(yv, nz)
mesh_array = np.c_[a,b ]
mesh_map =  autoencoder.predict(mesh_array)

fig01 = plt.figure();
plt.scatter(x=dec_map[:,0], y=dec_map[:,1], c='red',  cmap='winter', s=3)
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("Decoder mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

fig01 = plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c='red',  cmap='winter', s=3)
for i in range(nz):
    plt.plot( [mesh_array[i,0], mesh_map[i,0]],  [mesh_array[i,1], mesh_map[i,1]] , 'bo', linestyle="--", markersize=0.1 ,linewidth=0.1)
#plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.scatter( mesh_array[:,0],   mesh_array[:,1], c='green', s=1)
plt.title("autoencoder random input mapping (red) vs input mesh autoencoder mapping (blue)")
plt.show()

# plot the gap related to the set above
z_b = encoder.predict(mesh_array)
for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,1000), y=z_b , c=mesh_array[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

fig01 =plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=z,  cmap='winter', s=3)
fig01.colorbar(p)



########################################################################################################
#
ID10 = 'Split_demo_2D_to_1D_by_latent_circle_nodes_6_8_tanh'
epochs=100000

nrow = 10000
s=1
inp_d = 2
#TODO: uncomment later
R = 1
num_points = nrow
np.random.seed(1)
theta = np.random.uniform(0,2*np.pi, num_points)
radius = np.random.uniform(0,R, num_points) ** 0.5
x = radius * np.cos(theta)
y = radius * np.sin(theta)
# visualize the points:
#plt.scatter(x,y, s=1)


aFrame=np.column_stack((x,y))+1
#plt.scatter(aFrame[:,0] ,aFrame[:,1] , s=1)
np.savez(output_dir + '/' + ID10 + "_" +  'epochs' + str(epochs) + '_aFrame.npz', aFrame=aFrame)
npzfile = np.load(output_dir + '/' +ID10 + "_" +  'epochs' + str(epochs) + '_aFrame.npz')
aFrame = npzfile['aFrame']

lam = 0.1
latent_dim = 1
original_dim = inp_d
intermediate_dim = original_dim * 3
intermediate_dim2= original_dim * 4
# remove one unit to see of splitting stops -1 -still splits
#                                           -2

initializer = tf.keras.initializers.he_normal(12345)

X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(X)
#h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean', kernel_initializer=initializer)(h)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='relu', name='intermediate3', kernel_initializer=initializer)(z_mean)
#decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)(decoder_h)
autoencoder = Model(inputs=X, outputs=decoder_mean)

# loss for 2 layer encoder
def DCAE_2l(y_true, y_pred):
    U = encoder.get_layer('intermediate').trainable_weights[0]
    Z = encoder.get_layer('z_mean').trainable_weights[0]  # N x N_hidden
    U = K.transpose(U);
    Z = K.transpose(Z);  # N_hidden x N

    u = encoder.get_layer('intermediate').output
    du = relu_derivative(u)
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

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID10, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)
#
encoder.save_weights(output_dir + '/' +ID10 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.save_weights(output_dir + '/autoencoder_' +ID10 + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
np.savez(output_dir + '/' +ID10+ "_linear_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
with open(output_dir + '/' +ID10 + "_linear"+'epochs' + str(epochs) + '_history', 'wb') as file_pi:
    pickle.dump(history_multiple.history, file_pi)

encoder.load_weights(output_dir + '/' +ID10 +  "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' +ID10  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict(aFrame)
z = encoder.predict(aFrame)
print(output_dir + '/autoencoder_' +ID10  + "_linear_"  + 'epochs' + str(epochs) + '_3D.h5')

fig01 = plt.figure();
plt.hist(z,500)

fig02 = plt.figure();
idx1 = z>0.012
plt.hist(aFrame[idx1[:,0],1],200)

fig03 = plt.figure();
idx0 = z<0.012
plt.hist(aFrame[idx0[:,0],1],200)

fig02 = plt.figure();
idx1 = z>0.012
plt.hist(aFrame[idx1[:,0],0],200)

fig03 = plt.figure();
idx0 = z<0.012
plt.hist(aFrame[idx0[:,0],0],200)



from scipy.stats import spearmanr
spearmanr(z, aFrame[:,0])
spearmanr(z[z<0.012],aFrame[np.where(z<0.012)[0],0] )


for w in autoencoder.trainable_weights:
    print(K.eval(w))

fig01 = plt.figure();
col = 0
plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for col in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(np.random.uniform(-0.2,0.2,nrow), y=z, c=aFrame[:,col], cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

for dim in range(original_dim):
    fig01 = plt.figure();
    plt.scatter(aFrame[:,dim], y=z, s=0.1)
    plt.title('dimension' + str(dim))
    plt.colorbar()




fig01 = plt.figure();
col = 0
plt.scatter(aFrame[:, col], y=z,  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col = 1
plt.scatter(aFrame[:, col], y=z,  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

for w in encoder.trainable_weights:
    print(K.eval(w))


fig01 = plt.figure();
col=0
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()

fig01 = plt.figure();
col=1
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()


history = pickle.load(open(output_dir + '/' +ID10 + "_linear"+'epochs' + str(epochs) + '_history',  "rb"))
st = 10000;
stp = 100000 #len(history['loss'])
fig01 = plt.figure();
plt.plot((history['loss'][st:stp]));
plt.title('loss')
fig = plt.figure();
plt.plot((history['DCAE_2l'][st:stp]));
plt.title('DCAE_2l')

fig01 = plt.figure();
plt.scatter(x=A_rest[:,0], y=A_rest[:,1], c=z,  cmap='winter', s=0.1)
plt.title('color ' + str(col))
plt.colorbar()
