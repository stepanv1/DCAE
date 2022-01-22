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
import pickle
pio.renderers.default = "browser"

from utils_evaluation import plot3D_cluster_colors, table


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

DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Split_demo/'
output_dir  = DATA_ROOT + 'Artificial_sets/Split_demo/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
ID = 'Split_demo'
#ID = 'ELU_' + '_DCAE_norm_0.5' + 'lam_'  + str(lam) + 'batch_' + str(batch_size) + 'alp_' + str(alp) + 'm_' + str(m)
#output_dir  = DATA_ROOT + 'Artificial_sets/DCAE_output'
#load earlier generated data

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
#tf.compat.v1.disable_eager_execution()
#bl = list_of_branches[0]
# possibly final parameters: m=10 ; lam = 0.1; g=0.1
# worst: lam = 0.01; # g=0.1; lam = 0.01; # g=0.01, seems like lam =0.001 is to small

# generate data
nrow = 30000
s=1
#aFrame = np.random.uniform(low=0.0, high=1, size=nrow)
#aFrame  = np.c_[np.sort(np.random.uniform(low=0.0, high=1, size=nrow)), aFrame]
inp_d =10
aFrame = np.random.uniform(low=np.zeros(inp_d), high=np.ones(inp_d), size=(nrow,inp_d))

# fig00 = plt.figure();
# plt.scatter(x=aFrame[:,0], y=aFrame[:,1],c=aFrame[:,0],  cmap='winter', s=s)
# plt.colorbar()
#
# fig000 = plt.figure();
# plt.scatter(x=aFrame[:,0], y=aFrame[:,1],c=aFrame[:,1],  cmap='winter', s=s)
# plt.colorbar()

latent_dim = 2
original_dim = aFrame.shape[1]
intermediate_dim = original_dim*2
intermediate_dim2 = original_dim * 1

initializer = tf.keras.initializers.he_normal(12345)



X = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='elu', name='intermediate', kernel_initializer=initializer)(X)
h2= Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
z_mean = Dense(latent_dim, activation='sigmoid', name='z_mean', kernel_initializer=initializer)(h2)

encoder = Model(X, z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='elu', name='intermediate3', kernel_initializer=initializer)(z_mean)
decoder_h2 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)(decoder_h)
decoder_mean = Dense(original_dim, activation='elu', name='output', kernel_initializer=initializer)(decoder_h2)
autoencoder = Model(inputs=X, outputs=decoder_mean)

def loss(y_true, y_pred):
    msew =  tf.keras.losses.mean_squared_error(y_true, y_pred)
    return msew

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
)

autoencoder.compile(optimizer=opt, loss=loss,
                    metrics=[loss])

autoencoder.summary()

save_period = 100
batch_size = 32
epochs=30000

start = timeit.default_timer()
history_multiple = autoencoder.fit(aFrame, aFrame,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[saveEncoder(encoder=encoder, ID=ID, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period)],
                                   verbose=1)
stop = timeit.default_timer()
z = encoder.predict([aFrame])
print(stop - start)

encoder.save_weights(output_dir + '/' + ID + "_"  + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.save_weights(output_dir + '/autoencoder_' + ID + "_"  + 'epochs' + str(epochs) + '_3D.h5')
np.savez(output_dir + '/' + ID + "_" +  'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)


for i in range(original_dim):
    fig01 = plt.figure();
    col=i
    plt.scatter(z[:,0], z[:,1], c=aFrame[:,col],  cmap='winter', s=0.1)
    plt.title('color ' + str(col))
    plt.colorbar()

plt.hist(z[:,0],50)


import seaborn as sns
fig01 = plt.figure();
plt.scatter(range(nrow), z[:,0], c=aFrame[:,1],  cmap='winter', s=0.1)
plt.title('color 1')
plt.colorbar()
fig02 = plt.figure();
plt.scatter(range(nrow), z[:,0], c=aFrame[:,0],  cmap='winter', s=0.1)
plt.title('color 0')
plt.colorbar()

fig03 = plt.figure();
plt.scatter(z[:,0], y=np.zeros(nrow), c=aFrame[:,1],  cmap='winter', s=0.1)
plt.title('color 1')
plt.colorbar()

fig03 = plt.figure();
plt.scatter(z[:,0], y=np.zeros(nrow), c=aFrame[:,0],  cmap='winter', s=0.1)
plt.title('color 0')
plt.colorbar()

his= history_multiple.history
st = 500;
stp = epochs #len(history['loss'])
fig04 = plt.figure();
plt.plot(his['loss'][st:stp]);
plt.title('loss')

Apred = autoencoder.predict([aFrame])
fig05 = plt.figure();
plt.scatter(Apred[:,0], y=Apred[:,1], c=aFrame[:,1],  cmap='winter', s=0.1)
plt.title('color 1')
plt.colorbar()



encoder.save_weights(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.save_weights(output_dir + '/autoencoder_' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
np.savez(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
# np.savez(output_dir + '/' + str(bl) + 'epochs'+str(epochs)+ '_history.npz', history_multiple)
with open(output_dir + '/' + ID + str(bl) + 'epochs' + str(epochs) + '_history', 'wb') as file_pi:
    pickle.dump(history_multiple.history, file_pi)

fig = plot3D_cluster_colors(z, lbls=lbls)
html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                   include_mathjax=False, post_script=None, full_html=True,
                   animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = output_dir
Html_file = open(
    html_dir + "/" + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_epoch=' + 'Final' + '_' + "_Buttons.html",
    "w")
Html_file.write(html_str)
Html_file.close()


        #bl = list_of_branches[20]

npzfile = np.load(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz')
z = npzfile['z']
history = pickle.load(open(output_dir + '/'  + ID +  str(bl) + 'epochs'+str(epochs)+ '_history',  "rb"))
encoder.load_weights(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
autoencoder.load_weights(output_dir + '/autoencoder_' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict([aFrame, Sigma])

plt.hist(np.sum(z**2, axis=1),500)


from sklearn.preprocessing import normalize
def sample_sphere3D(npoints, ndim=3):
    vec = np.random.normal(size=(npoints, ndim))
    vec = normalize(vec, axis=1, norm='l2')
    return vec
sph1 = sample_sphere3D(5000, ndim=3)
import plotly.graph_objects as go
fig = plot3D_cluster_colors(z, lbls=lbls, msize=1)
from plotly.graph_objs import Scatter3d
fig.add_trace(Scatter3d(x=sph1[:,0], y=sph1[:,1], z=sph1[:,2],
                                name='sphere',
                                mode='markers',
                                marker=dict(
                                    size=0.5,
                                    color='black',  # set color to an array/list of desired values
                                    opacity=0.5,
                                ),
                                text=None,
                                hoverinfo=None))
fig.show()

st = 5;
stp = 500 #len(history['loss'])
fig01 = plt.figure();
plt.plot(history['loss'][st:stp]);
plt.title('loss')
fig015 = plt.figure();
plt.plot(history['graph_diff'][st:stp]);
plt.title('graph_diff')
fig02 = plt.figure();
plt.plot(history['DCAE_loss'][st:stp]);
plt.title('DCAE_loss')
fig03 = plt.figure();
plt.plot(history['loss_mmd'][st:stp]);
plt.title('loss_mmd')
fig04 = plt.figure();
plt.plot(history['mean_square_error_NN'][st:stp]);
plt.title('mean_square_error')


fig = plot3D_cluster_colors(z, lbls=lbls)
fig.show()

fig1 = plt.figure();
plt.plot(history['val_loss'][st:stp]);
plt.title('val_loss')
fig2 = plt.figure();
plt.plot(history['val_DCAE_loss'][st:stp]);
plt.title('val_DCAE_loss')
fig3 = plt.figure();
plt.plot(history['val_loss_mmd'][st:stp]);
plt.title('val_loss_mmd')
fig4 = plt.figure();
plt.plot(history['val_mean_square_error_NN'][st:stp]);
plt.title('val_mean_square_error')

import umap
# experiment with umap. Neede if some post-[rpcessing the output gives better plot
# Initial idea run umap on z few epochs to see if splitting is cured
from scipy.optimize import curve_fit
def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]



fss, _, _ = umap.umap_.fuzzy_simplicial_set(aFrame, n_neighbors=15, random_state=1,
                                            metric='euclidean', metric_kwds={},
                                            knn_indices=Idx[:, 0:90], knn_dists=Dist[:, 0:90],
                                            angular=False, set_op_mix_ratio=1.0, local_connectivity=1.0,
                                            apply_set_operations=True, verbose=True)

spread, min_dist = 1, 0.001
a, b = find_ab_params(spread, min_dist)
zzz = umap.umap_.simplicial_set_embedding(aFrame, fss,  n_components =3, n_epochs = 15, init = z,  random_state=np.random.RandomState(1),
                                          initial_alpha=1, gamma= 0.1, negative_sample_rate=150, a=a, b=b,densmap=False,
                                          densmap_kwds = {'lambda':2.0, 'frac':0.3, 'var_shift':0.1, 'n_neighbors':15, "graph_dists":Dist},output_dens= False,
                                                metric='euclidean', metric_kwds={} )

fig = plot3D_cluster_colors(zzz[0], lbls=lbls)
fig.show()

fig0 = plot3D_cluster_colors(z, lbls=lbls)
fig0.show()




autoencoder.load_weights(output_dir + '/autoencoder_' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
A_rest = autoencoder.predict([aFrame, Sigma, ])
import seaborn as sns

lblsC = [7 if i == -7 else i for i in lbls]
l_list = np.unique(lblsC)
fig, axs = plt.subplots(nrows=8)
yl = aFrame.min()
yu = aFrame.max()
for i in l_list:
    sns.violinplot(data=A_rest[lblsC == i, :], ax=axs[int(i)])
    axs[int(i)].set_ylim(yl, yu)
    axs[int(i)].set_title(str(int(i)), rotation=-90, x=1.05, y=0.5)
fig.savefig(PLOTS + 'Sensitivity/' + str(bl) + "Signal_violinplot" + ".eps", format='eps', dpi=350)
plt.close()

sns.violinplot(data=A_rest)
    sns.violinplot(data=aFrame)