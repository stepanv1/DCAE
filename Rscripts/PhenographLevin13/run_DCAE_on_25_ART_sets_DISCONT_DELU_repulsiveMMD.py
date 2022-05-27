'''
Runs DCAE
experiment with KL differen
'''
import timeit
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import tensorflow as tf
from plotly.io import to_html
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
import pickle
pio.renderers.default = "browser"

from utils_evaluation import plot3D_cluster_colors, table
from utils_model import plotCallback, AnnealingCallback, saveEncoder
from utils_model import frange_anneal, relu_derivative, elu_derivative, leaky_relu_derivative, linear_derivative

from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)



@tf.custom_gradient
def DELU_activation(x): # step RELU activation, see this post for references:
    # https://ai.stackexchange.com/questions/17609/in-deep-learning-is-it-possible-to-use-discontinuous-activation-functions
    #ones = tf.ones(tf.shape(x), dtype=x.dtype.base_dtype)
    #zeros = tf.zeros(tf.shape(x), dtype=x.dtype.base_dtype)
    cond = tf.math.greater_equal(x, tf.constant(0.0))
    expx = tf.math.exp(x)
    def grad(dy): # see https://localcoder.org/how-to-apply-guided-backprop-in-tensorflow-2-0#credit_1 for the details
        return tf.where(cond, tf.constant(1.0), expx) * dy  # TODO check gradient

    return tf.where(cond, x + tf.constant(0.2), expx - 1), grad

k = 30
epochs_list = [250]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 0.1
alp = 0.2
m = 10
patience = 500
min_delta = 1e-4
g=10
#epochs=100
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
output_dir  = DATA_ROOT + 'Artificial_sets/DCAE_output/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
ID = 'DICSCONT_DELU_0.2_repulsive_MMD_0.05_experiment' + '_g_'  + str(g) +  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)
#ID = 'DICSCONT_DELU_0.2' + '_g_'  + str(g) +  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)
#ID = 'ELU_' + '_DCAE_norm_0.5' + 'lam_'  + str(lam) + 'batch_' + str(batch_size) + 'alp_' + str(alp) + 'm_' + str(m)
#output_dir  = DATA_ROOT + 'Artificial_sets/DCAE_output'
#load earlier generated data
#epochs=500
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()
#bl = list_of_branches[0]
# epochs =250
# possibly final parameters: m=10 ; lam = 0.1; g=0.1
# worst: lam = 0.01; # g=0.1; lam = 0.01; # g=0.01, seems like lam =0.001 is to small
for epochs in epochs_list:
    for bl in list_of_branches[11:12]:
        infile = source_dir + 'set_' + str(bl) + '.npz'
        # markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
        # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
        npzfile = np.load(infile)
        weight_distALL = npzfile['Dist'];
        # = weight_distALL[IDX,:]
        aFrame = npzfile['aFrame'];

        #aFrame = np.random.uniform(low=-2.0, high=[2.0, 2.0, 2.0, -1.8, -1.8], size=(5000, 5))

        Dist = npzfile['Dist']
        Idx = npzfile['Idx']
        # cut_neibF = npzfile['cut_neibF'];
        # cut_neibF = cut_neibF[IDX,:]
        neibALL = npzfile['neibALL']
        Sigma = npzfile['Sigma']
        lbls = npzfile['lbls'];

        nrow = np.shape(aFrame)[0]

        from numpy import nanmax, argmax, unravel_index
        from scipy.spatial.distance import pdist, squareform

        IDX = np.random.choice(nrow, size=2000,replace=False)
        D = pdist(aFrame[IDX, :])
        D = squareform(D);
        max_dist, [I_row, I_col] = nanmax(D), unravel_index(argmax(D), D.shape)
        np.fill_diagonal(D, 10000)
        min_dist = np.min(D)
        np.fill_diagonal(D, 0)
        mean_dist = np.mean(D)

        # max_dist
        # sns.distplot(D)
        # convex hull
        # import numpy as np
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import cdist

        MMD_weight = K.variable(value=0)

        MMD_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=0.2)))

        DCAE_weight = K.variable(value=0)

        DCAE_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=1)))

        latent_dim = 3
        original_dim = aFrame.shape[1]
        intermediate_dim = original_dim * 3
        intermediate_dim2 = original_dim * 2

        initializer = tf.keras.initializers.he_normal(12345)


        SigmaTsq = Input(shape=(1,))
        X = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='elu', name='intermediate', kernel_initializer=initializer)(X)
        h1 = Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
        z_mean = Dense(latent_dim, activation=None, name='z_mean', kernel_initializer=initializer)(h1)

        encoder = Model([X, SigmaTsq], z_mean, name='encoder')

        decoder_h = Dense(intermediate_dim2, activation= DELU_activation, name='intermediate3', kernel_initializer=initializer)
        decoder_h1 = Dense(intermediate_dim, activation= DELU_activation, name='intermediate4', kernel_initializer=initializer)
        decoder_mean = Dense(original_dim, activation= DELU_activation, name='output', kernel_initializer=initializer)
        h_decoded = decoder_h(z_mean)
        h_decoded2 = decoder_h1(h_decoded)
        x_decoded_mean = decoder_mean(h_decoded2)
        autoencoder = Model(inputs=[X, SigmaTsq], outputs=x_decoded_mean)

        normSigma = 1


        # optimize matrix multiplication
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
            ds = linear_derivative(s)

            r = tf.linalg.einsum('aj->a', s ** 2)  # R^2 in reality. to think this trough

            # pot = 500 * tf.math.square(alp - r) * tf.dtypes.cast(tf.less(r, alp), tf.float32) + \
            #      500 * (r - 1) * tf.dtypes.cast(tf.greater_equal(r, 1), tf.float32) + 1
            # pot=1
            pot = tf.math.square(r - 1) + 1

            ds = tf.einsum('ak,a->ak', ds, pot)
            diff_tens = tf.einsum('al,lj->alj', ds, Z)
            diff_tens = tf.einsum('al,ajl->ajl', dm, diff_tens)
            diff_tens = tf.einsum('ajl,lk->ajk', diff_tens, W)
            u_U = tf.einsum('al,lj->alj', du, U)
            diff_tens = tf.einsum('ajl,alk->ajk', diff_tens, u_U)
            return lam * SigmaTsq[:, 0] * K.sqrt(K.sum(diff_tens ** 2, axis=[1, 2]))
            # return lam * K.sum(diff_tens ** 2, axis=[1, 2])

        meanS = np.mean(Sigma)
        neib_dist = np.mean(Dist[:,30])
        #plt.hist(Dist[:,30],50)


        def compute_graph_weights_Inp(x):
            x_size = tf.shape(x)[0]
            dim = tf.shape(x)[1]
            # TODO: x = x/meanS we nead update this function in the fashion that weights are computed
            #from w_ij = kernel((x_i-x_j)/sigma_i) and then symmetrized
            #x=x/max_dist
            tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, x_size, 1]))
            tiled_y = tf.tile(tf.reshape(x, tf.stack([1, x_size, dim])), tf.stack([x_size, 1, 1]))
            D = tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2)#/ tf.cast(x_size**2, tf.float32)
            #apply monotone f to sqeeze  and normalize
            D = K.sqrt(D)
            #D = 2 * (D - min_dist)/(max_dist - min_dist)
            D = 2 * D  / max_dist
            D = tf.linalg.set_diag(D, tf.zeros(x_size), name=None)
            return D
            #no log version
        def compute_graph_weights_enc(x):
            x_size = tf.shape(x)[0]
            dim = tf.shape(x)[1]
            #x = tf.linalg.normalize(x, ord=2, axis=1)[0]
            tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, x_size, 1]))
            tiled_y = tf.tile(tf.reshape(x, tf.stack([1, x_size, dim])), tf.stack([x_size, 1, 1]))
            D =  tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2)
            D = K.sqrt(D)
            #D = tf.linalg.set_diag(D, tf.zeros(x_size), name=None)# / tf.cast(dim, tf.float32))
            return D
        # long range version
        #def graph_diff(x, y):
            #pot = tf.math.square(r - 1)
        #    return g * K.sqrt(K.sum(K.square(1 - K.exp ( compute_graph_weights_Inp(X) -  compute_graph_weights_enc(z_mean)))) /tf.cast( batch_size**2, tf.float32))

        #short range version
        def graph_diff(x, y):
            x_size = tf.shape(x)[0]
            D = compute_graph_weights_Inp(X)
            d = compute_graph_weights_enc(z_mean)
            denom = K.exp(D) - 1
            denom = tf.linalg.set_diag(denom, tf.ones(x_size), name=None)  # / tf.cast(dim, tf.float32))
            enrg = K.square(D - d)  / denom
            enrg_no_nan = tf.linalg.set_diag(enrg, tf.zeros(x_size), name=None)
            return g * K.sum(enrg_no_nan) / K.sum(D)


        #idx =np.random.choice(nrow, size=30,replace=False)
        #LL = lbls[idx]
        #KL = K.eval(- K.sum(compute_graph_weights_Inp(aFrame[idx,:].astype('float32')) * K.log(compute_graph_weights_enc(z[idx,:]))) + K.log(K.sum(compute_graph_weights_enc(z[idx,:]))))
        #SigmaTsq =Sigma[idx]
        # K.eval(- K.sum(compute_graph_weights_Inp(aFrame[idx, :].astype('float32')) * K.log(compute_graph_weights_enc(z[idx, :]))))
        #g_diff = K.eval(-1 *  (compute_graph_weights_Inp(aFrame[idx,:].astype('float32')) * K.log(compute_graph_weights_enc(z[idx,:]))))
        #g_diff = K.eval(g * K.sqrt(((compute_graph_weights_Inp(aFrame[idx,:].astype('float32')) - compute_graph_weights_enc(z[idx,:])) ** 2)))
        #g_diff = K.eval(K.square(1 - K.exp ( compute_graph_weights_Inp(aFrame[idx,:].astype('float32')) -  compute_graph_weights_enc(z[idx,:]))))
        #g_enc =  K.eval(compute_graph_weights_enc(z[idx,:]))
        #g_inp =  K.eval(compute_graph_weights_Inp(aFrame[idx,:].astype('float32')))
        enrg = K.eval(K.square(compute_graph_weights_Inp(aFrame[idx, :].astype('float32') - compute_graph_weights_enc(z[idx,:]))) /
                        (K.exp(compute_graph_weights_Inp(aFrame[idx, :].astype('float32'))) - 1))
        #g_diff  = K.eval( (K.square((compute_graph_weights_Inp(aFrame[idx, :].astype('float32')) - compute_graph_weights_enc(z[idx,:]))) /
        #                   (K.exp(compute_graph_weights_Inp(aFrame[idx, :].astype('float32')))-1.0))
        #                  / K.sum(compute_graph_weights_Inp(aFrame[idx, :].astype('float32'))))

        def compute_kernel(x, y):
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]
            tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
            tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
            return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

        def compute_mmd(x, y):  # [batch_size, z_dim] [batch_size, z_dim]
            x_kernel = compute_kernel(x, x)
            y_kernel = compute_kernel(y, y)
            xy_kernel = compute_kernel(x, y)
            return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

        # a = 0.8; b = 1.1; ndim = 3; npoints = 100
        def sample_shell(npoints, a, b, ndim=3):
            """
            samples points uniformly in a spherical shell between radii a and b
            """
            # first sample spherical
            vec = K.random_normal(shape=(npoints, ndim))
            # K.get_value(vec)
            vec = tf.linalg.normalize(vec, axis=1)[0]

            R = tf.pow(K.random_uniform(shape=[npoints], minval=a ** 3, maxval=b ** 3), 1 / 3)
            #sns.displot( np.power(np.random.uniform(a ** 3, b ** 3, 50000), 1 / 3))
            return tf.einsum('a,aj->aj', R, vec)

        # a=0.8; b=1.1; ndim=3; npoints=100
        def loss_mmd(y_true, y_pred):
            batch_sz = K.shape(z_mean)[0]
            # latent_dim = K.int_shape(z_mean)[1]
            # true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.)
            true_samples = sample_shell(batch_sz, 0.9, 1.1)
            # true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
            # true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=0.0, maxval=1.0)
            return m * compute_mmd(true_samples, z_mean)
        #
        # y_true = np.random.normal(loc=0, scale=0, size=(250, 30))
        # y_pred = np.random.normal(loc=0, scale=0, size=(250, 30))
        def mean_square_error_NN(y_true, y_pred):
            # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
            msew_nw = tf.keras.losses.mean_squared_error(y_true, y_pred)
            # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
            return normSigma / SigmaTsq[:,0] * msew_nw
            # return msew_nw


        def ae_loss(weight, MMD_weight_lst):
            def loss(y_true, y_pred):
                msew = mean_square_error_NN(y_true, y_pred)
                # return coeffMSE * msew + (1 - MMD_weight) * loss_mmd(x, x_decoded_mean)
                # return coeffMSE * msew + (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + (MMD_weight + coeffCAE) * DCAE_loss(x, x_decoded_mean)
                # return coeffMSE * msew + 0.5 * (2 - MMD_weight) * loss_mmd(x, x_decoded_mean)
                return coeffMSE * msew +  loss_mmd(y_true, y_pred) +  coeffCAE * (DCAE_loss(y_true, y_pred)) +\
                       (1-MMD_weight+0.05)* graph_diff(y_true, y_pred)
                # return  loss_mmd(x, x_decoded_mean)

            return loss
            # return K.switch(tf.equal(Epoch_count, 10),  loss1(x, x_decoded_mean), loss1(x, x_decoded_mean))


        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, clipvalue=1.0
        )

        autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst),
                            metrics=[DCAE_loss, graph_diff, loss_mmd, mean_square_error_NN])

        autoencoder.summary()

        save_period = 10
        DCAEStop = EarlyStopping(monitor='DCAE_loss', min_delta=min_delta, patience=patience, mode='min',
                                 restore_best_weights=False)
        start = timeit.default_timer()
        history_multiple = autoencoder.fit([aFrame, Sigma], aFrame,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           shuffle=True,
                                           callbacks=[AnnealingCallback(MMD_weight, MMD_weight_lst),
                                                      plotCallback(aFrame=aFrame, Sigma=Sigma, lbls=lbls, encoder=encoder,
                                                                  ID=ID, bl=bl, epochs=epochs, output_dir=output_dir,
                                                        save_period=save_period,),
                                                      saveEncoder(encoder=encoder, ID=ID, bl=bl, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period),
                                                       DCAEStop],
                                           #callbacks=[AnnealingCallback(MMD_weight, MMD_weight_lst),
                                           #           callSave, callPlot, DCAEStop],
                                           #           #callSave, callPlot],
                                           verbose=2)
        stop = timeit.default_timer()
        z = encoder.predict([aFrame, Sigma])
        print(stop - start)

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