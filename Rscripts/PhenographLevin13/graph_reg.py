'''
Runs DCAE
mappings for artificial clusters
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

k = 30
coeffCAE = 1
coeffMSE = 1
epochs_list = [500]
batch_size = 128
lam = 0.1
alp = 0.2
m = 10
patience = 500
min_delta = 1e-4
g=1000
#epochs=100
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
output_dir  = DATA_ROOT + 'Artificial_sets/DCAE_output/temp'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
ID = 'ELU_graph_KL_DCAE_0_' + '_g_'  + str(g) +  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)
#ID = 'ELU_' + '_DCAE_norm_0.5' + 'lam_'  + str(lam) + 'batch_' + str(batch_size) + 'alp_' + str(alp) + 'm_' + str(m)
#load earlier generated data

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()
bl = list_of_branches[0]
epochs=500
for epochs in epochs_list:
    for bl in list_of_branches[0:7]:
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

        MMD_weight = K.variable(value=0)

        MMD_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=1)))

        latent_dim = 3
        original_dim = aFrame.shape[1]
        intermediate_dim = original_dim * 3
        intermediate_dim2 = original_dim * 2

        initializer = tf.keras.initializers.he_normal(12345)
        # initializer = None
        '''
        SigmaTsq = Input(shape=(1,))
        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dim,  name='intermediate', kernel_initializer=initializer)(x)
        h = LeakyReLU(alpha=0.3)(h)
        h1 = Dense(intermediate_dim2, name='intermediate2', kernel_initializer=initializer)(h)
        h1 = LeakyReLU(alpha=0.3)(h1)
        z_mean = Dense(latent_dim, activation=None, name='z_mean', kernel_initializer=initializer)(h1)

        encoder = Model([x, SigmaTsq], z_mean, name='encoder')

        decoder_h = Dense(intermediate_dim2,  name='intermediate3', kernel_initializer=initializer)
        decoder_h1 = Dense(intermediate_dim,  name='intermediate4', kernel_initializer=initializer)
        decoder_mean = Dense(original_dim,  name='output', kernel_initializer=initializer)
        h_decoded = decoder_h(z_mean)
        h_decoded = LeakyReLU(alpha=0.3)(h_decoded )
        h_decoded2 = decoder_h1(h_decoded)
        h_decoded2 = LeakyReLU(alpha=0.3)(h_decoded2)
        x_decoded_mean = decoder_mean(h_decoded2)
        x_decoded_mean = LeakyReLU(alpha=0.3)(x_decoded_mean)
        autoencoder = Model(inputs=[x, SigmaTsq], outputs=x_decoded_mean)
        '''

        SigmaTsq = Input(shape=(1,))
        X = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='elu', name='intermediate', kernel_initializer=initializer)(X)
        h1 = Dense(intermediate_dim2, activation='elu', name='intermediate2', kernel_initializer=initializer)(h)
        z_mean = Dense(latent_dim, activation=None, name='z_mean', kernel_initializer=initializer)(h1)

        encoder = Model([X, SigmaTsq], z_mean, name='encoder')

        decoder_h = Dense(intermediate_dim2, activation='elu', name='intermediate3', kernel_initializer=initializer)
        decoder_h1 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)
        decoder_mean = Dense(original_dim, activation='elu', name='output', kernel_initializer=initializer)
        h_decoded = decoder_h(z_mean)
        h_decoded2 = decoder_h1(h_decoded)
        x_decoded_mean = decoder_mean(h_decoded2)
        autoencoder = Model(inputs=[X, SigmaTsq], outputs=x_decoded_mean)

        normSigma = 1


        #optimize matrix multiplication
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
            diff_tens = tf.einsum('ajl,lk->ajk',  diff_tens, W)
            u_U = tf.einsum('al,lj->alj', du, U)
            diff_tens = tf.einsum('ajl,alk->ajk', diff_tens, u_U)
            # multiply by configning potential
            # Try L2,1 metric
            # first, take square
            #diff_tens = diff_tens**2
            #   second, sum over latent indices
            #   diff_tens = tf.einsum('ajl->al', diff_tens)
            #   return lam * SigmaTsq[:, 0] * K.sum(K.sqrt(diff_tens), axis=[1])
            # return lam * K.sum(K.sqrt(K.abs(diff_tens) + 1e-9), axis=[1, 2])
            #return lam * SigmaTsq[:, 0] * K.pow(K.sum(diff_tens ** 2, axis=[1, 2]), 0.25)
            return lam * SigmaTsq[:, 0] * K.sqrt(K.sum(diff_tens ** 2, axis=[1, 2]))
            #return lam * K.sum(diff_tens ** 2, axis=[1, 2])


        def DCAE2_loss(y_true, y_pred):
            U = encoder.get_layer('intermediate').trainable_weights[0]
            U = K.transpose(U);
            u = encoder.get_layer('intermediate').output
            du = elu_derivative(u)

            S_U = tf.einsum('al,lj->alj', du, U)
            # f = tf.where(K.abs(diff_tens) > 0, K.abs(diff_tens), 0.0)# to prevent nans in loss
            # return tf.transpose(1 / normSigma * SigmaTsq * lam) * K.sum(K.sqrt(K.abs(diff_tens)), axis=[1, 2])
            # return  (1 / normSigma * SigmaTsq * lam)[0,:] * K.sum(diff_tens**2, axis=[1, 2])
            return lam * (K.sum(K.abs(S_U), axis=[1, 2]) / intermediate_dim / original_dim)


        # mmd staff TODO: try approximation for this
        def compute_kernel(x, y):
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]
            tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
            tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
            return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))
            #return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(0.01, tf.float32))


        meanS = np.mean(Sigma)
        def compute_graph_weights_Inp(x):
            x_size = tf.shape(x)[0]
            #dim = tf.shape(x)[1]
            #TODO: x = x/meanS we nead update this function in the fashion that weights are computed
            # from w_ij = kernel((x_i-x_j)/sigma_i) and then symmetrized
            #tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, x_size, 1]))
            #tiled_y = tf.tile(tf.reshape(x, tf.stack([1, x_size, dim])), tf.stack([x_size, 1, 1]))
            #return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2)/meanS)
            r = tf.reduce_sum(x * x, 1)
            # turn r into column vector
            r = tf.reshape(r, [-1, 1])
            D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
            #multiply each row over tsk
            #D = tf.einsum('ik,i ->ik', D, 1/ SigmaTsq[:,0])
            #D = tf.einsum('ik,i ->ik', D, 1 / Sigma[idx])
            D = D/98
            D = tf.exp(-D)
            D = tf.linalg.set_diag(D, tf.zeros(x_size), name=None)
            #D = K.sqrt(D)
            D= tf.linalg.normalize(D, ord=1, axis=1)[0]
            D= (D + tf.transpose(D))/2
            return D


        def compute_graph_weights_enc(x):
            x_size = tf.shape(x)[0]
            #dim = tf.shape(x)[1]
            #x = x / .1
            #tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, x_size, 1]))
            #iled_y = tf.tile(tf.reshape(x, tf.stack([1, x_size, dim])), tf.stack([x_size, 1, 1]))
            #return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(1, tf.float32))
            x = x / 2
            r = tf.reduce_sum(x*x, 1)
            # turn r into column vector
            r = tf.reshape(r, [-1, 1])
            D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
            D = 1/(1+D)
            D = tf.linalg.set_diag(D, tf.zeros(x_size), name=None)
            #D = K.sqrt(D)
            D= tf.linalg.normalize(D, ord=1, axis=1)[0]
            D = tf.linalg.set_diag(D, tf.ones(x_size), name=None) # to prent nans because of log(0)
            return D

        def graph_diff(x, y):
            #return g * K.sqrt(
            #tf.reduce_mean((compute_graph_weights_Inp(X) - compute_graph_weights_enc(z_mean)) ** 2))
            return -g * tf.reduce_mean(compute_graph_weights_Inp(X) * K.log(compute_graph_weights_enc(z_mean)))


        #idx = np.random.randint(0, high=nrow, size=20)
        #m_diff = K.eval(-g  *  (compute_graph_weights_Inp(aFrame[idx,:].astype('float32')) * K.log(compute_graph_weights_enc(z[idx,:]))))
        #g_enc =  K.eval(compute_graph_weights_enc(z[idx,:]))
        #g_inp =  K.eval(compute_graph_weights_Inp(aFrame[idx,:].astype('float32')))


        # from sklearn.preprocessing import normalize
        # def np_compute_graph_weights_Inp(x):
        #     #x_size = np.shape(x)[0]
        #     dim = np.shape(x)[1]
        #     #TODO: x = x/meanS we nead update this function in the fashion that weights are computed
        #     # from w_ij = kernel((x_i-x_j)/sigma_i) and then symmetrized
        #     #tiled_x = np.tile(np.reshape(x, np.stack([x_size, 1, dim])), np.stack([1, x_size, 1]))
        #     #tiled_y = np.tile(np.reshape(x, np.stack([1, x_size, dim])), np.stack([x_size, 1, 1]))
        #     #return np.exp(-np.mean(np.square(tiled_x - tiled_y), axis=2)/meanS)
        #     s = Sigma[idx]
        #     r = np.sum(x * x, 1)
        #     # turn r into column vector
        #     r = np.reshape(r, [-1, 1])
        #     D = r - 2 * np.matmul(x, np.transpose(x)) + np.transpose(r)
        #     #D = np.einsum('ik,i ->ik', D, 1 / (dim * s))
        #     D = D/98
        #     D = np.exp(-D)
        #     D = normalize(D, norm='l1', axis=1)
        #     return(D)
        #
        # def np_compute_graph_weights_enc(x):
        #     #x_size = np.shape(x)[0]
        #     #dim = np.shape(x)[1]
        #     #x = x / .1
        #     #tiled_x = np.tile(np.reshape(x, np.stack([x_size, 1, dim])), np.stack([1, x_size, 1]))
        #     #tiled_y = np.tile(np.reshape(x, np.stack([1, x_size, dim])), np.stack([x_size, 1, 1]))
        #     #return np.exp(-np.mean(np.square(tiled_x - tiled_y), axis=2) /1)
        #     x=z[idx,:]
        #     #s=Sigma[idx]
        #     x = x / 2
        #     r = np.sum(x * x, 1)
        #     # turn r into column vector
        #     r = np.reshape(r, [-1, 1])
        #     D = r - 2 * np.matmul(x, np.transpose(x)) + np.transpose(r)
        #     D = 1/(1+D)
        #     D = normalize(D, norm='l1', axis=1)
        #     return D
        #
        # def np_graph_diff(x,y):
        #     return ((np_compute_graph_weights_Inp(x) - np_compute_graph_weights_enc(y))**2)
        #
        #
        # idx = np.random.randint(0, high=nrow, size=2000)
        # idx =  np.where(lbls== 0)[0][0:2000]
        # X_inp = aFrame[idx,:].astype('float32')
        # z_enc = z[idx,:]
        # m_diff=np_graph_diff(X_inp, z_enc)
        # g_enc =  np_compute_graph_weights_enc(z_enc)
        # g_inp =  np_compute_graph_weights_Inp(X_inp)
        # sns.distplot(g_enc  - g_inp)
        # tt=g_enc - g_inp
        # # check matrix distances
        # import seaborn as sns
        # from sklearn.metrics.pairwise import euclidean_distances
        # D_X_inp = euclidean_distances(X_inp, X_inp)**2
        # D_X_inp.max()
        # mask = np.ones(D_X_inp.shape, dtype=bool)
        # np.fill_diagonal(mask, 0)
        # D_X_inp[mask].min()
        # D_z_enc = euclidean_distances(z_enc, z_enc)**2
        # D_z_enc.max()
        # mask = np.ones(D_z_enc.shape, dtype=bool)
        # np.fill_diagonal(mask, 0)
        # D_z_enc[mask].min()
        #
        #
        # sns.distplot(D_X_inp[mask])
        # sns.distplot(D_z_enc[mask])
        #
        # fig = plot3D_cluster_colors(z_enc, lbls=lbls[idx], msize=5)
        # fig.show()
        # fig = plot3D_cluster_colors(z, lbls=lbls, msize=1)
        # fig.show()
        # fig = plot3D_cluster_colors(aFrame[:,0:3], lbls=lbls, msize=1)
        # fig.show()




# '''
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
            return tf.einsum('a,aj->aj', R, vec)


        # a=0.8; b=1.1; ndim=3; npoints=100
        def loss_mmd(y_true, y_pred):
            batch_sz = K.shape(z_mean)[0]
            # latent_dim = K.int_shape(z_mean)[1]
            # true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.)
            true_samples = sample_shell(batch_sz, 0.8, 1.2)
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
                return coeffMSE * msew + 0.5 * (2 - MMD_weight) * loss_mmd(y_true, y_pred) + 0*(
                        5 * MMD_weight + coeffCAE) * (DCAE_loss(y_true, y_pred)) + (
                        10 * MMD_weight + 1) * graph_diff(y_true, y_pred)
                # return  loss_mmd(x, x_decoded_mean)

            return loss
            # return K.switch(tf.equal(Epoch_count, 10),  loss1(x, x_decoded_mean), loss1(x, x_decoded_mean))


        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
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