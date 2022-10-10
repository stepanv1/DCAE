'''
Runs DCAE on artificial data
'''
import timeit
import numpy as np
import plotly.io as pio
import tensorflow as tf
from plotly.io import to_html
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pickle
pio.renderers.default = "browser"

from utils_evaluation import plot3D_cluster_colors
from utils_model import plotCallback, AnnealingCallback, saveEncoder
from utils_model import frange_anneal, elu_derivative, linear_derivative

from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

k = 30
epochs_list = [500]
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 0.1
alp = 0.5
m = 10
patience = 500
min_delta = 1e-4


DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
output_dir  = DATA_ROOT + 'Artificial_sets/DCAE_output/'
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])

ID = 'DCAE' +  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()

for epochs in epochs_list:
    for bl in list_of_branches:
        infile = source_dir + 'set_' + str(bl) + '.npz'
        npzfile = np.load(infile)
        aFrame = npzfile['aFrame'];
        lbls = npzfile['lbls'];
        Dist = npzfile['Dist']
        Idx = npzfile['Idx']
        neibALL = npzfile['neibALL']
        Sigma = npzfile['Sigma']

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

        MSE_weight = K.variable(value=0)
        MSE_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=0.2)))

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

        decoder_h = Dense(intermediate_dim2, activation='elu', name='intermediate3', kernel_initializer=initializer)
        decoder_h1 = Dense(intermediate_dim, activation='elu', name='intermediate4', kernel_initializer=initializer)
        decoder_mean = Dense(original_dim, activation='elu', name='output', kernel_initializer=initializer)
        h_decoded = decoder_h(z_mean)
        h_decoded2 = decoder_h1(h_decoded)
        x_decoded_mean = decoder_mean(h_decoded2)
        autoencoder = Model(inputs=[X, SigmaTsq], outputs=x_decoded_mean)

        normSigma = 1

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
            ds = linear_derivative(s)

            r = tf.linalg.einsum('aj->a', s ** 2)  # R^2 in reality. to think this trough

            pot = tf.math.square(r - 1) + 1

            ds = tf.einsum('ak,a->ak', ds, pot)
            diff_tens = tf.einsum('al,lj->alj', ds, Z)
            diff_tens = tf.einsum('al,ajl->ajl', dm, diff_tens)
            diff_tens = tf.einsum('ajl,lk->ajk', diff_tens, W)
            u_U = tf.einsum('al,lj->alj', du, U)
            diff_tens = tf.einsum('ajl,alk->ajk', diff_tens, u_U)
            return lam * SigmaTsq[:, 0] * K.sum(diff_tens ** 2, axis=[1, 2])

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

        def sample_shell(npoints, a, b, ndim=3):
            """
            samples points uniformly in a spherical shell between radii a and b
            """
            # first sample spherical
            vec = K.random_normal(shape=(npoints, ndim))
            vec = tf.linalg.normalize(vec, axis=1)[0]
            R = tf.pow(K.random_uniform(shape=[npoints], minval=a ** 3, maxval=b ** 3), 1 / 3)
            return tf.einsum('a,aj->aj', R, vec)

        def loss_mmd(y_true, y_pred):
            batch_sz = K.shape(z_mean)[0]
            true_samples = sample_shell(batch_sz, 0.99, 1.01)
            return m * compute_mmd(true_samples, z_mean)

        def mean_square_error_NN(y_true, y_pred):
            msew_nw = tf.keras.losses.mean_squared_error(y_true, y_pred)
            return normSigma / SigmaTsq[:,0] * msew_nw

        def ae_loss(weight, MMD_weight_lst):
            def loss(y_true, y_pred):
                msew = mean_square_error_NN(y_true, y_pred)

                return coeffMSE * (1 - MSE_weight + 0.1) * msew + 0.5 * (MSE_weight + 1) * loss_mmd(y_true, y_pred) + (
                        2 * MSE_weight + 0.1) * (DCAE_loss(y_true, y_pred))
            return loss

        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, clipvalue=1.0
        )


        autoencoder.compile(optimizer=opt, loss=ae_loss(MSE_weight, MSE_weight_lst),
                            metrics=[loss_mmd, DCAE_loss, mean_square_error_NN])
        autoencoder.summary()

        save_period = 10
        DCAEStop = EarlyStopping(monitor='DCAE_loss', min_delta=min_delta, patience=patience, mode='min',
                                 restore_best_weights=False)
        start = timeit.default_timer()
        history_multiple = autoencoder.fit([aFrame, Sigma], aFrame,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           shuffle=True,
                                           callbacks=[AnnealingCallback(MSE_weight, MSE_weight_lst),
                                                      plotCallback(aFrame=aFrame, Sigma=Sigma, lbls=lbls, encoder=encoder,
                                                                  ID=ID, bl=bl, epochs=epochs, output_dir=output_dir,
                                                        save_period=save_period,),
                                                      saveEncoder(encoder=encoder, ID=ID, bl=bl, epochs=epochs,
                                                                  output_dir=output_dir, save_period=save_period),
                                                       DCAEStop],

                                           verbose=2)
        stop = timeit.default_timer()
        z = encoder.predict([aFrame, Sigma])
        print(stop - start)

        encoder.save_weights(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
        autoencoder.save_weights(output_dir + '/autoencoder_' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
        np.savez(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz', z=z)
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
