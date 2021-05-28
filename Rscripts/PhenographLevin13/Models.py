# functions to fit deep autoencoder (DCAE)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import h5py
import timeit
from plotly.io import to_html
import plotly.io as pio
pio.renderers.default = "browser"

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from utils_evaluation import table,  plot3D_marker_colors, plot3D_cluster_colors


def frange_anneal(n_epoch, ratio=0.25, shape='sin'):
    L = np.ones(n_epoch)
    if ratio ==0:
        return L
    for c in range(n_epoch):
        if c <= np.floor(n_epoch*ratio):
            if shape=='sqrt':
                norm = np.sqrt(np.floor(n_epoch*ratio))
                L[c] = np.sqrt(c)/norm
            if shape=='sin':
                Om = (np.pi/2/(n_epoch*ratio))
                L[c] =  np.sin(Om*c)**4
        else:
            L[c]=1
    return L

class AnnealingCallback(Callback):
    def __init__(self, weight, kl_weight_lst):
        self.weight = weight
        self.kl_weight_lst = kl_weight_lst

    def on_epoch_end(self, epoch, logs={}):
        new_weight = K.eval(self.kl_weight_lst[epoch])
        K.set_value(self.weight, new_weight)
        print("  Current AP is " + str(K.get_value(self.weight)))

class EpochCounterCallback(Callback):
    def __init__(self, count, count_lst):
        self.count = count
        self.count_lst = count_lst

    def on_epoch_end(self, epoch, logs={}):
        new_count = K.eval(self.count_lst[epoch])
        K.set_value(self.count, new_count)
        #print("  Current AP is " + str(K.get_value(self.count)))

def fit_DCAE(data, epochs, coeffCAE,  Sigma, batch_size = 256, savePeriod = None, fileID='test', annealing_ratio = 0.95,
             output_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/simulatedData/output'):

    MMD_weight = K.variable(value=0)
    MMD_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=annealing_ratio)))

    nrow = data.shape[0]
    batch_size = 256
    latent_dim = 3
    original_dim = data.shape[1]
    intermediate_dim = original_dim * 3
    intermediate_dim2 = original_dim
    # var_dims = Input(shape = (original_dim,))
    #
    initializer = tf.keras.initializers.he_normal(12345)
    # initializer = None
    SigmaTsq = Input(shape=(1,))
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer=initializer)(x)
    h1 = Dense(intermediate_dim2, activation='relu', name='intermediate2', kernel_initializer=initializer)(h)
    z_mean = Dense(latent_dim, activation=None, name='z_mean', kernel_initializer=initializer)(h1)

    encoder = Model([x, SigmaTsq], z_mean, name='encoder')

    decoder_h = Dense(intermediate_dim2, activation='relu', name='intermediate3', kernel_initializer=initializer)
    decoder_h1 = Dense(intermediate_dim, activation='relu', name='intermediate4', kernel_initializer=initializer)
    decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer=initializer)
    h_decoded = decoder_h(z_mean)
    h_decoded2 = decoder_h1(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded2)
    autoencoder = Model(inputs=[x, SigmaTsq], outputs=x_decoded_mean)

    # Loss and optimizer ------------------------------------------------------
    # rewrite this based on recommendations here
    # https://www.tensorflow.org/guide/keras/train_and_evaluate

    normSigma = nrow / sum(1 / Sigma)

    lam = 1e-4

    def pot(alp, x):
        return np.select([(x < alp), (x >= alp) * (x <= 1), x > 1], [10, 0, 10])

    alp = 0.2

    def DCAE_loss(x, x_decoded_mean):  # attempt to avoid vanishing derivative of sigmoid
        U = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
        W = K.variable(value=encoder.get_layer('intermediate2').get_weights()[0])  # N x N_hidden
        Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
        U = K.transpose(U);
        W = K.transpose(W);
        Z = K.transpose(Z);  # N_hidden x N

        u = encoder.get_layer('intermediate').output
        du = tf.linalg.diag((tf.math.sign(u) + 1) / 2)
        m = encoder.get_layer('intermediate2').output
        dm = tf.linalg.diag((tf.math.sign(m) + 1) / 2)  # N_batch x N_hidden
        s = encoder.get_layer('z_mean').output

        r = tf.linalg.einsum('aj->a', s ** 2)

        ds = 500 * tf.math.square(tf.math.abs(alp - r)) * tf.dtypes.cast(tf.less(r, alp), tf.float32) + \
             (r ** 2 - 1) * tf.dtypes.cast(tf.greater_equal(r, 1), tf.float32) + 0.1
        # 0 * tf.dtypes.cast(tf.math.logical_and(tf.greater_equal(r, alp), tf.less(r, 1)), tf.float32) + \
        # ds = pot(0.1, r)
        S_0W = tf.einsum('akl,lj->akj', du, U)
        S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
        # tf.print((S_1W).shape) #[None, 120]
        S_2Z = tf.einsum('a,lj->alj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
        # tf.print((S_2Z).shape)
        diff_tens = tf.einsum('akl,alj->akj', S_2Z,
                              S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
        diff_tens = tf.einsum('akl,alj->akj', diff_tens, S_0W)
        # tf.Print(K.sum(diff_tens ** 2))
        return 1 / normSigma * (SigmaTsq) * lam * (K.sum(diff_tens ** 2))

    # 1000.0*  np.less(r, alp).astype(int)  + \
    #        0* (np.logical_and(np.greater_equal(r, alp), np.less(r, 1))).astype(int) + \
    #        1000.0* np.greater_equal(r, 1).astype(int)

    # mmd staff TODO: try approximation for this
    def compute_kernel(x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def compute_mmd(x, y):  # [batch_size, z_dim] [batch_size, z_dim]
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def loss_mmd(x, x_decoded_mean):
        batch_size = K.shape(z_mean)[0]
        latent_dim = K.int_shape(z_mean)[1]
        # true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.)
        true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=-1.5, maxval=1.5)
        # true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=0.0, maxval=1.0)
        return compute_mmd(true_samples, z_mean)

    def mean_square_error_NN(y_true, y_pred):
        # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
        msew = tf.keras.losses.mean_squared_error(y_true, y_pred)
        # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
        return tf.multiply(msew,
                           normSigma * 1 / SigmaTsq)  # TODO Sigma -denomiator or nominator? try reverse, schek hpw sigma computed in UMAP

    def ae_loss(weight, MMD_weight_lst):
        def loss(x, x_decoded_mean):
            msew = mean_square_error_NN(x, x_decoded_mean)
            # return msew + 1*(1-MMD_weight) * loss_mmd(x, x_decoded_mean) + (MMD_weight + coeffCAE) * DCAE_loss(x, x_decoded_mean) #TODO: try 1-MMD insted 2-MMD
            return msew + 1 * (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + (coeffCAE) * DCAE_loss(x,
                                                                                                      x_decoded_mean)
            # return K.mean(msew)

        return loss
        # return K.switch(tf.equal(Epoch_count, 10),  loss1(x, x_decoded_mean), loss1(x, x_decoded_mean))

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
    )

    autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst),
                        metrics=[DCAE_loss, loss_mmd, mean_square_error_NN])

    autoencoder.summary()

    if savePeriod != None:
        save_period =savePeriod

        class plotCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % save_period == 0 or epoch in range(200):
                    z = encoder.predict([data, Sigma])
                    fig = plot3D_cluster_colors(z, lbls=lbls)
                    html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                                       include_mathjax=False, post_script=None, full_html=True,
                                       animation_opts=None, default_width='100%', default_height='100%', validate=True)
                    html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
                    Html_file = open(html_dir + "/" + fileID + '_epoch=' + str(epoch) + '_' + "_Buttons.html", "w")
                    Html_file.write(html_str)
                    Html_file.close()

        callPlot = plotCallback()

        start = timeit.default_timer()
        history_multiple = autoencoder.fit([data, Sigma], data,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           shuffle=True,
                                           callbacks=[AnnealingCallback(MMD_weight, MMD_weight_lst),
                                                      callPlot], verbose=2)
        stop = timeit.default_timer()
    else:
        start = timeit.default_timer()
        history_multiple = autoencoder.fit([data, Sigma], data,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           shuffle=True,
                                           callbacks=[AnnealingCallback(MMD_weight, MMD_weight_lst)], verbose=2)
        stop = timeit.default_timer()

    z = encoder.predict([data, Sigma])

    print(stop - start)
    st = 10;
    stp = epochs
    fig01 = plt.figure();
    plt.plot(history_multiple.history['loss'][st:stp]);
    plt.title('loss')
    fig02 = plt.figure();
    plt.plot(history_multiple.history['DCAE_loss'][st:stp]);
    plt.title('DCAE_loss')
    fig03 = plt.figure();
    plt.plot(history_multiple.history['loss_mmd'][st:stp]);
    plt.title('loss_mmd')
    fig04 = plt.figure();
    plt.plot(history_multiple.history['mean_square_error_NN'][st:stp]);
    plt.title('mean_square_error')
    fig = plot3D_cluster_colors(z, lbls=lbls)
    fig.show()

    fig01 = plt.figure();
    plt.plot(history_multiple.history['loss'][st:stp], label='loss', c='red');
    plt.plot(history_multiple.history['DCAE_loss'][st:stp], label='DCAE_loss', c='green');
    plt.plot(history_multiple.history['loss_mmd'][st:stp], label='loss_mmd', c='blue');
    plt.plot(history_multiple.history['mean_square_error_NN'][st:stp], label='mean_square_error_NN', c='black');
    plt.legend(loc="upper right")

    encoder.save_weights(output_dir + '/' + fileID + '_3D.h5')
    autoencoder.save_weights(output_dir + '/autoencoder_' + fileID + '_3D.h5')
    np.savez(output_dir + '/' + fileID + '_latent_rep_3D.npz', z=z)
    return z



