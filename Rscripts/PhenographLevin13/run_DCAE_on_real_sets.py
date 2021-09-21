'''
Runs DCAE
mappings fro Leine32, same netweork as fro art sets
'''

import timeit
import numpy as np
import plotly.io as pio
import tensorflow as tf
from plotly.io import to_html
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

pio.renderers.default = "browser"

from utils_evaluation import plot3D_cluster_colors

from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

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

import ctypes
from numpy.ctypeslib import ndpointer
lib = ctypes.cdll.LoadLibrary("/home/grinek/PycharmProjects/BIOIBFO25L/Clibs/perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #Sigma
                ctypes.c_size_t]


k = 30
k3 = k * 3
coeffCAE = 1
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'CyTOFdataPreprocess/'
output_dir  = DATA_ROOT + 'Real_sets/DCAE_output/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']
# TEMP
#list_of_inputs = 'Shenkareuclid_not_scaled.npz'


#load earlier preprocessed data

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()
#bl = list_of_inputs[0]
epochs_list = [100, 200, 500, 1000]
for epochs in epochs_list:
    for bl in list_of_inputs:
        infile = source_dir + bl
        #markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
        # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
        npzfile = np.load(infile,  allow_pickle=True)
        # = weight_distALL[IDX,:]
        aFrame = npzfile['aFrame'];
        #cut_neibF = npzfile['cut_neibF'];
        #cut_neibF = cut_neibF[IDX,:]
        #neibALL  = neibALL [IDX,:]
        #np.sum(cut_neibF != 0)
        # plt.hist(cut_neibF[cut_neibF!=0],50)
        Sigma = npzfile['Sigma']
        lbls = npzfile['lbls'];
        #neib_weight = npzfile['neib_weight']
        k = 30
        nrow= np.shape(aFrame)[0]
        # TODO try downweight mmd to the end of computation
        #DCAE_weight = K.variable(value=0)
        #DCAE_weight_lst = K.variable(np.array(frange_anneal(epochs, ratio=0)))
        # TODO: Possible bug in here ast ther end of training after the switch
        # check for possible discontinuities/singularities o last epochs, is shape of potenatial  to narroe at the end?

        MMD_weight = K.variable(value=0)

        MMD_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=1)))

        batch_size = 256
        latent_dim = 3
        original_dim = aFrame.shape[1]
        intermediate_dim = original_dim * 3
        intermediate_dim2 = original_dim * 2 #corrected wideness to be the same with artificial sets
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

        #def pot(alp, x):
        #    return np.select([(x < alp), (x >= alp) * (x <= 1), x > 1], [10, 0, 10])
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
            true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.)
            #true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=-1.5, maxval=1.5)
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
                #return msew + 1 * (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + 1 * (MMD_weight + coeffCAE) * DCAE_loss(
                #    x, x_decoded_mean)
                return  (1 - MMD_weight+0.1) * (loss_mmd(x, x_decoded_mean)+msew) + 1 * (MMD_weight + coeffCAE) * DCAE_loss(
                    x, x_decoded_mean)

            return loss
            # return K.switch(tf.equal(Epoch_count, 10),  loss1(x, x_decoded_mean), loss1(x, x_decoded_mean))


        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        )

        autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst), metrics=[DCAE_loss, loss_mmd, mean_square_error_NN])

        autoencoder.summary()
        # opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
        # opt=tf.keras.optimizers.Adam(
        #    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        # )
        # autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst), metrics=[DCAE_loss, loss_mmd,  mean_square_error_NN])

        from tensorflow.keras.callbacks import Callback

        save_period = 10


        class plotCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % save_period == 0 or epoch in range(20):
                    z = encoder.predict([aFrame, Sigma])
                    fig = plot3D_cluster_colors(z, lbls=lbls)
                    html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                                       include_mathjax=False, post_script=None, full_html=True,
                                       animation_opts=None, default_width='100%', default_height='100%', validate=True)
                    html_dir = output_dir
                    Html_file = open(html_dir + "/" + str(bl) +'epochs'+str(epochs)+ '_epoch=' + str(epoch) + '_' + "Buttons.html", "w")
                    Html_file.write(html_str)
                    Html_file.close()


        callPlot = plotCallback()

        start = timeit.default_timer()
        history_multiple = autoencoder.fit([aFrame, Sigma], aFrame,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           shuffle=True,
                                           callbacks=[AnnealingCallback(MMD_weight, MMD_weight_lst),
                                                      callPlot], verbose=2)
        stop = timeit.default_timer()
        z = encoder.predict([aFrame, Sigma])


        print(stop - start)
        # st = 5;
        # stp = epochs
        # fig01 = plt.figure();
        # plt.plot(history_multiple.history['loss'][st:stp]);
        # plt.title('loss')
        # fig02 = plt.figure();
        # plt.plot(history_multiple.history['DCAE_loss'][st:stp]);
        # plt.title('DCAE_loss')
        # fig03 = plt.figure();
        # plt.plot(history_multiple.history['loss_mmd'][st:stp]);
        # plt.title('loss_mmd')
        # fig04 = plt.figure();
        # plt.plot(history_multiple.history['mean_square_error_NN'][st:stp]);
        # plt.title('mean_square_error')
        # fig = plot3D_cluster_colors(z, lbls=lbls)
        # fig.show()

        encoder.save_weights(output_dir + '/' + str(bl) +'epochs'+str(epochs)+ '_3D.h5')
        autoencoder.save_weights(output_dir + '/autoencoder_' + str(bl) +'epochs'+str(epochs)+ '_3D.h5')
        np.savez(output_dir + '/' + str(bl) + 'epochs'+str(epochs)+ '_latent_rep_3D.npz', z=z)
        #np.savez(output_dir + '/' + str(bl) + 'epochs' + str(epochs) + '_history.npz', history_multiple)

        '''
        encoder.load_weights(output_dir + '/'+ str(bl) +'epochs'+str(epochs)+ '_3D.h5')
        autoencoder.load_weights(output_dir + '/autoencoder_' + str(bl) +'epochs'+str(epochs)+ '_3D.h5')
        encoder.summary()
        z = encoder.predict([aFrame, Sigma, ])
    
        from keract import get_activations, display_activations
        from random import random
        def decision(probability):
            return random() < probability
    
        ns_sample= 10
        l_list = np.unique(lbls)
        size_list = [sum(lbls == l) for l in l_list]
        zip_iterator = zip(l_list, size_list)
        size_dict = dict(zip_iterator)
        indx = [True if size_dict[l] <= ns_sample else decision(ns_sample / size_dict[l]) for l in lbls]
        indx = np.arange(len(lbls))[indx]
        table(lbls[indx])
        activation_lists = [[], [], []]
        for j in range(len(indx)):
            keract_inputs = [aFrame[indx[j]:(indx[j]+1), :], Sigma[indx[j]:(indx[j]+1)] ]
        #keract_targets = target_test[:1]
            activations = get_activations(encoder,keract_inputs, layer_names=['input_2', 'intermediate', 'intermediate2'])
            activation_lists[0].append(list(activations['input_2']))
            activation_lists[1].append(list(activations['intermediate']))
            activation_lists[2].append(list(activations['intermediate2']))
            #display_activations(activations, cmap="gray", save=True, directory = output_dir + '/autoencoder_' + str(bl))
    
        #stack lists into matrices of activations
        activations_matrices = [np.array(i).squeeze() for i in activation_lists]
        '''
