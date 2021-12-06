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
from tensorflow.keras.callbacks import Callback, EarlyStopping
import pickle
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
#list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
#'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']
# TEMP
#list_of_inputs = 'Shenkareuclid_not_scaled.npz'

k = 30
k3 = k * 3
coeffCAE = 1
coeffMSE = 1 # normally 1, 5 for  Shekhar
epochs_list = [200, 500]
alp = 0.2
alp = 0.2
patience = 50
min_delta = 1e-5
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'CyTOFdataPreprocess/'
output_dir  = DATA_ROOT + 'Real_sets/DCAE_output/'
list_of_inputs = ['Levine32euclid_scaled_no_negative_removed.npz',
'Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz',  'Shenkareuclid_shifted.npz']
ID = 'ELU_sqrtD'

#load earlier preprocessed data

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()
#bl = list_of_inputs[0]
#epochs=500
for epochs in epochs_list:
    for bl in list_of_inputs:
        infile = source_dir + bl

        # markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
        # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
        npzfile = np.load(infile, allow_pickle=True)
        weight_distALL = npzfile['Dist'];
        # = weight_distALL[IDX,:]
        aFrame = npzfile['aFrame'];
        Dist = npzfile['Dist']
        Idx = npzfile['Idx']
        # cut_neibF = npzfile['cut_neibF'];
        # cut_neibF = cut_neibF[IDX,:]
        neibALL = npzfile['neibALL']
        # neibALL  = neibALL [IDX,:]
        # np.sum(cut_neibF != 0)
        # plt.hist(cut_neibF[cut_neibF!=0],50)
        Sigma = npzfile['Sigma']
        lbls = npzfile['lbls'];
        # neib_weight = npzfile['neib_weight']
        # [aFrame, neibF, cut_neibF, weight_neibF]
        # training set
        # targetTr = np.repeat(aFrame, r, axis=0)
        targetTr = aFrame
        neibF_Tr = neibALL
        weight_neibF_Tr = weight_distALL
        sourceTr = aFrame

        # Model-------------------------------------------------------------------------
        ######################################################
        # targetTr = np.repeat(aFrame, r, axis=0)

        # Model-------------------------------------------------------------------------
        ######################################################
        # targetTr = np.repeat(aFrame, r, axis=0)
        k = 30
        k3 = k * 3
        nrow = np.shape(aFrame)[0]
        # TODO try downweight mmd to the end of computation
        # DCAE_weight = K.variable(value=0)
        # DCAE_weight_lst = K.variable(np.array(frange_anneal(epochs, ratio=0)))
        # TODO: Possible bug in here ast ther end of training after the switch
        # check for possible discontinuities/singularities o last epochs, is shape of potenatial  to narroe at the end?

        MMD_weight = K.variable(value=0)

        MMD_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=1)))

        batch_size = 256
        latent_dim = 3
        original_dim = aFrame.shape[1]
        intermediate_dim = original_dim * 3
        intermediate_dim2 = original_dim * 2
        # intermediate_dim = original_dim
        # intermediate_dim2 = original_dim

        # var_dims = Input(shape = (original_dim,))
        #
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

        # normSigma = nrow / sum(1 / Sigma)
        normSigma = 1
        lam = 10

        def DCAE_loss(x, x_decoded_mean):  # attempt to avoid vanishing derivative of sigmoid
            U = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
            W = K.variable(value=encoder.get_layer('intermediate2').get_weights()[0])  # N x N_hidden
            Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
            U = K.transpose(U);
            W = K.transpose(W);
            Z = K.transpose(Z);  # N_hidden x N

            # derivative of leaky relu

            # relu
            def relu_derivative(a):
                cond = tf.math.greater_equal(a, tf.constant(0.0))
                return tf.where(cond, tf.constant(1.0), tf.constant(0.0))

            # elu
            def elu_derivative(a):
                cond = tf.math.greater_equal(a, tf.constant(0.0))
                return tf.where(cond, tf.constant(1.0), tf.math.exp(a))

            # def leaky relu
            def leaky_relu_derivative(a):
                cond = tf.math.greater_equal(a, tf.constant(0.0))
                return tf.where(cond, tf.constant(1.0), tf.constant(0.3))

            u = encoder.get_layer('intermediate').output
            du = elu_derivative(u)
            m = encoder.get_layer('intermediate2').output
            dm = elu_derivative(m)  # N_batch x N_hidden
            s = encoder.get_layer('z_mean').output
            ds = elu_derivative(s)  # N_batch x N_hidden

            r = tf.linalg.einsum('aj->a', s ** 2)  # R^2 in reality. to think this trough

            pot = 50 * tf.math.square(alp - r) * tf.dtypes.cast(tf.less(r, alp), tf.float32) + \
                  5 * (r ** 2 - 1) * tf.dtypes.cast(tf.greater_equal(r, 1), tf.float32) + 0.1
            dsU = tf.einsum('ak,a->ak', ds, pot)

            # 0 * tf.dtypes.cast(tf.math.logical_and(tf.greater_equal(r, alp), tf.less(r, 1)), tf.float32) + \
            # ds = pot(0.1, r)
            S_0W = tf.einsum('al,lj->alj', du, U)
            S_1W = tf.einsum('al,lj->alj', dm, W)  # N_batch x N_input ??
            # tf.print((S_1W).shape) #[None, 120]
            S_2Z = tf.einsum('al,lj->alj', dsU, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
            # tf.print((S_2Z).shape)
            diff_tens = tf.einsum('akl,alj->akj', S_2Z,
                                  S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
            diff_tens = tf.einsum('akl,alj->akj', diff_tens, S_0W)
            # f = tf.where(K.abs(diff_tens) > 0, K.abs(diff_tens), 0.0)# to prevent nans in loss
            # return tf.transpose(1 / normSigma * SigmaTsq * lam) * K.sum(K.sqrt(K.abs(diff_tens)), axis=[1, 2])
            # return  (1 / normSigma * SigmaTsq * lam)[0,:] * K.sum(diff_tens**2, axis=[1, 2])
            return (1 / normSigma * SigmaTsq * lam)[0, :] * K.sum(K.sqrt(K.abs(diff_tens) + 1e-9), axis=[1, 2])
            # return tf.transpose(1 / normSigma * SigmaTsq * lam) * K.sum(K.abs(diff_tens), axis=[1, 2])


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

        # experimensting with matching distributions in high and different dimensions
        # def compute_mmd_HDLD(x, y):  # [batch_size, z_dim] [batch_size, z_dim]
        #     x_kernel = compute_kernel(x, x)
        #     y_kernel = compute_kernel_Y(y, y)
        #     xy_kernel = compute_kernel_XY(x, y)
        #     return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)






        def loss_mmd(x, x_decoded_mean):
            batch_size = K.shape(z_mean)[0]
            latent_dim = K.int_shape(z_mean)[1]
            # true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.)
            true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=-1.5, maxval=1.5)
            # true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval=0.0, maxval=1.0)
            return compute_mmd(true_samples, z_mean)

        #
        # y_true = np.random.normal(loc=0, scale=0, size=(250, 30))
        # y_pred = np.random.normal(loc=0, scale=0, size=(250, 30))
        def mean_square_error_NN(y_true, y_pred):
            # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
            msew = tf.keras.losses.mean_squared_error(y_true, y_pred)
            # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
            return tf.multiply(tf.transpose(normSigma * 1 / SigmaTsq),
                               msew)  # TODO Sigma -denomiator or nominator? try reverse, schek hpw sigma computed in UMAP


        def ae_loss(weight, MMD_weight_lst):
            def loss(x, x_decoded_mean):
                msew = mean_square_error_NN(x, x_decoded_mean)
                #return coeffMSE * msew + (1 - MMD_weight) * loss_mmd(x, x_decoded_mean)
                #return coeffMSE * msew + (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + (MMD_weight + coeffCAE) * DCAE_loss(x, x_decoded_mean)
                return coeffMSE * msew + loss_mmd(x, x_decoded_mean) + (
                            MMD_weight + coeffCAE) * DCAE_loss(x, x_decoded_mean)
            return loss
            # return K.switch(tf.equal(Epoch_count, 10),  loss1(x, x_decoded_mean), loss1(x, x_decoded_mean))


        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        )

        autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst),
                            metrics=[DCAE_loss, loss_mmd, mean_square_error_NN])

        autoencoder.summary()
        # opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
        # opt=tf.keras.optimizers.Adam(
        #    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        # )
        # autoencoder.compile(optimizer=opt, loss=ae_loss(MMD_weight, MMD_weight_lst), metrics=[DCAE_loss, loss_mmd,  mean_square_error_NN])

        save_period = 10

        DCAEStop = EarlyStopping(monitor='DCAE_loss', min_delta=min_delta, patience=patience, mode = 'min',
                                 restore_best_weights =False)

        class plotCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % save_period == 0 or epoch in range(0,20):
                    z = encoder.predict([aFrame, Sigma])
                    fig = plot3D_cluster_colors(z, lbls=lbls)
                    html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                                       include_mathjax=False, post_script=None, full_html=True,
                                       animation_opts=None, default_width='100%', default_height='100%', validate=True)
                    html_dir = output_dir
                    Html_file = open(html_dir + "/" + ID + "_" + str(bl) + 'epochs'+str(epochs)+ '_epoch=' + str(epoch) + '_' + "_Buttons.html", "w")
                    Html_file.write(html_str)
                    Html_file.close()

        class saveEncoder(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % save_period == 0 and epoch != 0:
                    encoder.save_weights(output_dir + '/' + ID + "_encoder_" + str(bl) + 'epochs' + str(epochs) + '_epoch=' + str(epoch) + '_3D.h5')


        callPlot = plotCallback()
        callSave = saveEncoder()

        from sklearn.model_selection import train_test_split

        #X_train, X_val, Sigma_train, Sigma_val = train_test_split(
        #    aFrame, Sigma, test_size=0.25, random_state=0,  shuffle= True)



        start = timeit.default_timer()
        history_multiple = autoencoder.fit([aFrame, Sigma], aFrame,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           shuffle=True,
                                           callbacks=[AnnealingCallback(MMD_weight, MMD_weight_lst),
                                                      callSave, callPlot, DCAEStop],
                                           verbose=2)
        stop = timeit.default_timer()
        z = encoder.predict([aFrame, Sigma])
        print(stop - start)

        encoder.save_weights(output_dir + '/' + ID + "_" + str(bl)+ 'epochs'+str(epochs) + '_3D.h5')
        autoencoder.save_weights(output_dir + '/autoencoder_' + ID + "_" + str(bl) + 'epochs'+str(epochs)+ '_3D.h5')
        np.savez(output_dir + '/' + ID + "_" + str(bl) + 'epochs'+str(epochs)+ '_latent_rep_3D.npz', z=z)
        #np.savez(output_dir + '/' + str(bl) + 'epochs'+str(epochs)+ '_history.npz', history_multiple)
        with open(output_dir + '/' + ID + str(bl) + 'epochs'+str(epochs)+ '_history', 'wb') as file_pi:
            pickle.dump(history_multiple.history , file_pi)

        fig = plot3D_cluster_colors(z, lbls=lbls)
        html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                           include_mathjax=False, post_script=None, full_html=True,
                           animation_opts=None, default_width='100%', default_height='100%', validate=True)
        html_dir = output_dir
        Html_file = open(html_dir + "/" + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_epoch=' + 'Final' + '_' + "_Buttons.html", "w")
        Html_file.write(html_str)
        Html_file.close()

    # import umap
    # # experiment with umap. Neede if some post-[rpcessing the output gives better plot
    # # Initial idea run umap on z few epochs to see if splitting is cured
    # from scipy.optimize import curve_fit
    #
    #
    # def find_ab_params(spread, min_dist):
    #     """Fit a, b params for the differentiable curve used in lower
    #     dimensional fuzzy simplicial complex construction. We want the
    #     smooth curve (from a pre-defined family with simple gradient) that
    #     best matches an offset exponential decay.
    #     """
    #
    #     def curve(x, a, b):
    #         return 1.0 / (1.0 + a * x ** (2 * b))
    #
    #     xv = np.linspace(0, spread * 3, 300)
    #     yv = np.zeros(xv.shape)
    #     yv[xv < min_dist] = 1.0
    #     yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    #     params, covar = curve_fit(curve, xv, yv)
    #     return params[0], params[1]
    #
    #
    # fss, _, _ = umap.umap_.fuzzy_simplicial_set(aFrame, n_neighbors=15, random_state=1,
    #                                             metric='euclidean', metric_kwds={},
    #                                             knn_indices=Idx[:, 0:90], knn_dists=Dist[:, 0:90],
    #                                             angular=False, set_op_mix_ratio=1.0, local_connectivity=1.0,
    #                                             apply_set_operations=True, verbose=True)
    #
    # spread, min_dist = 1, 0.001
    # a, b = find_ab_params(spread, min_dist)
    # zzz = umap.umap_.simplicial_set_embedding(aFrame, fss, n_components=3, n_epochs=50, init=z,
    #                                           random_state=np.random.RandomState(1),
    #                                           initial_alpha=0.1, gamma=0.1, negative_sample_rate=1, a=a, b=b,
    #                                           densmap=False,
    #                                           densmap_kwds={'lambda': 2.0, 'frac': 0.3, 'var_shift': 0.1,
    #                                                         'n_neighbors': 15, "graph_dists": Dist}, output_dens=False,
    #                                           metric='euclidean', metric_kwds={})
    #
    # fig = plot3D_cluster_colors(zzz[0], lbls=lbls)
    # fig.show()
    #
    # fig0 = plot3D_cluster_colors(z, lbls=lbls)
    # fig0.show()
    #     bl = list_of_inputs[2]
    #
    #     npzfile = np.load(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_latent_rep_3D.npz')
    #     z = npzfile['z']
    #
    #     history = pickle.load(open(output_dir + '/'  + ID + str(bl) + 'epochs'+str(epochs)+ '_history',  "rb"))
    #
    #
    #     st = 50;
    #     stp = len(history['loss'])
    #     fig01 = plt.figure();
    #     plt.plot(history['loss'][st:stp]);
    #     plt.title('loss')
    #     fig02 = plt.figure();
    #     plt.plot(history['DCAE_loss'][st:stp]);
    #     plt.title('DCAE_loss')
    #     fig03 = plt.figure();
    #     plt.plot(history['loss_mmd'][st:stp]);
    #     plt.title('loss_mmd')
    #     fig04 = plt.figure();
    #     plt.plot(history['mean_square_error_NN'][st:stp]);
    #     plt.title('mean_square_error')
    #     fig = plot3D_cluster_colors(z, lbls=lbls)
    #     fig.show()
    #
    # autoencoder.load_weights(output_dir + '/autoencoder_' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
    # A_rest = autoencoder.predict([aFrame, Sigma, ])
    # import seaborn as sns
    #
    # lblsC = [7 if i == -7 else i for i in lbls]
    # l_list = np.unique(lblsC)
    # fig, axs = plt.subplots(nrows=8)
    # yl = aFrame.min()
    # yu = aFrame.max()
    # for i in l_list:
    #     sns.violinplot(data=A_rest[lblsC == i, :], ax=axs[int(i)])
    #     axs[int(i)].set_ylim(yl, yu)
    #     axs[int(i)].set_title(str(int(i)), rotation=-90, x=1.05, y=0.5)
    # fig.savefig(PLOTS + 'Sensitivity/' + str(bl) + "Signal_violinplot" + ".eps", format='eps', dpi=350)
    # plt.close()
    #
    # sns.violinplot(data=A_rest)
    # sns.violinplot(data=aFrame)
