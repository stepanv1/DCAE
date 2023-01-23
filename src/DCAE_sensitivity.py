'''
DCAE sensitivity analysis in artificial sets
'''

import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

pio.renderers.default = "browser"
import pandas as pd
from utils_model import frange_anneal, elu_derivative, linear_derivative

from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
output_dir  = DATA_ROOT + 'Artificial_sets/DCAE_output/'
sensitivity_dir = DATA_ROOT + 'Artificial_sets/Sensitivity/'
PLOTS = DATA_ROOT + "Artificial_sets/PLOTS/"
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
#load  generated data
k = 30
coeffCAE = 1
coeffMSE = 1
batch_size = 128
lam = 0.1
alp = 0.5
m = 10
patience = 500
min_delta = 1e-4
g=0#0.1
ID = 'DCAE' +  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)#ID = 'clip_grad_exp_MDS' + '_g_'  + str(g) +  '_lam_'  + str(lam) + '_batch_' + str(batch_size) + '_alp_' + str(alp) + '_m_' + str(m)

epochs = 500

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

#tf.compat.v1.disable_eager_execution() # commented out  to use to_numpy function of tf
#bl = list_of_branches[24]
for bl in list_of_branches:
    print(bl)
    infile = source_dir + 'set_'+ str(bl)+'.npz'
    npzfile = np.load(infile)
    weight_distALL = npzfile['Dist'];
    aFrame = npzfile['aFrame'];
    Dist = npzfile['Dist']
    Idx = npzfile['Idx']
    neibALL = npzfile['neibALL']
    Sigma = npzfile['Sigma']
    lbls = npzfile['lbls'];


    # Model-------------------------------------------------------------------------
    nrow = np.shape(aFrame)[0]

    from numpy import nanmax, argmax, unravel_index
    from scipy.spatial.distance import pdist, squareform

    IDX = np.random.choice(nrow, size=2000, replace=False)
    D = pdist(aFrame[IDX, :])
    D = squareform(D);
    max_dist, [I_row, I_col] = nanmax(D), unravel_index(argmax(D), D.shape)
    np.fill_diagonal(D, 1000)
    min_dist = np.min(D)
    np.fill_diagonal(D, 0)
    mean_dist = np.mean(D)

    MSE_weight = K.variable(value=0)
    MSE_weight_lst = K.variable(np.array(frange_anneal(int(epochs), ratio=1)))

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

    def compute_graph_weights_Inp(x):
        x_size = tf.shape(x)[0]
        dim = tf.shape(x)[1]
        # from w_ij = kernel((x_i-x_j)/sigma_i) and then symmetrized
        # x=x/max_dist
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, x_size, 1]))
        tiled_y = tf.tile(tf.reshape(x, tf.stack([1, x_size, dim])), tf.stack([x_size, 1, 1]))
        D = tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2)  # / tf.cast(x_size**2, tf.float32)
        # apply monotone f to sqeeze  and normalize
        D = K.sqrt(D)
        D = 2 * (D - min_dist) / (max_dist)
        D = tf.linalg.set_diag(D, tf.zeros(x_size), name=None)
        return D
        # no log version


    def compute_graph_weights_enc(x):
        x_size = tf.shape(x)[0]
        dim = tf.shape(x)[1]
        # x = tf.linalg.normalize(x, ord=2, axis=1)[0]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, x_size, 1]))
        tiled_y = tf.tile(tf.reshape(x, tf.stack([1, x_size, dim])), tf.stack([x_size, 1, 1]))
        D = tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2)
        D = K.sqrt(D)
        # D = tf.linalg.set_diag(D, tf.zeros(x_size), name=None)# / tf.cast(dim, tf.float32))
        return D


    # no log version
    def graph_diff(x, y):
        # pot = tf.math.square(r - 1)
        return g * K.sqrt(
            K.sum(K.square(1 - K.exp(compute_graph_weights_Inp(X) - compute_graph_weights_enc(z_mean)))) / tf.cast(
                batch_size ** 2, tf.float32))


    def compute_kernel(x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


    def compute_mmd(x, y):
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
        # sns.displot( np.power(np.random.uniform(a ** 3, b ** 3, 50000), 1 / 3))
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
        return normSigma / SigmaTsq[:, 0] * msew_nw
        # return msew_nw


    def ae_loss(weight, MMD_weight_lst):
        def loss(y_true, y_pred):
            msew = mean_square_error_NN(y_true, y_pred)
            # return coeffMSE * msew + (1 - MMD_weight) * loss_mmd(x, x_decoded_mean)
            # return coeffMSE * msew + (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + (MMD_weight + coeffCAE) * DCAE_loss(x, x_decoded_mean)
            # return coeffMSE * msew + 0.5 * (2 - MMD_weight) * loss_mmd(x, x_decoded_mean)
            return coeffMSE * (1 - MSE_weight + 0.1) * msew + 0.5 * (MSE_weight + 1) * loss_mmd(y_true, y_pred) + (
                    2 * MSE_weight + 0.1) * (DCAE_loss(y_true, y_pred))
            # return  loss_mmd(x, x_decoded_mean)

        return loss
        # return K.switch(tf.equal(Epoch_count, 10),  loss1(x, x_decoded_mean), loss1(x, x_decoded_mean))


    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, clipvalue=1.0
    )

    autoencoder.compile(optimizer=opt, loss=ae_loss(MSE_weight, MSE_weight_lst),
                        metrics=[DCAE_loss,  loss_mmd, mean_square_error_NN])

    autoencoder.summary()


    encoder.load_weights(output_dir + '/' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
    #utoencoder.load_weights(output_dir + '/autoencoder_' + ID + "_" + str(bl) + 'epochs' + str(epochs) + '_3D.h5')
    encoder.summary()
    z = encoder.predict([aFrame, Sigma ])
    lbls = [i if i != -7 else 7 for i in lbls]

    # to remove Sigma layer will need re-create encoder
    encoder2 = Model([X], z_mean, name='encoder2')
    encoder2.summary()

    x_tensor = tf.convert_to_tensor(aFrame, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as t:
        t.watch(x_tensor)
        output = encoder2(x_tensor)

        result = output
        gradients0 = t.gradient(output[:, 0], x_tensor).numpy()
        gradients1 = t.gradient(output[:, 1], x_tensor).numpy()
        gradients2 = t.gradient(output[:, 2], x_tensor).numpy()

    import seaborn as sns

    #Mean square elasticity
    l_list = np.unique(lbls)
    #n_z = normalize(z)
    SCgrad = np.sqrt(gradients0 ** 2 + gradients1 ** 2 + gradients2 ** 2)

    from scipy.stats import iqr
    SC = np.vstack(
        [SCgrad[lbls == i, :] / iqr(aFrame[lbls == i, :], axis=0) for i in l_list])

    SC_mean = np.stack(
        [np.median(SCgrad[lbls == i, :] /iqr(aFrame[lbls == i, :], axis=0),
                   axis=0) for i in l_list], axis=0)

    # plot a heatmap of this and peform  statistical tests, showing that data is interpreted correctly,
    # removing influence of non-informative dimensions

    SC_mean_norm =SC_mean
    plt.figure(figsize=(14, 10))
    g =  sns.heatmap(SC_mean_norm, center=0.1, linewidths=.2, cmap="GnBu",  annot=True, fmt='1.2f',  annot_kws={"fontsize":8})
    plt.savefig(PLOTS+'Sensitivity/' + ID +'_'+str(bl)+ 'epochs' + str(epochs) + "Sensitivity_heatmap"+".tif", format='tif', dpi=350)
    plt.close()


    fig, axs = plt.subplots(nrows=8)
    yl = SC.min()
    yu = SC.max()
    for i in l_list:
        sns.violinplot(data=SC[lbls == i, :], ax=axs[int(i)])
        axs[int(i)].set_ylim(yl, yu)
        axs[int(i)].set_title(str(int(i)), rotation=-90, x=1.05, y =0.5)
    fig.savefig(PLOTS+'Sensitivity/' + ID +'_'+ str(bl)+ 'epochs' + str(epochs) + "Sensitivity_violinplot"+".eps", format='eps', dpi=350)
    fig.savefig(
        PLOTS + 'Sensitivity/' + ID + '_' + str(bl) + 'epochs' + str(epochs) + "Sensitivity_violinplot" + ".tif",  format='tif', dpi=350)
    plt.close()

    fig, axs = plt.subplots(nrows=8)
    yl = aFrame.min()
    yu = aFrame.max()
    for i in l_list:
        sns.violinplot(data=aFrame[lbls == i, :], ax=axs[int(i)])
        axs[int(i)].set_ylim(yl, yu)
        axs[int(i)].set_title(str(int(i)), rotation=-90, x=1.05, y =0.5)
    fig.savefig(PLOTS + 'Sensitivity/' + ID +'_'+ str(bl) + "Signal_violinplot" + ".eps", format='eps', dpi=350)
    fig.savefig(PLOTS + 'Sensitivity/' + ID + '_' + str(bl) + "Signal_violinplot" + ".tif", format='tif', dpi=350)
    plt.close()

    from scipy.stats import brunnermunzel

    #create list of column names
    inform_dim_list = np.arange(0, 5)
    noisy_dim_list = np.arange(5, 30)
    col_names = ['bl', 'cluster']
    dim_comb =   sum([[str(dim_inf)+' '+ str(dim) for dim in noisy_dim_list] for dim_inf in inform_dim_list], [])
    col_names = col_names + dim_comb

    Pvals = []
    for i in l_list[0:7]:
       #iterate over all couples of informative versus non-informative
       #U1, p = mannwhitneyu(SC[lbls == i, 26], SC[lbls == i, 17], method="asymptotic")
       inform_dim_list  = np.arange(0,5)
       noisy_dim_list  = np.arange(5,30)
       test_res =  [[ brunnermunzel(SC[lbls == i, dim_inf], SC[lbls == i, dim],
                     alternative='greater', distribution='normal', nan_policy='propagate')
                     for dim in  noisy_dim_list] for dim_inf in inform_dim_list ]

       test_res = [res[1]  for res in  sum(test_res, [])]
       test_res = [bl, i] + test_res

       Pvals.append(test_res)

    df = pd.DataFrame(Pvals, columns=col_names)

    #8th cluster
    Pvals = []
    inform_dim_list = np.insert(np.arange(26,30), 0, 4, axis=0)
    noisy_dim_list = np.insert(np.arange(5, 26), 0, np.arange(0, 4), axis=0)
    col_names = ['bl', 'cluster']
    dim_comb = sum([[str(dim_inf) + ' ' + str(dim) for dim in noisy_dim_list] for dim_inf in inform_dim_list], [])
    col_names = col_names + dim_comb

    i = l_list[7]
    test_res = [[brunnermunzel(SC[lbls == i, dim_inf], SC[lbls == i, dim],
                               alternative='greater', distribution='normal', nan_policy='propagate')
                 for dim in noisy_dim_list] for dim_inf in inform_dim_list]

    test_res = [res[1] for res in sum(test_res, [])]
    test_res = [bl, i] + test_res

    Pvals.append(test_res)
    df7 = pd.DataFrame(Pvals, columns=col_names)

    outfile = sensitivity_dir + ID +'_'+   str(bl) + 'epochs' + str(epochs) + '_df_sensitivity_table_experiment.csv'
    df.to_csv(outfile,encoding ='utf-8')
    outfile = sensitivity_dir  + ID +'_'+  str(bl) + 'epochs' + str(epochs) + '_df7_sensitivity_table_experiment.csv'
    df7.to_csv(outfile,encoding ='utf-8')

#combine dataframes and create summary information on sensitivity in all 25 clusters
df_all = list()
df7_all = list()
#bl = list_of_branches[0]
for bl in list_of_branches:
    outfile = sensitivity_dir  + ID + '_'+  str(bl) + 'epochs' + str(epochs) + '_df7_sensitivity_table_experiment.csv'
    df7  = pd.read_csv(outfile).iloc[: , 1:]
    outfile = sensitivity_dir + ID + '_'+   str(bl) + 'epochs' + str(epochs) + '_df_sensitivity_table_experiment.csv'
    df = pd.read_csv(outfile).iloc[: , 1:]
    df_all.append(df)
    df7_all.append(df7)
df7_all = pd.concat(df7_all)
df_all = pd.concat(df_all)

#total summ of pairs where p<1e-10 in pentagone clusters
df7_num = df7_all.iloc[:, 2:].astype('float64').to_numpy()
np.sum(df7_num>1e-10)
np.sum(df7_num>1e-10)/df7_num.shape[0]/df7_num.shape[1]
#  0.1424
df_num = df_all.iloc[:, 2:].astype('float64').to_numpy()
np.sum(df_num>1e-10)
np.sum(df_num>1e-10)/df_num.shape[0]/df_num.shape[1]
# 0.0800

#exclude from the above calculation dimensions taking part in defining pentagone, namely dim 4
# dim 4 is not used, since its mode is the same across all clusters
inform_dim_list = np.arange(0, 4)
noisy_dim_list = np.arange(5, 30)
col_names = ['bl', 'cluster']
dim_comb = sum([[str(dim) + ' ' + str(dim_noi) for dim in  inform_dim_list ] for dim_noi in noisy_dim_list], [])
col_names = col_names + dim_comb

#df_num_pentagon_dims = df_all[col_names[1:10]]
# stats without 4th dimension
df_all_no4 = df_all[col_names]
df_num = df_all[dim_comb].astype('float64').to_numpy()
np.sum(df_num>1e-10)
np.sum(df_num>1e-10)/df_num.shape[0]/df_num.shape[1]
# 0.0718

# remove 4th dim as informative from the 7th cluster
df7_no4 = df7_all.loc[:, ~df7_all.columns.str.contains('4 ')]
df7_num = df7_no4.iloc[:, 2:].astype('float64').to_numpy()
np.sum(df7_num>1e-10)
np.sum(df7_num>1e-10)/df7_num.shape[0]/df7_num.shape[1]
#  0.1628
