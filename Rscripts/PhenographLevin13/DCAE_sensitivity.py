'''
DCAE
explains by clusters
Deep Taylor decomposition
https://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.4%20Deep%20Taylor%20Decomposition%20%281%29.ipynb
Look there for code: Constrained input space X=Rd+ and the z+-rule
Sensitivity
https://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.1%20Sensitivity%20Analysis.ipynb
SA_scores = [tf.square(tf.gradients(logits[:,i], X)) for i in range(10)]
Newer methods
https://github.com/PAIR-code/saliency
Patternet
https://github.com/albermax/innvestigate

'''

import timeit

import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import tensorflow as tf
from plotly.io import to_html
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import keract
pio.renderers.default = "browser"
import pandas as pd

from utils_evaluation import plot3D_cluster_colors

from pathos import multiprocessing
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

from tensorflow.keras.callbacks import Callback


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


k = 30
k3 = k * 3
coeffCAE = 1
epochs = 50
DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'Artificial_sets/Art_set25/'
output_dir  = DATA_ROOT + 'Artificial_sets/DCAE_output/'
sensitivity_dir = DATA_ROOT + 'Artificial_sets/Sensitivity/'
PLOTS = DATA_ROOT + "Artificial_sets/PLOTS/"
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
#load earlier generated data

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
#tf.compat.v1.disable_eager_execution() disabled to use to_numpy function of tf
#bl = list_of_branches[1]
for bl in list_of_branches:

    infile = source_dir + 'set_'+ str(bl)+'.npz'
    #markers = pd.read_csv(source_dir + "/Levine32_data.csv" , nrows=1).columns.to_list()
    # np.savez(outfile, weight_distALL=weight_distALL, cut_neibF=cut_neibF,neibALL=neibALL)
    npzfile = np.load(infile)
    weight_distALL = npzfile['Dist'];
    # = weight_distALL[IDX,:]
    aFrame = npzfile['aFrame'];
    Dist = npzfile['Dist']
    Idx = npzfile['Idx']
    #cut_neibF = npzfile['cut_neibF'];
    #cut_neibF = cut_neibF[IDX,:]
    neibALL = npzfile['neibALL']
    #neibALL  = neibALL [IDX,:]
    #np.sum(cut_neibF != 0)
    # plt.hist(cut_neibF[cut_neibF!=0],50)
    Sigma = npzfile['Sigma']
    lbls = npzfile['lbls'];
    #neib_weight = npzfile['neib_weight']
    # [aFrame, neibF, cut_neibF, weight_neibF]
    # training set
    # targetTr = np.repeat(aFrame, r, axis=0)
    targetTr = aFrame
    neibF_Tr = neibALL
    weight_neibF_Tr =weight_distALL
    sourceTr = aFrame




    # Model-------------------------------------------------------------------------
    ######################################################
    # targetTr = np.repeat(aFrame, r, axis=0)

    # Model-------------------------------------------------------------------------
    ######################################################
    # targetTr = np.repeat(aFrame, r, axis=0)
    k = 30
    k3 = k * 3
    nrow= np.shape(aFrame)[0]
    # TODO try downweight mmd to the end of computation
    #DCAE_weight = K.variable(value=0)
    #DCAE_weight_lst = K.variable(np.array(frange_anneal(epochs, ratio=0)))
    # TODO: Possible bug in here ast ther end of training after the switch
    # check for possible discontinuities/singularities o last epochs, is shape of potenatial  to narroe at the end?

    MMD_weight = K.variable(value=0)

    MMD_weight_lst = K.variable( np.array(frange_anneal(int(epochs), ratio=0.80)) )

    batch_size = 256
    latent_dim = 3
    original_dim = aFrame.shape[1]
    intermediate_dim = original_dim * 3
    intermediate_dim2 = original_dim * 2
    # var_dims = Input(shape = (original_dim,))
    #
    initializer = tf.keras.initializers.he_normal(12345)
    # initializer = None
    SigmaTsq = Input(shape=(1,),  name="Sigma_Layer")
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


    def DCAE3D_loss(x, x_decoded_mean):  # attempt to avoid vanishing derivative of sigmoid
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
        # r = tf.linalg.einsum('aj->a', s ** 2)
        ds = tf.linalg.diag(tf.math.scalar_mul(0, s) + 1)

        S_0W = tf.einsum('akl,lj->akj', du, U)
        S_1W = tf.einsum('akl,lj->akj', dm, W)  # N_batch x N_input ??
        # tf.print((S_1W).shape) #[None, 120]
        S_2Z = tf.einsum('akl,lj->akj', ds, Z)  # N_batch ?? TODO: use tf. and einsum and/or tile
        # tf.print((S_2Z).shape)
        diff_tens = tf.einsum('akl,alj->akj', S_2Z,
                              S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
        diff_tens = tf.einsum('akl,alj->akj', diff_tens, S_0W)
        # tf.Print(K.sum(diff_tens ** 2))
        return 1 / normSigma * (SigmaTsq) * lam * (K.sum(diff_tens ** 2))


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
            return msew + 1 * (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + 1 * (MMD_weight + coeffCAE) * DCAE_loss(
                x, x_decoded_mean)  # TODO: try 1-MMD insted 2-MMD
            # return msew + 1 * (1 - MMD_weight) * loss_mmd(x, x_decoded_mean) + (coeffCAE) * DCAE_loss(x,
            # x_decoded_mean)
            # return K.mean(msew)

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

    encoder.load_weights(output_dir + '/'+ str(bl) + '_3D.h5')
    autoencoder.load_weights(output_dir + '/autoencoder_' + str(bl) + '_3D.h5')
    encoder.summary()
    z = encoder.predict([aFrame, Sigma, ])
    lbls = [i if i != -7 else 7 for i in lbls]

    # implement LRP for vector output
    # relevance score would be defined as R_i= sqrt(sum_i (h^i_j**2))
    from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
    #encoder2 = Model([aFrame, Sigma], h, name='encoder2')
    #encoder2.layers[1].set_weights(encoder.get_weights())
    #endcoder_1 =  [aFrame, Sigma(encoder, encoder.layers[1], np.arange(intermediate_dim)[np.max(activations0,axis=0) < 0.05])
    #from kerassurgeon.operations import delete_layer, insert_layer, delete_channels

    # to remove Sigma layer will need re-create encoder
    encoder2 = Model([x], z_mean, name='encoder2')
    encoder2.summary()
    #excl=[0, 1]
    #encoder2 = delete_channels(encoder2, encoder2.layers[3], excl)
    #encoder2 = delete_layer(encoder2, encoder2.get_layer('Sigma_Layer') )
    #encoder3 = tf.keras.Model(inputs=encoder2.layers[-3].input, outputs=encoder2.output)
    # to remoce Sigmalayer will need reload weights in layer one by one
     ##from kerassurgeon import Surgeon
    #surgeon = Surgeon(encoder2)
    #surgeon.add_job('delete_layer', encoder2.layers[3])
    #new_model = surgeon.operate()
    #new_model = surgeon.operate()

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
    SC = np.sqrt(gradients0 ** 2 + gradients1 ** 2 + gradients2 ** 2)
    lmbd = np.sqrt(z[:, 0]**2 +  z[:, 1]** 2 +z[:, 2]**2)
    #elasticity score per cluster
    SC_mean = np.stack([np.median(SC[lbls == i, :] * aFrame[lbls == i, :] * np.expand_dims(1/lmbd[lbls == i], -1)  , axis=0) for i in l_list], axis=0)
    # plot a heatmap of this and peform  statistical tests, showing that data is interpreted correctly,
    # removing influence of non-informative dimensions

    from sklearn import preprocessing
    SC_mean_norm = preprocessing.normalize(SC_mean, norm = 'l1')
    plt.figure(figsize=(14, 10))
    g =  sns.heatmap(SC_mean_norm, center=0.1, linewidths=.2, cmap="GnBu",  annot=True, fmt='1.2f',  annot_kws={"fontsize":8})
    plt.savefig(PLOTS+'Sensitivity/' + str(bl)+ "Sensitivity_heatmap"+".eps", format='eps', dpi=350)
    plt.close()
    #elasticity score per point
    SC = np.concatenate([SC[lbls == i, :] * aFrame[lbls == i, :] * np.expand_dims(1/lmbd[lbls == i], -1) for i in l_list], axis=0)

    fig, axs = plt.subplots(nrows=8)
    yl = SC.min()
    yu = SC.max()
    for i in l_list:
        sns.violinplot(data=SC[lbls == i, :], ax=axs[int(i)])
        axs[int(i)].set_ylim(yl, yu)
        axs[int(i)].set_title(str(int(i)), rotation=-90, x=1.05, y =0.5)
    fig.savefig(PLOTS+'Sensitivity/' + str(bl)+ "Sensitivity_violinplot"+".eps", format='eps', dpi=700)
    plt.close()

    fig, axs = plt.subplots(nrows=8)
    yl = aFrame.min()
    yu = aFrame.max()
    for i in l_list:
        sns.violinplot(data=aFrame[lbls == i, :], ax=axs[int(i)])
        axs[int(i)].set_ylim(yl, yu)
        axs[int(i)].set_title(str(int(i)), rotation=-90, x=1.05, y =0.5)
    fig.savefig(PLOTS + 'Sensitivity/' + str(bl) + "Signal_violinplot" + ".eps", format='eps', dpi=350)
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
    outfile = output_dir + '/' + str(bl) + '_BOREALIS_PerformanceMeasures.npz'
    #TODO save 2 dataframes  in spartete files  times 25

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

    outfile = sensitivity_dir + str(bl) + '_df_sensitivity_table.csv'
    df.to_csv(outfile,encoding ='utf-8')
    outfile = sensitivity_dir + str(bl) + '_df7_sensitivity_table.csv'
    df7.to_csv(outfile,encoding ='utf-8')

#combine dataframes and create summary information on sensitivity in all 25 clusters
df_all = list()
df7_all = list()
#bl = list_of_branches[0]
for bl in list_of_branches:
    outfile = sensitivity_dir + str(bl) + '_df7_sensitivity_table.csv'
    df7  = pd.read_csv(outfile).iloc[: , 1:]
    outfile = sensitivity_dir + str(bl) + '_df_sensitivity_table.csv'
    df = pd.read_csv(outfile).iloc[: , 1:]
    df_all.append(df)
    df7_all.append(df7)
df7_all = pd.concat(df7_all)
df_all = pd.concat(df_all)

#total summ of pairs where p<1e-10 in pentagone clusters
df7_num = df7_all.iloc[:, 2:127].astype('float64').to_numpy()
np.sum(df7_num>1e-10)
np.sum(df7_num>1e-10)/df7_num.shape[0]/df7_num.shape[1]

df_num = df_all.iloc[:, 2:127].astype('float64').to_numpy()
np.sum(df_num>1e-10)
np.sum(df_num>1e-10)/df_num.shape[0]/df_num.shape[1]

#exclude from the above calculation dimensions not varyyin in pentagone, namely dim 4
inform_dim_list = np.arange(0, 4)
noisy_dim_list = np.arange(5, 30)
col_names = ['bl', 'cluster']
dim_comb = sum([[str(dim_inf) + ' ' + str(dim) for dim in  inform_dim_list ] for dim_inf in noisy_dim_list], [])
col_names = col_names + dim_comb

df_num_pentagon_dims = df_all[col_names[1:10]]









