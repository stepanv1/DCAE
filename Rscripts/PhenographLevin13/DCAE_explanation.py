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
list_of_branches = sum([[(x,y) for x in range(5)] for y in range(5) ], [])
#load earlier generated data

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.compat.v1.disable_eager_execution()
#bl = list_of_branches[2]
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
    '''
    from tensorflow.keras.callbacks import Callback

    save_period = 10


    class plotCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % save_period == 0 or epoch in range(200):
                z = encoder.predict([aFrame, Sigma])
                fig = plot3D_cluster_colors(z, lbls=lbls)
                html_str = to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                                   include_mathjax=False, post_script=None, full_html=True,
                                   animation_opts=None, default_width='100%', default_height='100%', validate=True)
                html_dir = output_dir
                Html_file = open(html_dir + "/" + str(bl) + '_epoch=' + str(epoch) + '_' + "_Buttons.html", "w")
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

    encoder.save_weights(output_dir + '/' + str(bl) + '_3D.h5')
    autoencoder.save_weights(output_dir + '/autoencoder_' + str(bl) + '_3D.h5')
    np.savez(output_dir + '/' + str(bl) + '_latent_rep_3D.npz', z=z)
    '''
    encoder.load_weights(output_dir + '/'+ str(bl) + '_3D.h5')
    autoencoder.load_weights(output_dir + '/autoencoder_' + str(bl) + '_3D.h5')
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
        activations = get_activations(encoder,keract_inputs, layer_names=['input_1', 'intermediate', 'intermediate2'])
        activation_lists[0].append(list(activations['input_1']))
        activation_lists[1].append(list(activations['intermediate']))
        activation_lists[2].append(list(activations['intermediate2']))
        #display_activations(activations, cmap="gray", save=True, directory = output_dir + '/autoencoder_' + str(bl))

    #stack lists into matrices of activations
    activations_matrices = [np.array(i).squeeze() for i in activation_lists]
    weights=[]
    for layer in encoder.layers:
        weights.append(layer.get_weights())

    # implement LRP for vector output
    # relevance score would be defined as R_i= sqrt(sum_i (h^i_j**2))
    from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
    #encoder2 = Model([aFrame, Sigma], h, name='encoder2')
    #encoder2.layers[1].set_weights(encoder.get_weights())
    #endcoder_1 =  [aFrame, Sigma(encoder, encoder.layers[1], np.arange(intermediate_dim)[np.max(activations0,axis=0) < 0.05])
    from kerassurgeon.operations import delete_layer, insert_layer, delete_channels

    # to remoce Sigmalayer will need reload weights in layer one by one
    encoder2 = Model([x], z_mean, name='encoder2')
    encoder2.summary()
    #excl=[0, 1]
    #encoder2 = delete_channels(encoder2, encoder2.layers[3], excl)
    #encoder2 = delete_layer(encoder2, encoder2.get_layer('Sigma_Layer') )
    #encoder3 = tf.keras.Model(inputs=encoder2.layers[-3].input, outputs=encoder2.output)
    encoder2.summary() # to remoce Sigmalayer will need reload weights in layer one by one
     ##from kerassurgeon import Surgeon
    #surgeon = Surgeon(encoder2)
    #surgeon.add_job('delete_layer', encoder2.layers[3])
    #new_model = surgeon.operate()
    #new_model = surgeon.operate()
    
    # create nearual network for suprevides clustering. Probailities will be used
    # for explanatory analysis
    # Initialising the ANN
       
    classifier =tf.keras.Sequential()
    # Input layer and the first hidden layer
    classifier.add(Dense(units=10, kernel_initializer= "uniform", activation = "relu", input_dim = 3))
    # Then Adding the second hidden layer
    classifier.add(Dense(units=40, kernel_initializer= "uniform", activation = "relu"))
    classifier.add(Dense(units=30, kernel_initializer="uniform", activation="relu"))
    # and the output layer
    classifier.add(Dense(units=len(np.unique(lbls)), kernel_initializer= "uniform", activation = "relu"))
    lblsC = [7 if i==-7 else i for i in lbls]
        classifier.summary()
    # split into train test sets

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(z, lblsC, test_size=0.33)

    classifier.compile(optimizer= "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])
    classifier.fit(X_train, y_train, batch_size=256, epochs=1000)
    test_loss, test_acc = classifier.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([classifier, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(z)

    #now add it to the encoder and backpropagate
    
    
    activation_lists = [[], [], []]
    for j in range(len(indx)):
        keract_inputs = [aFrame[indx[j]:(indx[j] + 1), :]]
        # keract_targets = target_test[:1]
        activations = get_activations(encoder2, keract_inputs, layer_names=['input_1', 'intermediate', 'intermediate2'])
        activation_lists[0].append(list(activations['input_1']))
        activation_lists[1].append(list(activations['intermediate']))
        activation_lists[2].append(list(activations['intermediate2']))

    #stack lists into matrices of activations
    activations_matrices = [np.array(i).squeeze() for i in activation_lists]
    weights=[]
    for layer in encoder2.layers:
        weights.append(layer.get_weights())

    # do LRP on component 0
    #https: // git.tu - berlin.de / gmontavon / lrp - tutorial yhis contains relu network
    # get separate lists for weights and biases
    W =  [weights[l][0] for l in range(1,4)]
    B = [weights[l][1] for l in range(1, 4)]
    L = len(W)
    # compute forward pass
    A = [aFrame]+[None]*L
    for l in range(L-1):
        A[l+1] = np.maximum(0,A[l].dot(W[l])+B[l])
    #top layer has linear activation:
    A[L] = A[L-1].dot(W[L-1]) + B[L-1]
    #  ρ is a function that transform the weights, and ϵ is a small positive increment
    def rho(w, l):
        return w + [0.1, 0.1, 0.0, 0.0][l] * np.maximum(0, w)
    def incr(z, l):
        return z + [0.0, 0.0, 0.1, 0.0][l] * (z ** 2).mean() ** .5 + 1e-9

    #create a list to store relevance scores at each layer
    R = [None] * L + [(A[L])]# using activations of as top layer relevances is not agood idea, as they change sign?
    for l in range(0, L)[::-1]:
        w = rho(W[l], l)
        b = rho(B[l], l)

        z = incr(A[l].dot(w) + b, l)  # step 1
        s = R[l + 1] / z  # step 2
        c = s.dot(w.T)  # step 3
        R[l] = A[l] * c

    R[0].max()
    R[0].min()
    SC = (np.arcsinh(R[0]))
    yl = SC.max()
    yb = SC.min(); cut=5; gridsize=1000; inner =None
    fig, axs = plt.subplots(nrows=8)
    sns.violinplot(data=SC[lbls == 0, :], cut=cut, gridsize=gridsize, inner =inner, ax=axs[0]).set_title('0', rotation=-90, position=(1, 1), ha='left',
                                                               va='bottom')
    axs[0].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 1, :], cut=cut,gridsize=gridsize, inner =inner,ax=axs[1]).set_title('1', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[1].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 2, :], cut=cut,gridsize=gridsize,inner =inner, ax=axs[2]).set_title('2', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[2].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 3, :], cut=cut,gridsize=gridsize, inner =inner,ax=axs[3]).set_title('3', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[3].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 4, :], cut=cut,gridsize=gridsize,inner =inner,ax=axs[4]).set_title('4', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[4].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 5, :], cut=cut,gridsize=gridsize, inner =inner,ax=axs[5]).set_title('5', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[5].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 6, :], cut=cut,gridsize=gridsize,inner =inner,ax=axs[6]).set_title('6', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[6].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == -7, :], cut=cut,gridsize=gridsize, inner =inner,ax=axs[7]).set_title('7', rotation=-90, position=(1, 2), ha='left',
                                                                va='center')
    axs[7].set_ylim(yb, yl)

    import seaborn as sns
    import umap.umap_ as umap
    import plotly.io as pio
    pio.renderers.default = "browser"
    yl =7
    fig, axs = plt.subplots(nrows=8)
    sns.violinplot(data=aFrame[lbls==0,:],  ax=axs[0]).set_title('0', rotation=-90, position=(1, 1), ha='left', va='bottom')
    axs[0].set_ylim(0, yl)
    sns.violinplot(data=aFrame[lbls==1,:],  ax=axs[1]).set_title('1', rotation=-90, position=(1, 2), ha='left', va='center')
    axs[1].set_ylim(0, yl)
    sns.violinplot(data=aFrame[lbls==2,:],  ax=axs[2]).set_title('2', rotation=-90, position=(1, 2), ha='left', va='center')
    axs[2].set_ylim(0, yl)
    sns.violinplot(data=aFrame[lbls==3,:],  ax=axs[3]).set_title('3', rotation=-90, position=(1, 2), ha='left', va='center')
    axs[3].set_ylim(0, yl)
    sns.violinplot(data=aFrame[lbls==4,:],  ax=axs[4]).set_title('4', rotation=-90, position=(1, 2), ha='left', va='center')
    axs[4].set_ylim(0, yl)
    sns.violinplot(data=aFrame[lbls==5,:],  ax=axs[5]).set_title('5', rotation=-90, position=(1, 2), ha='left', va='center')
    axs[5].set_ylim(0, yl)
    sns.violinplot(data=aFrame[lbls==6,:],  ax=axs[6]).set_title('6', rotation=-90, position=(1, 2), ha='left', va='center')
    axs[6].set_ylim(0, yl)
    sns.violinplot(data=aFrame[lbls==-7,:], ax=axs[7]).set_title('7', rotation=-90, position=(1, 2), ha='left', va='center')
    axs[7].set_ylim(0, yl)
    plt.show()
    

    [for lbls in lbl_list]
    
    



    mapper = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=True).fit(aFrame)
    yUMAP =  mapper.transform(aFrame)

    from sklearn import decomposition
    pca = decomposition.PCA(n_components=2)
    pca.fit(aFrame)
    yPCA = pca.transform(aFrame)

    pca = decomposition.PCA(n_components=10)
    pca.fit(aFrame)
    yPCA3 = pca.transform(aFrame)



    fig = plot2D_cluster_colors(yUMAP, lbls=lbls, msize=5)
    fig.show()
    fig = plot2D_cluster_colors(yPCA, lbls=lbls, msize=5)
    fig.show()





    x_tensor = tf.convert_to_tensor(aFrame, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as t:
        t.watch(x_tensor)
        output = encoder2(x_tensor)

        result = output
        gradients0 = t.gradient(output[:, 0], x_tensor).numpy()
        gradients1 = t.gradient(output[:, 1], x_tensor).numpy()
        gradients2 = t.gradient(output[:, 2], x_tensor).numpy()

    import seaborn as sns


    SC = np.sqrt(gradients0**2+ gradients1**2 +gradients2**2)
    yl = SC.max()
    yb = SC.min()
    fig, axs = plt.subplots(nrows=8)
    sns.violinplot(data= SC[lbls == 0, :]/np.std(aFrame[lbls == 0, :], axis=0), ax=axs[0]).set_title('0', rotation=-90, position=(1, 1), ha='left',
                                                               va='bottom')
    axs[0].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 1, :]/np.std(aFrame[lbls == 1, :], axis=0), ax=axs[1]).set_title('1', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[1].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 2, :]/np.std(aFrame[lbls == 2, :], axis=0), ax=axs[2]).set_title('2', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[2].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 3, :]/np.std(aFrame[lbls == 3, :], axis=0), ax=axs[3]).set_title('3', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[3].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 4, :]/np.std(aFrame[lbls == 4, :], axis=0), ax=axs[4]).set_title('4', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[4].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 5, :]/np.std(aFrame[lbls == 5, :], axis=0), ax=axs[5]).set_title('5', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[5].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == 6, :]/np.std(aFrame[lbls == 6, :], axis=0), ax=axs[6]).set_title('6', rotation=-90, position=(1, 2), ha='left',
                                                               va='center')
    axs[6].set_ylim(yb, yl)
    sns.violinplot(data=SC[lbls == -7, :]/np.std(aFrame[lbls == -7, :], axis=0), ax=axs[7]).set_title('7', rotation=-90, position=(1, 2), ha='left',
                                                                va='center')
    axs[7].set_ylim(yb, yl)
    plt.show()

    zzz=np.ptp(aFrame, axis=0)
    np.median(SC[lbls == -7, :], axis=0)
    np.median(SC[lbls == 0, :], axis=0)
    np.mean(SC[lbls == -7, :], axis=0)
    np.mean(SC[lbls == 0, :], axis=0)

    np.std(SC[lbls == -7, :], axis=0)
    np.std(SC[lbls == 0, :], axis=0)

    np.std(aFrame[lbls == -7, :], axis=0)
    np.std(aFrame[lbls == 0, :], axis=0)





