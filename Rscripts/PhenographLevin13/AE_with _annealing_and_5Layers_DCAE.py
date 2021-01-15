'''This script is a summary of DCAE model with annealing
This code should be repeatred on all other data sets
Innate‐like CD8+ T‐cells and NK cells: converging functions and phenotypes
Ayako Kurioka,corresponding author 1 , 2 Paul Klenerman, 1 , 2 and Christian B. Willbergcorresponding author 1 , 2
CD8+ T Cells and NK Cells: Parallel and Complementary Soldiers of Immunotherapy
Jillian Rosenberg1 and Jun Huang1,2
'''

#import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import sys
from tensorflow.keras import metrics
import random
# from keras.utils import multi_gpu_model
import timeit
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.constraints import maxnorm
import sklearn
from sklearn.preprocessing import MinMaxScaler
from plotly.io import to_html
import plotly.io as pio
pio.renderers.default = "browser"
# from kerasvis import DBLogger
# Start the keras visualization server with
# export FLASK_APP=kerasvis.runserver
# flask run
import seaborn as sns
import warnings
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                        write_graph=True, write_images=True)

from utils_evaluation import compute_f1, table, find_neighbors, compare_neighbours, compute_cluster_performance, projZ,\
    plot3D_marker_colors, plot3D_cluster_colors, plot2D_cluster_colors, neighbour_marker_similarity_score, neighbour_onetomany_score, \
    get_wsd_scores, neighbour_marker_similarity_score_per_cell, plot3D_performance_colors, plot2D_performance_colors



import umap.umap_ as umap
def table(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('%d %d', np.asarray((unique, counts)).T)
    return {'unique': unique, 'counts': counts}

markers = ["CD3", "CD45", "pNFkB", "pp38", "CD4", "CD20", "CD33", "pStat5", "CD123", "pAkt", "pStat1", "pSHP2",
           "pZap70",
           "pStat3", "CD14", "pSlp76", "pBtk", "pPlcg2", "pErk", "pLat", "IgM", "pS6", "HLA-DR", "CD7"]


def frange_anneal(n_epoch, ratio=0.25, shape='sin'):
    L = np.ones(n_epoch)
    for c in range(n_epoch):
        if c <= np.floor(n_epoch*ratio):
            if shape=='sqrt':
                norm = np.sqrt(np.floor(n_epoch*ratio))
                L[c] = np.sqrt(c)/norm
            if shape=='sin':
                Om = (np.pi/2/(n_epoch*ratio))
                L[c] =  np.sin(Om*c)
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
        print("  Current DCAE Weight is " + str(K.get_value(self.weight)))


from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from pathos import multiprocessing


# results = Parallel(n_jobs=12, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in range(100))

# load data
k = 30
perp = k
k3 = k * 3
coeffCAE = 5
epochs = 10000
ID = 'Nowizk24_MMD_01_3D_DCAE_h120_hidden_5_layers_anneal'+ str(coeffCAE) + '_' + str(epochs) + '_kernelInit_tf2'
source_dir = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/'
output_dir  = '/media/grines02/vol1/Box Sync/Box Sync/CyTOFdataPreprocess/'
epsilon_std = 1.0
'''
data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816
'''
source_dir = "/home/grines02/PycharmProjects/BIOIBFO25L/data/data/"
'''
#file_list = glob.glob(source_dir + '/*.txt')
data0 = np.genfromtxt(source_dir + "d_matrix.txt"
, names=None, dtype=float, skip_header=1)
aFrame = data0[:,1:]
# set negative values to zero
aFrame[aFrame < 0] = 0
patient_table = np.genfromtxt(source_dir + "label_patient.txt", names=None, dtype='str', skip_header=1, delimiter=" ", usecols = (1, 2, 3))
lbls=patient_table[:,0]
len(lbls)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
scaler.fit_transform(aFrame)
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=12)
Idx = nb['idx']; Dist = nb['dist']
'''
#data0 = np.genfromtxt(source_dir + "d_matrix.txt"
                 #######     , names=None, dtype=float, skip_header=1)
#clust = np.genfromtxt(source_dir + "label_patient.txt", names=None, dtype='str', skip_header=1, delimiter=" ",
#                      usecols=(1, 2, 3))[:, 0]
outfile = '/home/grines02/PycharmProjects/BIOIBFO25L/data/WeberLabels/Nowicka2017euclid_scaled.npz'
# np.savez(outfile, Idx=Idx, aFrame=aFrame, lbls=lbls,  Dist=Dist)
npzfile = np.load(outfile)
lbls = npzfile['lbls'];
Idx = npzfile['Idx'];
aFrame = npzfile['aFrame'];
Dist = npzfile['Dist']
Sigma = npzfile['Sigma']

# lbls2=npzfile['lbls'];Idx2=npzfile['Idx'];aFrame2=npzfile['aFrame'];
# cutoff2=npzfile['cutoff']; Dist2 =npzfile['Dist']


epochs = 10000
# annealing schedule
DCAE_weight = K.variable(value=0)
DCAE_weight_lst = K.variable(np.array(frange_anneal(epochs, ratio=0.5)))
# number of replicas in validation data set

nrow = aFrame.shape[0]
batch_size = 256
original_dim = 24
latent_dim = 3
intermediate_dim = 120
intermediate_dim2=120
nb_hidden_layers = [original_dim, intermediate_dim, latent_dim, intermediate_dim, original_dim]

#define the model
SigmaTsq = Input(shape=(1,))
initializer = tf.keras.initializers.he_normal(12345)
x = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu', name='intermediate', kernel_initializer = initializer)(x)
z_mean =  Dense(latent_dim, activation=None, name='z_mean', kernel_initializer = initializer)(h)

encoder = Model([x, SigmaTsq], z_mean, name='encoder')

decoder_h = Dense(intermediate_dim2, activation='relu', name='intermediate2', kernel_initializer = initializer)
decoder_mean = Dense(original_dim, activation='relu', name='output', kernel_initializer = initializer)
h_decoded = decoder_h(z_mean)
x_decoded_mean = decoder_mean(h_decoded)
#autoencoder = Model(inputs=[x ], outputs=x_decoded_mean)
autoencoder = Model(inputs=[x, SigmaTsq], outputs=x_decoded_mean)

normSigma = nrow / sum(1 / Sigma)
lam=1e-4
def DCAE_loss(x, x_decoded_mean):  # 5 layers case
    W = K.variable(value=encoder.get_layer('intermediate').get_weights()[0])  # N x N_hidden
    Z = K.variable(value=encoder.get_layer('z_mean').get_weights()[0])  # N x N_hidden
    W = K.transpose(W);
    Z = K.transpose(Z);  # N_hidden x N
    m = encoder.get_layer('intermediate').output
    dm = tf.linalg.diag((tf.math.sign(m)+1)/2)  # N_batch x N_hidden
    s = encoder.get_layer('z_mean').output
    r = tf.linalg.einsum('aj->a', s**2)
    ds  = DCAE_weight * (-2 * r + 1.5*r **2) + 1.5 + 1.2*(DCAE_weight-1)

    S_1W = tf.einsum('akl,lj->akj', dm, W)
    S_2Z = tf.einsum('a,lj->alj', ds, Z)
    # tf.print((S_2Z).shape)
    diff_tens = tf.einsum('akl,alj->akj', S_2Z, S_1W)  # Batch matrix multiplication: out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
    # tf.Print(K.sum(diff_tens ** 2))
    return 1 / normSigma * (SigmaTsq) * lam * K.sum(diff_tens ** 2)
    #return  lam * K.sum(diff_tens ** 2)


#mmd staff TODO: try approximation for this
def compute_kernel(x,y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))
def compute_mmd(x, y):   # [batch_size, z_dim] [batch_size, z_dim]
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
def loss_mmd(x, x_decoded_mean):
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    #true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    true_samples = K.random_uniform(shape=(batch_size, latent_dim), minval = -1., maxval = 1.0)
    return compute_mmd(true_samples, z_mean)


def mean_square_error_NN(y_true, y_pred):
    # dst = K.mean(K.square((neib - K.expand_dims(y_pred, 1)) / (tf.expand_dims(cut_neib,1) + 1)), axis=-1)
    msew = tf.keras.losses.mean_squared_error(y_true, y_pred)
    # return 0.5 * (tf.multiply(weightedN, 1/SigmaTsq))
    return  tf.multiply(msew, normSigma * 1/SigmaTsq ) # TODO Sigma -denomiator or nominator? try reverse, schek hpw sigma computed in UMAP

def ae_loss(weight, DCAE_weight_lst):
    def loss(x, x_decoded_mean):
        msew = mean_square_error_NN(x, x_decoded_mean)
        return msew + loss_mmd(x, x_decoded_mean) + coeffCAE * DCAE_loss(x, x_decoded_mean)
        # return K.mean(msew)
    return loss

autoencoder.summary()
autoencoder.compile(optimizer='adam', loss=ae_loss(DCAE_weight, DCAE_weight_lst), metrics=[DCAE_loss, loss_mmd])

start = timeit.default_timer()
history = autoencoder.fit([aFrame, Sigma], aFrame,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[AnnealingCallback(DCAE_weight, DCAE_weight_lst)])
stop = timeit.default_timer()
z = encoder.predict([aFrame,  Sigma])
# vae.save('WEBERCELLS3D32lambdaPerpCorr0.01h5')
# vae.load('WEBERCELLS3D.h5')

# ae=load_model('Wang0_modell1.h5', custom_objects={'mean_square_error_weighted':mean_square_error_weighted, 'ae_loss':
#  ae_loss, 'mean_square_error_weightedNN' : mean_square_error_weightedNN})
# ae = load_model('Wang0_modell1.h5')

print(stop - start)
fig0 = plt.figure();
plt.plot(history.history['DCAE_loss'][1200:]);

fig01 = plt.figure();
plt.plot(history.history['loss'][1200:]);

fig03 = plt.figure();
plt.plot(history.history['loss_mmd'][1200:]);

#encoder.save_weights(output_dir +'/'+ID + '_3D.h5')
#autoencoder.save_weights(output_dir +'/autoencoder_'+ID + '_3D.h5')
#np.savez(output_dir +'/'+ ID + '_latent_rep_3D.npz', z = z)

encoder.load_weights(output_dir +''+ID + '_3D.h5')
autoencoder.load_weights(output_dir +'autoencoder_'+ID + '_3D.h5')
encoder.summary()
z = encoder.predict([aFrame, Sigma])



fig = plot3D_cluster_colors(z, camera = dict(eye = dict(x=-0.2,y=0.2,z=1.5)), lbls=lbls)
fig.show()
fig =plot3D_marker_colors(z, data=aFrame, markers=markers, sub_s = 50000, lbls=lbls)
fig.show()
# sretch low values
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
aFrame_scaled=  scaler.fit_transform(aFrame)
#z=np.sqrt(z)
fig =plot3D_marker_colors(z, data=aFrame_scaled, markers=markers, sub_s = 50000, lbls=lbls)
fig.write_html('temp2.html', auto_open=True)
# predict and extract latent variables
fig = plot3D_cluster_colors(z, camera = dict(eye = dict(x=-0.2,y=0.2,z=1.5)), lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Buttons.html","w")
Html_file.write(html_str)
Html_file.close()


fig =plot3D_marker_colors(z, data=aFrame, markers=markers, sub_s = 50000, lbls=lbls)
fig.show()
html_str=to_html(fig, config=None, auto_play=True, include_plotlyjs=True,
                  include_mathjax=False, post_script=None, full_html=True,
                  animation_opts=None, default_width='100%', default_height='100%', validate=True)
html_dir = "/media/grines02/vol1/Box Sync/Box Sync/github/stepanv1.github.io/_includes"
Html_file= open(html_dir + "/"+ID + "_Markers.html","w")
Html_file.write(html_str)
Html_file.close()

#stretch low signal
aFramesqrt =np.sqrt(aFrame)
fig =plot3D_marker_colors(z, data=aFramesqrt, markers=markers, sub_s = 50000, lbls=lbls)
fig.show()

# clustering UMAP representation
#mapper = umap.UMAP(n_neighbors=30, n_components=2, metric='euclidean', random_state=42, min_dist=0, low_memory=False).fit(aFrame)
#embedUMAP =  mapper.transform(aFrame)
#np.savez('Nowizka_' + 'embedUMAP.npz', embedUMAP=embedUMAP)
embedUMAP = np.load('LEVINE32_' + 'embedUMAP.npz')['embedUMAP']



# cluster with phenograph
#communities, graph, Q = phenograph.cluster(aFrame)
#np.savez('Phenograph.npz', communities=communities, graph=graph, Q=Q)
communities =np.load('Phenograph.npz')['communities']
print(compute_cluster_performance(lbls, communities))
print(compute_cluster_performance(lbls[lbls!='"unassigned"'], communities[lbls!='"unassigned"']))

######################################3
# try SAUCIE
sys.path.append("/home/grines02/SAUCIE")
sys.path.append("/home/grines02/PycharmProjects/BIOIBFO25L/SAUCIE")
data = aFrame
from importlib import reload
import SAUCIE
#reload(SAUCIE)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
saucie = SAUCIE.SAUCIE(data.shape[1])
loadtrain = SAUCIE.Loader(data, shuffle=True)
saucie.train(loadtrain, steps=100000)

loadeval = SAUCIE.Loader(data, shuffle=False)
embedding = saucie.get_embedding(loadeval)
np.savez('Nowizka_' + 'embedSAUCIE_100000.npz', embedding=embedding)
embedding = np.load('Nowizka_' + 'embedSAUCIE.npz')['embedding'] #TODO redo LEVINE32
#number_of_clusters, clusters = saucie.get_clusters(loadeval)
#print(compute_cluster_performance(lbls,  clusters))
#clusters= [str(x) for  x in clusters]
#fig = plot3D_cluster_colors(x=embedding[:, 0],y=embedding[:, 1],z=np.zeros(len(clusters)), lbls=np.asarray(clusters))
#fig.show()
fig = plot2D_cluster_colors(embedding, lbls=lbls)
fig.show()


z_mr =  neighbour_marker_similarity_score(z, aFrame, kmax=90)
embedding_mr =  neighbour_marker_similarity_score(embedding, aFrame, kmax=90)
embedUMAP_mr = neighbour_marker_similarity_score(embedUMAP, aFrame, kmax=90)
#np.savez(ID + '_marker_similarity.npz', z_mr = z_mr,  embedding_mr=embedding_mr, embedUMAP_mr=embedUMAP_mr)
npobj =  np.load(ID + '_marker_similarity.npz')
z_mr,embedding_mr,embedUMAP_mr  = npobj ['z_mr'] , npobj['embedding_mr'],  npobj['embedUMAP_mr'],
z_mr[89]
embedding_mr[89]
embedUMAP_mr[89]
# plot
df = pd.DataFrame({'k':range(0,90)[2:],  'DCAE': z_mr[2:], 'SAUCIE': embedding_mr[2:], 'UMAP': embedUMAP_mr[2:]})

# multiple line plot
plt.plot('k', 'DCAE', data=df, marker='o', markerfacecolor='blue', markersize=2, color='skyblue', linewidth=4)
plt.plot('k', 'SAUCIE', data=df, marker='', color='olive', linewidth=2)
plt.plot('k', 'UMAP', data=df, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.legend()

# create performance plots for paper
embedding = np.load('LEVINE32_' + 'embedSAUCIE.npz')['embedding']
embedUMAP = np.load('LEVINE32_' + 'embedUMAP.npz')['embedUMAP']
PAPERPLOTS  = './PAPERPLOTS/'
#3 plots for paper
# how to export as png: https://plotly.com/python/static-image-export/ 2D
fig = plot3D_cluster_colors(z[lbls !='"unassigned"', :  ], camera = dict(eye = dict(x=-0.2,y=0.2,z=1.5)),
                            lbls=lbls[lbls !='"unassigned"'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ "LEVINE32.png")

fig = plot2D_cluster_colors(embedding[lbls !='"unassigned"', :  ], lbls=lbls[lbls !='"unassigned"'],legend=False)
fig.show()
fig.write_image(PAPERPLOTS+ "LEVINE32_SAUCIE.png")

fig = plot2D_cluster_colors(embedUMAP[lbls !='"unassigned"', :  ], lbls=lbls[lbls !='"unassigned"'],legend=True)
fig.show()
fig.write_image(PAPERPLOTS+ "LEVINE32_UMAP.png")


#TODO:very importmant!!! scale all the output to be in unite square (or cube)
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
embedding=  scaler.fit_transform(embedding)
embedUMAP= scaler.fit_transform(embedUMAP)
z= scaler.fit_transform(z)
z = z/np.sqrt(3.1415)
prZ = projZ(z)
prZ = scaler.fit_transform(prZ)
prZ =prZ/np.sqrt(3.1415)
#DCAE
discontinuityDCAE, manytooneDCAE = get_wsd_scores(aFrame, z, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreDCAE = neighbour_onetomany_score(z, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreDCAE = neighbour_marker_similarity_score_per_cell(z, aFrame, kmax=90, num_cores=12)

#discontinuityDCAE_prZ, manytooneDCAE_prZ = get_wsd_scores(aFrame, prZ, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
#onetomany_scoreDCAE_prZ = neighbour_onetomany_score(prZ, Idx, kmax=90, num_cores=12)[1]
#marker_similarity_scoreDCAE_prZ = neighbour_marker_similarity_score_per_cell(prZ, aFrame, kmax=90, num_cores=12)

#UMAP
discontinuityUMAP, manytooneUMAP = get_wsd_scores(aFrame, embedUMAP, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreUMAP= neighbour_onetomany_score(embedUMAP, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreUMAP = neighbour_marker_similarity_score_per_cell(embedUMAP, aFrame, kmax=90, num_cores=12)

#SAUCIE
discontinuitySAUCIE, manytooneSAUCIE = get_wsd_scores(aFrame, embedding, 90, num_meandist=10000, compute_knn_x=False, x_knn=Idx)
onetomany_scoreSAUCIE= neighbour_onetomany_score(embedding, Idx, kmax=90, num_cores=12)[1]
marker_similarity_scoreSAUCIE = neighbour_marker_similarity_score_per_cell(embedding, aFrame, kmax=90, num_cores=12)

outfile2 = source_dir + '/' + ID+ '_PerformanceMeasures.npz'
#p.savez(outfile2, discontinuityDCAE = discontinuityDCAE, manytooneDCAE= manytooneDCAE, onetomany_scoreDCAE= onetomany_scoreDCAE, marker_similarity_scoreDCAE= marker_similarity_scoreDCAE[1],
#         discontinuityUMAP= discontinuityUMAP, manytooneUMAP= manytooneUMAP, onetomany_scoreUMAP= onetomany_scoreUMAP, marker_similarity_scoreUMAP= marker_similarity_scoreUMAP[1],
#         discontinuitySAUCIE= discontinuitySAUCIE, manytooneSAUCIE= manytooneSAUCIE, onetomany_scoreSAUCIE= onetomany_scoreSAUCIE, marker_similarity_scoreSAUCIE= marker_similarity_scoreSAUCIE[1])

npzfile = np.load(outfile2)
discontinuityDCAE = npzfile['discontinuityDCAE']; manytooneDCAE= npzfile['manytooneDCAE']; onetomany_scoreDCAE= npzfile['onetomany_scoreDCAE']; marker_similarity_scoreDCAE= npzfile['marker_similarity_scoreDCAE'];
discontinuityUMAP= npzfile['discontinuityUMAP']; manytooneUMAP= npzfile['manytooneUMAP']; onetomany_scoreUMAP= npzfile['onetomany_scoreUMAP']; marker_similarity_scoreUMAP= npzfile['marker_similarity_scoreUMAP'];
discontinuitySAUCIE= npzfile['discontinuitySAUCIE']; manytooneSAUCIE= npzfile['manytooneSAUCIE']; onetomany_scoreSAUCIE= npzfile['onetomany_scoreSAUCIE']; marker_similarity_scoreSAUCIE=  npzfile['marker_similarity_scoreSAUCIE']
#Quick look into results
# TODO: data normalization by normalize_data_by_mean_pdist in y space
np.mean(discontinuityDCAE)
np.mean(manytooneDCAE)
np.mean(discontinuityUMAP)
np.mean(manytooneUMAP)
np.mean(discontinuitySAUCIE)
np.mean(manytooneSAUCIE)
np.mean(onetomany_scoreDCAE[29,:])
np.mean(marker_similarity_scoreDCAE[29])
np.mean(onetomany_scoreUMAP[29,:])
np.mean(marker_similarity_scoreUMAP[29])
np.mean(onetomany_scoreSAUCIE[29,:])
np.mean(marker_similarity_scoreSAUCIE[29])

np.mean(discontinuityDCAE_prZ)
np.mean(manytooneDCAE_prZ)
np.mean(onetomany_scoreDCAE_prZ[29,:])
np.mean(marker_similarity_scoreDCAE_prZ[1][29])

np.median(discontinuityDCAE)
np.median(manytooneDCAE)
np.median(discontinuityUMAP)
np.median(manytooneUMAP)
np.median(discontinuitySAUCIE)
np.median(manytooneSAUCIE)
np.median(onetomany_scoreDCAE[29,:])
np.median(marker_similarity_scoreDCAE[29])
np.median(onetomany_scoreUMAP[29,:])
np.median(marker_similarity_scoreUMAP[29])
np.median(onetomany_scoreSAUCIE[29,:])
np.median(marker_similarity_scoreSAUCIE[29])

np.median(discontinuityDCAE_prZ)
np.median(manytooneDCAE_prZ)
np.median(onetomany_scoreDCAE_prZ[29,:])
np.median(marker_similarity_scoreDCAE_prZ[29])


plt.hist(onetomany_scoreSAUCIE[90,:],250)
plt.hist(onetomany_scoreDCAE[90,:],250)
plt.hist(onetomany_scoreUMAP[90,:],250)
plt.hist(discontinuityDCAE,250)
plt.hist(discontinuitySAUCIE,250)
plt.hist(discontinuityUMAP,250)

plt.hist(marker_similarity_scoreSAUCIE[29],250)
plt.hist(marker_similarity_scoreDCAE[29],250)
plt.hist(marker_similarity_scoreUMAP[29],250)
plt.hist(manytooneSAUCIE,250)
plt.hist(manytooneDCAE,250)
plt.hist(manytooneUMAP,250)



plt.hist(z,250)
plt.hist(embedding,250)
#build grpahs using above data
# now build plots and tables. 2 plots: 1 for onetomany_score, 1 marker_similarity_scoreDCAE on 2 methods
# table: Discontinuity and manytoone (2 columns) with 3 rows, each per method. Save as a table then stack with output on other  data , to create the final table
median_marker_similarity_scoreDCAE = np.median(marker_similarity_scoreDCAE, axis=1);median_marker_similarity_scoreSAUCIE = np.median(marker_similarity_scoreSAUCIE, axis=1);
median_marker_similarity_scoreUMAP = np.median(marker_similarity_scoreUMAP, axis=1);
df_sim = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_marker_similarity_scoreDCAE[1:], 'SAUCIE': median_marker_similarity_scoreSAUCIE[1:], 'UMAP': median_marker_similarity_scoreUMAP[1:]})
#fig1, fig2 = plt.subplots()
plt.plot('k', 'DCAE', data=df_sim, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_sim, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_sim, marker='x', color='olive', linewidth=2)
plt.legend()
plt.savefig(PAPERPLOTS  + 'LEVINE32_' + ID+ 'performance_marker_similarity_score.png')
plt.show()
plt.clf()
median_onetomany_scoreDCAE = np.median(onetomany_scoreDCAE, axis=1);median_onetomany_scoreSAUCIE = np.median(onetomany_scoreSAUCIE, axis=1);
median_onetomany_scoreUMAP = np.median(onetomany_scoreUMAP, axis=1);
df_otm = pd.DataFrame({'k':range(0,91)[1:],  'DCAE': median_onetomany_scoreDCAE[1:], 'SAUCIE': median_onetomany_scoreSAUCIE[1:], 'UMAP': median_onetomany_scoreUMAP[1:]})
plt.plot('k', 'DCAE', data=df_otm, marker='o',  markersize=5, color='skyblue', linewidth=3)
plt.plot('k', 'SAUCIE', data=df_otm, marker='v', color='orange', linewidth=2)
plt.plot('k', 'UMAP', data=df_otm, marker='x', color='olive', linewidth=2)
plt.savefig(PAPERPLOTS  + 'LEVINE32_' + ID+'_performance_onetomany_score.png')
plt.show()
# tables
df_BORAI = pd.DataFrame({'Method':['DCAE', 'SAUCIE', 'UMAP'],  'manytoone': [0.1064, 0.1785, 0.1177], 'discontinuity': [0.0008, 0.0099, 0.0013]})
df_BORAI.to_csv(PAPERPLOTS  + 'LEVINE32_' + ID+ 'Borealis_measures.csv', index=False)
np.median(discontinuityDCAE)
#0.01565989388359918
np.median(manytooneDCAE)
#0.10611877287197075
np.median(discontinuityUMAP)
#0.0013421323564317491
np.median(manytooneUMAP)
#0.11770417201150978
np.median(discontinuitySAUCIE)
#0.009914790259467234
np.median(manytooneSAUCIE)
#0.17852087116020135

#plot perf
fig =plot3D_performance_colors(z, perf=discontinuityDCAE, lbls=lbls)
fig.show()
fig =plot3D_performance_colors(z, perf=onetomany_scoreDCAE[30,:], lbls=lbls)
fig.show()
fig =plot3D_performance_colors(z, perf=manytooneDCAE, lbls=lbls)
fig.show()
fig =plot3D_performance_colors(z, perf=marker_similarity_scoreDCAE, lbls=lbls)
fig.show()

fig =plot2D_performance_colors(embedding, perf=discontinuitySAUCIE, lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedding, perf=onetomany_scoreSAUCIE[30,:], lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedding, perf=manytooneSAUCIE, lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedding, perf=marker_similarity_scoreSAUCIE, lbls=lbls)
fig.show()

fig =plot2D_performance_colors(embedUMAP, perf=discontinuityUMAP, lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedUMAP, perf=onetomany_scoreSAUCIE[30,:], lbls=lbls)
fig.show()
fig =plot2D_performance_colors(embedUMAP, perf=manytooneSAUCIE, lbls=lbls)
fig.show()