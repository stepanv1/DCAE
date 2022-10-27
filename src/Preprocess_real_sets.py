'''
Preprocesses real sets
'''
import pandas as pd
import numpy as np

import ctypes
from numpy.ctypeslib import ndpointer
lib = ctypes.cdll.LoadLibrary("./perp.so")
perp = lib.Perplexity
perp.restype = None
perp.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t, ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,  ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #Sigma
                ctypes.c_size_t]




DATA_ROOT = '/media/grinek/Seagate/'
source_dir = DATA_ROOT + 'CyTOFdataPreprocess/'
output_dir = DATA_ROOT + 'CyTOFdataPreprocess/temp'


#Levine32

#Pregnancy

#Shekhar
#Get Shekhar 2016 following  https://colab.research.google.com/github/KrishnaswamyLab/SingleCellWorkshop/blob/master/
#/exercises/Deep_Learning/notebooks/02_Answers_Exploratory_analysis_of_single_cell_data_with_SAUCIE.ipynb#scrollTo=KgaZZU8r4drd
import scprep
scprep.io.download.download_google_drive("1GYqmGgv-QY6mRTJhOCE1sHWszRGMFpnf", "/media/grinek/Seagate/CyTOFdataPreprocess/data.pickle.gz")
scprep.io.download.download_google_drive("1q1N1s044FGWzYnQEoYMDJOjdWPm_Uone", "/media/grinek/Seagate/CyTOFdataPreprocess/metadata.pickle.gz")
data_raw = pd.read_pickle("data.pickle.gz")
data_raw.head()
metadata = pd.read_pickle("metadata.pickle.gz")
# the batch ids are in the cell barcode names
metadata['batch'] = [int(index[7]) for index in metadata.index]
# for simplicity, we'll split the six batches into two groups -- 1-3 and 4-6
metadata['sample_id'] = np.where(metadata['batch'] < 4, 1, 2)
metadata['sample_name'] = np.where(metadata['batch'] < 4, 'Samples 1-3', 'Samples 4-6')
metadata.head()
pca_op = sklearn.decomposition.PCA(100)
pca = sklearn.decomposition.PCA(100)
pca = pca.fit(data_raw.to_numpy())
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

data = pca_op.fit_transform(data_raw.to_numpy())

n_features = data.shape[1]
n_features
scprep.plot.scatter2d(data, c=metadata['sample_name'], ticks=False, label_prefix="PC", legend_title="Batch")

# get labels
table(metadata['CELLTYPE'])
aFrame = data
aFrame.shape
# set negative values to zero
#aFrame[aFrame < 0] = 0
lbls= metadata['CELLTYPE']
#randomize order
IDX = np.random.choice(aFrame.shape[0], aFrame.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
aFrame= aFrame[IDX,:]
lbls = lbls[IDX]
len(lbls)
#scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
#scaler.fit_transform(aFrame)

aFrame = aFrame  - aFrame.min(axis=0)
aFrame= aFrame/np.max(aFrame)
#
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=16)
Idx = nb['idx']; Dist = nb['dist']
#Dist = Dist[IDX]
#Idx = Idx[IDX]
nrow=Idx.shape[0]
# find nearest neighbours
def singleInput(i):
    nei = noisy_clus[Idx[i, :], :]
    return [nei, i]
# find nearest neighbours
nn=30
rk=range(k3)
def singleInput(i):
     nei =  aFrame[Idx[i,:],:]
     di = [np.sqrt(sum(np.square(aFrame[i] - nei[k_i,]))) for k_i in rk]
     return [nei, di, i]
nrow = len(lbls)
inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
num_cores = 16
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=6, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
original_dim=100
neibALL = np.zeros((nrow, k3, original_dim))
Distances = np.zeros((nrow, k3))
neib_weight = np.zeros((nrow, k3))
Sigma = np.zeros(nrow, dtype=float)
for i in range(nrow):
    neibALL[i,] = results[i][0]
for i in range(nrow):
    Distances[i,] = results[i][1]
#Compute perpelexities
nn=30
perp((Distances[:,0:k3]),       nrow,     original_dim,   neib_weight,          nn,          k3,   Sigma,    12)
      #(     double* dist,      int N,    int D,       double* P,     double perplexity,    int K, int num_threads)
np.shape(neib_weight)
plt.plot(neib_weight[1,])
#sort and normalise weights
topk = np.argsort(neib_weight, axis=1)[:,-nn:]
topk= np.apply_along_axis(np.flip, 1, topk,0)
neib_weight=np.array([ neib_weight[i, topk[i]] for i in range(len(topk))])
neib_weight=sklearn.preprocessing.normalize(neib_weight, axis=1, norm='l1')
neibALL=np.array([ neibALL[i, topk[i,:],:] for i in range(len(topk))])
plt.plot(neib_weight[1,:]);plt.show()
outfile = source_dir + '/Shenkareuclid_shifted.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)


#Samusik_01