'''
Preprocesses real sets
'''
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from utils_evaluation import  table, find_neighbors
import seaborn as sns



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
output_dir = DATA_ROOT + 'CyTOFdataPreprocess/'

k = 30
k3 = k * 3
#Levine32
#data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
#https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816

data0 = np.genfromtxt(source_dir + "/Levine32_data.csv" , names=None, dtype=float, skip_header=1, delimiter=',')
aFrame = data0[:,:]
aFrame.shape
aFrame.min(axis=0)
aFrame.max(axis=0)
sns.violinplot(data= aFrame, bw = 0.1);plt.show()
# set negative values to zero and shift minima to zero
#aFrame = aFrame  - aFrame.min(axis=0)

aFrame.min(axis=0)
aFrame.max(axis=0)
lbls= np.genfromtxt(source_dir + "Levine32_population.csv" , names=None, skip_header=0, delimiter=',', dtype='U100')
#randomize order
IDX = np.random.choice(aFrame.shape[0], aFrame.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
aFrame= aFrame[IDX,:]
lbls = lbls[IDX]
len(lbls)

aFrame = aFrame  - aFrame.min(axis=0)
aFrame= aFrame/np.max(aFrame)
#sns.violinplot(data= aFrame2, bw = 0.1);plt.show()

nb=find_neighbors(aFrame, k3, metric='euclidean', cores=48)
Idx = nb['idx']; Dist = nb['dist']

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
num_cores = 48
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=16, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
original_dim=32
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
#outfile = source_dir + '/Nowicka2017euclid.npz'
outfile = output_dir + '/Levine32euclid_scaled_no_negative_removed.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)



#Pregnancy
'''
data :CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets
https://scholar.google.com/scholar?biw=1586&bih=926&um=1&ie=UTF-8&lr&cites=8750634913997123816
'''


#data0 = np.genfromtxt(source_dir + "/Gates_PTLG008_1_Unstim.fcs.csv" , names=None, dtype=float,  delimiter=',')
data0 = pd.read_csv(source_dir + "/Gates_PTLG008_1_Unstim.fcs.csv")
gating_channels = ["CD57", "CD19", "CD4", "CD8", "IgD", "CD11c", "CD16", "CD3", "CD38", "CD27", "CD14", "CXCR5", "CCR7", "CD45RA", "CD20", "CD127", "CD33", "CD28", "CD161", "TCRgd", "CD123", "CD56", "HLADR", "CD25", "CD235ab_CD61", "CD66", "CD45", "Tbet", "CD7", "FoxP3", "CD11b"]
MarkersMap = pd.read_csv(source_dir + "/MarkersInfo.csv")
data1 = data0.rename(columns=MarkersMap.set_index('Channels')['Markers'].to_dict())
markers = ['CD235ab_CD61',         'CD45',
                          'CD66',                   'CD7',
               'CD19',       'CD45RA',        'CD11b',          'CD4',
               'CD8a',        'CD11c',        'CD123',         'CREB',
              'STAT5',          'p38',        'TCRgd',        'STAT1',
              'STAT3',           'S6',        'CXCR3',        'CD161',
               'CD33',     'MAPKAPK2',         'Tbet',
              'FoxP3',                 'IkB',         'CD16',
               'NFkB',          'ERK',         'CCR9',         'CD25',
                'CD3',         'CCR7',         'CD15',         'CCR2',
              'HLADR',         'CD14',         'CD56']
data2 = data1[markers]
data3= data2-data2.min()
aFrame = data3.to_numpy()
aFrame.shape
# set negative values to zero
#sns.violinplot(data= aFrame, bw = 0.1);plt.show()
#aFrame[aFrame < 0] = 0
lbls= np.genfromtxt(source_dir + "/Gates_PTLG008_1_Unstim.fcs_LeafPopulations.csv" , names=None, skip_header=1, delimiter=',', dtype='U100')
#randomize order
IDX = np.random.choice(aFrame.shape[0], aFrame.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
aFrame= aFrame[IDX,:]
lbls = lbls[IDX]
len(lbls)

sns.violinplot(data= aFrame, bw = 0.1);plt.show()
# arcsinh transform
aFrame  = np.arcsinh(aFrame/5)

aFrame = aFrame  - aFrame.min(axis=0)
aFrame= aFrame/np.max(aFrame)

#sns.violinplot(data= aFrame[:, :], bw = 0.1);plt.show()
sns.violinplot(data= aFrame, bw = 0.1);plt.show()
#aFrame= aFrame/np.max(aFrame)
# find nearest neighbours
nb=find_neighbors(aFrame, k3, metric='euclidean', cores=16)
Idx = nb['idx']; Dist = nb['dist']
#Dist = Dist[IDX]
#Idx = Idx[IDX]
nrow=Idx.shape[0]
# find nearest neighbours
rk=range(k3)
def singleInput(i):
     nei =  aFrame[Idx[i,:],:]
     di = [np.sqrt(sum(np.square(aFrame[i] - nei[k_i,]))) for k_i in rk]
     return [nei, di, i]
nrow = len(lbls)
inputs = range(nrow)
from joblib import Parallel, delayed
from pathos import multiprocessing
results = Parallel(n_jobs=16, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
original_dim=37
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
perp((Distances[:,0:k3]),       nrow,     original_dim,   neib_weight,          nn,          k3,   Sigma,    4)
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
#outfile = source_dir + '/Nowicka2017euclid.npz'
outfile = output_dir + '/Pr_008_1_Unstim_euclid_scaled_asinh_div5.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma, markers=markers)

#Shekhar
#Get Shekhar 2016 following  https://colab.research.google.com/github/KrishnaswamyLab/SingleCellWorkshop/blob/master/
#/exercises/Deep_Learning/notebooks/02_Answers_Exploratory_analysis_of_single_cell_data_with_SAUCIE.ipynb#scrollTo=KgaZZU8r4drd
import scprep
scprep.io.download.download_google_drive("1GYqmGgv-QY6mRTJhOCE1sHWszRGMFpnf", "/media/grinek/Seagate/CyTOFdataPreprocess/data.pickle.gz")
scprep.io.download.download_google_drive("1q1N1s044FGWzYnQEoYMDJOjdWPm_Uone", "/media/grinek/Seagate/CyTOFdataPreprocess/metadata.pickle.gz")
# somehow script from Colab does not work anymore so unzip gz files by hand
data_raw = pd.read_pickle("/media/grinek/Seagate/CyTOFdataPreprocess/retinal_bipolar_data.pickle")
metadata = pd.read_pickle("/media/grinek/Seagate/CyTOFdataPreprocess/retinal_bipolar_metadata.pickle")
#data_raw = pd.read_pickle("data.pickle.gz")
#data_raw.head()
#metadata = pd.read_pickle("metadata.pickle.gz")
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

nrow=Idx.shape[0]
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
outfile = output_dir + '/Shenkareuclid_shifted.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma)

#Samusik_01
#read and process files generated by R-scipts in [Dimensionality reduction for visualizing single-cell data
#using UMAP, doi:10.1038/nbt.4314]
df = pd.read_csv(source_dir + "/Samusik_01.csv",  nrows=0,  delimiter="\t")
markers =list(df.columns)
data0 = np.genfromtxt(source_dir + "/Samusik_01.csv" , names=None, dtype=float, skip_header=1, delimiter="\t")
aFrame = data0[:,:]
aFrame.shape
aFrame.min(axis=0)
aFrame.max(axis=0)
sns.violinplot(data= aFrame, bw = 0.1);plt.show()
# set negative values to zero and shift minima to zero
#aFrame = aFrame  - aFrame.min(axis=0)

aFrame.min(axis=0)
aFrame.max(axis=0)
lbls= np.genfromtxt(source_dir + "Samusik_01_labels.csv" , names=None, skip_header=0, delimiter="\t", dtype='U100')
#randomize order
IDX = np.random.choice(aFrame.shape[0], aFrame.shape[0], replace=False)
#patient_table = patient_table[IDX,:]
aFrame= aFrame[IDX,:]
lbls = lbls[IDX]
len(lbls)

aFrame = aFrame  - aFrame.min(axis=0)
aFrame= aFrame/np.max(aFrame)
#sns.violinplot(data= aFrame2, bw = 0.1);plt.show()

nb=find_neighbors(aFrame, k3, metric='euclidean', cores=48)
Idx = nb['idx']; Dist = nb['dist']

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
num_cores = 48
#pool = multiprocessing.Pool(num_cores)
results = Parallel(n_jobs=16, verbose=0, backend="threading")(delayed(singleInput, check_pickle=False)(i) for i in inputs)
original_dim=aFrame.shape[1]
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
#outfile = source_dir + '/Nowicka2017euclid.npz'
outfile = output_dir + '/Samusik_01.npz'
np.savez(outfile, aFrame = aFrame, Idx=Idx, lbls=lbls,  Dist=Dist,
         neibALL=neibALL, neib_weight= neib_weight, Sigma=Sigma, markers=markers)

