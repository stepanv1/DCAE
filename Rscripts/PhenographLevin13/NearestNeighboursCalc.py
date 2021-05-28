#To evaluate Phenograph on uncurated data set
from sklearn.neighbors import KDTree
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import os
import sklearn
import fcsparser
import timeit
from multiprocessing import Pool
from contextlib import closing
from itertools import repeat

def calc_jaccard(i, idx):
    """Compute the Jaccard coefficient between i and i's direct neighbors"""
    coefficients = np.fromiter((len(set(idx[i]).intersection(set(idx[j]))) for j in idx[i]), dtype=float)
    coefficients /= (2 * idx.shape[1] - coefficients)
    return idx[i], coefficients

def parallel_jaccard_kernel(idx):
    """Compute Jaccard coefficient between nearest-neighbor sets in parallel
    :param idx: n-by-k integer matrix of k-nearest neighbors
    :return (i, j, s): row indices, column indices, and nonzero values for a sparse adjacency matrix
    """
    n = len(idx)
    with closing(Pool()) as pool:
        jaccard_values = pool.starmap(calc_jaccard, zip(range(n), repeat(idx)))
    return jaccard_values



#Set the variables
ROOT = '/mnt/f/Brinkman group/current/Stepan/RobinsonFlowComparisonALL'
DATA_DIR = "/benchmark_data_sets/"
DATA = 'Samusik_all.fcs'

MANUAL_PHENOGRAPH = "/results/manual/phenoGraph/"
RES_DIR_PHENOGRAPH = "/results/auto/PhenoGraph/"
CALC_NAME = "KdependencySamusik_840k"
SET_NAME = 'Samusik_all'

meta, data = fcsparser.parse(ROOT + DATA_DIR + DATA, meta_data_only=False, reformat_meta=True, output_format='ndarray')
meta['_channels_']
#extract markers
l=data[:, 53]
#data = data[~np.isnan(l), :]
data = data[:, 8:47]



#subset the data


start = timeit.default_timer()
tree = KDTree(data, metric='manhattan')
dist, ind = tree.query(data, k=76, dualtree = True)
ind += 1
stop = timeit.default_timer()
print(stop - start)

if not os.path.exists(ROOT  + '/results/'+ CALC_NAME):
    os.makedirs(ROOT + '/results/'+CALC_NAME)
np.savetxt(ROOT  + '/results/'+ CALC_NAME+"/"+"dist" + SET_NAME + ".txt", newline='\n',
           fmt='%f', X = dist)

if not os.path.exists(ROOT  + '/results/'+ CALC_NAME):
    os.makedirs(ROOT  + '/results/'+ CALC_NAME)
np.savetxt(ROOT  + '/results/'+ CALC_NAME+"/"+"ind" + SET_NAME + ".txt", newline='\n',
           fmt='%f', X = ind)



ind = np.loadtxt(ROOT + '/results/' + CALC_NAME+ "/" + "ind" + SET_NAME + ".txt")
dist = np.loadtxt(ROOT + '/results/' + CALC_NAME+ "/" + "dist" + SET_NAME + ".txt")
ind = ind-1#jaccard distances

pool = Pool(processes=10)
jaccard45 = parallel_jaccard_kernel(ind[:,1:46])

j45list = [list(y) for y in [zip([z]*45, jaccard45[z][0], jaccard45[z][1]) for z in [x for x in range(len(jaccard45))]]]

dt=np.dtype('int, int, float')
j45arrList = [[list(y) for y in x] for x in j45list]

j45 = [np.array(x) for x in j45arrList]
j45st = np.vstack(j45)

np.savetxt(ROOT + '/results/'+ CALC_NAME+"/"+"j45" + SET_NAME + ".txt", newline='\n', fmt=['%d', '%d', '%f'], X = j45st)